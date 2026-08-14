"""The QLoRA training pipeline (spec sections 13, 15, 45).

This is a real training pipeline. It loads the model, quantizes it, attaches a
LoRA adapter, tokenizes conversations with assistant-only loss masking, runs a
genuine ``transformers.Trainer`` loop, checkpoints, and saves adapter weights.
Nothing here prints "Training model..." and returns.

The orchestration follows spec section 15 step for step, and each stage updates
the experiment manifest so an interrupted run still leaves a truthful record.

``transformers``/``torch`` are imported inside functions, keeping the module
importable in a light environment for inspection and testing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.config import ExperimentConfig
from kleos_models.constants import (
    ADAPTER_DIRNAME,
    EFFECTIVE_CONFIG_FILENAME,
    IGNORE_INDEX,
    METRICS_FILENAME,
    STRUCTURED_LOG_FILENAME,
    TOKENIZER_DIRNAME,
    TRAINING_LOG_FILENAME,
)
from kleos_models.data.formatting import ConversationFormatter
from kleos_models.data.loaders import DatasetBundle
from kleos_models.data.schemas import TrainingExample
from kleos_models.errors import KleosError
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.logging_utils import EventLogger, get_logger, log_stage
from kleos_models.models.loading import LoadedModel, load_model
from kleos_models.models.peft_setup import PeftSetupResult, attach_lora
from kleos_models.training.arguments import build_training_arguments
from kleos_models.training.callbacks import (
    CheckpointMetadataCallback,
    EarlyOOMGuardCallback,
    MemoryMonitorCallback,
    ProgressCallback,
    StructuredLoggingCallback,
)
from kleos_models.training.checkpointing import resolve_resume_path
from kleos_models.training.memory import (
    diagnose_oom,
    free_memory,
    is_oom_error,
    render_environment_report,
    reset_peak_memory,
    snapshot_memory,
)

logger = get_logger(__name__)


class PaddingCollator:
    """Pad a batch of variable-length examples.

    Written rather than reused because the standard causal-LM collator builds
    labels by copying ``input_ids``, which would discard the assistant-only
    masking the formatter computed. Here labels are padded with
    ``IGNORE_INDEX`` so padded positions contribute nothing to the loss.
    """

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            remainder = longest % self.pad_to_multiple_of
            if remainder:
                longest += self.pad_to_multiple_of - remainder

        input_ids, labels, attention = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            lbl = list(feature["labels"])
            mask = list(feature.get("attention_mask") or [1] * len(ids))
            padding = longest - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            labels.append(lbl + [IGNORE_INDEX] * padding)
            attention.append(mask + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


class ListDataset:
    """Minimal map-style dataset over formatted examples.

    Avoids a hard dependency on ``datasets`` for the common in-memory case, which
    keeps Colab setup simpler and start-up faster.
    """

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


@dataclass
class TrainingResult:
    """Everything a completed run produced."""

    experiment_id: str
    output_dir: Path
    adapter_path: Path
    tokenizer_path: Path
    manifest: ExperimentManifest
    metrics: dict[str, Any] = field(default_factory=dict)
    formatting_stats: dict[str, Any] = field(default_factory=dict)
    peft_summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "=" * 72,
            "Training complete",
            "=" * 72,
            f"  experiment id : {self.experiment_id}",
            f"  output dir    : {self.output_dir}",
            f"  adapter       : {self.adapter_path}",
            f"  tokenizer     : {self.tokenizer_path}",
            "",
            "  Artifacts",
        ]
        lines.extend(f"    {name:<20} {path}" for name, path in sorted(self.artifacts.items()))
        if self.metrics:
            lines.extend(["", "  Metrics"])
            for key, value in self.metrics.items():
                rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"    {key:<28} {rendered}")
        lines.extend(
            [
                "",
                "  Next steps",
                f"    Evaluate : python scripts/evaluate.py --config <config> "
                f"--adapter {self.adapter_path}",
                "    Compare  : python scripts/compare.py --base <base.json> --finetuned <ft.json>",
                "",
            ]
        )
        return "\n".join(lines)


def format_split(
    examples: Sequence[TrainingExample],
    formatter: ConversationFormatter,
    *,
    split_name: str,
) -> tuple[ListDataset, dict[str, Any]]:
    """Tokenize a split into a trainer-ready dataset."""
    formatted, stats = formatter.format_dataset(examples)
    if not formatted:
        raise KleosError(
            f"No usable examples remain in the {split_name} split after formatting.",
            details=stats.summary(),
            suggestions=[
                "Every example produced zero supervised tokens.",
                "Check that assistant messages are non-empty and that "
                "model.max_seq_length is large enough to include them.",
                "Inspect masking: python scripts/inspect_dataset.py --dataset <dir> --show-masking",
            ],
        )
    rows = [item.to_dict() for item in formatted]
    summary = stats.summary()
    logger.info("Formatted %s split: %s", split_name, summary)
    return ListDataset(rows), summary


def _sample_batch(
    dataset: ListDataset, collator: PaddingCollator, *, size: int = 2
) -> dict[str, Any]:
    """Build a small batch for gradient verification."""
    rows = [dataset[i] for i in range(min(size, len(dataset)))]
    return collator(rows)


def run_training(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    manifest: ExperimentManifest,
    *,
    resume_from_checkpoint: str | None = None,
    verify_gradients: bool = True,
    loaded: LoadedModel | None = None,
) -> TrainingResult:
    """Execute the full training pipeline (spec section 15).

    Args:
        config: Resolved experiment configuration.
        bundle: Loaded and validated dataset.
        manifest: Manifest to update as the run progresses. Saved at every stage
            so an interrupted run still leaves a record.
        resume_from_checkpoint: ``None``, ``"auto"``, or an explicit path.
        verify_gradients: Run a forward/backward check before the real loop, so a
            silently-untrainable setup fails in seconds rather than hours.
        loaded: Pre-loaded model, used by tests to inject a tiny model.

    Returns:
        A :class:`TrainingResult`.

    Raises:
        KleosError: with actionable diagnostics on any failure. The manifest is
            marked failed and saved before the exception propagates.
    """
    from transformers import Trainer

    from kleos_models.compat import trainer_tokenizer_kwarg

    output_dir = Path(config.training.output_dir) / manifest.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = EventLogger(output_dir / STRUCTURED_LOG_FILENAME, experiment_id=manifest.experiment_id)
    stage = "setup"

    try:
        # --- 1-3. environment report -------------------------------------
        events.set_stage("environment")
        report = render_environment_report(config.model, config.training)
        logger.info("\n%s", report)
        (output_dir / "environment.txt").write_text(report, encoding="utf-8")
        manifest.mark_started()
        manifest.save(output_dir)

        # --- 4-6. model, tokenizer, quantization, PEFT --------------------
        stage = "model_loading"
        events.set_stage(stage)
        with log_stage(logger, "Loading base model"):
            if loaded is None:
                loaded = load_model(
                    config.model,
                    reasoning_mode=config.model.reasoning.default_mode,
                    for_training=True,
                )
            manifest.model = loaded.describe()
            manifest.quantization = loaded.quantization_metadata
            manifest.reasoning_mode = loaded.reasoning_mode.value
            manifest.save(output_dir)

        stage = "peft_setup"
        events.set_stage(stage)
        with log_stage(logger, "Attaching LoRA adapter"):
            peft_result: PeftSetupResult = attach_lora(
                loaded.model,
                config.model,
                config.training,
                loaded.adapter,
                is_quantized=config.model.quantization.enabled,
            )
            model = peft_result.model
            manifest.lora = peft_result.to_dict()
            manifest.save(output_dir)
            events.emit("peft_attached", **peft_result.to_dict())

        # --- 7-9. dataset formatting --------------------------------------
        stage = "data_formatting"
        events.set_stage(stage)
        with log_stage(logger, "Formatting dataset"):
            formatter = ConversationFormatter(
                loaded.tokenizer,
                max_seq_length=config.model.max_seq_length,
                template_kwargs=loaded.chat_template_kwargs(),
                strip_reasoning=config.model.reasoning.strip_thinking_from_targets,
            )
            train_dataset, train_stats = format_split(bundle.train, formatter, split_name="train")
            eval_dataset = None
            eval_stats: dict[str, Any] = {}
            if bundle.validation:
                eval_dataset, eval_stats = format_split(
                    bundle.validation, formatter, split_name="validation"
                )
            manifest.metrics["formatting"] = {"train": train_stats, "validation": eval_stats}
            manifest.save(output_dir)

        collator = PaddingCollator(pad_token_id=loaded.tokenizer.pad_token_id)

        # --- gradient verification ----------------------------------------
        if verify_gradients:
            stage = "gradient_check"
            events.set_stage(stage)
            with log_stage(logger, "Verifying gradients reach the adapter"):
                from kleos_models.models.peft_setup import verify_gradients_flow

                batch = _sample_batch(train_dataset, collator)
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                diagnostics = verify_gradients_flow(model, batch)
                logger.info(
                    "Gradient check passed: loss=%.4f, %d/%d LoRA tensors have non-zero gradients.",
                    diagnostics["loss"],
                    diagnostics["lora_tensors_with_nonzero_grad"],
                    diagnostics["lora_tensors"],
                )
                events.emit("gradient_check", **diagnostics)
                manifest.metrics["gradient_check"] = diagnostics
                free_memory()

        # --- 10. trainer ---------------------------------------------------
        stage = "trainer_init"
        events.set_stage(stage)
        arguments, argument_metadata = build_training_arguments(
            config.training,
            output_dir=output_dir,
            run_name=manifest.experiment_id,
            has_eval_dataset=eval_dataset is not None,
        )
        manifest.effective_config["training_arguments"] = argument_metadata
        for note in argument_metadata["compat_translations"]:
            manifest.note(f"transformers compat: {note}")
        if argument_metadata["optimizer"]["adjustment"]:
            manifest.add_adjustment(
                "training.optim",
                config.training.optim,
                argument_metadata["optimizer"]["resolved"],
                argument_metadata["optimizer"]["adjustment"],
            )
        if argument_metadata["eval"]["adjustment"]:
            manifest.add_adjustment(
                "training.eval_strategy",
                config.training.eval_strategy,
                argument_metadata["eval"]["strategy"],
                argument_metadata["eval"]["adjustment"],
            )

        callbacks = [
            StructuredLoggingCallback(events, experiment_id=manifest.experiment_id),
            ProgressCallback(log_every=config.training.logging_steps),
            MemoryMonitorCallback(),
            EarlyOOMGuardCallback(),
            CheckpointMetadataCallback(
                experiment_id=manifest.experiment_id,
                metadata={
                    "model": config.model.base_model,
                    # The manifest is the run's authoritative record; the bundle
                    # may be unversioned when data was supplied as loose files.
                    "dataset_version": manifest.dataset_version or bundle.version,
                    "dataset_hash": manifest.dataset_hash,
                    "config_hash": config.config_hash,
                    "seed": config.seed,
                },
                output_dir=output_dir,
                save_total_limit=config.training.save_total_limit,
                manifest=manifest,
            ),
        ]

        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
            **trainer_tokenizer_kwarg(loaded.tokenizer),
        )

        # --- 11. train ------------------------------------------------------
        stage = "training"
        events.set_stage(stage)
        resume_path, checkpoint_info = resolve_resume_path(
            resume_from_checkpoint or config.training.resume_from_checkpoint,
            output_dir,
        )
        if checkpoint_info is not None:
            manifest.note(
                f"Resumed from {checkpoint_info.path.name} at step {checkpoint_info.step}."
            )
            events.emit("resume", checkpoint=str(checkpoint_info.path), step=checkpoint_info.step)

        reset_peak_memory()
        manifest.save(output_dir)

        with log_stage(logger, "Training"):
            train_output = trainer.train(resume_from_checkpoint=resume_path)

        metrics: dict[str, Any] = dict(train_output.metrics)
        peak = snapshot_memory()
        if peak:
            metrics["peak_memory_gb"] = round(peak.max_allocated_gb, 2)

        # --- 12. evaluate ---------------------------------------------------
        if eval_dataset is not None:
            stage = "evaluation"
            events.set_stage(stage)
            with log_stage(logger, "Evaluating on the validation split"):
                metrics.update(trainer.evaluate())

        # --- 13-16. save artifacts -----------------------------------------
        stage = "saving"
        events.set_stage(stage)
        adapter_path = output_dir / ADAPTER_DIRNAME
        tokenizer_path = output_dir / TOKENIZER_DIRNAME

        with log_stage(logger, "Saving adapter and tokenizer"):
            trainer.model.save_pretrained(str(adapter_path))
            loaded.tokenizer.save_pretrained(str(tokenizer_path))
            config.save(output_dir / EFFECTIVE_CONFIG_FILENAME)

        import json

        (output_dir / METRICS_FILENAME).write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )

        artifacts = {
            "adapter": str(adapter_path),
            "tokenizer": str(tokenizer_path),
            "config": str(output_dir / EFFECTIVE_CONFIG_FILENAME),
            "metrics": str(output_dir / METRICS_FILENAME),
            "manifest": str(output_dir / "manifest.json"),
            "events": str(output_dir / STRUCTURED_LOG_FILENAME),
            "environment": str(output_dir / "environment.txt"),
            "training_log": str(output_dir / TRAINING_LOG_FILENAME),
        }
        for name, path in artifacts.items():
            manifest.add_artifact(name, path)

        # --- 17. manifest ---------------------------------------------------
        manifest.mark_completed(metrics)
        manifest.save(output_dir)
        events.emit(
            "run_completed", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )

        _write_run_readme(output_dir, config, manifest, metrics)

        return TrainingResult(
            experiment_id=manifest.experiment_id,
            output_dir=output_dir,
            adapter_path=adapter_path,
            tokenizer_path=tokenizer_path,
            manifest=manifest,
            metrics=metrics,
            formatting_stats={"train": train_stats, "validation": eval_stats},
            peft_summary=peft_result.to_dict(),
            artifacts=artifacts,
        )

    except BaseException as exc:
        # Spec section 36: failed runs stay on disk with their manifest.
        manifest.mark_failed(exc, stage=stage)
        try:
            manifest.save(output_dir)
            events.emit("run_failed", stage=stage, error=type(exc).__name__)
        except Exception:  # pragma: no cover - disk problems during teardown
            logger.exception("Could not persist the failure manifest.")

        if is_oom_error(exc):
            raise diagnose_oom(exc, config.model, config.training, stage=stage) from exc
        raise


def _write_run_readme(
    output_dir: Path,
    config: ExperimentConfig,
    manifest: ExperimentManifest,
    metrics: dict[str, Any],
) -> Path:
    """Write a README into the run directory (spec section 34)."""
    lines = [
        f"# Run `{manifest.experiment_id}`",
        "",
        "Generated by `scripts/train.py`. This directory is the complete record of",
        "one training run.",
        "",
        "## Provenance",
        "",
        f"- base model: `{config.model.base_model}` (revision `{config.model.revision}`)",
        f"- model family: {config.model.family}",
        f"- dataset version: `{manifest.dataset_version}`",
        f"- dataset hash: `{manifest.dataset_hash or 'n/a'}`",
        f"- config hash: `{manifest.config_hash}`",
        f"- seed: {manifest.seed}",
        f"- git commit: `{manifest.git.get('commit', 'unavailable')}`",
        "",
        "## Contents",
        "",
        "| Path | What it is |",
        "| --- | --- |",
        "| `adapter/` | Trained LoRA weights (this is the model artifact) |",
        "| `tokenizer/` | Tokenizer used for training |",
        "| `config.yaml` | Fully resolved effective configuration |",
        "| `manifest.json` | Complete run manifest |",
        "| `metrics.json` | Final metrics |",
        "| `events.jsonl` | Structured event stream |",
        "| `environment.txt` | Pre-flight hardware and software report |",
        "| `checkpoint-*/` | Resumable training checkpoints |",
        "",
        "## Metrics",
        "",
    ]
    if metrics:
        lines.extend(["| Metric | Value |", "| --- | ---: |"])
        for key, value in metrics.items():
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| `{key}` | {rendered} |")
    else:
        lines.append("_No metrics recorded._")

    if manifest.adjustments:
        lines.extend(
            [
                "",
                "## Automatic adjustments",
                "",
                "The requested configuration was changed automatically. Take these into",
                "account before comparing this run with another.",
                "",
                "| Field | From | To | Why |",
                "| --- | --- | --- | --- |",
            ]
        )
        for adjustment in manifest.adjustments:
            lines.append(
                f"| `{adjustment['field']}` | {adjustment['original']} | "
                f"{adjustment['adjusted']} | {adjustment['reason']} |"
            )

    lines.extend(
        [
            "",
            "## Interpreting this run",
            "",
            "A training loss curve is not a result. Whether this adapter is better than",
            "the base model is decided by `scripts/evaluate.py` and `scripts/compare.py`",
            "on a held-out split, not by the loss reaching a low number.",
            "",
        ]
    )

    path = output_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
