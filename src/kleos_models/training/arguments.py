"""Translate KLEOS config into ``transformers.TrainingArguments``.

All the version-dependent naming is handled by :mod:`kleos_models.compat`, so this
module states intent (``warmup_ratio``, ``eval_strategy``) and the shim emits
whatever the installed transformers actually accepts.

Precision resolution is the other job here: ``precision: auto`` becomes bf16 on
compute capability 8.0+ and fp16 below it, and the choice is *returned* so it can
be recorded rather than silently applied. A Colab T4 is compute capability 7.5 and
cannot do bf16, so this is a real branch on the canonical training environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kleos_models.compat import build_training_arguments_kwargs, require_transformers
from kleos_models.config import Precision, TrainingConfig
from kleos_models.errors import ConfigError
from kleos_models.logging_utils import get_logger
from kleos_models.models.feasibility import GPUInfo, probe_gpu

logger = get_logger(__name__)

#: Optimizers that require bitsandbytes.
_BNB_OPTIMIZERS = frozenset(
    {
        "adamw_bnb_8bit",
        "adamw_8bit",
        "paged_adamw_8bit",
        "paged_adamw_32bit",
        "lion_8bit",
        "paged_lion_8bit",
    }
)


def resolve_precision(
    training: TrainingConfig, *, gpu: GPUInfo | None = None
) -> tuple[bool, bool, str]:
    """Resolve the precision setting into ``(bf16, fp16, explanation)``."""
    gpu = gpu or probe_gpu()

    if training.precision is Precision.AUTO:
        if not gpu.available:
            return False, False, "auto → fp32 (no CUDA device; CPU training is fp32)"
        if gpu.bf16_supported:
            return True, False, "auto → bf16 (compute capability >= 8.0)"
        return (
            False,
            True,
            (f"auto → fp16 (compute capability {gpu.capability_string} cannot do bf16)"),
        )

    if training.precision is Precision.BF16:
        if gpu.available and not gpu.bf16_supported:
            raise ConfigError(
                f"training.precision='bf16' but this GPU ({gpu.name}, compute "
                f"capability {gpu.capability_string}) does not support bfloat16.",
                details={"gpu": gpu.render()},
                suggestions=[
                    "Use precision: auto to pick correctly for the assigned GPU.",
                    "Or set precision: fp16 explicitly.",
                    "Colab free-tier T4 GPUs are compute capability 7.5.",
                ],
            )
        return True, False, "explicitly configured as bf16"

    if training.precision is Precision.FP16:
        return False, True, "explicitly configured as fp16"

    return False, False, "explicitly configured as fp32"


def resolve_optimizer(
    training: TrainingConfig, *, has_bitsandbytes: bool | None = None
) -> tuple[str, str | None]:
    """Resolve the optimizer, falling back when bitsandbytes is unavailable.

    Returns:
        ``(optimizer_name, adjustment_reason_or_None)``. A non-None reason means
        the value changed and must be recorded in the manifest.
    """
    if has_bitsandbytes is None:
        from kleos_models.compat import package_version

        has_bitsandbytes = package_version("bitsandbytes") is not None

    if training.optim in _BNB_OPTIMIZERS and not has_bitsandbytes:
        fallback = "adamw_torch"
        reason = (
            f"optimizer {training.optim!r} requires bitsandbytes, which is not "
            f"installed; falling back to {fallback!r}. This uses more memory for "
            "optimizer state."
        )
        return fallback, reason

    return training.optim, None


def build_training_arguments(
    training: TrainingConfig,
    *,
    output_dir: Path | str,
    run_name: str | None = None,
    gpu: GPUInfo | None = None,
    has_eval_dataset: bool = False,
    extra: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build ``TrainingArguments`` from the KLEOS training config.

    Returns:
        ``(training_arguments, metadata)``. Metadata records precision and
        optimizer resolution plus every compat translation, and lands in the
        manifest so the effective configuration is always visible.
    """
    transformers = require_transformers()

    bf16, fp16, precision_note = resolve_precision(training, gpu=gpu)
    optimizer, optimizer_note = resolve_optimizer(training)

    logger.info("Precision: %s", precision_note)
    if optimizer_note:
        logger.warning("Optimizer: %s", optimizer_note)

    # Evaluation cannot be requested without an eval dataset; that would fail deep
    # inside the Trainer with a confusing message.
    eval_strategy = training.eval_strategy
    eval_note: str | None = None
    if eval_strategy != "no" and not has_eval_dataset:
        eval_note = (
            f"eval_strategy={eval_strategy!r} was requested but no validation split "
            "is available; evaluation is disabled for this run."
        )
        logger.warning("%s", eval_note)
        eval_strategy = "no"

    load_best = training.load_best_model_at_end and eval_strategy != "no"

    requested: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": run_name or training.run_name or Path(output_dir).name,
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "learning_rate": training.learning_rate,
        "lr_scheduler_type": training.lr_scheduler_type,
        "warmup_ratio": training.warmup_ratio,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "max_grad_norm": training.max_grad_norm,
        "weight_decay": training.weight_decay,
        "optim": optimizer,
        "adam_beta1": training.adam_beta1,
        "adam_beta2": training.adam_beta2,
        "adam_epsilon": training.adam_epsilon,
        "gradient_checkpointing": training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bf16,
        "fp16": fp16,
        "seed": training.seed,
        "data_seed": training.data_seed if training.data_seed is not None else training.seed,
        "logging_steps": training.logging_steps,
        "logging_first_step": True,
        "save_strategy": training.save_strategy,
        "save_steps": training.save_steps,
        "save_total_limit": max(1, training.save_total_limit),
        "eval_strategy": eval_strategy,
        "eval_steps": training.eval_steps,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": training.metric_for_best_model,
        "greater_is_better": training.greater_is_better,
        "dataloader_num_workers": training.dataloader_num_workers,
        "group_by_length": training.group_by_length,
        "report_to": training.report_to or "none",
        "remove_unused_columns": False,  # our collator needs the raw columns
        "label_names": ["labels"],
        "disable_tqdm": False,
    }
    if extra:
        requested.update(extra)

    kwargs, compat_notes = build_training_arguments_kwargs(requested)

    try:
        arguments = transformers.TrainingArguments(**kwargs)
    except TypeError as exc:
        raise ConfigError(
            "TrainingArguments rejected the generated arguments.",
            details={
                "transformers_version": getattr(transformers, "__version__", "unknown"),
                "error": str(exc),
                "compat_notes": "; ".join(compat_notes) or "(none)",
            },
            suggestions=[
                "The installed transformers may have changed its API again.",
                'Install a tested version: pip install "transformers>=4.56,<6"',
                "Then report the mismatch so kleos_models/compat.py can be updated.",
            ],
        ) from exc

    metadata = {
        "precision": {"bf16": bf16, "fp16": fp16, "reason": precision_note},
        "optimizer": {"resolved": optimizer, "adjustment": optimizer_note},
        "eval": {"strategy": eval_strategy, "adjustment": eval_note},
        "compat_translations": compat_notes,
        "effective_batch_size": training.effective_batch_size,
    }
    return arguments, metadata
