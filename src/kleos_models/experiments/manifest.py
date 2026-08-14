"""Experiment manifests (spec sections 16, 34, 36, 38).

Every run writes a manifest. Not optionally, and not only on success — a run that
crashes writes a manifest with ``status="failed"`` and the error recorded.

Spec section 36 is explicit that failed experiments must be recorded rather than
quietly deleted, because "fine-tuning did not improve performance" is a valid
result and the record of how it was obtained is what makes it credible.

A manifest identifies a run by everything needed to reproduce it:
model + dataset version + config hash + seed + code commit + environment, plus the
model-family details that keep incompatible runs from being pooled.

This module imports no torch.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kleos_models.constants import MANIFEST_FILENAME
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


class RunStatus(str, Enum):
    """Lifecycle state of a run."""

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ExperimentManifest(BaseModel):
    """The complete, mandatory record of one experiment run."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    # --- identity ----------------------------------------------------------
    experiment_id: str
    name: str = "kleos-experiment"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    hypothesis: str | None = None
    task: str | None = None
    kind: str = Field(default="training", description="training | evaluation | comparison")
    status: RunStatus = RunStatus.STARTED

    # --- reproducibility chain ---------------------------------------------
    config_hash: str = ""
    seed: int = 42
    seeding: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str = "unversioned"
    dataset_hash: str | None = None
    dataset_source: str | None = None
    dataset_counts: dict[str, int] = Field(default_factory=dict)
    split_strategy: str | None = None

    # --- model identity (spec section 38) ----------------------------------
    model: dict[str, Any] = Field(default_factory=dict)
    lora: dict[str, Any] = Field(default_factory=dict)
    quantization: dict[str, Any] = Field(default_factory=dict)
    reasoning_mode: str | None = None

    # --- run context --------------------------------------------------------
    hardware: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    git: dict[str, Any] = Field(default_factory=dict)

    # --- timings ------------------------------------------------------------
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None

    # --- what happened ------------------------------------------------------
    effective_config: dict[str, Any] = Field(default_factory=dict)
    adjustments: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Automatic changes to the requested configuration. Present so a "
            "fallback can never silently invalidate a comparison."
        ),
    )
    feasibility: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)

    # -- lifecycle ----------------------------------------------------------

    def mark_started(self) -> ExperimentManifest:
        self.started_at = datetime.now(UTC).isoformat()
        self.status = RunStatus.RUNNING
        return self

    def mark_completed(self, metrics: dict[str, Any] | None = None) -> ExperimentManifest:
        self.finished_at = datetime.now(UTC).isoformat()
        self.status = RunStatus.COMPLETED
        if metrics:
            self.metrics.update(metrics)
        self._compute_duration()
        return self

    def mark_failed(self, error: BaseException, *, stage: str = "unknown") -> ExperimentManifest:
        """Record a failure. The run stays on disk as evidence."""
        self.finished_at = datetime.now(UTC).isoformat()
        self.status = RunStatus.FAILED
        self.error = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
        self._compute_duration()
        return self

    def mark_interrupted(self) -> ExperimentManifest:
        self.finished_at = datetime.now(UTC).isoformat()
        self.status = RunStatus.INTERRUPTED
        self._compute_duration()
        return self

    def _compute_duration(self) -> None:
        if not (self.started_at and self.finished_at):
            return
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(self.finished_at)
        self.duration_seconds = round((finished - started).total_seconds(), 2)

    def add_adjustment(self, field: str, original: Any, adjusted: Any, reason: str) -> None:
        """Record an automatic configuration change."""
        self.adjustments.append(
            {
                "field": field,
                "original": original,
                "adjusted": adjusted,
                "reason": reason,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )

    def add_checkpoint(self, path: str | Path, step: int, **extra: Any) -> None:
        self.checkpoints.append(
            {
                "path": str(path),
                "step": step,
                "saved_at": datetime.now(UTC).isoformat(),
                **extra,
            }
        )

    def add_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = str(path)

    def note(self, message: str) -> None:
        self.notes.append(message)

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path | str, *, filename: str = MANIFEST_FILENAME) -> Path:
        """Write the manifest, atomically enough to survive an interrupted Colab."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> ExperimentManifest:
        """Read a manifest from a file or a run directory."""
        source = Path(path)
        if source.is_dir():
            source = source / MANIFEST_FILENAME
        return cls.model_validate(json.loads(source.read_text(encoding="utf-8")))

    def summary(self) -> str:
        """One-screen human summary."""
        lines = [
            "=" * 72,
            f"Experiment {self.experiment_id}",
            "=" * 72,
            f"  status        : {self.status.value}",
            f"  kind          : {self.kind}",
            f"  task          : {self.task or '-'}",
            f"  model         : {self.model.get('base_model', '-')}",
            f"  family        : {self.model.get('family', '-')}",
            f"  reasoning     : {self.reasoning_mode or '-'}",
            f"  dataset       : {self.dataset_version}",
            f"  config hash   : {self.config_hash[:16]}",
            f"  seed          : {self.seed}",
            f"  git commit    : {self.git.get('commit', 'unavailable')[:12]}",
            f"  duration      : "
            f"{f'{self.duration_seconds:.1f}s' if self.duration_seconds else '-'}",
        ]
        if self.adjustments:
            lines.append(f"  adjustments   : {len(self.adjustments)} (see manifest.json)")
        if self.metrics:
            lines.append("")
            lines.append("  Metrics")
            for key, value in list(self.metrics.items())[:12]:
                rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"    {key:<28} {rendered}")
        if self.error:
            lines.extend(
                [
                    "",
                    f"  FAILED at stage {self.error['stage']}: "
                    f"{self.error['type']}: {self.error['message'][:200]}",
                ]
            )
        return "\n".join(lines)


def generate_experiment_id(
    *, name: str = "kleos", model_name: str | None = None, config_hash: str | None = None
) -> str:
    """Build a readable, sortable, unique experiment id.

    Format: ``<name>-<model>-<YYYYmmdd-HHMMSS>-<confighash|random>``. Time-ordered
    so runs sort chronologically, and carrying the config hash so two runs of the
    same configuration are visibly related.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = config_hash[:8] if config_hash else uuid.uuid4().hex[:8]
    parts = [name]
    if model_name:
        parts.append(model_name)
    parts.extend([timestamp, suffix])
    return "-".join(part.replace(" ", "_").replace("/", "_") for part in parts)


def build_manifest(
    config: Any,
    *,
    kind: str = "training",
    experiment_id: str | None = None,
    dataset_version: str = "unversioned",
    dataset_hash: str | None = None,
    dataset_counts: dict[str, int] | None = None,
    model_description: dict[str, Any] | None = None,
    root: Path | str | None = None,
) -> ExperimentManifest:
    """Create a manifest pre-populated from an :class:`ExperimentConfig`.

    Args:
        config: The resolved experiment configuration.
        kind: ``training``, ``evaluation`` or ``comparison``.
        experiment_id: Override the generated id.
        dataset_version: Version string from the dataset manifest.
        dataset_hash: Content digest of the loaded data.
        dataset_counts: Per-split counts.
        model_description: Output of ``ModelFamilyAdapter.describe()``.
        root: Repository root for git capture.
    """
    from kleos_models.experiments.environment import capture_environment

    snapshot = capture_environment(root)
    config_hash = config.config_hash

    manifest = ExperimentManifest(
        experiment_id=experiment_id
        or generate_experiment_id(
            name=config.name, model_name=config.model.name, config_hash=config_hash
        ),
        name=config.name,
        description=config.description,
        tags=list(config.tags),
        hypothesis=config.hypothesis,
        task=config.task,
        kind=kind,
        config_hash=config_hash,
        seed=config.seed,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        dataset_source=str(config.dataset.path) if config.dataset and config.dataset.path else None,
        dataset_counts=dataset_counts or {},
        split_strategy=config.dataset.split.strategy if config.dataset else None,
        model=model_description or _describe_model_from_config(config.model),
        lora=config.model.lora.model_dump(mode="json"),
        quantization=config.model.quantization.model_dump(mode="json"),
        reasoning_mode=config.model.reasoning.default_mode.value,
        hardware=snapshot.gpu,
        software=snapshot.libraries,
        environment={
            "python": snapshot.python_version,
            "platform": snapshot.platform,
            "machine": snapshot.machine,
            "in_colab": snapshot.in_colab,
            "hostname_hash": snapshot.hostname_hash,
            "flags": snapshot.environment_flags,
        },
        git=snapshot.git,
        effective_config=config.canonical_dict(),
    )

    if not snapshot.git.get("available"):
        manifest.note(
            "Git commit unavailable: "
            f"{snapshot.git.get('reason', 'unknown')}. This run cannot be traced to "
            "a code revision."
        )
    elif snapshot.git.get("dirty"):
        manifest.note(
            "Working tree was dirty at run time; the recorded commit does not fully "
            "describe the code that ran."
        )

    return manifest


def _describe_model_from_config(model_config: Any) -> dict[str, Any]:
    """Fallback model description when no adapter was constructed."""
    return {
        "name": model_config.name,
        "family": model_config.family,
        "base_model": model_config.base_model,
        "revision": model_config.revision,
        "tokenizer": model_config.tokenizer_id,
        "model_type": model_config.model_type,
        "architecture": model_config.architecture,
        "parameter_count": model_config.parameter_count,
        "active_parameter_count": model_config.active_parameter_count,
        "is_moe": model_config.is_moe,
        "is_multimodal": model_config.is_multimodal,
        "max_seq_length": model_config.max_seq_length,
    }
