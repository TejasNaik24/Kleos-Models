"""Training layer: QLoRA pipeline, arguments, callbacks, checkpointing, memory.

Requires the ``[train]`` extra to run, but stays importable without it — torch and
transformers are imported inside functions.
"""

from kleos_models.training.arguments import (
    build_training_arguments,
    resolve_optimizer,
    resolve_precision,
)
from kleos_models.training.checkpointing import (
    CheckpointInfo,
    discover_checkpoints,
    find_latest_checkpoint,
    prune_checkpoints,
    resolve_resume_path,
    summarize_checkpoints,
    validate_checkpoint,
)
from kleos_models.training.memory import (
    MemorySnapshot,
    diagnose_oom,
    is_oom_error,
    render_environment_report,
    snapshot_memory,
)
from kleos_models.training.trainer import (
    ListDataset,
    PaddingCollator,
    TrainingResult,
    format_split,
    run_training,
)

__all__ = [
    "CheckpointInfo",
    "ListDataset",
    "MemorySnapshot",
    "PaddingCollator",
    "TrainingResult",
    "build_training_arguments",
    "diagnose_oom",
    "discover_checkpoints",
    "find_latest_checkpoint",
    "format_split",
    "is_oom_error",
    "prune_checkpoints",
    "render_environment_report",
    "resolve_optimizer",
    "resolve_precision",
    "resolve_resume_path",
    "run_training",
    "snapshot_memory",
    "summarize_checkpoints",
    "validate_checkpoint",
]
