"""Checkpoint discovery, resume and retention (spec section 33).

Written for the reality of free Colab: the runtime can disappear at any moment,
usually without warning and often overnight. Three consequences shape this module.

1. **Resume must be trivial.** ``--resume-from-checkpoint auto`` finds the newest
   valid checkpoint without the user needing to remember a step number.
2. **A checkpoint must never be assumed valid.** A run killed mid-save leaves a
   partial directory; resuming from it fails confusingly. Checkpoints are
   validated before being offered.
3. **Retention never empties the directory.** ``save_total_limit`` is honoured with
   a hard floor of one, so cleanup cannot leave a user with nothing to resume from.

This module imports no torch.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_models.constants import CHECKPOINT_PREFIX
from kleos_models.errors import CheckpointError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

_CHECKPOINT_PATTERN = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}(\d+)$")

#: Files a transformers Trainer checkpoint must contain to be resumable.
_REQUIRED_MARKERS = ("trainer_state.json",)

#: At least one of these must exist for the weights to be present.
_WEIGHT_MARKERS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
)


@dataclass
class CheckpointInfo:
    """A discovered checkpoint."""

    path: Path
    step: int
    valid: bool
    reason: str = ""
    modified_at: float = 0.0
    size_mb: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "step": self.step,
            "valid": self.valid,
            "reason": self.reason,
            "size_mb": round(self.size_mb, 1),
            "modified_at": datetime.fromtimestamp(self.modified_at, tz=UTC).isoformat()
            if self.modified_at
            else None,
        }


def _directory_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:  # pragma: no cover - race with cleanup
                continue
    return total / (1024 * 1024)


def validate_checkpoint(path: Path) -> tuple[bool, str]:
    """Check a checkpoint directory looks complete.

    Returns:
        ``(is_valid, reason)``.
    """
    if not path.is_dir():
        return False, "not a directory"

    missing = [marker for marker in _REQUIRED_MARKERS if not (path / marker).exists()]
    if missing:
        return False, f"missing {', '.join(missing)} (likely an interrupted save)"

    if not any((path / marker).exists() for marker in _WEIGHT_MARKERS):
        return False, "no model or adapter weight file present"

    state_file = path / "trainer_state.json"
    try:
        json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"trainer_state.json is unreadable ({exc})"

    return True, "ok"


def discover_checkpoints(
    output_dir: Path | str, *, include_invalid: bool = False
) -> list[CheckpointInfo]:
    """List checkpoints under an output directory, newest step first."""
    directory = Path(output_dir)
    if not directory.exists():
        return []

    found: list[CheckpointInfo] = []
    for candidate in directory.iterdir():
        if not candidate.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.match(candidate.name)
        if not match:
            continue
        valid, reason = validate_checkpoint(candidate)
        if not valid and not include_invalid:
            logger.warning("Ignoring incomplete checkpoint %s: %s", candidate.name, reason)
            continue
        found.append(
            CheckpointInfo(
                path=candidate,
                step=int(match.group(1)),
                valid=valid,
                reason=reason,
                modified_at=candidate.stat().st_mtime,
                size_mb=_directory_size_mb(candidate),
                metadata=read_checkpoint_metadata(candidate),
            )
        )

    found.sort(key=lambda c: c.step, reverse=True)
    return found


def find_latest_checkpoint(output_dir: Path | str) -> CheckpointInfo | None:
    """Newest valid checkpoint, or ``None``."""
    checkpoints = discover_checkpoints(output_dir)
    return checkpoints[0] if checkpoints else None


def resolve_resume_path(
    resume: str | None, output_dir: Path | str
) -> tuple[str | None, CheckpointInfo | None]:
    """Resolve a ``--resume-from-checkpoint`` value into a concrete path.

    Args:
        resume: ``None``, ``"auto"``, or an explicit path.
        output_dir: Where to search when ``auto``.

    Returns:
        ``(path_or_None, checkpoint_info_or_None)``.

    Raises:
        CheckpointError: when an explicit path is missing or invalid.
    """
    if resume is None:
        return None, None

    if resume == "auto":
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            logger.info(
                "resume-from-checkpoint=auto: no checkpoint found in %s, starting fresh.",
                output_dir,
            )
            return None, None
        logger.info(
            "Resuming from %s (step %d, %.0f MB).",
            latest.path.name,
            latest.step,
            latest.size_mb,
        )
        return str(latest.path), latest

    path = Path(resume)
    if not path.exists():
        available = discover_checkpoints(output_dir)
        raise CheckpointError(
            f"Checkpoint not found: {path}",
            details={
                "available": ", ".join(c.path.name for c in available) or "(none)",
                "searched_output_dir": str(output_dir),
            },
            suggestions=[
                "Use --resume-from-checkpoint auto to pick the newest automatically.",
                "List checkpoints: ls <output_dir>",
            ],
        )

    valid, reason = validate_checkpoint(path)
    if not valid:
        raise CheckpointError(
            f"Checkpoint {path} is not resumable: {reason}",
            details={"path": str(path), "reason": reason},
            suggestions=[
                "This usually means the run was killed mid-save.",
                "Resume from the previous checkpoint, or delete this directory and "
                "use --resume-from-checkpoint auto.",
            ],
        )

    match = _CHECKPOINT_PATTERN.match(path.name)
    return str(path), CheckpointInfo(
        path=path,
        step=int(match.group(1)) if match else -1,
        valid=True,
        modified_at=path.stat().st_mtime,
        size_mb=_directory_size_mb(path),
    )


def write_checkpoint_metadata(checkpoint_dir: Path | str, metadata: dict[str, Any]) -> Path:
    """Write KLEOS metadata alongside a Trainer checkpoint.

    Keeps experiment context with the checkpoint, so a directory recovered from
    Drive months later still identifies the run that produced it.
    """
    path = Path(checkpoint_dir) / "kleos_checkpoint.json"
    payload = {"written_at": datetime.now(UTC).isoformat(), **metadata}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_checkpoint_metadata(checkpoint_dir: Path | str) -> dict[str, Any] | None:
    """Read KLEOS checkpoint metadata, if present."""
    path = Path(checkpoint_dir) / "kleos_checkpoint.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def prune_checkpoints(output_dir: Path | str, *, keep: int, dry_run: bool = False) -> list[Path]:
    """Delete old checkpoints, always keeping at least one.

    Args:
        output_dir: Directory to prune.
        keep: How many to retain. Values below 1 are raised to 1.
        dry_run: Report what would be deleted without deleting.

    Returns:
        Paths removed (or that would be removed).
    """
    floor = max(1, keep)
    if floor != keep:
        logger.warning(
            "save_total_limit=%d would risk leaving no checkpoint to resume from; "
            "keeping %d instead.",
            keep,
            floor,
        )

    checkpoints = discover_checkpoints(output_dir, include_invalid=True)
    valid = [c for c in checkpoints if c.valid]
    invalid = [c for c in checkpoints if not c.valid]

    removed: list[Path] = []

    # Incomplete checkpoints are never useful; remove them regardless of the limit.
    for checkpoint in invalid:
        logger.info(
            "Removing incomplete checkpoint %s (%s)", checkpoint.path.name, checkpoint.reason
        )
        if not dry_run:
            shutil.rmtree(checkpoint.path, ignore_errors=True)
        removed.append(checkpoint.path)

    for checkpoint in valid[floor:]:
        logger.info("Pruning checkpoint %s (keeping %d newest)", checkpoint.path.name, floor)
        if not dry_run:
            shutil.rmtree(checkpoint.path, ignore_errors=True)
        removed.append(checkpoint.path)

    return removed


def summarize_checkpoints(output_dir: Path | str) -> str:
    """Human-readable checkpoint listing."""
    checkpoints = discover_checkpoints(output_dir, include_invalid=True)
    if not checkpoints:
        return f"No checkpoints in {output_dir}."
    lines = [f"Checkpoints in {output_dir}:"]
    for checkpoint in checkpoints:
        marker = "✓" if checkpoint.valid else "✗"
        detail = "" if checkpoint.valid else f" — {checkpoint.reason}"
        lines.append(
            f"  {marker} {checkpoint.path.name:<24} step={checkpoint.step:<8} "
            f"{checkpoint.size_mb:>8.0f} MB{detail}"
        )
    return "\n".join(lines)
