"""Trainer callbacks: structured logging, memory tracking, checkpoint metadata."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kleos_models.logging_utils import EventLogger, get_logger
from kleos_models.training.checkpointing import (
    prune_checkpoints,
    write_checkpoint_metadata,
)
from kleos_models.training.memory import snapshot_memory

logger = get_logger(__name__)


def _base_callback_class() -> Any:
    """Return ``transformers.TrainerCallback``, or a stub when unavailable."""
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:  # pragma: no cover - light environment only

        class _StubCallback:
            """Stand-in used when transformers is not installed."""

        return _StubCallback


TrainerCallbackBase = _base_callback_class()


class StructuredLoggingCallback(TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Write training events to a JSONL stream."""

    def __init__(self, event_logger: EventLogger, *, experiment_id: str) -> None:
        self.events = event_logger
        self.experiment_id = experiment_id
        self._train_started: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._train_started = time.perf_counter()
        self.events.set_stage("train")
        self.events.emit(
            "train_begin",
            max_steps=getattr(state, "max_steps", None),
            num_train_epochs=getattr(args, "num_train_epochs", None),
            per_device_batch_size=getattr(args, "per_device_train_batch_size", None),
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", None),
        )

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        # One payload, not two splatted dicts: `logs` carries `epoch`, which would raise TypeError.
        payload: dict[str, Any] = {
            "step": getattr(state, "global_step", None),
            "epoch": round(getattr(state, "epoch", 0.0) or 0.0, 4),
        }
        payload.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        memory = snapshot_memory()
        if memory:
            payload["memory"] = memory.to_dict()
        self.events.emit("log", **payload)

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not metrics:
            return
        payload: dict[str, Any] = {"step": getattr(state, "global_step", None)}
        payload.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        self.events.emit("evaluate", **payload)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.events.emit("checkpoint_saved", step=getattr(state, "global_step", None))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._train_started if self._train_started else None
        self.events.emit(
            "train_end",
            step=getattr(state, "global_step", None),
            elapsed_seconds=round(elapsed, 2) if elapsed else None,
        )


class MemoryMonitorCallback(TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Track peak VRAM and warn as the run approaches the limit."""

    def __init__(self, *, warn_threshold: float = 0.92, log_every: int = 50) -> None:
        self.warn_threshold = warn_threshold
        self.log_every = log_every
        self.peak_gb = 0.0
        self._warned = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = getattr(state, "global_step", 0)
        if step % self.log_every != 0:
            return
        memory = snapshot_memory()
        if memory is None:
            return
        self.peak_gb = max(self.peak_gb, memory.max_allocated_gb)
        if memory.utilization >= self.warn_threshold and not self._warned:
            self._warned = True
            logger.warning(
                "GPU memory is %.0f%% reserved (%s). An OOM is likely at evaluation "
                "or checkpoint time. Consider reducing max_seq_length or "
                "per_device_eval_batch_size.",
                memory.utilization * 100,
                memory.render(),
            )
        else:
            logger.debug("step %d memory: %s", step, memory.render())


class CheckpointMetadataCallback(TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Attach KLEOS metadata to each checkpoint and enforce retention."""

    def __init__(
        self,
        *,
        experiment_id: str,
        metadata: dict[str, Any],
        output_dir: Path | str,
        save_total_limit: int,
        manifest: Any = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.metadata = metadata
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.manifest = manifest

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = getattr(state, "global_step", 0)
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            return

        payload = {
            "experiment_id": self.experiment_id,
            "global_step": step,
            "epoch": getattr(state, "epoch", None),
            "best_metric": getattr(state, "best_metric", None),
            **self.metadata,
        }
        try:
            write_checkpoint_metadata(checkpoint_dir, payload)
        except OSError as exc:  # pragma: no cover - disk-full on Colab
            logger.warning("Could not write checkpoint metadata: %s", exc)

        if self.manifest is not None:
            self.manifest.add_checkpoint(checkpoint_dir, step)
            try:
                self.manifest.save(self.output_dir)
            except OSError as exc:  # pragma: no cover
                logger.warning("Could not update manifest after checkpoint: %s", exc)

        # Never leave zero checkpoints, nor delete the one load_best_model_at_end loads (H-F9).
        prune_checkpoints(
            self.output_dir,
            keep=self.save_total_limit,
            protect=(getattr(state, "best_model_checkpoint", None),),
        )


class ProgressCallback(TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Concise console progress with an ETA."""

    def __init__(self, *, log_every: int = 10) -> None:
        self.log_every = log_every
        self._start: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._start = time.perf_counter()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        step = getattr(state, "global_step", 0)
        if step % self.log_every != 0:
            return

        max_steps = getattr(state, "max_steps", 0) or 0
        elapsed = time.perf_counter() - self._start if self._start else 0.0
        eta = ""
        if max_steps and step:
            remaining = (elapsed / step) * (max_steps - step)
            eta = f" | eta {remaining / 60:.1f}m"

        parts = [f"step {step}"]
        if max_steps:
            parts.append(f"/{max_steps}")
        parts.append(f" | loss {logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f" | lr {logs['learning_rate']:.2e}")
        if "grad_norm" in logs:
            parts.append(f" | grad {logs['grad_norm']:.2f}")
        logger.info("%s%s", "".join(parts), eta)


class EarlyOOMGuardCallback(TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Fail fast when the first steps already exhaust memory."""

    def __init__(self, *, check_steps: int = 5, threshold: float = 0.97) -> None:
        self.check_steps = check_steps
        self.threshold = threshold

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = getattr(state, "global_step", 0)
        if step > self.check_steps:
            return
        memory = snapshot_memory()
        if memory and memory.utilization >= self.threshold:
            logger.warning(
                "GPU memory is already %.0f%% reserved at step %d. This run is very "
                "likely to OOM later, when evaluation or checkpointing adds a spike. "
                "Consider stopping now and reducing max_seq_length.",
                memory.utilization * 100,
                step,
            )
