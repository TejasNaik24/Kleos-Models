"""Runtime memory reporting and OOM diagnostics (spec sections 13, 14, 32).

Estimation and feasibility planning live in :mod:`kleos_models.models.feasibility`.
This module covers what happens *during* a run: peak-memory tracking, the
pre-flight environment report, and turning a bare ``CUDA out of memory`` into
something a user can act on.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kleos_models.config import ModelConfig, TrainingConfig
from kleos_models.errors import InsufficientMemoryError
from kleos_models.logging_utils import get_logger
from kleos_models.models.feasibility import BYTES_PER_GB, GPUInfo, probe_gpu

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Allocator state at a moment in time."""

    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float
    total_gb: float

    @property
    def utilization(self) -> float:
        return self.reserved_gb / self.total_gb if self.total_gb else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated_gb": round(self.allocated_gb, 2),
            "reserved_gb": round(self.reserved_gb, 2),
            "max_allocated_gb": round(self.max_allocated_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "utilization": round(self.utilization, 3),
        }

    def render(self) -> str:
        return (
            f"allocated {self.allocated_gb:.2f} GB | reserved {self.reserved_gb:.2f} GB | "
            f"peak {self.max_allocated_gb:.2f} GB | free {self.free_gb:.2f} GB "
            f"({self.utilization:.0%} of {self.total_gb:.1f} GB reserved)"
        )


def snapshot_memory() -> MemorySnapshot | None:
    """Current CUDA allocator state, or ``None`` without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return MemorySnapshot(
            allocated_gb=torch.cuda.memory_allocated(device) / BYTES_PER_GB,
            reserved_gb=torch.cuda.memory_reserved(device) / BYTES_PER_GB,
            max_allocated_gb=torch.cuda.max_memory_allocated(device) / BYTES_PER_GB,
            free_gb=free_bytes / BYTES_PER_GB,
            total_gb=total_bytes / BYTES_PER_GB,
        )
    except ImportError:
        return None


def reset_peak_memory() -> None:
    """Reset peak-memory tracking on every GPU, so a phase's peak is attributable to it."""
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
    except ImportError:
        pass


def peak_allocated_all_gb() -> float | None:
    """The fullest GPU's peak allocation, or ``None`` without CUDA.

    On one GPU this is ``torch.cuda.max_memory_allocated``. A model spread over
    several GPUs (``device_map: auto``) is limited by whichever fills first.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    peaks = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    return max(peaks, default=0) / BYTES_PER_GB


def model_devices(model: Any) -> list[int]:
    """The CUDA devices holding the model's parameters, in ascending order."""
    return sorted(
        {
            parameter.device.index
            for parameter in model.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )


@dataclass(frozen=True)
class DeviceReading:
    """One GPU's memory at the peak of the probe's training passes, in GiB."""

    index: int
    peak_allocated_gb: float
    peak_reserved_gb: float
    total_gb: float
    free_at_peak_gb: float
    #: Paged optimizer state for the trainable parameters on this GPU, which
    #: bitsandbytes allocates outside PyTorch's allocator at the first step.
    paged_optimizer_gb: float

    @property
    def spare_after_optimizer_gb(self) -> float:
        return self.free_at_peak_gb - self.paged_optimizer_gb

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_allocated_gb": round(self.peak_allocated_gb, 3),
            "peak_reserved_gb": round(self.peak_reserved_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "free_at_peak_gb": round(self.free_at_peak_gb, 3),
            "paged_optimizer_gb": round(self.paged_optimizer_gb, 3),
            "spare_after_optimizer_gb": round(self.spare_after_optimizer_gb, 3),
        }


def probe_result(
    readings: Sequence[DeviceReading],
    *,
    batch_size: int,
    sequence_length: int,
    precision: str,
) -> dict[str, Any]:
    """The memory probe's record, gated on the GPU with the least to spare.

    With one GPU the record is exactly what it always was. With several, its
    top-level figures are the tightest GPU's (``scripts/check_smoke_gate.py``
    gates on them unchanged), and every GPU's figures are listed beside them.
    """
    tightest = min(readings, key=lambda r: r.spare_after_optimizer_gb)
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "precision": precision,
        **tightest.to_dict(),
    }
    if len(readings) > 1:
        result["gated_on_device"] = tightest.index
        result["devices"] = [{"index": r.index, **r.to_dict()} for r in readings]
    return result


def free_memory() -> None:
    """Release cached allocator blocks. Useful between load and train phases."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def probe_training_peak(
    model: Any,
    dataset: Any,
    collator: Any,
    training_config: TrainingConfig,
) -> dict[str, Any] | None:
    """Measure a training step's memory on the longest micro-batch, before step 1.

    The pre-flight estimate is arithmetic; this is the measurement. Two forward
    and backward passes on the ``per_device_train_batch_size`` longest examples,
    under the autocast the Trainer will use, keeping the gradients between them as
    gradient accumulation does. A configuration that cannot survive its longest
    batch then fails in the first minute rather than hours in, and a smoke run
    reports the peak its real run will reach whichever examples its ten steps
    happen to draw.

    Nothing trains: gradients are discarded, and the Trainer re-seeds on
    construction, so the passes leave no trace on the run.

    Returns:
        The measurement, or ``None`` without CUDA.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    from kleos_models.training.arguments import resolve_precision

    bf16, fp16, _ = resolve_precision(training_config)
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None

    size = max(1, training_config.per_device_train_batch_size)
    longest = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]), reverse=True)
    batch = collator([dataset[i] for i in longest[:size]])
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    # A model spread over several GPUs (device_map: auto) is measured on each;
    # on one GPU, exactly as before.
    spread = model_devices(model)
    measured = [torch.device("cuda", index) for index in spread] if len(spread) > 1 else [device]

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    free_memory()
    for target in measured:
        torch.cuda.reset_peak_memory_stats(target)
    try:
        for _ in range(2):
            autocast = (
                torch.autocast("cuda", dtype=dtype)
                if dtype is not None
                else contextlib.nullcontext()
            )
            with autocast:
                loss = model(**batch).loss
            loss.backward()
        raw: list[tuple[Any, int, int, float, float]] = []
        for target in measured:
            torch.cuda.synchronize(target)
            free_bytes, total_bytes = torch.cuda.mem_get_info(target)
            raw.append(
                (
                    target,
                    free_bytes,
                    total_bytes,
                    torch.cuda.max_memory_allocated(target) / BYTES_PER_GB,
                    torch.cuda.max_memory_reserved(target) / BYTES_PER_GB,
                )
            )
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        free_memory()

    # bitsandbytes' paged optimizer allocates its state outside PyTorch's
    # allocator at the first optimizer step, from what is free now, on the GPU
    # that holds each trainable parameter.
    paged = "paged" in training_config.optim
    readings = []
    for target, free_bytes, total_bytes, peak_allocated, peak_reserved in raw:
        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad and (len(measured) == 1 or p.device == target)
        )
        readings.append(
            DeviceReading(
                index=target.index if target.index is not None else 0,
                peak_allocated_gb=peak_allocated,
                peak_reserved_gb=peak_reserved,
                total_gb=total_bytes / BYTES_PER_GB,
                free_at_peak_gb=free_bytes / BYTES_PER_GB,
                paged_optimizer_gb=trainable * 2 / BYTES_PER_GB if paged else 0.0,
            )
        )
    result = probe_result(
        readings,
        batch_size=int(batch["input_ids"].shape[0]),
        sequence_length=int(batch["input_ids"].shape[1]),
        precision="bf16" if bf16 else "fp16" if fp16 else "fp32",
    )
    for entry in result.get("devices", [result]):
        logger.info(
            "Memory probe (longest batch %d x %d tokens, %s)%s: peak %.2f GB allocated, "
            "%.2f GB reserved of %.2f GB; %.2f GB spare once the optimizer state exists.",
            result["batch_size"],
            result["sequence_length"],
            result["precision"],
            f" on GPU {entry['index']}" if "index" in entry else "",
            entry["peak_allocated_gb"],
            entry["peak_reserved_gb"],
            entry["total_gb"],
            entry["spare_after_optimizer_gb"],
        )
    return result


def render_environment_report(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    gpu: GPUInfo | None = None,
    seq_length: int | None = None,
) -> str:
    """The pre-flight report the training script prints (spec section 13).

    Everything the spec requires before an expensive run: GPU, VRAM, CUDA, library
    versions, model id, quantization, LoRA settings, and the memory estimate.

    Args:
        seq_length: The longest training sequence, when measured; the memory
            estimate otherwise assumes ``model.max_seq_length``.
    """
    from kleos_models.compat import library_versions
    from kleos_models.models.feasibility import estimate_memory

    gpu = gpu or probe_gpu()
    versions = library_versions()
    from kleos_models.models.adapters import get_adapter

    targets = get_adapter(model_config).resolve_target_modules(model_config.lora)
    estimate = estimate_memory(
        model_config, training_config, target_modules=targets, seq_length=seq_length
    )

    lines = [
        "=" * 72,
        "KLEOS training pre-flight",
        "=" * 72,
        "",
        "Hardware",
        f"  GPU                   : {gpu.name}",
        f"  VRAM                  : {gpu.total_memory_gb:.1f} GB total, "
        f"{gpu.free_memory_gb:.1f} GB free",
        f"  compute capability    : {gpu.capability_string}",
        f"  bfloat16 supported    : {'yes' if gpu.bf16_supported else 'NO (float16 will be used)'}",
        f"  CUDA                  : {gpu.cuda_version or 'not detected'}",
        f"  devices               : {gpu.device_count}",
        "",
        "Software",
    ]
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"):
        lines.append(f"  {name:<22}: {versions.get(name) or 'not installed'}")

    lines.extend(
        [
            "",
            "Model",
            f"  base model            : {model_config.base_model}",
            f"  revision              : {model_config.revision}",
            f"  family / type         : {model_config.family} / {model_config.model_type or 'auto'}",
            f"  parameters            : "
            f"{f'{model_config.parameter_count / 1e9:.1f}B' if model_config.parameter_count else 'unknown'}",
        ]
    )
    if model_config.active_parameter_count:
        lines.append(
            f"  active parameters     : {model_config.active_parameter_count / 1e9:.1f}B "
            "(MoE — do not plan compute as if dense)"
        )
    lines.extend(
        [
            f"  max sequence length   : {model_config.max_seq_length}",
            f"  quantization          : {model_config.quantization.mode.value} "
            f"(double_quant={model_config.quantization.double_quant}, "
            f"compute_dtype={model_config.quantization.compute_dtype.value})",
            "",
            "LoRA",
            f"  r / alpha / dropout   : {model_config.lora.r} / "
            f"{model_config.lora.alpha} / {model_config.lora.dropout}",
            f"  scaling (alpha/r)     : {model_config.lora.scaling:.2f}",
            f"  target modules        : {model_config.lora.target_modules}",
            "",
            "Training",
            f"  epochs / max steps    : {training_config.num_train_epochs} / "
            f"{training_config.max_steps}",
            f"  learning rate         : {training_config.learning_rate}",
            f"  batch (device x accum): {training_config.per_device_train_batch_size} x "
            f"{training_config.gradient_accumulation_steps} = "
            f"{training_config.effective_batch_size}",
            f"  optimizer             : {training_config.optim}",
            f"  gradient checkpointing: {training_config.gradient_checkpointing}",
            f"  precision             : {training_config.precision.value}",
            f"  seed                  : {training_config.seed}",
            f"  strict config         : {training_config.strict_config}",
            "",
            "Estimated memory",
            estimate.render(),
            "",
            "  Assumptions:",
        ]
    )
    lines.extend(f"    - {assumption}" for assumption in estimate.assumptions)
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def diagnose_oom(
    error: BaseException,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    stage: str = "training",
) -> InsufficientMemoryError:
    """Convert a CUDA OOM into an actionable error (spec section 32).

    A bare ``CUDA error`` tells a user nothing. This reports what was configured,
    what the hardware is, and an ordered list of things to change — cheapest and
    least experiment-distorting first.
    """
    gpu = probe_gpu()
    snapshot = snapshot_memory()

    suggestions = [
        f"Reduce model.max_seq_length (currently {model_config.max_seq_length}) — "
        "activation memory scales linearly with it.",
    ]
    if not training_config.gradient_checkpointing:
        suggestions.append(
            "Enable training.gradient_checkpointing — usually the single largest "
            "saving, at roughly 20-30% slower steps."
        )
    if training_config.per_device_train_batch_size > 1:
        suggestions.append(
            f"Reduce training.per_device_train_batch_size (currently "
            f"{training_config.per_device_train_batch_size}) and raise "
            "gradient_accumulation_steps by the same factor, which keeps the "
            "effective batch size and the learning dynamics unchanged."
        )
    if not model_config.quantization.enabled:
        suggestions.append("Enable 4-bit quantization: model.quantization.mode='nf4'.")
    if training_config.optim not in ("paged_adamw_8bit", "adamw_bnb_8bit"):
        suggestions.append(
            f"Use a paged 8-bit optimizer (currently {training_config.optim!r}): "
            "training.optim='paged_adamw_8bit'."
        )
    suggestions.extend(
        [
            "Check nothing else holds GPU memory: run nvidia-smi, and in a notebook "
            "restart the runtime to clear a previous model.",
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.",
            "Verify feasibility before retrying: python scripts/plan_run.py --config <your config>",
        ]
    )

    details: dict[str, Any] = {
        "stage": stage,
        "detected_gpu": gpu.render(),
        "model": model_config.base_model,
        "quantization": model_config.quantization.mode.value,
        "max_seq_length": model_config.max_seq_length,
        "per_device_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation": training_config.gradient_accumulation_steps,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "original_error": str(error)[:300],
    }
    if snapshot:
        details["memory_at_failure"] = snapshot.render()

    return InsufficientMemoryError(
        f"CUDA ran out of memory during {stage}.",
        details=details,
        suggestions=suggestions,
    )


def is_oom_error(error: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory condition."""
    message = str(error).lower()
    if "out of memory" in message or "cuda oom" in message:
        return True
    return type(error).__name__ == "OutOfMemoryError"
