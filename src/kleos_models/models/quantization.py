"""4-bit / 8-bit quantization configuration (spec section 13).

bitsandbytes is CUDA-only in practice. This module therefore separates two
questions that are easy to conflate:

* *Can this environment quantize at all?* — :func:`quantization_support`
* *What config implements the requested mode?* — :func:`build_quantization_config`

If quantization is requested and unavailable, the pipeline fails with a
diagnostic. It never silently loads in full precision, because that turns a
"QLoRA run" into a run that either OOMs confusingly or trains something other
than what the manifest claims.

Compute dtype
-------------
``compute_dtype: auto`` resolves to bfloat16 on compute capability >= 8.0 and
float16 below it. This matters on the free Colab tier: a T4 is compute capability
7.5 and **cannot** do bfloat16. The resolved choice is always recorded, never
silently applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kleos_models.config import DType, ModelConfig, QuantizationMode
from kleos_models.errors import ModelCompatibilityError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Below this compute capability, bfloat16 is unsupported by the hardware.
BF16_MIN_COMPUTE_CAPABILITY = (8, 0)


@dataclass
class QuantizationSupport:
    """What the current environment can actually do."""

    bitsandbytes_installed: bool
    bitsandbytes_version: str | None
    cuda_available: bool
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    reason: str = ""

    @property
    def can_quantize(self) -> bool:
        return self.bitsandbytes_installed and self.cuda_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "bitsandbytes_installed": self.bitsandbytes_installed,
            "bitsandbytes_version": self.bitsandbytes_version,
            "cuda_available": self.cuda_available,
            "compute_capability": (
                f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability
                else None
            ),
            "bf16_supported": self.bf16_supported,
            "can_quantize": self.can_quantize,
            "reason": self.reason,
        }


def quantization_support() -> QuantizationSupport:
    """Probe the environment for quantization capability."""
    from kleos_models.compat import package_version

    bnb_version = package_version("bitsandbytes")
    cuda_available = False
    capability: tuple[int, int] | None = None
    bf16 = False
    reason = ""

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            capability = torch.cuda.get_device_capability(0)
            bf16 = capability >= BF16_MIN_COMPUTE_CAPABILITY
            if not bf16:
                reason = (
                    f"GPU compute capability {capability[0]}.{capability[1]} < 8.0, "
                    "so bfloat16 is unsupported; float16 will be used."
                )
        else:
            reason = "No CUDA device is visible to torch."
    except ImportError:
        reason = "torch is not installed."

    if bnb_version is None and not reason:
        reason = "bitsandbytes is not installed."

    return QuantizationSupport(
        bitsandbytes_installed=bnb_version is not None,
        bitsandbytes_version=bnb_version,
        cuda_available=cuda_available,
        compute_capability=capability,
        bf16_supported=bf16,
        reason=reason,
    )


def resolve_compute_dtype(
    requested: DType, support: QuantizationSupport | None = None
) -> tuple[Any, str]:
    """Resolve a compute dtype to a real ``torch.dtype``.

    Returns:
        ``(torch_dtype, explanation)``. The explanation is recorded in the run
        manifest, so an automatic choice is always visible.
    """
    import torch

    support = support or quantization_support()

    if requested is DType.AUTO:
        if support.bf16_supported:
            return torch.bfloat16, "auto → bfloat16 (compute capability >= 8.0)"
        capability = (
            f"{support.compute_capability[0]}.{support.compute_capability[1]}"
            if support.compute_capability
            else "unknown"
        )
        return torch.float16, (
            f"auto → float16 (compute capability {capability} does not support bfloat16)"
        )

    mapping = {
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.FLOAT32: torch.float32,
    }
    dtype = mapping[requested]

    if requested is DType.BFLOAT16 and support.cuda_available and not support.bf16_supported:
        capability = (
            f"{support.compute_capability[0]}.{support.compute_capability[1]}"
            if support.compute_capability
            else "unknown"
        )
        raise ModelCompatibilityError(
            f"bfloat16 was requested but this GPU (compute capability {capability}) "
            "does not support it.",
            details={"gpu_compute_capability": capability, "required": ">= 8.0"},
            suggestions=[
                "Set model.quantization.compute_dtype to 'float16'.",
                "Or use 'auto', which picks the right dtype for the assigned GPU.",
                "Colab free-tier T4 GPUs are compute capability 7.5 and need float16.",
            ],
        )

    return dtype, f"explicitly configured as {requested.value}"


def build_quantization_config(
    config: ModelConfig,
    *,
    extra_skip_modules: list[str] | None = None,
    support: QuantizationSupport | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Build a ``BitsAndBytesConfig`` for the requested mode.

    Args:
        config: Model configuration.
        extra_skip_modules: Modules the family adapter requires to stay in full
            precision (a vision tower, an MoE router).
        support: Pre-probed environment support, to avoid probing twice.

    Returns:
        ``(quantization_config_or_None, metadata)``. Metadata is recorded in the
        manifest so the exact quantization is always reproducible.

    Raises:
        ModelCompatibilityError: when quantization is requested but impossible.
    """
    quant = config.quantization
    support = support or quantization_support()

    metadata: dict[str, Any] = {
        "mode": quant.mode.value,
        "requested_compute_dtype": quant.compute_dtype.value,
        "double_quant": quant.double_quant,
        "support": support.to_dict(),
    }

    if not quant.enabled:
        metadata["applied"] = False
        metadata["note"] = "Quantization disabled; the base model loads in full precision."
        return None, metadata

    if not support.bitsandbytes_installed:
        raise ModelCompatibilityError(
            f"Quantization mode {quant.mode.value!r} needs bitsandbytes, which is not installed.",
            details={"requested_mode": quant.mode.value, "probe": support.reason},
            suggestions=[
                'Install it: pip install -e ".[train,quant]"',
                "On Colab: python scripts/colab_setup.py",
                "Or set model.quantization.mode='none' — but note that an unquantized "
                "8B model needs roughly 16GB just for weights, and a 24B model ~48GB.",
                "The pipeline will not silently drop quantization: that would change "
                "what the experiment measures.",
            ],
        )

    if not support.cuda_available:
        raise ModelCompatibilityError(
            f"Quantization mode {quant.mode.value!r} requires a CUDA GPU; none is visible.",
            details={"probe": support.reason},
            suggestions=[
                "Run training on a CUDA machine or a Colab GPU runtime.",
                "In Colab: Runtime → Change runtime type → GPU.",
                "For a CPU-only pipeline check, use configs/training/debug.yaml with "
                "model.quantization.mode='none' and a tiny model.",
            ],
        )

    from transformers import BitsAndBytesConfig

    compute_dtype, dtype_note = resolve_compute_dtype(quant.compute_dtype, support)
    metadata["resolved_compute_dtype"] = str(compute_dtype).replace("torch.", "")
    metadata["compute_dtype_reason"] = dtype_note
    logger.info("Quantization compute dtype: %s", dtype_note)

    skip_modules = sorted({*quant.skip_modules, *(extra_skip_modules or [])})
    metadata["skip_modules"] = skip_modules

    if quant.mode is QuantizationMode.INT8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip_modules or None,
        )
        metadata["applied"] = True
        return bnb_config, metadata

    storage_dtype = (
        compute_dtype
        if quant.quant_storage_dtype is DType.AUTO
        else resolve_compute_dtype(quant.quant_storage_dtype, support)[0]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4" if quant.mode is QuantizationMode.NF4 else "fp4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.double_quant,
        bnb_4bit_quant_storage=storage_dtype,
        llm_int8_skip_modules=skip_modules or None,
    )
    metadata["applied"] = True
    metadata["quant_type"] = "nf4" if quant.mode is QuantizationMode.NF4 else "fp4"
    logger.info(
        "4-bit quantization: %s, double_quant=%s, compute_dtype=%s",
        metadata["quant_type"],
        quant.double_quant,
        metadata["resolved_compute_dtype"],
    )
    return bnb_config, metadata
