"""Model layer: family adapters, loading, quantization, PEFT, feasibility.

This package needs the ``[train]`` extra for anything that touches weights.
Module import itself is cheap: torch and transformers are imported inside
functions, so ``ModelFamilyAdapter`` metadata, memory estimation and feasibility
planning all work in a light environment. That is what lets a Colab notebook print
a feasibility table before installing or downloading anything heavy.
"""

from kleos_models.models.adapters import (
    ADAPTER_REGISTRY,
    Mistral3VLMAdapter,
    MistralDenseAdapter,
    ModelCapabilities,
    ModelFamilyAdapter,
    QwenDenseAdapter,
    QwenMoEAdapter,
    TargetModuleResolution,
    get_adapter,
    register_adapter,
    resolve_adapter_from_hf_config,
)
from kleos_models.models.feasibility import (
    Adjustment,
    FeasibilityReport,
    GPUInfo,
    MemoryEstimate,
    ModelShape,
    assess_feasibility,
    enforce_feasibility,
    estimate_memory,
    probe_gpu,
    render_feasibility_table,
)
from kleos_models.models.quantization import (
    QuantizationSupport,
    build_quantization_config,
    quantization_support,
    resolve_compute_dtype,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "Adjustment",
    "FeasibilityReport",
    "GPUInfo",
    "MemoryEstimate",
    "Mistral3VLMAdapter",
    "MistralDenseAdapter",
    "ModelCapabilities",
    "ModelFamilyAdapter",
    "ModelShape",
    "QuantizationSupport",
    "QwenDenseAdapter",
    "QwenMoEAdapter",
    "TargetModuleResolution",
    "assess_feasibility",
    "build_quantization_config",
    "enforce_feasibility",
    "estimate_memory",
    "get_adapter",
    "probe_gpu",
    "quantization_support",
    "register_adapter",
    "render_feasibility_table",
    "resolve_adapter_from_hf_config",
    "resolve_compute_dtype",
]
