"""GPU probing, memory estimation and feasibility assessment (spec sections 3, 13, 14).

The requirement this implements: *architectural support is not the same as
trainability on a free GPU*, and the pipeline must never silently shrink a
configuration until the experiment stops being comparable.

So there are two separable things here:

1. **Estimation** — how much memory would this run need? Pure arithmetic over the
   model's shape, runnable with no GPU and no torch, which is what lets a Colab
   notebook print a feasibility table for every model config *before* downloading
   28GB of weights.
2. **Policy** — what to do when it does not fit. Adjustments are proposed, logged
   and recorded in the manifest. Under ``training.strict_config`` they are refused
   outright.

The estimates are approximations. They are documented as such, they state their
assumptions, and they are deliberately a little pessimistic: telling someone a run
fits when it does not wastes a Colab session.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from kleos_models.config import (
    FeasibilityTier,
    ModelConfig,
    QuantizationMode,
    TrainingConfig,
)
from kleos_models.errors import InsufficientMemoryError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

BYTES_PER_GB = 1024**3

#: Non-parameter CUDA overhead: context, cuBLAS workspaces, allocator fragmentation.
CUDA_OVERHEAD_GB = 1.2

#: Multiplier on stored activations when gradient checkpointing is OFF. Real values
#: depend on the attention implementation; this is a mid-range estimate.
ACTIVATION_MULTIPLIER_NO_CHECKPOINT = 12.0
#: With checkpointing, roughly one tensor per layer boundary is retained.
ACTIVATION_MULTIPLIER_CHECKPOINT = 1.5

#: Fraction of total VRAM that must remain free for a run to count as
#: FULL_RESEARCH rather than merely fitting.
RESEARCH_HEADROOM_FRACTION = 0.15

#: Optimizer state bytes per trainable parameter.
OPTIMIZER_BYTES = {
    "adamw_torch": 8.0,  # two fp32 moments
    "adamw_hf": 8.0,
    "adamw_torch_fused": 8.0,
    "adamw_bnb_8bit": 2.0,  # two int8 moments
    "paged_adamw_8bit": 2.0,
    "paged_adamw_32bit": 8.0,
    "sgd": 4.0,
    "adafactor": 4.0,
}


@dataclass
class GPUInfo:
    """Facts about the accelerator actually assigned to this session."""

    available: bool
    name: str = "none"
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    device_count: int = 0
    cuda_version: str | None = None
    driver_version: str | None = None
    torch_version: str | None = None
    source: str = "unavailable"

    @property
    def bf16_supported(self) -> bool:
        """bfloat16 needs compute capability 8.0+. A Colab T4 is 7.5."""
        return self.compute_capability is not None and self.compute_capability >= (8, 0)

    @property
    def capability_string(self) -> str:
        if self.compute_capability is None:
            return "unknown"
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "free_memory_gb": round(self.free_memory_gb, 2),
            "compute_capability": self.capability_string,
            "bf16_supported": self.bf16_supported,
            "device_count": self.device_count,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "torch_version": self.torch_version,
            "source": self.source,
        }

    def render(self) -> str:
        if not self.available:
            return f"No GPU detected ({self.source})."
        return (
            f"{self.name} | {self.total_memory_gb:.1f} GB total, "
            f"{self.free_memory_gb:.1f} GB free | compute capability "
            f"{self.capability_string} | bf16 {'yes' if self.bf16_supported else 'NO'} | "
            f"CUDA {self.cuda_version or 'unknown'}"
        )


def probe_gpu() -> GPUInfo:
    """Detect the available accelerator.

    Tries torch first, then ``nvidia-smi``, so a notebook can report hardware even
    before the training extra is installed. Never raises.
    """
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            return GPUInfo(
                available=True,
                name=properties.name,
                total_memory_gb=total_bytes / BYTES_PER_GB,
                free_memory_gb=free_bytes / BYTES_PER_GB,
                compute_capability=(properties.major, properties.minor),
                device_count=torch.cuda.device_count(),
                cuda_version=torch.version.cuda,
                torch_version=torch.__version__,
                source="torch.cuda",
            )
        # Apple Silicon. Usable for small CPU/MPS checks, not for bitsandbytes QLoRA.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUInfo(
                available=False,
                name="Apple MPS",
                torch_version=torch.__version__,
                source="torch.mps (bitsandbytes 4-bit training is CUDA-only)",
            )
        return GPUInfo(
            available=False, torch_version=torch.__version__, source="torch reports no CUDA device"
        )
    except ImportError:
        pass

    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
            first = output.splitlines()[0]
            name, total_mb, free_mb, driver = (part.strip() for part in first.split(","))
            return GPUInfo(
                available=True,
                name=name,
                total_memory_gb=float(total_mb) / 1024,
                free_memory_gb=float(free_mb) / 1024,
                device_count=len(output.splitlines()),
                driver_version=driver,
                source="nvidia-smi (torch not installed)",
            )
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:  # pragma: no cover
            logger.debug("nvidia-smi probe failed: %s", exc)

    return GPUInfo(available=False, source="torch not installed and no nvidia-smi")


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------


@dataclass
class ModelShape:
    """Enough architecture to estimate memory without downloading weights."""

    parameter_count: float
    hidden_size: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14336
    vocab_size: int = 32000
    active_parameter_count: float | None = None
    is_moe: bool = False

    @classmethod
    def from_config(cls, config: ModelConfig) -> ModelShape:
        """Derive a shape from a model config, using known checkpoint facts.

        The table below is verified against each checkpoint's published
        ``config.json``. Unknown models fall back to a generic 7B-ish shape and the
        estimate is flagged as low-confidence.
        """
        known: dict[str, dict[str, Any]] = {
            "Qwen/Qwen3-8B": {
                "parameter_count": 8.2e9,
                "hidden_size": 4096,
                "num_layers": 36,
                "intermediate_size": 12288,
                "vocab_size": 151936,
            },
            "Qwen/Qwen3-30B-A3B-Thinking-2507": {
                "parameter_count": 30.5e9,
                "active_parameter_count": 3.3e9,
                "hidden_size": 2048,
                "num_layers": 48,
                "intermediate_size": 768,  # per-expert
                "vocab_size": 151936,
                "is_moe": True,
            },
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
                "parameter_count": 24.0e9,
                "hidden_size": 5120,
                "num_layers": 40,
                "intermediate_size": 32768,
                "vocab_size": 131072,
            },
            "mistralai/Ministral-8B-Instruct-2410": {
                "parameter_count": 8.0e9,
                "hidden_size": 4096,
                "num_layers": 36,
                "intermediate_size": 12288,
                "vocab_size": 131072,
            },
        }

        facts = known.get(config.base_model, {})
        parameter_count = config.parameter_count or facts.get("parameter_count") or 7.0e9
        return cls(
            parameter_count=float(parameter_count),
            hidden_size=int(facts.get("hidden_size", 4096)),
            num_layers=int(facts.get("num_layers", 32)),
            intermediate_size=int(facts.get("intermediate_size", 14336)),
            vocab_size=int(facts.get("vocab_size", 32000)),
            active_parameter_count=(
                config.active_parameter_count or facts.get("active_parameter_count")
            ),
            is_moe=config.is_moe or bool(facts.get("is_moe", False)),
        )

    @classmethod
    def from_hf_config(cls, hf_config: Any, parameter_count: float) -> ModelShape:
        """Build a shape from a downloaded HF config (most accurate path)."""
        text_config = getattr(hf_config, "text_config", hf_config)

        def read(name: str, default: int) -> int:
            value = getattr(text_config, name, None)
            if value is None:
                value = getattr(hf_config, name, None)
            return int(value) if value is not None else default

        return cls(
            parameter_count=parameter_count,
            hidden_size=read("hidden_size", 4096),
            num_layers=read("num_hidden_layers", 32),
            intermediate_size=read("intermediate_size", 14336),
            vocab_size=read("vocab_size", 32000),
            is_moe=hasattr(text_config, "num_experts"),
        )


@dataclass
class MemoryEstimate:
    """Estimated peak VRAM, broken down so a user can see what to change."""

    base_weights_gb: float
    lora_params: int
    lora_weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    confidence: str = "medium"
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return (
            self.base_weights_gb
            + self.lora_weights_gb
            + self.gradients_gb
            + self.optimizer_gb
            + self.activations_gb
            + self.overhead_gb
        )

    #: Activation budget for single-sequence inference, computed independently of
    #: the training batch size. Set by :func:`estimate_memory`.
    inference_activations_gb: float = 0.0

    @property
    def inference_gb(self) -> float:
        """Weights plus a single-sequence activation budget, without training state.

        Deliberately independent of the *training* batch size and sequence length:
        whether a model can be loaded at all is a property of the weights, not of
        how large a training batch someone asked for.
        """
        return self.base_weights_gb + self.overhead_gb + self.inference_activations_gb

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_weights_gb": round(self.base_weights_gb, 2),
            "lora_trainable_params": self.lora_params,
            "lora_weights_gb": round(self.lora_weights_gb, 3),
            "gradients_gb": round(self.gradients_gb, 3),
            "optimizer_gb": round(self.optimizer_gb, 3),
            "activations_gb": round(self.activations_gb, 2),
            "overhead_gb": round(self.overhead_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "inference_gb": round(self.inference_gb, 2),
            "confidence": self.confidence,
            "assumptions": self.assumptions,
        }

    def render(self) -> str:
        return "\n".join(
            [
                f"  base weights (quantized)  {self.base_weights_gb:6.2f} GB",
                f"  LoRA weights              {self.lora_weights_gb:6.3f} GB "
                f"({self.lora_params:,} trainable params)",
                f"  gradients                 {self.gradients_gb:6.3f} GB",
                f"  optimizer state           {self.optimizer_gb:6.3f} GB",
                f"  activations               {self.activations_gb:6.2f} GB",
                f"  CUDA overhead             {self.overhead_gb:6.2f} GB",
                f"  {'-' * 40}",
                f"  estimated peak            {self.total_gb:6.2f} GB "
                f"(confidence: {self.confidence})",
            ]
        )


def estimate_lora_parameters(shape: ModelShape, rank: int, target_modules: list[str]) -> int:
    """Count trainable LoRA parameters for a target set.

    A LoRA on a ``d_in x d_out`` matrix adds ``r * (d_in + d_out)`` parameters.
    """
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    per_layer = 0

    for module in target_modules:
        if module in ("q_proj", "o_proj"):
            per_layer += rank * (hidden + hidden)
        elif module in ("k_proj", "v_proj"):
            # Grouped-query attention makes these narrower; approximate at 1/4.
            per_layer += rank * (hidden + max(hidden // 4, 1))
        elif module in ("gate_proj", "up_proj"):
            per_layer += rank * (hidden + intermediate)
        elif module == "down_proj":
            per_layer += rank * (intermediate + hidden)
        else:
            per_layer += rank * (hidden + hidden)

    return per_layer * shape.num_layers


def estimate_memory(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    shape: ModelShape | None = None,
    target_modules: list[str] | None = None,
    seq_length: int | None = None,
    batch_size: int | None = None,
) -> MemoryEstimate:
    """Estimate peak training VRAM for a configuration.

    The arithmetic is deliberately explicit rather than a fitted heuristic, so a
    user can see which term is dominating and change the right knob.
    """
    shape = shape or ModelShape.from_config(model_config)
    targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    seq = seq_length or model_config.max_seq_length
    batch = batch_size or training_config.per_device_train_batch_size

    assumptions: list[str] = []

    # --- base weights -------------------------------------------------------
    quant = model_config.quantization
    bytes_per_param = quant.mode.bits / 8.0
    if quant.mode.is_4bit:
        # Double quantization still stores absmax scales; ~0.03 bytes/param extra.
        bytes_per_param += 0.0 if quant.double_quant else 0.03
    base_weights_gb = shape.parameter_count * bytes_per_param / BYTES_PER_GB

    # The embedding and LM head are typically not quantized.
    unquantized_gb = 2 * shape.vocab_size * shape.hidden_size * 2 / BYTES_PER_GB
    if quant.mode is not QuantizationMode.NONE:
        base_weights_gb += unquantized_gb
        assumptions.append(f"embedding and lm_head kept unquantized (+{unquantized_gb:.2f} GB)")
    assumptions.append(
        f"{shape.parameter_count / 1e9:.1f}B parameters at "
        f"{quant.mode.bits}-bit ({bytes_per_param:.2f} bytes/param)"
    )
    if shape.is_moe:
        assumptions.append(
            "MoE: ALL experts are resident in memory even though only a subset "
            "activates per token, so weight memory tracks total, not active, params"
        )

    # --- LoRA, gradients, optimizer ----------------------------------------
    lora_params = estimate_lora_parameters(shape, model_config.lora.r, targets)
    lora_weights_gb = lora_params * 2 / BYTES_PER_GB  # bf16/fp16 adapter weights
    gradients_gb = lora_params * 2 / BYTES_PER_GB  # only adapters get gradients
    optimizer_bytes = OPTIMIZER_BYTES.get(training_config.optim, 8.0)
    optimizer_gb = lora_params * optimizer_bytes / BYTES_PER_GB
    assumptions.append(
        f"optimizer {training_config.optim} at {optimizer_bytes:.0f} bytes per trainable parameter"
    )

    # --- activations --------------------------------------------------------
    multiplier = (
        ACTIVATION_MULTIPLIER_CHECKPOINT
        if training_config.gradient_checkpointing
        else ACTIVATION_MULTIPLIER_NO_CHECKPOINT
    )
    per_token_bytes = shape.hidden_size * shape.num_layers * 2
    activations_gb = batch * seq * per_token_bytes * multiplier / BYTES_PER_GB
    assumptions.append(
        f"activations for batch={batch} x seq={seq}, gradient_checkpointing="
        f"{training_config.gradient_checkpointing} (x{multiplier})"
    )

    # Logits over the vocabulary are a real and often-forgotten spike.
    logits_gb = batch * seq * shape.vocab_size * 4 / BYTES_PER_GB
    activations_gb += logits_gb
    assumptions.append(f"output logits in fp32 (+{logits_gb:.2f} GB)")

    # Single-sequence inference footprint, independent of the training batch.
    # A KV cache over a 2048-token context plus one row of logits.
    inference_seq = min(seq, 2048)
    inference_activations_gb = (
        inference_seq * shape.hidden_size * shape.num_layers * 2 * 2 / BYTES_PER_GB
        + shape.vocab_size * 4 / BYTES_PER_GB
    )

    confidence = "medium"
    if model_config.parameter_count is None:
        confidence = "low"
        assumptions.append(
            "parameter_count was not configured; a generic shape was assumed. "
            "Set model.parameter_count for a better estimate."
        )

    return MemoryEstimate(
        base_weights_gb=base_weights_gb,
        lora_params=lora_params,
        lora_weights_gb=lora_weights_gb,
        gradients_gb=gradients_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        overhead_gb=CUDA_OVERHEAD_GB,
        inference_activations_gb=inference_activations_gb,
        confidence=confidence,
        assumptions=assumptions,
    )


@dataclass
class Adjustment:
    """A recorded change to the requested configuration.

    Spec section 14: automatic fallbacks are permitted, silent ones are not.
    Every adjustment lands in the experiment manifest.
    """

    field: str
    original: Any
    adjusted: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "original": self.original,
            "adjusted": self.adjusted,
            "reason": self.reason,
        }

    def render(self) -> str:
        return f"{self.field}: {self.original} → {self.adjusted} ({self.reason})"


@dataclass
class FeasibilityReport:
    """Whether a run fits, and what to do about it."""

    tier: FeasibilityTier
    fits: bool
    gpu: GPUInfo
    estimate: MemoryEstimate
    model_name: str
    headroom_gb: float
    blocking_reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    proposed_adjustments: list[Adjustment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "tier": self.tier.value,
            "fits": self.fits,
            "headroom_gb": round(self.headroom_gb, 2),
            "gpu": self.gpu.to_dict(),
            "estimate": self.estimate.to_dict(),
            "blocking_reasons": self.blocking_reasons,
            "recommendations": self.recommendations,
            "proposed_adjustments": [a.to_dict() for a in self.proposed_adjustments],
        }

    def render(self) -> str:
        icon = {
            FeasibilityTier.FULL_RESEARCH: "✓",
            FeasibilityTier.ADAPTER_TRAIN: "✓",
            FeasibilityTier.SMOKE: "~",
            FeasibilityTier.INFERENCE_ONLY: "!",
            FeasibilityTier.INFEASIBLE: "✗",
        }[self.tier]
        lines = [
            f"{icon} {self.model_name}: {self.tier.value.replace('_', ' ').upper()}",
            f"  GPU: {self.gpu.render()}",
            "",
            self.estimate.render(),
            "",
            f"  headroom: {self.headroom_gb:+.2f} GB",
        ]
        if self.blocking_reasons:
            lines.extend(["", "  Blocking:"])
            lines.extend(f"    - {reason}" for reason in self.blocking_reasons)
        if self.recommendations:
            lines.extend(["", "  Suggested actions:"])
            lines.extend(f"    {i}. {rec}" for i, rec in enumerate(self.recommendations, start=1))
        return "\n".join(lines)


def assess_feasibility(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    gpu: GPUInfo | None = None,
    shape: ModelShape | None = None,
    target_modules: list[str] | None = None,
) -> FeasibilityReport:
    """Decide which feasibility tier a configuration reaches on the given GPU."""
    gpu = gpu or probe_gpu()
    estimate = estimate_memory(
        model_config,
        training_config,
        shape=shape,
        target_modules=target_modules,
    )

    budget = gpu.free_memory_gb if gpu.free_memory_gb > 0 else gpu.total_memory_gb
    headroom = budget - estimate.total_gb
    blocking: list[str] = []
    recommendations: list[str] = []
    adjustments: list[Adjustment] = []

    if not gpu.available:
        return FeasibilityReport(
            tier=FeasibilityTier.INFEASIBLE,
            fits=False,
            gpu=gpu,
            estimate=estimate,
            model_name=model_config.name,
            headroom_gb=0.0,
            blocking_reasons=[f"No CUDA GPU is available ({gpu.source})."],
            recommendations=[
                "Run training on a CUDA machine or a Colab GPU runtime.",
                "In Colab: Runtime → Change runtime type → T4 GPU.",
                "The data, validation, splitting and offline evaluation pipelines "
                "all run fine on CPU.",
            ],
        )

    # A smoke-sized variant: the smallest configuration still exercising the path.
    smoke_estimate = estimate_memory(
        model_config,
        training_config,
        shape=shape,
        target_modules=target_modules,
        seq_length=min(512, model_config.max_seq_length),
        batch_size=1,
    )

    if estimate.inference_gb > budget:
        tier = FeasibilityTier.INFEASIBLE
        blocking.append(
            f"Even inference needs about {estimate.inference_gb:.1f} GB, more than "
            f"the {budget:.1f} GB available."
        )
        recommendations.extend(
            [
                f"{model_config.base_model} does not fit this runtime at "
                f"{model_config.quantization.mode.value}.",
                "Use a smaller model config: configs/models/qwen3_8b.yaml or "
                "ministral_8b.yaml fit a 16GB T4 in 4-bit.",
                "A 24B or 30B model needs an A100-class GPU (40GB+) for adapter training.",
            ]
        )
    elif smoke_estimate.total_gb > budget:
        tier = FeasibilityTier.INFERENCE_ONLY
        blocking.append(
            f"Training needs about {smoke_estimate.total_gb:.1f} GB even at "
            f"seq=512/batch=1, more than the {budget:.1f} GB available."
        )
        recommendations.extend(
            [
                "This model can be evaluated here but not trained.",
                "Run the base-model evaluation arm on this runtime and train elsewhere.",
                "Or switch to an 8B config for training.",
            ]
        )
    elif estimate.total_gb > budget:
        tier = FeasibilityTier.SMOKE
        blocking.append(
            f"The requested seq={model_config.max_seq_length}/"
            f"batch={training_config.per_device_train_batch_size} needs about "
            f"{estimate.total_gb:.1f} GB, more than the {budget:.1f} GB available."
        )
        # Propose, do not apply. The caller decides, and strict_config can refuse.
        if model_config.max_seq_length > 512:
            adjustments.append(
                Adjustment(
                    field="model.max_seq_length",
                    original=model_config.max_seq_length,
                    adjusted=min(1024, model_config.max_seq_length // 2),
                    reason="reduce activation memory to fit the assigned GPU",
                )
            )
        if not training_config.gradient_checkpointing:
            adjustments.append(
                Adjustment(
                    field="training.gradient_checkpointing",
                    original=False,
                    adjusted=True,
                    reason="trades compute for a large activation-memory reduction",
                )
            )
        if training_config.per_device_train_batch_size > 1:
            adjustments.append(
                Adjustment(
                    field="training.per_device_train_batch_size",
                    original=training_config.per_device_train_batch_size,
                    adjusted=1,
                    reason=(
                        "reduce activation memory; raise gradient_accumulation_steps "
                        "to keep the effective batch size unchanged"
                    ),
                )
            )
        recommendations.extend(
            [
                "A smoke run fits, but the configured research settings do not.",
                "Applying the proposed adjustments changes what the run measures; "
                "they are recorded in the manifest so comparisons stay honest.",
                "Set training.strict_config=true to refuse adjustment instead.",
            ]
        )
    elif headroom < budget * RESEARCH_HEADROOM_FRACTION:
        tier = FeasibilityTier.ADAPTER_TRAIN
        recommendations.append(
            f"Fits with only {headroom:.1f} GB spare. Memory spikes during "
            "evaluation or checkpointing may still OOM; consider a shorter "
            "sequence length or per_device_eval_batch_size=1."
        )
    else:
        tier = FeasibilityTier.FULL_RESEARCH

    fits = tier in (FeasibilityTier.ADAPTER_TRAIN, FeasibilityTier.FULL_RESEARCH)

    return FeasibilityReport(
        tier=tier,
        fits=fits,
        gpu=gpu,
        estimate=estimate,
        model_name=f"{model_config.name} ({model_config.base_model})",
        headroom_gb=headroom,
        blocking_reasons=blocking,
        recommendations=recommendations,
        proposed_adjustments=adjustments,
    )


def enforce_feasibility(
    report: FeasibilityReport, training_config: TrainingConfig
) -> list[Adjustment]:
    """Apply the feasibility policy.

    Args:
        report: Assessment for this run.
        training_config: Supplies ``strict_config``.

    Returns:
        Adjustments that should be applied and recorded.

    Raises:
        InsufficientMemoryError: when the run cannot proceed, or when adjustments
            would be required but ``strict_config`` forbids them.
    """
    if report.tier is FeasibilityTier.INFEASIBLE:
        raise InsufficientMemoryError(
            f"{report.model_name} cannot run on the available hardware.",
            details={
                "gpu": report.gpu.render(),
                "estimated_peak_gb": round(report.estimate.total_gb, 2),
                "available_gb": round(report.gpu.free_memory_gb or report.gpu.total_memory_gb, 2),
            },
            suggestions=report.recommendations,
        )

    if report.tier is FeasibilityTier.INFERENCE_ONLY:
        raise InsufficientMemoryError(
            f"{report.model_name} fits for inference but not for training on this GPU.",
            details={
                "gpu": report.gpu.render(),
                "estimated_training_peak_gb": round(report.estimate.total_gb, 2),
            },
            suggestions=report.recommendations,
        )

    if report.proposed_adjustments and training_config.strict_config:
        raise InsufficientMemoryError(
            f"{report.model_name} does not fit the requested configuration, and "
            "training.strict_config=true forbids automatic adjustment.",
            details={
                "gpu": report.gpu.render(),
                "required_adjustments": "; ".join(a.render() for a in report.proposed_adjustments),
            },
            suggestions=[
                "Change the config yourself so the run stays comparable by design.",
                "Or set strict_config=false to accept recorded adjustments.",
                "strict_config exists so a research comparison cannot be invalidated "
                "by a silent fallback.",
            ],
        )

    for adjustment in report.proposed_adjustments:
        logger.warning("Auto-adjusting %s", adjustment.render())
    return report.proposed_adjustments


def render_feasibility_table(reports: list[FeasibilityReport]) -> str:
    """Render a comparison table across model configs for one runtime."""
    if not reports:
        return "(no models assessed)"

    gpu = reports[0].gpu
    lines = [
        "=" * 78,
        "Feasibility for the currently assigned runtime",
        "=" * 78,
        f"GPU: {gpu.render()}",
        "",
        f"{'model':<30} {'tier':<16} {'est. peak':>10} {'headroom':>10}",
        f"{'-' * 30} {'-' * 16} {'-' * 10} {'-' * 10}",
    ]
    for report in reports:
        name = report.model_name.split(" (")[0][:30]
        lines.append(
            f"{name:<30} {report.tier.value:<16} "
            f"{report.estimate.total_gb:>8.1f}GB {report.headroom_gb:>+9.1f}GB"
        )
    lines.extend(
        [
            "",
            "Tiers: full_research = fits with headroom | adapter_train = fits |",
            "       smoke = only a reduced config fits | inference_only = cannot train |",
            "       infeasible = cannot load",
            "",
            "Estimates are approximate (±20%) and intentionally slightly pessimistic.",
            "",
        ]
    )
    return "\n".join(lines)
