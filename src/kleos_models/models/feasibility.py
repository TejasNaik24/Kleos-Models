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

The estimate is arithmetic over explicit terms, each stating its assumption, and
it is checked against peaks measured on real runs (:data:`EMPIRICAL_ANCHORS`,
enforced by the tests). The fit decision then adds overheads measured on the same
hardware rather than a round-number margin. A run whose estimate is within that
overhead of the limit is reported as *marginal*: allowed to start, because the
trainer's memory probe measures the longest batch before the first step.
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

#: Memory PyTorch's caching allocator holds beyond the peak it hands out
#: (fragmentation slack). Measured on Hermes' T4 run: 13.58 GiB reserved against
#: a 13.09 GiB peak allocation (docs/experiments/kleos-v006-mistralnemo12b-run1-
#: report.md, section 4). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#: shrinks it; it is the one overhead a user can influence.
ALLOCATOR_RESERVE_GB = 0.5

#: Device memory in use outside PyTorch's allocator on that run: the CUDA context
#: plus bitsandbytes' paged optimizer state (57,016,320 LoRA parameters x 2 bytes
#: = 0.106 GiB of it). Measured at step 299: 14.56 total - 13.58 reserved - 0.84
#: free = 0.14 GiB. Subtracted from a budget that is total memory; free memory
#: measured live already excludes the context.
NON_ALLOCATOR_GB = 0.14
#: The paged optimizer state included in ``NON_ALLOCATOR_GB``.
REFERENCE_PAGED_GB = 57_016_320 * 2 / BYTES_PER_GB

#: Headroom below which a fit is reported as marginal: half the measured
#: allocator reserve, the least certain overhead (measured once). A policy
#: threshold, not a measurement.
MARGINAL_HEADROOM_GB = ALLOCATOR_RESERVE_GB / 2

#: bitsandbytes 4-bit state per quantized weight: one absmax per 64-weight block,
#: in fp32 (4/64 bytes); with double quantization, a uint8 per block plus an fp32
#: second-level absmax per 256 blocks.
NF4_STATE_BYTES = 4 / 64
NF4_DOUBLE_QUANT_STATE_BYTES = 1 / 64 + 4 / (64 * 256)

#: Bytes per logit at the loss: the 16-bit logits, their fp32 upcast inside the
#: loss, and the fp32 gradient flowing back.
LOGIT_BYTES = 2 + 4 + 4

#: Activations one decoder layer keeps for its backward pass, per token, in 16-bit:
#: about sixteen hidden-width tensors (norms, projections, attention output, LoRA
#: dropout copies) and eight intermediate-width ones (gate, up, activation,
#: product). With gradient checkpointing only one layer's worth is live at a time.
#: Modelling assumptions; the anchors below check the total.
LAYER_HIDDEN_TENSORS = 16
LAYER_INTERMEDIATE_TENSORS = 8

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
        free = (
            f"{self.free_memory_gb:.1f} GB free" if self.free_memory_gb > 0 else "free not measured"
        )
        return (
            f"{self.name} | {self.total_memory_gb:.2f} GB total, "
            f"{free} | compute capability "
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
    """Enough architecture to estimate memory without downloading weights.

    ``parameter_count`` counts the language model only; a vision tower is
    ``vision_parameter_count`` and costs memory only when it is loaded.
    """

    parameter_count: float
    hidden_size: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14336
    vocab_size: int = 32000
    active_parameter_count: float | None = None
    is_moe: bool = False
    #: Attention geometry. When known, LoRA counts are exact; otherwise k/v
    #: projections are approximated at a quarter of the hidden width.
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    tie_word_embeddings: bool = False
    vision_parameter_count: float = 0.0
    #: Where the shape came from: a verified table entry, a downloaded config,
    #: or the generic fallback.
    source: str = "generic"

    @property
    def embedding_parameters(self) -> float:
        """``embed_tokens`` plus an untied ``lm_head``."""
        return self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)

    @property
    def norm_parameters(self) -> float:
        """RMSNorm weights: two per layer and a final one."""
        return (2 * self.num_layers + 1) * self.hidden_size

    @property
    def quantizable_parameters(self) -> float:
        """Linear-layer weights: everything bitsandbytes quantizes."""
        return max(0.0, self.parameter_count - self.embedding_parameters - self.norm_parameters)

    @property
    def projection_dims(self) -> dict[str, tuple[int, int]]:
        """``(in, out)`` of each LoRA-targetable projection."""
        hidden = self.hidden_size
        if self.num_attention_heads and self.head_dim:
            q_out = self.num_attention_heads * self.head_dim
            kv_heads = self.num_key_value_heads or self.num_attention_heads
            kv_out = kv_heads * self.head_dim
        else:
            q_out = hidden
            kv_out = max(hidden // 4, 1)
        return {
            "q_proj": (hidden, q_out),
            "k_proj": (hidden, kv_out),
            "v_proj": (hidden, kv_out),
            "o_proj": (q_out, hidden),
            "gate_proj": (hidden, self.intermediate_size),
            "up_proj": (hidden, self.intermediate_size),
            "down_proj": (self.intermediate_size, hidden),
        }

    @classmethod
    def from_config(cls, config: ModelConfig) -> ModelShape:
        """Derive a shape from a model config, using known checkpoint facts.

        :data:`KNOWN_SHAPES` is verified against each checkpoint's published
        ``config.json``. Unknown models fall back to a generic 7B-ish shape and the
        estimate is flagged as low-confidence.
        """
        facts = KNOWN_SHAPES.get(config.base_model)
        if facts is not None:
            return cls(
                **{
                    **facts,
                    "active_parameter_count": (
                        config.active_parameter_count or facts.get("active_parameter_count")
                    ),
                    "is_moe": config.is_moe or bool(facts.get("is_moe", False)),
                    "source": f"verified shape for {config.base_model}",
                }
            )
        return cls(
            parameter_count=float(config.parameter_count or 7.0e9),
            active_parameter_count=config.active_parameter_count,
            is_moe=config.is_moe,
        )

    @classmethod
    def from_hf_config(cls, hf_config: Any, parameter_count: float) -> ModelShape:
        """Build a shape from a downloaded HF config (most accurate path)."""
        text_config = getattr(hf_config, "text_config", None) or hf_config

        def read(name: str, default: int | None) -> int | None:
            value = getattr(text_config, name, None)
            if value is None:
                value = getattr(hf_config, name, None)
            return int(value) if value is not None else default

        return cls(
            parameter_count=parameter_count,
            hidden_size=read("hidden_size", 4096) or 4096,
            num_layers=read("num_hidden_layers", 32) or 32,
            intermediate_size=read("intermediate_size", 14336) or 14336,
            vocab_size=read("vocab_size", 32000) or 32000,
            is_moe=hasattr(text_config, "num_experts"),
            num_attention_heads=read("num_attention_heads", None),
            num_key_value_heads=read("num_key_value_heads", None),
            head_dim=read("head_dim", None),
            tie_word_embeddings=bool(getattr(text_config, "tie_word_embeddings", False)),
            source="downloaded config.json",
        )


#: Architecture facts per checkpoint, from its ``config.json``. The Mistral
#: entries carry full attention geometry (so LoRA counts are exact) and the
#: text-only parameter count; the vision counts come from instantiating the
#: container on the meta device at the pinned revision (vision tower 403,305,472
#: plus projector 35,652,608).
KNOWN_SHAPES: dict[str, dict[str, Any]] = {
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
        "intermediate_size": 768,  # per expert
        "vocab_size": 151936,
        "is_moe": True,
    },
    # Verified against config.json; the Ministral-8B run's manifest reports the
    # 43,646,976 LoRA parameters this geometry gives at r=16 on all seven
    # projections.
    "mistralai/Ministral-8B-Instruct-2410": {
        "parameter_count": 8.0e9,
        "hidden_size": 4096,
        "num_layers": 36,
        "intermediate_size": 12288,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
    },
    # KLEOS Hermes. config.json @ 04d8a905; 12,247,782,400 parameters.
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "parameter_count": 12_247_782_400,
        "hidden_size": 5120,
        "num_layers": 40,
        "intermediate_size": 14336,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
    },
    # KLEOS Logos: the ministral3 text tower of the container, config.json @
    # 3cea74c1. 13,506,073,600 text parameters counted on the meta device.
    "mistralai/Ministral-3-14B-Instruct-2512-BF16": {
        "parameter_count": 13_506_073_600,
        "hidden_size": 5120,
        "num_layers": 40,
        "intermediate_size": 16384,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "vision_parameter_count": 438_958_080,
    },
    # Text 23,572,403,200 + vision 438,958,080 = 24.0B total.
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "parameter_count": 23_572_403_200,
        "hidden_size": 5120,
        "num_layers": 40,
        "intermediate_size": 32768,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "vision_parameter_count": 438_958_080,
    },
}


@dataclass
class MemoryEstimate:
    """Estimated peak VRAM, broken down so a user can see what to change.

    ``peak_allocated_gb`` is comparable with ``torch.cuda.max_memory_allocated``.
    ``total_gb`` is what PyTorch's allocator must be able to reserve: the peak
    plus the fragmentation reserve. ``minimum_gb`` is the peak alone. A paged
    optimizer's state lives outside the allocator and is taken from the budget
    instead (:func:`memory_budget_gb`).
    """

    base_weights_gb: float
    lora_params: int
    lora_weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    #: Part of ``base_weights_gb`` kept unquantized (embeddings, lm_head, norms,
    #: a loaded vision tower).
    unquantized_weights_gb: float = 0.0
    paged_optimizer: bool = False
    reserve_gb: float = ALLOCATOR_RESERVE_GB
    sequence_length: int = 0
    batch_size: int = 1
    confidence: str = "medium"
    assumptions: list[str] = field(default_factory=list)
    #: Weights as loaded for inference (no k-bit upcast), and a single-sequence
    #: activation budget computed independently of the training batch size.
    inference_weights_gb: float = 0.0
    inference_activations_gb: float = 0.0

    @property
    def paged_gb(self) -> float:
        return self.optimizer_gb if self.paged_optimizer else 0.0

    @property
    def peak_allocated_gb(self) -> float:
        optimizer = 0.0 if self.paged_optimizer else self.optimizer_gb
        return (
            self.base_weights_gb
            + self.lora_weights_gb
            + self.gradients_gb
            + optimizer
            + self.activations_gb
        )

    @property
    def minimum_gb(self) -> float:
        """Allocator memory needed if it did not fragment at all."""
        return self.peak_allocated_gb

    @property
    def total_gb(self) -> float:
        """Allocator memory needed with the fragmentation measured on a real run."""
        return self.minimum_gb + self.reserve_gb

    @property
    def inference_gb(self) -> float:
        """Weights plus a single-sequence activation budget, without training state.

        Deliberately independent of the *training* batch size and sequence length:
        whether a model can be loaded at all is a property of the weights, not of
        how large a training batch someone asked for.
        """
        return self.inference_weights_gb + self.inference_activations_gb + self.reserve_gb

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_weights_gb": round(self.base_weights_gb, 2),
            "unquantized_weights_gb": round(self.unquantized_weights_gb, 2),
            "lora_trainable_params": self.lora_params,
            "lora_weights_gb": round(self.lora_weights_gb, 3),
            "gradients_gb": round(self.gradients_gb, 3),
            "optimizer_gb": round(self.optimizer_gb, 3),
            "paged_optimizer": self.paged_optimizer,
            "activations_gb": round(self.activations_gb, 2),
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "peak_allocated_gb": round(self.peak_allocated_gb, 2),
            "reserve_gb": round(self.reserve_gb, 2),
            "minimum_gb": round(self.minimum_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "inference_gb": round(self.inference_gb, 2),
            "confidence": self.confidence,
            "assumptions": self.assumptions,
        }

    def render(self) -> str:
        optimizer_note = " (paged, outside the allocator)" if self.paged_optimizer else ""
        return "\n".join(
            [
                f"  base weights              {self.base_weights_gb:6.2f} GB "
                f"(of which unquantized {self.unquantized_weights_gb:.2f} GB)",
                f"  LoRA weights              {self.lora_weights_gb:6.3f} GB "
                f"({self.lora_params:,} trainable params)",
                f"  gradients                 {self.gradients_gb:6.3f} GB",
                f"  optimizer state           {self.optimizer_gb:6.3f} GB{optimizer_note}",
                f"  activations and logits    {self.activations_gb:6.2f} GB "
                f"(batch {self.batch_size} x seq {self.sequence_length})",
                f"  {'-' * 40}",
                f"  estimated peak allocated  {self.peak_allocated_gb:6.2f} GB "
                f"(confidence: {self.confidence})",
                f"  + allocator reserve       {self.reserve_gb:6.2f} GB",
                f"  required                  {self.total_gb:6.2f} GB",
            ]
        )


def estimate_lora_parameters(shape: ModelShape, rank: int, target_modules: list[str]) -> int:
    """Count trainable LoRA parameters for a target set.

    A LoRA on a ``d_in x d_out`` matrix adds ``r * (d_in + d_out)`` parameters.
    Exact when the shape carries its attention geometry; unknown module names
    are counted as hidden x hidden.
    """
    dims = shape.projection_dims
    fallback = (shape.hidden_size, shape.hidden_size)
    per_layer = sum(rank * sum(dims.get(module, fallback)) for module in target_modules)
    return per_layer * shape.num_layers


@dataclass(frozen=True)
class MemoryTerms:
    """The arithmetic behind one estimate, in bytes."""

    quantized_weights: float
    unquantized_weights: float
    lora_weights: float
    gradients: float
    optimizer: float
    activations: float
    inference_weights: float
    inference_activations: float


def memory_terms(
    shape: ModelShape,
    *,
    quantization_mode: QuantizationMode,
    double_quant: bool,
    lora_params: int,
    optim: str,
    gradient_checkpointing: bool,
    batch: int,
    seq: int,
    loads_vision_tower: bool,
) -> MemoryTerms:
    """Every memory term of a QLoRA/LoRA training step, from first principles.

    What the terms encode, beyond the obvious:

    * ``prepare_model_for_kbit_training`` upcasts every weight it cannot
      quantize (embeddings, ``lm_head``, norms, a loaded vision tower) to fp32.
      For a 131k-token vocabulary that is ~5 GiB, the term earlier versions of
      this estimator missed (finding H-F10).
    * Under 16-bit autocast the fp32 ``lm_head`` is cast to a 16-bit copy for its
      matmul, and that copy is saved for the backward pass.
    * PEFT keeps adapter weights, and so their gradients, in fp32.
    * The residual stream is fp32 after the embedding upcast, so each layer
      boundary saved by gradient checkpointing costs 4 bytes per value.
    """
    quantized = quantization_mode is not QuantizationMode.NONE
    if quantization_mode.is_4bit:
        per_weight = 0.5 + (NF4_DOUBLE_QUANT_STATE_BYTES if double_quant else NF4_STATE_BYTES)
    else:
        per_weight = quantization_mode.bits / 8.0
    upcast_bytes = 4.0 if quantized else 2.0

    unquantized_params = shape.embedding_parameters + shape.norm_parameters
    if loads_vision_tower:
        unquantized_params += shape.vision_parameter_count

    quantized_weights = shape.quantizable_parameters * per_weight
    unquantized_weights = unquantized_params * upcast_bytes

    lora_weights = lora_params * 4.0
    gradients = lora_params * 4.0
    optimizer = lora_params * OPTIMIZER_BYTES.get(optim, 8.0)

    tokens = batch * seq
    hidden, inter, layers = shape.hidden_size, shape.intermediate_size, shape.num_layers
    residual_bytes = 4.0 if quantized else 2.0
    one_layer = tokens * (LAYER_HIDDEN_TENSORS * hidden + LAYER_INTERMEDIATE_TENSORS * inter) * 2
    boundaries = tokens * hidden * layers * residual_bytes
    layer_activations = boundaries + (one_layer if gradient_checkpointing else one_layer * layers)
    logits = tokens * shape.vocab_size * LOGIT_BYTES
    head_cast = shape.vocab_size * hidden * 2.0 if quantized else 0.0
    dequantized_weight = hidden * inter * 2.0 if quantized else 0.0

    # Inference: weights as loaded (no fp32 upcast), a KV cache over up to 2048
    # tokens and one row of fp32 logits.
    inference_seq = min(seq, 2048)
    if shape.num_key_value_heads and shape.head_dim:
        kv_width = shape.num_key_value_heads * shape.head_dim
    else:
        kv_width = hidden
    inference_activations = 2 * layers * kv_width * 2 * inference_seq + shape.vocab_size * 4

    return MemoryTerms(
        quantized_weights=quantized_weights,
        unquantized_weights=unquantized_weights,
        lora_weights=lora_weights,
        gradients=gradients,
        optimizer=optimizer,
        activations=layer_activations + logits + head_cast + dequantized_weight,
        inference_weights=quantized_weights + unquantized_params * 2.0,
        inference_activations=float(inference_activations),
    )


@dataclass(frozen=True)
class EmpiricalAnchor:
    """A training peak measured on a real run, used to check the estimator.

    ``measured_peak_gb`` is ``torch.cuda.max_memory_allocated`` over the run, in
    GiB. ``sequence_length`` is the longest training sequence after padding,
    which is what sets the activation peak at batch size 1.
    """

    run: str
    base_model: str
    measured_peak_gb: float
    sequence_length: int
    source: str
    lora_r: int = 16
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    optim: str = "paged_adamw_8bit"

    def estimate_gb(self) -> float:
        """The estimator's peak allocation for this run's configuration."""
        shape = ModelShape(**KNOWN_SHAPES[self.base_model])
        terms = memory_terms(
            shape,
            quantization_mode=QuantizationMode.NF4,
            double_quant=True,
            lora_params=estimate_lora_parameters(shape, self.lora_r, list(self.target_modules)),
            optim=self.optim,
            gradient_checkpointing=True,
            batch=1,
            seq=self.sequence_length,
            loads_vision_tower=False,
        )
        paged = "paged" in self.optim
        allocated = (
            terms.quantized_weights
            + terms.unquantized_weights
            + terms.lora_weights
            + terms.gradients
            + (0.0 if paged else terms.optimizer)
            + terms.activations
        )
        return allocated / BYTES_PER_GB

    def deviation(self) -> float:
        """Relative error of the estimate: positive means pessimistic."""
        return self.estimate_gb() / self.measured_peak_gb - 1.0


#: Peaks measured on the project's own T4 runs: QLoRA NF4 with double
#: quantization, r=16 on all seven projections, batch 1, gradient
#: checkpointing, paged 8-bit AdamW, fp16.
EMPIRICAL_ANCHORS: tuple[EmpiricalAnchor, ...] = (
    EmpiricalAnchor(
        run="kleos-v006-mistralnemo12b-run1 (Hermes)",
        base_model="mistralai/Mistral-Nemo-Instruct-2407",
        measured_peak_gb=13.09,
        # Measured: the repo's formatter over the v0.0.6 train split with Nemo's
        # tokenizer at the pinned revision gives a longest example of 440 tokens.
        sequence_length=440,
        source="docs/experiments/kleos-v006-mistralnemo12b-run1-report.md, section 4",
    ),
    EmpiricalAnchor(
        run="kleos-v006-ministral8b-run1",
        base_model="mistralai/Ministral-8B-Instruct-2410",
        measured_peak_gb=9.67,
        # Not measured directly (the tokenizer is gated): same data, the same
        # Tekken vocabulary and [INST] template as Nemo, and the run's config
        # records a longest example of ~440 tokens.
        sequence_length=440,
        source="docs/experiments/kleos-v006-ministral8b-run1-artifact-audit.md",
    ),
)


def estimate_memory(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    shape: ModelShape | None = None,
    target_modules: list[str] | None = None,
    seq_length: int | None = None,
    batch_size: int | None = None,
    loads_vision_tower: bool | None = None,
) -> MemoryEstimate:
    """Estimate peak training VRAM for a configuration.

    Args:
        seq_length: The longest training sequence. Defaults to
            ``model.max_seq_length``, the worst case the configuration allows.
            With batch size 1 and padding to the longest in the batch, the real
            peak follows the longest example actually present, so pass it when
            known (``scripts/train.py`` measures it before loading weights).
        loads_vision_tower: Whether a vision tower is resident. Defaults to
            ``model.is_multimodal`` (a text-only view loads none).

    The arithmetic is deliberately explicit rather than a fitted heuristic, so a
    user can see which term is dominating and change the right knob.
    """
    shape = shape or ModelShape.from_config(model_config)
    targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    seq = seq_length or model_config.max_seq_length
    batch = batch_size or training_config.per_device_train_batch_size
    vision = model_config.is_multimodal if loads_vision_tower is None else loads_vision_tower
    quant = model_config.quantization
    paged = "paged" in training_config.optim

    lora_params = estimate_lora_parameters(shape, model_config.lora.r, targets)
    terms = memory_terms(
        shape,
        quantization_mode=quant.mode,
        double_quant=quant.double_quant,
        lora_params=lora_params,
        optim=training_config.optim,
        gradient_checkpointing=training_config.gradient_checkpointing,
        batch=batch,
        seq=seq,
        loads_vision_tower=vision,
    )

    assumptions: list[str] = []
    quantized_gb = terms.quantized_weights / BYTES_PER_GB
    unquantized_gb = terms.unquantized_weights / BYTES_PER_GB
    per_weight = (
        terms.quantized_weights / shape.quantizable_parameters
        if shape.quantizable_parameters
        else 0.0
    )
    assumptions.append(
        f"{shape.quantizable_parameters / 1e9:.2f}B linear-layer parameters at "
        f"{quant.mode.bits}-bit ({per_weight:.3f} bytes/param including quantization state)"
    )
    if quant.mode is not QuantizationMode.NONE:
        assumptions.append(
            f"embeddings, lm_head and norms upcast to fp32 by k-bit preparation "
            f"(+{unquantized_gb:.2f} GB)"
        )
    if vision and shape.vision_parameter_count:
        assumptions.append(
            f"vision tower resident and unquantized "
            f"({shape.vision_parameter_count / 1e9:.2f}B parameters)"
        )
    if shape.is_moe:
        assumptions.append(
            "MoE: ALL experts are resident in memory even though only a subset "
            "activates per token, so weight memory tracks total, not active, params"
        )
    optimizer_bytes = OPTIMIZER_BYTES.get(training_config.optim, 8.0)
    assumptions.append(
        f"LoRA weights and gradients in fp32; optimizer {training_config.optim} at "
        f"{optimizer_bytes:.0f} bytes per trainable parameter"
        + (" (paged: held outside PyTorch's allocator)" if paged else "")
    )
    assumptions.append(
        f"activations for batch={batch} x seq={seq}, gradient_checkpointing="
        f"{training_config.gradient_checkpointing}, including fp32 logits at the loss"
        + (
            " and the 16-bit autocast copy of lm_head"
            if quant.mode is not QuantizationMode.NONE
            else ""
        )
    )
    if seq_length is None:
        assumptions.append(
            f"seq={seq} is model.max_seq_length, the worst case; the real peak "
            "follows the longest example in the data"
        )
    assumptions.append(
        f"allocator reserve {ALLOCATOR_RESERVE_GB} GB, as measured on a T4 run; "
        + "; ".join(
            f"{anchor.run}: measured {anchor.measured_peak_gb:.2f} GB, "
            f"estimated {anchor.estimate_gb():.2f} GB"
            for anchor in EMPIRICAL_ANCHORS
        )
    )

    confidence = "medium" if shape.source != "generic" else "low"
    if model_config.parameter_count is None and shape.source == "generic":
        assumptions.append(
            "parameter_count was not configured; a generic shape was assumed. "
            "Set model.parameter_count for a better estimate."
        )
    elif shape.source == "generic":
        assumptions.append(
            f"no verified shape for {model_config.base_model}: generic layer "
            "geometry assumed around the configured parameter count"
        )

    return MemoryEstimate(
        base_weights_gb=quantized_gb + unquantized_gb,
        unquantized_weights_gb=unquantized_gb,
        lora_params=lora_params,
        lora_weights_gb=terms.lora_weights / BYTES_PER_GB,
        gradients_gb=terms.gradients / BYTES_PER_GB,
        optimizer_gb=terms.optimizer / BYTES_PER_GB,
        paged_optimizer=paged,
        activations_gb=terms.activations / BYTES_PER_GB,
        sequence_length=seq,
        batch_size=batch,
        inference_weights_gb=terms.inference_weights / BYTES_PER_GB,
        inference_activations_gb=terms.inference_activations / BYTES_PER_GB,
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
    #: Memory the run may use: free memory when probed live, otherwise total
    #: memory less the CUDA context.
    budget_gb: float = 0.0
    #: Fits only if the allocator fragments less than on the reference run: the
    #: estimate without the reserve fits, the estimate with it does not.
    marginal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "tier": self.tier.value,
            "fits": self.fits,
            "marginal": self.marginal,
            "budget_gb": round(self.budget_gb, 2),
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
        title = self.tier.value.replace("_", " ").upper()
        if self.marginal:
            title += " (MARGINAL)"
        lines = [
            f"{icon} {self.model_name}: {title}",
            f"  GPU: {self.gpu.render()}",
            "",
            self.estimate.render(),
            "",
            f"  budget:   {self.budget_gb:.2f} GB",
            f"  headroom: {self.headroom_gb:+.2f} GB",
        ]
        if self.blocking_reasons:
            lines.extend(["", "  Blocking:"])
            lines.extend(f"    - {reason}" for reason in self.blocking_reasons)
        if self.recommendations:
            lines.extend(["", "  Suggested actions:"])
            lines.extend(f"    {i}. {rec}" for i, rec in enumerate(self.recommendations, start=1))
        return "\n".join(lines)


def memory_budget_gb(gpu: GPUInfo, *, paged_gb: float = 0.0) -> float:
    """Memory PyTorch's allocator may use on this GPU.

    Measured free memory, less the paged optimizer state that will be allocated
    outside the allocator later. Without a measurement: total memory less the
    non-allocator use of the reference run, plus any paged state beyond its.
    """
    if gpu.free_memory_gb > 0:
        return max(0.0, gpu.free_memory_gb - paged_gb)
    excess_paged = max(0.0, paged_gb - REFERENCE_PAGED_GB)
    return max(0.0, gpu.total_memory_gb - NON_ALLOCATOR_GB - excess_paged)


def assess_feasibility(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    gpu: GPUInfo | None = None,
    shape: ModelShape | None = None,
    target_modules: list[str] | None = None,
    seq_length: int | None = None,
) -> FeasibilityReport:
    """Decide which feasibility tier a configuration reaches on the given GPU.

    Args:
        seq_length: The longest training sequence, when measured. Defaults to
            ``model.max_seq_length`` (see :func:`estimate_memory`).
    """
    gpu = gpu or probe_gpu()
    estimate = estimate_memory(
        model_config,
        training_config,
        shape=shape,
        target_modules=target_modules,
        seq_length=seq_length,
    )

    budget = memory_budget_gb(gpu, paged_gb=estimate.paged_gb)
    headroom = budget - estimate.total_gb
    marginal = False
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
            budget_gb=0.0,
            blocking_reasons=[f"No CUDA GPU is available ({gpu.source})."],
            recommendations=[
                "Run training on a CUDA machine or a Colab GPU runtime.",
                "In Colab: Runtime → Change runtime type → T4 GPU.",
                "The data, validation, splitting and offline evaluation pipelines "
                "all run fine on CPU.",
            ],
        )

    # A smoke-sized variant: the smallest configuration still exercising the path,
    # which includes the gradient checkpointing an adjustment would switch on.
    smoke_estimate = estimate_memory(
        model_config,
        training_config.model_copy(update={"gradient_checkpointing": True}),
        shape=shape,
        target_modules=target_modules,
        seq_length=min(512, seq_length or model_config.max_seq_length),
        batch_size=1,
    )

    if estimate.inference_gb > memory_budget_gb(gpu):
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
    elif smoke_estimate.minimum_gb > budget:
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
    elif estimate.minimum_gb > budget:
        tier = FeasibilityTier.SMOKE
        blocking.append(
            f"The requested seq={estimate.sequence_length}/"
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
    elif headroom < MARGINAL_HEADROOM_GB:
        # Too close to call: the answer depends on allocator fragmentation, which
        # was measured once. Not refused: the trainer's memory probe measures the
        # longest batch before the first step, so a wrong call costs minutes.
        tier = FeasibilityTier.ADAPTER_TRAIN
        marginal = True
        recommendations.extend(
            [
                f"MARGINAL: the estimated peak ({estimate.peak_allocated_gb:.2f} GB allocated) "
                f"leaves {budget - estimate.minimum_gb:.2f} GB of the {budget:.2f} GB budget "
                f"for allocator fragmentation; the reference run used "
                f"{ALLOCATOR_RESERVE_GB} GB.",
                "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting "
                "Python. It changes allocation only, not the computation.",
                "Treat the trainer's memory probe (longest batch, before step 1) as "
                "the decision; this estimate cannot settle a margin this small.",
            ]
        )
    elif headroom < budget * RESEARCH_HEADROOM_FRACTION:
        tier = FeasibilityTier.ADAPTER_TRAIN
        recommendations.append(
            f"Fits with only {headroom:.1f} GB spare. Memory spikes during "
            "evaluation or checkpointing may still OOM; consider a shorter "
            "sequence length or per_device_eval_batch_size=1."
        )
        if headroom < 1.0:
            recommendations.append(
                "Under 1 GB spare: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                "and check the trainer's memory probe before a long run."
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
        budget_gb=budget,
        marginal=marginal,
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
            "Estimates reproduce the measured peaks of the project's own T4 runs; the",
            "budget allows for the allocator reserve and non-allocator memory measured there.",
            "",
        ]
    )
    return "\n".join(lines)


#: Runtimes this project has measured, for planning without the hardware
#: (``plan_run.py --simulate-gpu``). Only measured capacities belong here.
GPU_PRESETS: dict[str, GPUInfo] = {
    "t4-colab": GPUInfo(
        available=True,
        name="Tesla T4 (Colab free tier, simulated)",
        total_memory_gb=14.56,
        # Unmeasured at start-up, so the budget comes from total memory.
        free_memory_gb=0.0,
        compute_capability=(7, 5),
        device_count=1,
        cuda_version="12.8",
        source=(
            "preset t4-colab: the capacity torch reported on Colab's T4 "
            "(Hermes run report, section 4)"
        ),
    ),
}


def simulated_gpu(spec: str) -> GPUInfo:
    """A GPU to plan against: a preset name, or ``NAME:TOTAL_GIB:MAJOR.MINOR``.

    The custom form is for hardware without a measured preset, e.g.
    ``L4:22.0:8.9``; its budget is the stated total less the measured
    non-allocator overhead.

    Raises:
        ValueError: for an unknown preset or a malformed custom spec.
    """
    if spec in GPU_PRESETS:
        return GPU_PRESETS[spec]
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Unknown GPU {spec!r}. Use a preset ({', '.join(sorted(GPU_PRESETS))}) "
            "or NAME:TOTAL_GIB:MAJOR.MINOR, e.g. L4:22.0:8.9."
        )
    name, total, capability = parts
    try:
        total_gb = float(total)
        major, minor = (int(x) for x in capability.split("."))
    except ValueError as exc:
        raise ValueError(f"Malformed GPU spec {spec!r}: {exc}") from exc
    if total_gb <= 0:
        raise ValueError(f"GPU memory must be positive in {spec!r}.")
    return GPUInfo(
        available=True,
        name=f"{name} (simulated)",
        total_memory_gb=total_gb,
        free_memory_gb=0.0,
        compute_capability=(major, minor),
        device_count=1,
        source=f"custom spec {spec!r}: stated capacity, not measured",
    )
