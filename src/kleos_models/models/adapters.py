"""Model-family adapters (spec sections 3, 38, 39, 46).

Everything that differs between Qwen and Mistral lives behind this interface.
The training and evaluation pipelines never branch on model family; the task
definition, dataset, split and rubric stay identical across families so that a
Qwen-vs-Mistral comparison measures the model, not the harness.

Why a family abstraction is not optional
----------------------------------------
These are real, verified differences between the four supported checkpoints:

======================================  ==========================  ========================
Checkpoint                              Auto class                  LoRA targeting
======================================  ==========================  ========================
Qwen/Qwen3-8B                           AutoModelForCausalLM        attention + MLP
Qwen/Qwen3-30B-A3B-Thinking-2507        AutoModelForCausalLM        attention only (128 experts)
mistralai/Mistral-Small-3.2-24B-...     AutoModelForImageTextToText language_model.* only
mistralai/Ministral-8B-Instruct-2410    AutoModelForCausalLM        attention + MLP
======================================  ==========================  ========================

Mistral Small 3.2 declares ``Mistral3ForConditionalGeneration`` with
``model_type: mistral3``, and transformers registers ``mistral3`` **only** in
``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``. ``AutoModelForCausalLM`` cannot
load it. It is a vision-language model, so LoRA must be scoped to the language
tower and the vision tower must stay frozen and unquantized.

Qwen3-30B-A3B is a mixture of experts: 48 layers x 128 experts. Targeting
``mlp.experts.*`` would create roughly 18,000 adapter modules, which is not a
sensible default. Attention-only targeting is.

Reasoning is a capability, not a prompt hack
--------------------------------------------
Qwen3-8B accepts ``enable_thinking``; Qwen3-30B-A3B-Thinking-2507 is thinking-only
and its template does not accept the flag at all; Mistral models have no reasoning
mode. Requesting an unsupported mode raises rather than silently producing a
mangled prompt, and we never inject ``<think>`` strings to fake reasoning.

This module imports torch/transformers lazily, inside methods, so it stays
importable for introspection in the light environment.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kleos_models.config import (
    LoRAConfig,
    ModelConfig,
    ReasoningCapability,
    ReasoningMode,
)
from kleos_models.errors import ModelCompatibilityError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCapabilities:
    """What a checkpoint can do, independent of what a config asks for."""

    family: str
    model_type: str
    auto_class: str
    reasoning: ReasoningCapability
    is_moe: bool = False
    is_multimodal: bool = False
    supports_gradient_checkpointing: bool = True
    supports_4bit: bool = True
    supports_flash_attention: bool = True
    default_context_limit: int | None = None
    #: Prefix under which the trainable language model lives ("" for plain LMs).
    language_model_prefix: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_type": self.model_type,
            "auto_class": self.auto_class,
            "reasoning": self.reasoning.value,
            "is_moe": self.is_moe,
            "is_multimodal": self.is_multimodal,
            "supports_4bit": self.supports_4bit,
            "default_context_limit": self.default_context_limit,
            "language_model_prefix": self.language_model_prefix,
            "notes": self.notes,
        }


@dataclass
class TargetModuleResolution:
    """Outcome of resolving LoRA target modules against a real model."""

    requested: list[str]
    matched: list[str]
    missing: list[str]
    excluded: list[str]
    matched_module_count: int
    candidates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and self.matched_module_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "matched": self.matched,
            "missing": self.missing,
            "excluded_patterns": self.excluded,
            "matched_module_count": self.matched_module_count,
        }


class ModelFamilyAdapter(ABC):
    """Per-family behaviour: loading, templates, reasoning, LoRA targeting."""

    #: ``config.model_type`` values this adapter handles.
    model_types: tuple[str, ...] = ()
    #: Family label recorded in manifests.
    family: str = "unknown"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # -- capability declaration --------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Static facts about this family."""

    @property
    @abstractmethod
    def default_target_modules(self) -> list[str]:
        """Architecture-appropriate LoRA targets.

        Chosen by inspecting the actual architecture, not copied from a Llama
        tutorial. They are still validated against the loaded model.
        """

    @property
    def excluded_module_patterns(self) -> list[str]:
        """Name fragments that must never receive an adapter."""
        return ["lm_head", "embed_tokens"]

    @property
    def modules_to_not_quantize(self) -> list[str]:
        """Modules kept in full precision when quantizing."""
        return ["lm_head"]

    # -- loading ------------------------------------------------------------

    def auto_model_class(self) -> Any:
        """Return the transformers auto class able to load this checkpoint.

        Raises:
            ModelCompatibilityError: when the named class does not exist.
        """
        from kleos_models.compat import require_transformers

        transformers = require_transformers()
        name = self.capabilities.auto_class
        cls = getattr(transformers, name, None)
        if cls is None:
            raise ModelCompatibilityError(
                f"transformers has no auto class {name!r}.",
                details={
                    "installed_transformers": getattr(transformers, "__version__", "unknown"),
                    "model": self.config.base_model,
                },
                suggestions=[
                    f"{name} requires a newer transformers; "
                    'install with: pip install -U "transformers>=4.56,<6"',
                ],
            )
        return cls

    def model_load_kwargs(self) -> dict[str, Any]:
        """Extra ``from_pretrained`` kwargs for this family."""
        kwargs: dict[str, Any] = {}
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        return kwargs

    # -- reasoning ----------------------------------------------------------

    def resolve_reasoning_mode(self, requested: ReasoningMode | None = None) -> ReasoningMode:
        """Validate a requested reasoning mode against real capability.

        Raises:
            ModelCompatibilityError: when the checkpoint cannot honour the request.
        """
        capability = self.capabilities.reasoning
        mode = requested or self.config.reasoning.default_mode

        if capability is ReasoningCapability.UNSUPPORTED:
            if mode is ReasoningMode.THINKING:
                raise ModelCompatibilityError(
                    f"{self.config.base_model} has no reasoning mode, but "
                    f"reasoning mode {mode.value!r} was requested.",
                    details={"family": self.family, "capability": capability.value},
                    suggestions=[
                        "Set model.reasoning.default_mode to 'standard'.",
                        "Use a reasoning-capable checkpoint such as "
                        "Qwen/Qwen3-30B-A3B-Thinking-2507 for reasoning experiments.",
                        "Do not simulate reasoning by injecting <think> tags: that "
                        "measures formatting, not reasoning.",
                    ],
                )
            return ReasoningMode.STANDARD

        if capability is ReasoningCapability.ALWAYS_ON and mode is not ReasoningMode.THINKING:
            raise ModelCompatibilityError(
                f"{self.config.base_model} is a thinking-only checkpoint and cannot "
                f"run in {mode.value!r} mode.",
                details={"family": self.family, "capability": capability.value},
                suggestions=[
                    "Set model.reasoning.default_mode to 'thinking'.",
                    "For a switchable model, use Qwen/Qwen3-8B instead.",
                    "Comparing a thinking-only model against a non-thinking one is a "
                    "valid experiment, but it must be configured deliberately and "
                    "recorded in the manifest.",
                ],
            )
        return mode

    def chat_template_kwargs(self, mode: ReasoningMode) -> dict[str, Any]:
        """Template kwargs implementing the reasoning mode. Empty by default."""
        return {}

    # -- LoRA targeting -----------------------------------------------------

    def resolve_target_modules(self, lora: LoRAConfig) -> list[str]:
        """Config targets, or family defaults when set to ``auto``."""
        if lora.target_modules == "auto":
            targets = self.default_target_modules
            logger.info(
                "Resolved LoRA target_modules=auto to %s for family %r",
                targets,
                self.family,
            )
            return targets
        return list(lora.target_modules)

    def validate_target_modules(self, model: Any, lora: LoRAConfig) -> TargetModuleResolution:
        """Check requested targets exist in the loaded model (spec section 46).

        A target that matches nothing produces an adapter that trains nothing, and
        a loss curve that looks plausible while the model never changes. This
        turns that silent failure into a hard error.
        """
        requested = self.resolve_target_modules(lora)
        excluded = [*self.excluded_module_patterns, *lora.exclude_modules]
        prefix = self.capabilities.language_model_prefix

        module_names = [name for name, _ in model.named_modules() if name]
        leaf_names = [name for name in module_names if _is_leaf_linear(model, name)]

        matched: set[str] = set()
        matched_count = 0
        for name in leaf_names:
            suffix = name.rsplit(".", 1)[-1]
            if any(pattern in name for pattern in excluded):
                continue
            if prefix and not name.startswith(prefix):
                continue
            if suffix in requested:
                matched.add(suffix)
                matched_count += 1

        missing = sorted(set(requested) - matched)
        resolution = TargetModuleResolution(
            requested=requested,
            matched=sorted(matched),
            missing=missing,
            excluded=excluded,
            matched_module_count=matched_count,
            candidates=sorted({n.rsplit(".", 1)[-1] for n in leaf_names}),
        )

        if missing:
            raise ModelCompatibilityError(
                f"LoRA target module(s) {missing} do not exist in {self.config.base_model}.",
                details={
                    "requested": requested,
                    "matched": resolution.matched,
                    "missing": missing,
                    "available_leaf_module_names": ", ".join(resolution.candidates[:40]),
                    "architecture": type(model).__name__,
                },
                suggestions=[
                    "List real candidates: python scripts/inspect_model.py --model "
                    f"{self.config.base_model}",
                    "Set model.lora.target_modules to 'auto' to use family defaults.",
                    "Do not copy target modules from a tutorial for a different architecture.",
                ],
            )

        if matched_count == 0:
            raise ModelCompatibilityError(
                f"No modules matched the LoRA targets {requested} in "
                f"{self.config.base_model} after exclusions.",
                details={"excluded_patterns": excluded, "language_model_prefix": prefix},
                suggestions=[
                    "Check model.lora.exclude_modules is not filtering everything.",
                    "Run scripts/inspect_model.py to see the module tree.",
                ],
            )

        logger.info(
            "LoRA targets %s matched %d module(s) in %s",
            resolution.matched,
            matched_count,
            type(model).__name__,
        )
        return resolution

    def prepare_model_for_training(self, model: Any) -> Any:
        """Family-specific adjustments before attaching the adapter."""
        return model

    # -- metadata -----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Manifest-ready description of the model.

        Recorded on every run so results cannot be pooled across incompatible
        checkpoints (spec section 38).
        """
        capabilities = self.capabilities
        return {
            "name": self.config.name,
            "family": self.family,
            "base_model": self.config.base_model,
            "revision": self.config.revision,
            "tokenizer": self.config.tokenizer_id,
            "model_type": capabilities.model_type,
            "architecture": self.config.architecture,
            "auto_class": capabilities.auto_class,
            "parameter_count": self.config.parameter_count,
            "active_parameter_count": self.config.active_parameter_count,
            "is_moe": capabilities.is_moe,
            "is_multimodal": capabilities.is_multimodal,
            "reasoning_capability": capabilities.reasoning.value,
            "context_limit": self.config.context_limit or capabilities.default_context_limit,
            "max_seq_length": self.config.max_seq_length,
            "notes": capabilities.notes,
        }


def _is_leaf_linear(model: Any, name: str) -> bool:
    """Whether the named module is a leaf that LoRA can wrap.

    Checks for a ``weight`` attribute with 2 dimensions, which covers ``nn.Linear``
    and the bitsandbytes 4-bit/8-bit replacements without importing either.
    """
    module = model.get_submodule(name) if hasattr(model, "get_submodule") else None
    if module is None:
        return False
    if len(list(module.children())) > 0:
        return False
    weight = getattr(module, "weight", None)
    if weight is None:
        return False
    shape = getattr(weight, "shape", ())
    return len(shape) == 2


# ---------------------------------------------------------------------------
# Qwen dense (Qwen3-8B)
# ---------------------------------------------------------------------------


class QwenDenseAdapter(ModelFamilyAdapter):
    """Qwen3 dense causal LMs, e.g. ``Qwen/Qwen3-8B``.

    Qwen3 supports switchable thinking through the chat template's
    ``enable_thinking`` argument.
    """

    model_types = ("qwen3", "qwen2", "qwen2_5")
    family = "qwen"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            family="qwen",
            model_type=self.config.model_type or "qwen3",
            auto_class="AutoModelForCausalLM",
            reasoning=ReasoningCapability.SWITCHABLE,
            is_moe=False,
            is_multimodal=False,
            supports_4bit=True,
            default_context_limit=40960,
            notes=[
                "Thinking mode is switchable via the chat template's enable_thinking argument.",
            ],
        )

    @property
    def default_target_modules(self) -> list[str]:
        # Attention projections plus the dense MLP. Standard for Qwen3 dense and
        # verified against the published architecture.
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def chat_template_kwargs(self, mode: ReasoningMode) -> dict[str, Any]:
        # Qwen3's template reads enable_thinking; passing it explicitly keeps the
        # mode recorded and reproducible rather than relying on a template default.
        return {"enable_thinking": mode is ReasoningMode.THINKING}


# ---------------------------------------------------------------------------
# Qwen MoE (Qwen3-30B-A3B-Thinking-2507)
# ---------------------------------------------------------------------------


class QwenMoEAdapter(ModelFamilyAdapter):
    """Qwen3 mixture-of-experts checkpoints.

    ``Qwen/Qwen3-30B-A3B-Thinking-2507`` has 48 layers and 128 experts with 8
    active per token. Two consequences drive this adapter:

    1. **LoRA targets attention only.** Adapting ``mlp.experts.*`` would create
       roughly 48 x 128 x 3 = 18,432 adapter modules. Beyond being impractical,
       each expert sees only a fraction of tokens, so per-expert adapters train on
       very little data.
    2. **The router must never be adapted or quantized.** Perturbing ``mlp.gate``
       changes expert routing, which destabilizes training in a way that is hard
       to attribute.

    It is also thinking-only: the template always opens ``<think>`` and does not
    accept ``enable_thinking``.
    """

    model_types = ("qwen3_moe",)
    family = "qwen"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            family="qwen",
            model_type=self.config.model_type or "qwen3_moe",
            auto_class="AutoModelForCausalLM",
            reasoning=ReasoningCapability.ALWAYS_ON,
            is_moe=True,
            is_multimodal=False,
            supports_4bit=True,
            default_context_limit=262144,
            notes=[
                "Mixture of experts: ~30B total parameters, ~3B active per token. "
                "Do not plan compute as if it were a dense 30B model.",
                "Thinking-only checkpoint: the chat template does not accept "
                "enable_thinking and always emits a reasoning span.",
                "LoRA targets attention only; adapting 128 experts per layer is "
                "neither practical nor well-conditioned.",
            ],
        )

    @property
    def default_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    @property
    def excluded_module_patterns(self) -> list[str]:
        return [
            *super().excluded_module_patterns,
            "mlp.gate",  # MoE router: routing must stay fixed
            "mlp.experts",  # per-expert FFNs
            "shared_expert_gate",
        ]

    @property
    def modules_to_not_quantize(self) -> list[str]:
        # Quantizing the router degrades routing decisions disproportionately to
        # the memory it saves.
        return ["lm_head", "mlp.gate"]

    def chat_template_kwargs(self, mode: ReasoningMode) -> dict[str, Any]:
        # Passing enable_thinking to a Thinking-2507 template is an error, not a
        # no-op. The mode is implicit in the checkpoint.
        return {}


# ---------------------------------------------------------------------------
# Mistral dense (Ministral-8B)
# ---------------------------------------------------------------------------


class MistralDenseAdapter(ModelFamilyAdapter):
    """Text-only Mistral causal LMs, e.g. ``mistralai/Ministral-8B-Instruct-2410``.

    This is the scale-matched counterpart to Qwen3-8B: comparing an 8B Qwen
    against a 24B multimodal Mistral confounds family with scale and modality.

    ``ministral`` is registered alongside ``mistral`` because transformers gave
    Ministral its own ``model_type`` (it uses interleaved sliding-window
    attention, where classic Mistral does not). transformers 4.x reports these
    checkpoints as ``mistral`` and 5.x reports ``ministral``, and the loader
    trusts whatever the checkpoint says — so both have to resolve here or the
    same config breaks on a version bump. The distinction does not affect
    adapter placement: the projection names are identical, so the LoRA targets
    below are correct for both.
    """

    model_types = ("mistral", "ministral")
    family = "mistral"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            family="mistral",
            model_type=self.config.model_type or "mistral",
            auto_class="AutoModelForCausalLM",
            reasoning=ReasoningCapability.UNSUPPORTED,
            is_moe=False,
            is_multimodal=False,
            supports_4bit=True,
            default_context_limit=32768,
            notes=[
                "Standard instruction model with no reasoning mode.",
                "Uses interleaved sliding-window attention; very long contexts "
                "behave differently from Qwen3.",
            ],
        )

    @property
    def default_target_modules(self) -> list[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


# ---------------------------------------------------------------------------
# Mistral 3 vision-language (Mistral-Small-3.2-24B)
# ---------------------------------------------------------------------------


class Mistral3VLMAdapter(ModelFamilyAdapter):
    """``Mistral3ForConditionalGeneration`` checkpoints (``model_type: mistral3``).

    ``mistralai/Mistral-Small-3.2-24B-Instruct-2506`` is a vision-language model:
    a 40-layer text tower plus a 24-layer vision tower behind a projector.

    Two things follow, and both are easy to get wrong:

    * ``AutoModelForCausalLM`` **cannot load it**. In transformers, ``mistral3``
      is registered only in ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``.
      Loading must go through ``AutoModelForImageTextToText``.
    * LoRA must be scoped to ``language_model.*``. Adapting the vision tower while
      training on text-only KLEOS data would update parameters that receive no
      meaningful gradient signal and inflate the adapter for no benefit.

    KLEOS training data is text-only, so the vision tower is frozen and left
    unquantized.
    """

    model_types = ("mistral3",)
    family = "mistral"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            family="mistral",
            model_type=self.config.model_type or "mistral3",
            auto_class="AutoModelForImageTextToText",
            reasoning=ReasoningCapability.UNSUPPORTED,
            is_moe=False,
            is_multimodal=True,
            supports_4bit=True,
            default_context_limit=131072,
            language_model_prefix="language_model",
            notes=[
                "Vision-language model. AutoModelForCausalLM cannot load it: "
                "transformers maps mistral3 to image-text-to-text only.",
                "LoRA is scoped to language_model.*; the vision tower and "
                "projector stay frozen and unquantized for text-only training.",
                "The official repository ships a tekken tokenizer; the HF "
                "tokenizer's chat template is used here for training.",
            ],
        )

    @property
    def default_target_modules(self) -> list[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    @property
    def excluded_module_patterns(self) -> list[str]:
        return [
            *super().excluded_module_patterns,
            "vision_tower",
            "multi_modal_projector",
            "patch_merger",
            "vision_encoder",
        ]

    @property
    def modules_to_not_quantize(self) -> list[str]:
        # Quantizing a tower that is never trained and (for text-only data) never
        # even used buys nothing and risks load-time errors.
        return ["lm_head", "vision_tower", "multi_modal_projector"]

    def prepare_model_for_training(self, model: Any) -> Any:
        """Freeze the vision tower and projector explicitly.

        PEFT already freezes everything outside the adapter, but making this
        explicit means a misconfigured ``modules_to_save`` cannot silently start
        training a 24-layer vision encoder.
        """
        frozen = 0
        for name, parameter in model.named_parameters():
            if any(fragment in name for fragment in ("vision_tower", "multi_modal_projector")):
                parameter.requires_grad = False
                frozen += 1
        if frozen:
            logger.info(
                "Froze %d vision-tower/projector parameter tensor(s); KLEOS training "
                "data is text-only.",
                frozen,
            )
        return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Concrete adapters, in resolution order.
_ADAPTER_CLASSES: tuple[type[ModelFamilyAdapter], ...] = (
    QwenMoEAdapter,
    QwenDenseAdapter,
    Mistral3VLMAdapter,
    MistralDenseAdapter,
)

#: ``model_type`` → adapter class.
ADAPTER_REGISTRY: dict[str, type[ModelFamilyAdapter]] = {
    model_type: cls for cls in _ADAPTER_CLASSES for model_type in cls.model_types
}


def register_adapter(cls: type[ModelFamilyAdapter]) -> type[ModelFamilyAdapter]:
    """Register a new family adapter.

    Adding a model family should require a config plus an adapter, never a change
    to the training pipeline (spec section 3).
    """
    for model_type in cls.model_types:
        ADAPTER_REGISTRY[model_type] = cls
    return cls


def _infer_model_type(config: ModelConfig) -> str:
    """Best-effort ``model_type`` when the config does not state one.

    Prefers the explicit field. Falls back to checkpoint-name heuristics so
    ``scripts/inspect_model.py --model <id>`` works without a config file.
    """
    if config.model_type:
        return config.model_type

    name = config.base_model.lower()
    # Order matters: the MoE and VLM checks must precede the generic ones.
    if re.search(r"a3b|moe|mixture", name):
        return "qwen3_moe"
    if "mistral-small-3" in name or "mistral-small-24b" in name:
        return "mistral3"
    if "qwen3" in name or "qwen-3" in name:
        return "qwen3"
    if "qwen2.5" in name or "qwen2_5" in name:
        return "qwen2_5"
    if "qwen" in name:
        return "qwen2"
    if "mistral" in name or "ministral" in name or "mixtral" in name:
        return "mistral"
    return ""


def get_adapter(config: ModelConfig) -> ModelFamilyAdapter:
    """Resolve the family adapter for a model configuration.

    Raises:
        ModelCompatibilityError: when no adapter handles the model type.
    """
    model_type = _infer_model_type(config)
    adapter_class = ADAPTER_REGISTRY.get(model_type)

    if adapter_class is None:
        raise ModelCompatibilityError(
            f"No model-family adapter registered for model_type={model_type!r} "
            f"(base_model={config.base_model!r}).",
            details={
                "registered_model_types": sorted(ADAPTER_REGISTRY),
                "configured_family": config.family,
            },
            suggestions=[
                "Set model.model_type explicitly in the config.",
                "Add an adapter subclassing ModelFamilyAdapter and register it "
                "with @register_adapter — the training pipeline needs no changes.",
                "See src/kleos_models/models/adapters.py for the four built-in families.",
            ],
        )

    adapter = adapter_class(config)

    # A config claiming a family the adapter disagrees with usually means the
    # wrong checkpoint was pasted in. Catch it before a multi-hour run.
    if config.family and config.family != adapter.family:
        raise ModelCompatibilityError(
            f"Config declares family={config.family!r} but {config.base_model!r} "
            f"resolves to the {adapter.family!r} adapter.",
            details={"model_type": model_type, "adapter": adapter_class.__name__},
            suggestions=[
                "Fix model.family, or model.base_model if the wrong checkpoint was set.",
            ],
        )
    return adapter


def resolve_adapter_from_hf_config(hf_config: Any, config: ModelConfig) -> ModelFamilyAdapter:
    """Resolve an adapter using a downloaded HF config as the authority.

    Preferred over :func:`get_adapter` once the checkpoint's real config is
    available, because it removes all guesswork about the architecture.
    """
    model_type = getattr(hf_config, "model_type", None)
    architectures = list(getattr(hf_config, "architectures", None) or [])

    if model_type and model_type != config.model_type:
        if config.model_type:
            logger.warning(
                "Config declares model_type=%r but the checkpoint reports %r; "
                "trusting the checkpoint.",
                config.model_type,
                model_type,
            )
        config = config.model_copy(update={"model_type": model_type})

    if architectures and config.architecture and config.architecture not in architectures:
        raise ModelCompatibilityError(
            f"Config expects architecture {config.architecture!r} but "
            f"{config.base_model!r} reports {architectures}.",
            details={"model_type": model_type},
            suggestions=[
                "The config and the checkpoint disagree. Confirm which checkpoint "
                "you intend to train and fix model.architecture or model.base_model.",
                "This check exists so results are never attributed to the wrong model.",
            ],
        )
    if architectures and not config.architecture:
        config = config.model_copy(update={"architecture": architectures[0]})

    return get_adapter(config)
