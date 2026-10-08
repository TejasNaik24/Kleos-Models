"""Model-family adapters."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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


#: Key renames exposing a ``mistral3`` checkpoint's text tower as a plain causal LM.
#: Hub checkpoints use the pre-transformers-5 VLM layout, with an untied lm_head.
MISTRAL3_TEXT_KEY_MAPPING: dict[str, str] = {
    r"^language_model\.model\.": "model.",
    r"^language_model\.lm_head\.": "lm_head.",
}
#: Checkpoint keys a text-only view is expected to leave behind.
MISTRAL3_NON_TEXT_PREFIXES: tuple[str, ...] = ("vision_tower.", "multi_modal_projector.")


@dataclass(frozen=True)
class CheckpointView:
    """How to load one tower of a composite checkpoint as a standalone model."""

    kind: str
    container_model_type: str
    container_architectures: tuple[str, ...]
    text_model_type: str
    text_config: Any
    key_mapping: Mapping[str, str]
    ignored_prefixes: tuple[str, ...]
    dtype_inherited: str | None = None

    def load_kwargs(self) -> dict[str, Any]:
        """``from_pretrained`` kwargs that load the view."""
        return {
            "config": self.text_config,
            "key_mapping": dict(self.key_mapping),
            "output_loading_info": True,
        }

    def validate_loading_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse a load that did not map every expected weight."""
        missing = sorted(str(k) for k in info.get("missing_keys") or [])
        mismatched = [str(k) for k in info.get("mismatched_keys") or []]
        unexpected = sorted(str(k) for k in info.get("unexpected_keys") or [])
        stray = [k for k in unexpected if not k.startswith(self.ignored_prefixes)]
        if missing or mismatched or stray:
            raise ModelCompatibilityError(
                f"The {self.kind} view did not load cleanly from the "
                f"{self.container_model_type} checkpoint.",
                details={
                    "missing_keys": missing[:20],
                    "mismatched_keys": mismatched[:20],
                    "unexpected_non_vision_keys": stray[:20],
                },
                suggestions=[
                    "The checkpoint's key layout is not the one the view expects; "
                    "do not train on this load.",
                    "Confirm the pinned revision and the installed transformers version.",
                ],
            )
        return {
            "missing_keys": 0,
            "mismatched_keys": 0,
            "unexpected_keys": len(unexpected),
            "unexpected_by_prefix": {
                prefix: sum(1 for key in unexpected if key.startswith(prefix))
                for prefix in self.ignored_prefixes
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "container_model_type": self.container_model_type,
            "container_architectures": list(self.container_architectures),
            "text_model_type": self.text_model_type,
            "key_mapping": dict(self.key_mapping),
            "ignored_checkpoint_prefixes": list(self.ignored_prefixes),
            "dtype_inherited_from_container": self.dtype_inherited,
        }


class ModelFamilyAdapter(ABC):
    """Per-family behaviour: loading, templates, reasoning, LoRA targeting."""

    #: ``config.model_type`` values this adapter handles.
    model_types: tuple[str, ...] = ()
    #: Family label recorded in manifests.
    family: str = "unknown"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Static facts about this family."""

    @property
    @abstractmethod
    def default_target_modules(self) -> list[str]:
        """Architecture-appropriate LoRA targets."""

    @property
    def excluded_module_patterns(self) -> list[str]:
        """Name fragments that must never receive an adapter."""
        return ["lm_head", "embed_tokens"]

    @property
    def modules_to_not_quantize(self) -> list[str]:
        """Modules kept in full precision when quantizing."""
        return ["lm_head"]

    def auto_model_class(self) -> Any:
        """Return the transformers auto class able to load this checkpoint."""
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

    def checkpoint_view(self, hf_config: Any) -> CheckpointView | None:
        """A view to load when the checkpoint is a container for this family."""
        return None

    def resolve_reasoning_mode(self, requested: ReasoningMode | None = None) -> ReasoningMode:
        """Validate a requested reasoning mode against real capability."""
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
                        "Use a reasoning-capable checkpoint for reasoning experiments, "
                        "for example configs/models/ministral3_14b_reasoning.yaml.",
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
                    "For answers without thinking, use a checkpoint whose reasoning "
                    "capability is not thinking-only (for Ministral 3, the Instruct "
                    "release: configs/models/ministral3_14b.yaml).",
                    "Comparing a thinking-only model against a non-thinking one is a "
                    "valid experiment, but it must be configured deliberately and "
                    "recorded in the manifest.",
                ],
            )
        return mode

    def chat_template_kwargs(self, mode: ReasoningMode) -> dict[str, Any]:
        """Template kwargs implementing the reasoning mode. Empty by default."""
        return {}

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
        """Check requested targets exist in the loaded model."""
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

    def describe(self) -> dict[str, Any]:
        """Manifest-ready description of the model."""
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
    """Whether the named module is a leaf that LoRA can wrap."""
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


class MistralDenseAdapter(ModelFamilyAdapter):
    """Text-only Mistral causal LMs, e.g. ``mistralai/Ministral-8B-Instruct-2410``."""

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
                "behave differently from models with full attention.",
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


class Mistral3VLMAdapter(ModelFamilyAdapter):
    """``Mistral3ForConditionalGeneration`` checkpoints (``model_type: mistral3``)."""

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
        # Never trained or used on text data; quantizing it only risks load errors.
        return ["lm_head", "vision_tower", "multi_modal_projector"]

    def prepare_model_for_training(self, model: Any) -> Any:
        """Freeze the vision tower and projector explicitly."""
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


class Ministral3TextAdapter(ModelFamilyAdapter):
    """The text tower of a Ministral 3 checkpoint, loaded as a plain causal LM."""

    model_types = ("ministral3",)
    family = "mistral"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            family="mistral",
            model_type="ministral3",
            auto_class="AutoModelForCausalLM",
            reasoning=ReasoningCapability.UNSUPPORTED,
            is_moe=False,
            is_multimodal=False,
            supports_4bit=True,
            default_context_limit=262144,
            notes=[
                "Text tower of a Mistral3 vision-language checkpoint, loaded as "
                "Ministral3ForCausalLM; the vision tower is never loaded.",
                "The chat template renders system prompts in place, so no system "
                "merge is needed (unlike Mistral-Nemo, deviation D5).",
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
        # Defensive: the view loads no vision modules, but one must never get an adapter.
        return [*super().excluded_module_patterns, "vision_tower", "multi_modal_projector"]

    def checkpoint_view(self, hf_config: Any) -> CheckpointView | None:
        container_type = getattr(hf_config, "model_type", None)
        if container_type != "mistral3":
            return None  # a plain ministral3 checkpoint loads as itself

        text_config = getattr(hf_config, "text_config", None)
        text_type = getattr(text_config, "model_type", None)
        architectures = tuple(getattr(hf_config, "architectures", None) or ())
        problems: list[str] = []
        if text_type != "ministral3":
            problems.append(
                f"its text tower is {text_type!r}, not 'ministral3' (for example "
                "Mistral Small 3.x, whose tower is 'mistral')"
            )
        if "Mistral3ForConditionalGeneration" not in architectures:
            problems.append(f"it declares {list(architectures)}")
        if self.config.architecture and self.config.architecture not in architectures:
            problems.append(
                f"the config expects {self.config.architecture!r} but the checkpoint "
                f"declares {list(architectures)}"
            )
        if getattr(hf_config, "quantization_config", None):
            problems.append(
                "it is pre-quantized (for example the FP8 release), and a text-only "
                "view would silently drop that quantization"
            )
        if problems:
            raise ModelCompatibilityError(
                f"{self.config.base_model} is not a Ministral 3 checkpoint this "
                "text-only view can open.",
                details={"problems": problems},
                suggestions=[
                    "Use the BF16 Ministral 3 repository at a pinned revision.",
                    "For a different Mistral3 model, use model_type 'mistral3' and "
                    "the vision-language adapter.",
                ],
            )

        text: Any = copy.deepcopy(text_config)  # not None: checked via text_type above
        inherited: str | None = None
        container_dtype = getattr(hf_config, "dtype", None) or getattr(
            hf_config, "torch_dtype", None
        )
        if getattr(text, "dtype", None) is None and container_dtype is not None:
            # Sub-configs rarely state dtype; inherit the container's so 'auto' matches.
            text.dtype = container_dtype
            inherited = str(container_dtype).replace("torch.", "")

        return CheckpointView(
            kind="text_only",
            container_model_type="mistral3",
            container_architectures=architectures,
            text_model_type="ministral3",
            text_config=text,
            key_mapping=MISTRAL3_TEXT_KEY_MAPPING,
            ignored_prefixes=MISTRAL3_NON_TEXT_PREFIXES,
            dtype_inherited=inherited,
        )

    def prepare_model_for_training(self, model: Any) -> Any:
        """Prove the view held: not one vision parameter may be present."""
        leaked = [
            name
            for name, _ in model.named_parameters()
            if "vision_tower" in name or "multi_modal_projector" in name
        ]
        if leaked:
            raise ModelCompatibilityError(
                "The text-only view loaded vision parameters.",
                details={"parameters": leaked[:10]},
                suggestions=["Do not train: this is not the text tower alone."],
            )
        return model


class Ministral3ReasoningTextAdapter(Ministral3TextAdapter):
    """The text tower of Ministral 3 *Reasoning*: the same view, always thinking."""

    model_types = ("ministral3_reasoning",)

    @property
    def capabilities(self) -> ModelCapabilities:
        return replace(
            super().capabilities,
            reasoning=ReasoningCapability.ALWAYS_ON,
            notes=[
                "Text tower of Ministral 3 Reasoning, loaded as Ministral3ForCausalLM "
                "through the Instruct release's text-only view.",
                "Always thinks: completions are [THINK]trace[/THINK]answer. The "
                "trace is split off at the [/THINK] token before grading.",
                "The chat template adds a default thinking system prompt only to a "
                "conversation without a system message; KLEOS prompts always have one.",
            ],
        )


#: Concrete adapters, in resolution order.
_ADAPTER_CLASSES: tuple[type[ModelFamilyAdapter], ...] = (
    Mistral3VLMAdapter,
    Ministral3TextAdapter,
    Ministral3ReasoningTextAdapter,
    MistralDenseAdapter,
)

#: ``model_type`` → adapter class.
ADAPTER_REGISTRY: dict[str, type[ModelFamilyAdapter]] = {
    model_type: cls for cls in _ADAPTER_CLASSES for model_type in cls.model_types
}


def register_adapter(cls: type[ModelFamilyAdapter]) -> type[ModelFamilyAdapter]:
    """Register a new family adapter."""
    for model_type in cls.model_types:
        ADAPTER_REGISTRY[model_type] = cls
    return cls


def _infer_model_type(config: ModelConfig) -> str:
    """Best-effort ``model_type`` when the config does not state one."""
    if config.model_type:
        return config.model_type

    name = config.base_model.lower()
    # Order matters: the specific checks must precede the generic Mistral one.
    if "mistral-small-3" in name or "mistral-small-24b" in name:
        return "mistral3"
    if "ministral-3-" in name and "reasoning" in name:
        return "ministral3_reasoning"
    if "ministral-3-" in name:
        return "ministral3"
    if "mistral" in name or "ministral" in name or "mixtral" in name:
        return "mistral"
    return ""


def get_adapter(config: ModelConfig) -> ModelFamilyAdapter:
    """Resolve the family adapter for a model configuration."""
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
                "See src/kleos_models/models/adapters.py for the built-in Mistral adapters.",
            ],
        )

    adapter = adapter_class(config)

    # A family mismatch usually means the wrong checkpoint; catch it before a long run.
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


@dataclass
class LoadPlan:
    """What to instantiate for a config, given the checkpoint's own config."""

    config: ModelConfig
    adapter: ModelFamilyAdapter
    #: Set when a tower of a composite checkpoint is loaded on its own.
    view: CheckpointView | None = None
    #: The config's model_type when the checkpoint's was trusted instead.
    flipped_from: str | None = None


def resolve_load_plan(config: ModelConfig, hf_config: Any) -> LoadPlan:
    """Decide how to load ``config`` given the checkpoint's reported config."""
    reported = getattr(hf_config, "model_type", None)
    if reported and config.model_type and reported != config.model_type:
        try:
            configured = get_adapter(config)
        except ModelCompatibilityError:
            configured = None
        view = configured.checkpoint_view(hf_config) if configured is not None else None
        if configured is not None and view is not None:
            return LoadPlan(config=config, adapter=configured, view=view)

    flipped: str | None = None
    resolved = config
    if reported and reported != config.model_type:
        if config.model_type:
            logger.warning(
                "Config says model_type=%r; the checkpoint reports %r. Trusting the checkpoint.",
                config.model_type,
                reported,
            )
            flipped = config.model_type
        resolved = config.model_copy(update={"model_type": reported})
    return LoadPlan(config=resolved, adapter=get_adapter(resolved), flipped_from=flipped)


def resolve_adapter_from_hf_config(hf_config: Any, config: ModelConfig) -> ModelFamilyAdapter:
    """Resolve an adapter using a downloaded HF config as the authority."""
    architectures = list(getattr(hf_config, "architectures", None) or [])
    plan = resolve_load_plan(config, hf_config)
    if plan.view is not None:
        return plan.adapter
    config = plan.config
    model_type = config.model_type

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
