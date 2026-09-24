"""Model and tokenizer loading (spec sections 13, 45, 46).

The load path is deliberately explicit about *which* auto class is used, because
that is where the Mistral Small 3.2 trap lives: it is a
``Mistral3ForConditionalGeneration`` registered only for image-text-to-text, so
``AutoModelForCausalLM`` fails on it. The family adapter names the right class,
and the checkpoint's own config is consulted before loading so a mismatch is
caught in seconds rather than after a 28GB download.

Everything torch-specific is imported inside functions so the module stays
importable in the light environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.compat import dtype_kwarg, mistral_regex_kwarg, require_transformers
from kleos_models.config import DType, ModelConfig, ReasoningMode
from kleos_models.errors import ModelCompatibilityError
from kleos_models.logging_utils import get_logger
from kleos_models.models.adapters import ModelFamilyAdapter, get_adapter, resolve_load_plan
from kleos_models.models.quantization import (
    build_quantization_config,
    quantization_support,
    resolve_compute_dtype,
)

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer, adapter and provenance metadata."""

    model: Any
    tokenizer: Any
    adapter: ModelFamilyAdapter
    config: ModelConfig
    reasoning_mode: ReasoningMode
    quantization_metadata: dict[str, Any] = field(default_factory=dict)
    load_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def describe(self) -> dict[str, Any]:
        """Manifest-ready description."""
        description = self.adapter.describe()
        description.update(
            {
                "reasoning_mode": self.reasoning_mode.value,
                "loaded_parameter_count": self.parameter_count,
                "trainable_parameter_count": self.trainable_parameter_count,
                "trainable_fraction": (
                    self.trainable_parameter_count / self.parameter_count
                    if self.parameter_count
                    else 0.0
                ),
                "quantization": self.quantization_metadata,
                "load": self.load_metadata,
            }
        )
        return description

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Template kwargs for the resolved reasoning mode."""
        return self.adapter.chat_template_kwargs(self.reasoning_mode)


def _hf_token() -> str | None:
    """Read a Hugging Face token from the environment.

    Never accepted as a function argument or written to a config, so it cannot end
    up in a manifest or a committed file.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_hf_config(config: ModelConfig) -> Any:
    """Download and return the checkpoint's own ``config.json``.

    Cheap (a few KB) and worth doing first: it identifies the real architecture
    before any weight download begins.
    """
    transformers = require_transformers()
    try:
        return transformers.AutoConfig.from_pretrained(
            config.base_model,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            token=_hf_token(),
        )
    except Exception as exc:
        message = str(exc)
        suggestions = [
            "Check the model id is spelled correctly.",
            "Check network access from this runtime.",
        ]
        if "gated" in message.lower() or "401" in message or "403" in message:
            suggestions = [
                f"{config.base_model} is a gated repository.",
                "Accept the licence on its Hugging Face model page while signed in.",
                "Then set HF_TOKEN (locally in .env, or as a Colab secret).",
                "See docs/colab.md for the Colab secret setup.",
            ]
        elif "trust_remote_code" in message:
            suggestions = [
                "This checkpoint ships custom code.",
                "Set model.trust_remote_code=true only if you trust the repository.",
            ]
        raise ModelCompatibilityError(
            f"Could not read the config for {config.base_model!r}.",
            details={"revision": config.revision, "error": message[:400]},
            suggestions=suggestions,
        ) from exc


def resolve_fix_mistral_regex(
    config: ModelConfig, requested: bool | None = None
) -> tuple[bool | None, str]:
    """The tokenizer regex flag to use, and where the decision came from.

    Two places can state it: the model config (``model.fix_mistral_regex``,
    Logos onwards) and a caller (serving passes the value its manifest records).
    They must agree. Two statements that disagree mean the model would be
    served with a tokenizer it was not trained with, which fails silently, so
    it is an error here.

    Returns:
        ``(value, source)`` with source one of ``unset``, ``config``, ``caller``
        or ``both``. ``None`` means pass nothing: the library default, which is
        what every run before Logos used.
    """
    configured = config.fix_mistral_regex
    if requested is None:
        return configured, "config" if configured is not None else "unset"
    if configured is None:
        return requested, "caller"
    if configured != requested:
        raise ModelCompatibilityError(
            "The tokenizer regex flag is stated twice and the two disagree.",
            details={"model_config": configured, "caller": requested},
            suggestions=[
                "Training, evaluation and serving must tokenize identically; a "
                "different pre-tokenizer regex is train/serve skew.",
                "Fix the manifest or the model config so they state the same value.",
            ],
        )
    return requested, "both"


def load_tokenizer(
    config: ModelConfig,
    adapter: ModelFamilyAdapter | None = None,
    *,
    fix_mistral_regex: bool | None = None,
) -> Any:
    """Load the tokenizer and validate it can format conversations.

    A tokenizer with no chat template cannot produce correct training targets, so
    that is an error rather than a silent fallback to a generic format.

    Args:
        fix_mistral_regex: Pin the Mistral pre-tokenizer regex behaviour
            explicitly. Combined with ``config.fix_mistral_regex`` by
            :func:`resolve_fix_mistral_regex`; when both are unset nothing is
            passed and the installed transformers decides, which is what every
            run before Logos used. Serving passes its recorded value so a library
            default can never move tokenization underneath a frozen adapter.
    """
    transformers = require_transformers()
    adapter = adapter or get_adapter(config)

    resolved_flag, _ = resolve_fix_mistral_regex(config, fix_mistral_regex)
    kwargs: dict[str, Any] = {}
    if resolved_flag is not None:
        kwargs = mistral_regex_kwarg(resolved_flag)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            token=_hf_token(),
            use_fast=True,
            **kwargs,
        )
    except Exception as exc:
        raise ModelCompatibilityError(
            f"Could not load the tokenizer for {config.tokenizer_id!r}.",
            details={"error": str(exc)[:400]},
            suggestions=[
                "For gated repositories, accept the licence and set HF_TOKEN.",
                "Some Mistral repositories ship a tekken tokenizer; if the fast "
                "tokenizer fails, pin model.tokenizer to a compatible id.",
            ],
        ) from exc

    if config.chat_template:
        tokenizer.chat_template = config.chat_template
        logger.info("Applied the chat template override from the model config.")

    if getattr(tokenizer, "chat_template", None) is None:
        raise ModelCompatibilityError(
            f"The tokenizer for {config.tokenizer_id!r} has no chat template.",
            details={"tokenizer_class": type(tokenizer).__name__},
            suggestions=[
                "KLEOS trains on conversations, so a chat template is required.",
                "Set model.chat_template in the config to supply one explicitly.",
                "Base (non-instruct) checkpoints usually lack a template; use the "
                "instruct variant.",
            ],
        )

    # Causal LMs need a pad token for batching. Reusing EOS is the standard
    # approach; the attention mask keeps padding out of the loss.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer had no pad token; using eos_token %r.", tokenizer.eos_token)
        else:  # pragma: no cover - very unusual
            raise ModelCompatibilityError(
                "Tokenizer has neither a pad token nor an eos token.",
                suggestions=["Add a pad token before training."],
            )

    # Right padding for training (left padding shifts label positions).
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    config: ModelConfig,
    *,
    reasoning_mode: ReasoningMode | None = None,
    for_training: bool = True,
    device_map: str | dict[str, Any] | None = None,
    fix_mistral_regex: bool | None = None,
) -> LoadedModel:
    """Load a base model with the correct auto class and quantization.

    Args:
        config: Model configuration.
        reasoning_mode: Requested mode; validated against real capability.
        for_training: Disable the KV cache and enable input grads for checkpointing.
        device_map: Override the config's device map.
        fix_mistral_regex: Passed to :func:`load_tokenizer`; ``None`` keeps the
            installed library's default, which is what research runs used.

    Raises:
        ModelCompatibilityError: on an architecture/auto-class mismatch, a gated
            repository, or an unsupported reasoning request.
    """
    from kleos_models.compat import check_transformers_version

    check_transformers_version()

    # Settle the tokenizer regex flag before any download, so a conflict
    # between the config and the caller fails in seconds.
    regex_flag, regex_source = resolve_fix_mistral_regex(config, fix_mistral_regex)

    # 1. Read the checkpoint's real config first. Cheap, and it removes guesswork.
    hf_config = load_hf_config(config)
    reported_type = getattr(hf_config, "model_type", None)
    architectures = list(getattr(hf_config, "architectures", None) or [])

    # The checkpoint is the authority on model_type, except for a deliberate
    # view of one tower of a composite checkpoint (KLEOS Logos).
    plan = resolve_load_plan(config, hf_config)
    resolved_config = plan.config
    view = plan.view

    adapter = plan.adapter
    capabilities = adapter.capabilities

    if architectures and capabilities.is_multimodal and "CausalLM" in str(architectures):
        logger.debug("Architecture %s resolved through the multimodal adapter.", architectures)

    mode = adapter.resolve_reasoning_mode(reasoning_mode)
    logger.info(
        "Loading %s (%s, %s) with reasoning mode %r",
        resolved_config.base_model,
        reported_type,
        capabilities.auto_class,
        mode.value,
    )

    # 2. Quantization, including family-required full-precision modules.
    support = quantization_support()
    bnb_config, quant_metadata = build_quantization_config(
        resolved_config,
        extra_skip_modules=adapter.modules_to_not_quantize,
        support=support,
    )

    # 3. Load weights through the family's auto class.
    auto_class = adapter.auto_model_class()
    dtype_value: Any = None
    if resolved_config.dtype is not DType.AUTO:
        dtype_value, _ = resolve_compute_dtype(resolved_config.dtype, support)
    elif bnb_config is None:
        dtype_value, _ = resolve_compute_dtype(DType.AUTO, support)

    load_kwargs: dict[str, Any] = {
        "revision": resolved_config.revision,
        "trust_remote_code": resolved_config.trust_remote_code,
        "token": _hf_token(),
        **adapter.model_load_kwargs(),
        **dtype_kwarg(dtype_value),
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    effective_device_map = device_map if device_map is not None else resolved_config.device_map
    if effective_device_map and support.cuda_available:
        load_kwargs["device_map"] = effective_device_map
    if view is not None:
        load_kwargs.update(view.load_kwargs())

    loading_info: dict[str, Any] | None = None
    try:
        loaded_object = auto_class.from_pretrained(resolved_config.base_model, **load_kwargs)
        if view is not None:
            model, raw_info = loaded_object
            loading_info = view.validate_loading_info(raw_info)
        else:
            model = loaded_object
    except ValueError as exc:
        message = str(exc)
        if "Unrecognized configuration class" in message or "AutoModel" in message:
            raise ModelCompatibilityError(
                f"{auto_class.__name__} cannot load {resolved_config.base_model!r}.",
                details={
                    "model_type": reported_type,
                    "architectures": architectures,
                    "attempted_auto_class": auto_class.__name__,
                    "error": message[:400],
                },
                suggestions=[
                    "The family adapter picked the wrong auto class for this architecture.",
                    "Vision-language checkpoints such as Mistral Small 3.2 need "
                    "AutoModelForImageTextToText, not AutoModelForCausalLM.",
                    "Inspect the checkpoint: python scripts/inspect_model.py --model "
                    f"{resolved_config.base_model}",
                ],
            ) from exc
        raise
    except (OSError, RuntimeError) as exc:
        raise _describe_load_failure(exc, resolved_config, support) from exc

    if (
        view is not None
        and getattr(getattr(model, "generation_config", None), "eos_token_id", None) is None
    ):
        # Without an end-of-sequence id every generation runs to the token cap.
        raise ModelCompatibilityError(
            f"{resolved_config.base_model} loaded through the {view.kind} view "
            "has no generation_config.eos_token_id.",
            suggestions=["Check the checkpoint's generation_config.json at the pinned revision."],
        )

    # 4. Family-specific preparation (freezing a vision tower, for instance).
    model = adapter.prepare_model_for_training(model)

    # The KV cache is useless during training and conflicts with gradient
    # checkpointing.
    if for_training and hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = getattr(
            model.generation_config, "pad_token_id", None
        )

    tokenizer = load_tokenizer(resolved_config, adapter, fix_mistral_regex=regex_flag)

    load_metadata: dict[str, Any] = {
        "auto_class": auto_class.__name__,
        "reported_model_type": reported_type,
        "reported_architectures": architectures,
        "device_map": effective_device_map,
        "dtype": str(dtype_value).replace("torch.", "") if dtype_value else "default",
        "attn_implementation": resolved_config.attn_implementation,
        "for_training": for_training,
    }
    if view is not None:
        load_metadata["instantiated_class"] = type(model).__name__
        load_metadata["view"] = {**view.to_dict(), "loading_info": loading_info}
    if plan.flipped_from is not None:
        load_metadata["checkpoint_model_type_flipped_from"] = plan.flipped_from
    if regex_source != "unset":
        load_metadata["fix_mistral_regex"] = regex_flag
        load_metadata["fix_mistral_regex_source"] = regex_source

    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=resolved_config,
        reasoning_mode=mode,
        quantization_metadata=quant_metadata,
        load_metadata=load_metadata,
    )
    logger.info(
        "Loaded %s: %.2fB parameters",
        type(model).__name__,
        loaded.parameter_count / 1e9,
    )
    return loaded


def _describe_load_failure(
    exc: Exception, config: ModelConfig, support: Any
) -> ModelCompatibilityError:
    """Turn an opaque load failure into an actionable one (spec section 32)."""
    message = str(exc)
    lowered = message.lower()

    if "out of memory" in lowered or "cuda out of memory" in lowered:
        from kleos_models.models.feasibility import probe_gpu

        gpu = probe_gpu()
        return ModelCompatibilityError(
            f"CUDA ran out of memory while loading {config.base_model!r}.",
            details={
                "detected_gpu": gpu.render(),
                "quantization": config.quantization.mode.value,
                "max_seq_length": config.max_seq_length,
            },
            suggestions=[
                "Check feasibility first: python scripts/plan_run.py --config <config>",
                "Reduce model.max_seq_length.",
                "Enable training.gradient_checkpointing.",
                "Reduce training.per_device_train_batch_size and raise "
                "gradient_accumulation_steps to keep the effective batch size.",
                "Confirm no other process holds GPU memory (nvidia-smi).",
                "A 24B or 30B model in 4-bit does not fit a 16GB T4 for training.",
            ],
        )

    if "no module named" in lowered and "bitsandbytes" in lowered:
        return ModelCompatibilityError(
            "bitsandbytes is required for quantized loading but is not importable.",
            details={"error": message[:300]},
            suggestions=[
                'pip install -e ".[train,quant]"',
                "On Colab: python scripts/colab_setup.py",
            ],
        )

    return ModelCompatibilityError(
        f"Failed to load {config.base_model!r}.",
        details={
            "error": message[:500],
            "revision": config.revision,
            "cuda_available": support.cuda_available,
        },
        suggestions=[
            "Check the model id and revision.",
            "For gated repositories, accept the licence and set HF_TOKEN.",
            "See docs/troubleshooting.md.",
        ],
    )


def load_adapter_model(
    config: ModelConfig,
    adapter_path: Path | str,
    *,
    reasoning_mode: ReasoningMode | None = None,
    merge: bool = False,
    fix_mistral_regex: bool | None = None,
    adapter_device: str | None = None,
) -> LoadedModel:
    """Load a base model and attach a trained LoRA adapter.

    This is the fine-tuned evaluation arm. The base model is loaded exactly as in
    the base arm, so the two differ only by the adapter.

    Args:
        merge: Merge adapter weights into the base. Not possible for a quantized
            base; the request is refused rather than silently ignored.
        fix_mistral_regex: Pin tokenizer regex behaviour explicitly. Serving
            passes the value recorded in the deployment manifest.
        adapter_device: Device PEFT reads the adapter file onto before copying
            it into the model. ``None`` lets PEFT choose (CUDA when available),
            which is what every research run did. ZeroGPU passes ``"cpu"``: its
            startup process emulates CUDA with no GPU attached, and reading the
            file straight onto CUDA fails there. The attached weights are
            identical either way; only where the file is read changes.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        from kleos_models.errors import MissingDependencyError

        raise MissingDependencyError("peft", extra="train", purpose="load a LoRA adapter") from exc

    path = Path(adapter_path)
    if not path.exists():
        raise ModelCompatibilityError(
            f"Adapter directory not found: {path}",
            suggestions=[
                "Point --adapter at the run's adapter/ directory.",
                "Training writes it to outputs/<experiment-id>/adapter/.",
            ],
        )

    loaded = load_model(
        config,
        reasoning_mode=reasoning_mode,
        for_training=False,
        fix_mistral_regex=fix_mistral_regex,
    )
    placement = {"torch_device": adapter_device} if adapter_device else {}
    loaded.model = PeftModel.from_pretrained(
        loaded.model, str(path), is_trainable=False, **placement
    )
    loaded.load_metadata["adapter_path"] = str(path)

    if merge:
        if config.quantization.enabled:
            raise ModelCompatibilityError(
                "Cannot merge a LoRA adapter into a quantized base model.",
                details={"quantization": config.quantization.mode.value},
                suggestions=[
                    "Merging requires loading the base unquantized: set "
                    "model.quantization.mode='none' (needs far more memory).",
                    "For evaluation, an unmerged adapter is mathematically equivalent "
                    "and much cheaper.",
                ],
            )
        loaded.model = loaded.model.merge_and_unload()
        loaded.load_metadata["merged"] = True

    loaded.model.eval()
    logger.info("Attached LoRA adapter from %s", path)
    return loaded


def list_candidate_modules(model: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    """List leaf modules that could receive a LoRA adapter (spec section 46).

    Powers ``scripts/inspect_model.py``, so target modules are chosen by looking
    at the architecture rather than copied from a tutorial.
    """
    candidates: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or len(getattr(weight, "shape", ())) != 2:
            continue
        out_features, in_features = tuple(weight.shape)
        candidates.append(
            {
                "name": name,
                "suffix": name.rsplit(".", 1)[-1],
                "class": type(module).__name__,
                "in_features": in_features,
                "out_features": out_features,
                "parameters": int(in_features) * int(out_features),
            }
        )
        if limit is not None and len(candidates) >= limit:
            break
    return candidates
