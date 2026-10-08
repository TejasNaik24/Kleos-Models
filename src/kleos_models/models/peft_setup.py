"""PEFT / LoRA attachment (spec sections 13, 45, 46)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kleos_models.config import LoRAConfig, ModelConfig, TrainingConfig
from kleos_models.errors import MissingDependencyError, ModelCompatibilityError
from kleos_models.logging_utils import get_logger
from kleos_models.models.adapters import ModelFamilyAdapter, TargetModuleResolution

logger = get_logger(__name__)


@dataclass
class PeftSetupResult:
    """Outcome of attaching an adapter."""

    model: Any
    resolution: TargetModuleResolution
    trainable_parameters: int
    total_parameters: int
    lora_config: dict[str, Any]

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / self.total_parameters if self.total_parameters else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_percent": round(self.trainable_fraction * 100, 4),
            "target_modules": self.resolution.to_dict(),
            "lora_config": self.lora_config,
        }

    def render(self) -> str:
        return (
            f"trainable parameters: {self.trainable_parameters:,} of "
            f"{self.total_parameters:,} ({self.trainable_fraction * 100:.4f}%)"
        )


def _require_peft() -> Any:
    try:
        import peft
    except ImportError as exc:
        raise MissingDependencyError("peft", extra="train", purpose="attach LoRA adapters") from exc
    return peft


def build_lora_config(
    lora: LoRAConfig, target_modules: list[str], *, adapter: ModelFamilyAdapter
) -> Any:
    """Construct a PEFT ``LoraConfig`` from the KLEOS config."""
    peft = _require_peft()

    kwargs: dict[str, Any] = {
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": target_modules,
        "bias": lora.bias,
        "task_type": getattr(peft.TaskType, lora.task_type),
        "use_rslora": lora.use_rslora,
        "init_lora_weights": lora.init_lora_weights,
    }
    if lora.modules_to_save:
        kwargs["modules_to_save"] = lora.modules_to_save

    # PEFT takes a regex or a list depending on version; a substring list is portable.
    exclusions = [*adapter.excluded_module_patterns, *lora.exclude_modules]
    if exclusions and "exclude_modules" in getattr(peft.LoraConfig, "__dataclass_fields__", {}):
        kwargs["exclude_modules"] = exclusions
    elif exclusions:
        logger.debug(
            "Installed peft has no exclude_modules field; exclusions are enforced "
            "during target validation instead."
        )

    return peft.LoraConfig(**kwargs)


def prepare_for_kbit_training(model: Any, training: TrainingConfig) -> Any:
    """Run PEFT's k-bit preparation on a quantized model."""
    peft = _require_peft()
    return peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def attach_lora(
    model: Any,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    adapter: ModelFamilyAdapter,
    *,
    is_quantized: bool,
) -> PeftSetupResult:
    """Validate targets, attach the LoRA adapter and verify it will train."""
    peft = _require_peft()
    lora = model_config.lora

    resolution = adapter.validate_target_modules(model, lora)

    if is_quantized:
        model = prepare_for_kbit_training(model, training_config)
    elif training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # The revision goes into adapter_config.json so AutoPeftModel finds the trained-against
    # base, not `main` (H-F1). Adapter files are still fetched at the caller's revision.
    lora_config = build_lora_config(lora, resolution.matched, adapter=adapter)
    model = peft.get_peft_model(model, lora_config, revision=model_config.revision)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if trainable == 0:
        raise ModelCompatibilityError(
            "LoRA attached but no parameter requires gradients.",
            details={
                "target_modules": resolution.matched,
                "matched_module_count": resolution.matched_module_count,
            },
            suggestions=[
                "This would train nothing while still producing a loss curve.",
                "Check model.lora.target_modules and exclude_modules.",
                "Inspect the architecture: python scripts/inspect_model.py --model "
                f"{model_config.base_model}",
            ],
        )

    fraction = trainable / total if total else 0.0
    if fraction > 0.5 and not training_config.allow_full_finetune:
        raise ModelCompatibilityError(
            f"{fraction:.1%} of parameters are trainable — this is a full "
            "fine-tune, not a LoRA run.",
            details={"trainable": trainable, "total": total},
            suggestions=[
                "The pipeline refuses to silently full-fine-tune (spec section 13).",
                "Check model.lora.modules_to_save is not listing large modules.",
                "Set training.allow_full_finetune=true only if that is genuinely "
                "the experiment you intend to run.",
            ],
        )

    result = PeftSetupResult(
        model=model,
        resolution=resolution,
        trainable_parameters=trainable,
        total_parameters=total,
        lora_config={
            "r": lora.r,
            "alpha": lora.alpha,
            "dropout": lora.dropout,
            "scaling": round(lora.scaling, 4),
            "bias": lora.bias,
            "task_type": lora.task_type,
            "use_rslora": lora.use_rslora,
            "target_modules": resolution.matched,
            "modules_to_save": lora.modules_to_save,
        },
    )
    logger.info("LoRA attached — %s", result.render())
    return result


def verify_gradients_flow(model: Any, sample_batch: dict[str, Any]) -> dict[str, Any]:
    """Run one forward/backward pass and confirm LoRA parameters receive gradients."""
    import torch

    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(**sample_batch)
    loss = outputs.loss
    if loss is None:
        raise ModelCompatibilityError(
            "The forward pass returned no loss.",
            suggestions=["Confirm the batch includes a 'labels' tensor."],
        )
    loss.backward()

    lora_with_grad = 0
    lora_total = 0
    nonzero = 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        lora_total += 1
        if parameter.grad is not None:
            lora_with_grad += 1
            if torch.any(parameter.grad != 0):
                nonzero += 1

    model.zero_grad(set_to_none=True)

    if lora_total == 0:
        raise ModelCompatibilityError(
            "No LoRA parameters were found on the model.",
            suggestions=["attach_lora() must run before gradient verification."],
        )
    if nonzero == 0:
        raise ModelCompatibilityError(
            "LoRA parameters exist but all gradients are zero after a backward pass.",
            details={"lora_tensors": lora_total, "with_grad": lora_with_grad},
            suggestions=[
                "Every label may be masked to -100: check the formatter's assistant-span masking.",
                "Confirm the model is in train mode and inputs require grad.",
            ],
        )

    return {
        "loss": float(loss.detach().item()),
        "lora_tensors": lora_total,
        "lora_tensors_with_grad": lora_with_grad,
        "lora_tensors_with_nonzero_grad": nonzero,
    }
