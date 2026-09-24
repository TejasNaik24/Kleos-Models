"""Finding L-F1: ``Mistral3VLMAdapter``'s LoRA targeting does not fit transformers 5.

Recorded in docs/logos.md, not fixed: the adapter serves only
``configs/models/mistral_small_3_2.yaml``, which no KLEOS run uses, and Logos
loads its checkpoint through ``Ministral3TextAdapter`` instead. Both tests are
strict xfails, so the day the adapter is fixed they fail and force the finding
to be closed rather than forgotten.

1. **Prefix.** Target validation keeps only modules whose names *start with*
   ``language_model``. transformers 5 names them ``model.language_model.*``, so
   nothing matches and attaching LoRA raises.
2. **Exclusion.** The vision tower is excluded by passing ``"vision_tower"`` in
   PEFT's ``exclude_modules`` list. PEFT treats list entries as exact names or
   ``.suffix`` matches, which no projection inside the tower has, so the tower's
   ``q_proj`` … ``down_proj`` would be adapted too.
"""

from __future__ import annotations

from typing import Any

import pytest

from kleos_models.config import LoRAConfig, ModelConfig
from kleos_models.errors import ModelCompatibilityError
from kleos_models.models.adapters import Mistral3VLMAdapter

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _Weight:
    shape = (4, 4)


class _Module:
    """Just enough of ``nn.Module`` for target validation, without torch."""

    def __init__(self, leaf: bool) -> None:
        self._leaf = leaf
        self.weight = _Weight() if leaf else None

    def children(self) -> list[Any]:
        return [] if self._leaf else [object()]


class FakeTransformers5VLM:
    """The module names transformers 5 gives ``Mistral3ForConditionalGeneration``."""

    def __init__(self, layers: int = 2) -> None:
        names: list[tuple[str, bool]] = [("model", False), ("model.language_model", False)]
        for i in range(layers):
            base = f"model.language_model.layers.{i}"
            names += [(f"{base}.self_attn.{p}", True) for p in PROJECTIONS[:4]]
            names += [(f"{base}.mlp.{p}", True) for p in PROJECTIONS[4:]]
            vision = f"model.vision_tower.transformer.layers.{i}"
            names += [(f"{vision}.attention.{p}", True) for p in PROJECTIONS[:4]]
            names += [(f"{vision}.feed_forward.{p}", True) for p in PROJECTIONS[4:]]
        names += [("model.multi_modal_projector.linear_1", True), ("lm_head", True)]
        self._modules = {name: _Module(leaf) for name, leaf in names}

    def named_modules(self) -> list[tuple[str, Any]]:
        return [("", self), *self._modules.items()]

    def get_submodule(self, name: str) -> Any:
        return self._modules[name]


def _adapter() -> Mistral3VLMAdapter:
    return Mistral3VLMAdapter(
        ModelConfig(
            name="mistral_small_3_2",
            family="mistral",
            base_model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            model_type="mistral3",
            is_multimodal=True,
        )
    )


@pytest.mark.xfail(
    strict=True,
    raises=ModelCompatibilityError,
    reason="L-F1: the language_model prefix check expects transformers 4 module names",
)
def test_targets_the_text_tower_of_a_transformers_5_checkpoint():
    resolution = _adapter().validate_target_modules(
        FakeTransformers5VLM(layers=2), LoRAConfig(target_modules="auto")
    )
    assert resolution.matched_module_count == 2 * len(PROJECTIONS)


@pytest.mark.requires_torch
@pytest.mark.requires_peft
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="L-F1: PEFT matches exclude_modules list entries by exact name or suffix",
)
def test_the_vision_tower_receives_no_lora_modules():
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForImageTextToText, Mistral3Config

    config = Mistral3Config(
        text_config={
            "model_type": "mistral",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_dim": 16,
            "image_size": 32,
            "patch_size": 16,
        },
    )
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config)
    assert any(
        "vision_tower" in name and name.endswith("q_proj") for name, _ in model.named_modules()
    )

    from kleos_models.models.peft_setup import build_lora_config

    adapter = _adapter()
    lora = build_lora_config(LoRAConfig(), list(PROJECTIONS), adapter=adapter)
    peft_model = get_peft_model(model, lora)
    adapted_vision = [
        name
        for name, _ in peft_model.named_modules()
        if "vision_tower" in name and name.endswith("lora_A")
    ]
    assert adapted_vision == []
