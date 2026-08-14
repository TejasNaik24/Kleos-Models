"""Model-family adapter tests (spec §3, §38, §39, §46).

These pin the architectural facts that are easy to get wrong and expensive to
discover late:

* Mistral Small 3.2 needs AutoModelForImageTextToText, not AutoModelForCausalLM.
* Qwen3-30B-A3B is MoE and thinking-only; its experts must not be adapted.
* Reasoning is a declared capability, validated, not a prompt string.

They need no torch: adapters expose their metadata without loading weights.
"""

from __future__ import annotations

import pytest
from tests.conftest import CONFIGS_DIR

from kleos_models.config import (
    LoRAConfig,
    ModelConfig,
    ReasoningCapability,
    ReasoningMode,
    load_model_config,
)
from kleos_models.errors import ModelCompatibilityError
from kleos_models.models.adapters import (
    ADAPTER_REGISTRY,
    Mistral3VLMAdapter,
    MistralDenseAdapter,
    ModelFamilyAdapter,
    QwenDenseAdapter,
    QwenMoEAdapter,
    get_adapter,
    register_adapter,
)


def config_for(name: str) -> ModelConfig:
    return load_model_config(CONFIGS_DIR / "models" / f"{name}.yaml")


class TestAdapterResolution:
    @pytest.mark.parametrize(
        ("config_name", "expected"),
        [
            ("qwen3_8b", QwenDenseAdapter),
            ("qwen3_30b_a3b_thinking", QwenMoEAdapter),
            ("mistral_small_3_2", Mistral3VLMAdapter),
            ("ministral_8b", MistralDenseAdapter),
        ],
    )
    def test_each_config_resolves_to_its_adapter(self, config_name, expected):
        assert isinstance(get_adapter(config_for(config_name)), expected)

    def test_model_type_is_inferred_from_the_checkpoint_name(self):
        adapter = get_adapter(ModelConfig(name="x", family="", base_model="Qwen/Qwen3-8B"))
        assert isinstance(adapter, QwenDenseAdapter)

    def test_moe_is_inferred_before_dense(self):
        adapter = get_adapter(
            ModelConfig(name="x", family="", base_model="Qwen/Qwen3-30B-A3B-Thinking-2507")
        )
        assert isinstance(adapter, QwenMoEAdapter)

    def test_mistral3_is_inferred_before_plain_mistral(self):
        adapter = get_adapter(
            ModelConfig(
                name="x", family="", base_model="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            )
        )
        assert isinstance(adapter, Mistral3VLMAdapter)

    def test_unknown_architecture_is_rejected_with_guidance(self):
        with pytest.raises(ModelCompatibilityError, match="No model-family adapter"):
            get_adapter(ModelConfig(name="x", family="", base_model="acme/unknown-arch-v1"))

    def test_family_mismatch_is_caught(self):
        # Usually means the wrong checkpoint was pasted into a config.
        with pytest.raises(ModelCompatibilityError, match="declares family"):
            get_adapter(
                ModelConfig(
                    name="x", family="qwen", base_model="mistralai/Ministral-8B-Instruct-2410"
                )
            )

    def test_a_new_family_can_be_registered_without_touching_the_pipeline(self):
        class FakeAdapter(ModelFamilyAdapter):
            model_types = ("fake_arch",)
            family = "fake"

            @property
            def capabilities(self):
                from kleos_models.models.adapters import ModelCapabilities

                return ModelCapabilities(
                    family="fake",
                    model_type="fake_arch",
                    auto_class="AutoModelForCausalLM",
                    reasoning=ReasoningCapability.UNSUPPORTED,
                )

            @property
            def default_target_modules(self):
                return ["q_proj"]

        register_adapter(FakeAdapter)
        try:
            adapter = get_adapter(
                ModelConfig(name="x", family="fake", base_model="acme/x", model_type="fake_arch")
            )
            assert isinstance(adapter, FakeAdapter)
        finally:
            ADAPTER_REGISTRY.pop("fake_arch", None)


class TestAutoClassSelection:
    def test_mistral_small_requires_the_image_text_to_text_class(self):
        # Verified against transformers: mistral3 appears only in
        # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, never the causal-LM mapping.
        adapter = get_adapter(config_for("mistral_small_3_2"))
        assert adapter.capabilities.auto_class == "AutoModelForImageTextToText"

    @pytest.mark.parametrize("config_name", ["qwen3_8b", "qwen3_30b_a3b_thinking", "ministral_8b"])
    def test_text_only_models_use_the_causal_lm_class(self, config_name):
        adapter = get_adapter(config_for(config_name))
        assert adapter.capabilities.auto_class == "AutoModelForCausalLM"


class TestLoRATargeting:
    def test_dense_models_target_attention_and_mlp(self):
        targets = get_adapter(config_for("qwen3_8b")).default_target_modules
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)

    def test_moe_targets_attention_only(self):
        # 48 layers x 128 experts x 3 projections would be ~18,400 adapters.
        targets = get_adapter(config_for("qwen3_30b_a3b_thinking")).default_target_modules
        assert set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj"}
        assert "gate_proj" not in targets

    def test_moe_excludes_experts_and_the_router(self):
        adapter = get_adapter(config_for("qwen3_30b_a3b_thinking"))
        excluded = adapter.excluded_module_patterns
        assert any("experts" in pattern for pattern in excluded)
        assert any("mlp.gate" in pattern for pattern in excluded)

    def test_moe_keeps_the_router_unquantized(self):
        adapter = get_adapter(config_for("qwen3_30b_a3b_thinking"))
        assert any("gate" in module for module in adapter.modules_to_not_quantize)

    def test_vlm_scopes_lora_to_the_language_tower(self):
        adapter = get_adapter(config_for("mistral_small_3_2"))
        assert adapter.capabilities.language_model_prefix == "language_model"

    def test_vlm_excludes_the_vision_tower(self):
        adapter = get_adapter(config_for("mistral_small_3_2"))
        excluded = adapter.excluded_module_patterns
        assert "vision_tower" in excluded
        assert "multi_modal_projector" in excluded

    def test_vlm_keeps_the_vision_tower_unquantized(self):
        adapter = get_adapter(config_for("mistral_small_3_2"))
        assert "vision_tower" in adapter.modules_to_not_quantize

    def test_every_adapter_excludes_the_lm_head(self):
        for name in ("qwen3_8b", "ministral_8b", "mistral_small_3_2", "qwen3_30b_a3b_thinking"):
            assert "lm_head" in get_adapter(config_for(name)).excluded_module_patterns

    def test_explicit_targets_override_auto(self):
        adapter = get_adapter(config_for("qwen3_8b"))
        resolved = adapter.resolve_target_modules(LoRAConfig(target_modules=["q_proj"]))
        assert resolved == ["q_proj"]

    def test_auto_resolves_to_family_defaults(self):
        adapter = get_adapter(config_for("qwen3_8b"))
        resolved = adapter.resolve_target_modules(LoRAConfig(target_modules="auto"))
        assert resolved == adapter.default_target_modules


class TestReasoningCapability:
    def test_qwen3_8b_is_switchable(self):
        adapter = get_adapter(config_for("qwen3_8b"))
        assert adapter.capabilities.reasoning is ReasoningCapability.SWITCHABLE
        assert adapter.resolve_reasoning_mode(ReasoningMode.THINKING) is ReasoningMode.THINKING
        assert (
            adapter.resolve_reasoning_mode(ReasoningMode.NON_THINKING) is ReasoningMode.NON_THINKING
        )

    def test_qwen3_8b_passes_enable_thinking_to_the_template(self):
        adapter = get_adapter(config_for("qwen3_8b"))
        assert adapter.chat_template_kwargs(ReasoningMode.THINKING) == {"enable_thinking": True}
        assert adapter.chat_template_kwargs(ReasoningMode.NON_THINKING) == {
            "enable_thinking": False
        }

    def test_thinking_only_model_refuses_non_thinking(self):
        adapter = get_adapter(config_for("qwen3_30b_a3b_thinking"))
        assert adapter.capabilities.reasoning is ReasoningCapability.ALWAYS_ON
        with pytest.raises(ModelCompatibilityError, match="thinking-only"):
            adapter.resolve_reasoning_mode(ReasoningMode.NON_THINKING)

    def test_thinking_only_model_sends_no_enable_thinking_kwarg(self):
        # Qwen3-30B-A3B-Thinking-2507's template does not accept the flag at all.
        adapter = get_adapter(config_for("qwen3_30b_a3b_thinking"))
        assert adapter.chat_template_kwargs(ReasoningMode.THINKING) == {}

    def test_non_reasoning_model_refuses_thinking(self):
        adapter = get_adapter(config_for("ministral_8b"))
        assert adapter.capabilities.reasoning is ReasoningCapability.UNSUPPORTED
        with pytest.raises(ModelCompatibilityError, match="no reasoning mode"):
            adapter.resolve_reasoning_mode(ReasoningMode.THINKING)

    def test_refusal_mentions_not_faking_reasoning(self):
        adapter = get_adapter(config_for("ministral_8b"))
        with pytest.raises(ModelCompatibilityError) as info:
            adapter.resolve_reasoning_mode(ReasoningMode.THINKING)
        assert "<think>" in str(info.value)

    def test_non_reasoning_model_sends_no_template_kwargs(self):
        adapter = get_adapter(config_for("ministral_8b"))
        assert adapter.chat_template_kwargs(ReasoningMode.STANDARD) == {}


class TestManifestDescription:
    def test_description_records_the_identity_fields(self):
        description = get_adapter(config_for("qwen3_8b")).describe()
        for field in ("family", "base_model", "revision", "model_type", "auto_class"):
            assert field in description

    def test_moe_description_records_active_parameters(self):
        description = get_adapter(config_for("qwen3_30b_a3b_thinking")).describe()
        assert description["is_moe"] is True
        assert description["active_parameter_count"] is not None
        assert description["active_parameter_count"] < description["parameter_count"]

    def test_vlm_description_is_flagged_multimodal(self):
        description = get_adapter(config_for("mistral_small_3_2")).describe()
        assert description["is_multimodal"] is True

    def test_notes_document_the_architectural_traps(self):
        moe = get_adapter(config_for("qwen3_30b_a3b_thinking")).describe()
        assert any("MoE" in note or "expert" in note for note in moe["notes"])

        vlm = get_adapter(config_for("mistral_small_3_2")).describe()
        assert any("AutoModelForCausalLM" in note for note in vlm["notes"])
