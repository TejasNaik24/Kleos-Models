"""Model-family adapter tests."""

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
    get_adapter,
    register_adapter,
)


def config_for(name: str) -> ModelConfig:
    return load_model_config(CONFIGS_DIR / "models" / f"{name}.yaml")


class TestAdapterResolution:
    @pytest.mark.parametrize(
        ("config_name", "expected"),
        [
            ("mistral_small_3_2", Mistral3VLMAdapter),
            ("ministral_8b", MistralDenseAdapter),
            ("mistral_nemo_12b", MistralDenseAdapter),
        ],
    )
    def test_each_config_resolves_to_its_adapter(self, config_name, expected):
        assert isinstance(get_adapter(config_for(config_name)), expected)

    def test_model_type_is_inferred_from_the_checkpoint_name(self):
        adapter = get_adapter(
            ModelConfig(name="x", family="", base_model="mistralai/Mistral-Nemo-Instruct-2407")
        )
        assert isinstance(adapter, MistralDenseAdapter)

    @pytest.mark.parametrize("model_type", ["mistral", "ministral"])
    def test_both_mistral_model_types_resolve(self, model_type):
        # transformers 4.x reports Ministral as 'mistral', 5.x as 'ministral'.
        config = config_for("ministral_8b").model_copy(update={"model_type": model_type})
        assert isinstance(get_adapter(config), MistralDenseAdapter)

    def test_ministral_is_registered(self):
        assert ADAPTER_REGISTRY["ministral"] is MistralDenseAdapter

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
        with pytest.raises(ModelCompatibilityError, match="declares family"):
            get_adapter(
                ModelConfig(
                    name="x", family="other", base_model="mistralai/Ministral-8B-Instruct-2410"
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
        # mistral3 is only in transformers' image-text-to-text mapping, not causal-LM.
        adapter = get_adapter(config_for("mistral_small_3_2"))
        assert adapter.capabilities.auto_class == "AutoModelForImageTextToText"

    @pytest.mark.parametrize("config_name", ["ministral_8b", "mistral_nemo_12b", "ministral3_14b"])
    def test_text_only_models_use_the_causal_lm_class(self, config_name):
        adapter = get_adapter(config_for(config_name))
        assert adapter.capabilities.auto_class == "AutoModelForCausalLM"


class TestLoRATargeting:
    def test_dense_models_target_attention_and_mlp(self):
        targets = get_adapter(config_for("ministral_8b")).default_target_modules
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)

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
        for name in ("ministral_8b", "mistral_nemo_12b", "mistral_small_3_2", "ministral3_14b"):
            assert "lm_head" in get_adapter(config_for(name)).excluded_module_patterns

    def test_explicit_targets_override_auto(self):
        adapter = get_adapter(config_for("ministral_8b"))
        resolved = adapter.resolve_target_modules(LoRAConfig(target_modules=["q_proj"]))
        assert resolved == ["q_proj"]

    def test_auto_resolves_to_family_defaults(self):
        adapter = get_adapter(config_for("ministral_8b"))
        resolved = adapter.resolve_target_modules(LoRAConfig(target_modules="auto"))
        assert resolved == adapter.default_target_modules


class TestReasoningCapability:
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

    def test_refusal_points_at_the_reasoning_config_shipped_here(self):
        adapter = get_adapter(config_for("ministral_8b"))
        with pytest.raises(ModelCompatibilityError) as info:
            adapter.resolve_reasoning_mode(ReasoningMode.THINKING)
        assert "ministral3_14b_reasoning" in str(info.value)
        assert "qwen" not in str(info.value).lower()


class TestOnlyMistralFamiliesShip:
    """The repository ships Mistral-family adapters only."""

    def test_no_non_mistral_model_type_is_registered(self):
        assert all(cls.family == "mistral" for cls in ADAPTER_REGISTRY.values())
        assert not any("qwen" in model_type for model_type in ADAPTER_REGISTRY)

    def test_an_unrecognized_checkpoint_name_has_no_adapter(self):
        with pytest.raises(ModelCompatibilityError, match="No model-family adapter"):
            get_adapter(ModelConfig(name="x", family="", base_model="acme/unknown-8B"))

    def test_the_removed_reasoning_values_are_gone(self):
        assert not hasattr(ReasoningCapability, "SWITCHABLE")
        assert not hasattr(ReasoningMode, "NON_THINKING")

    def test_non_reasoning_model_sends_no_template_kwargs(self):
        adapter = get_adapter(config_for("ministral_8b"))
        assert adapter.chat_template_kwargs(ReasoningMode.STANDARD) == {}


class TestManifestDescription:
    def test_description_records_the_identity_fields(self):
        description = get_adapter(config_for("ministral_8b")).describe()
        for field in ("family", "base_model", "revision", "model_type", "auto_class"):
            assert field in description

    def test_notes_document_the_architectural_traps(self):

        vlm = get_adapter(config_for("mistral_small_3_2")).describe()
        assert any("AutoModelForCausalLM" in note for note in vlm["notes"])
