"""Configuration tests (spec §6, §30).

Defaults work, invalid hyperparameters fail, model configs resolve, composition
behaves, and the config hash is a reliable identity for a run.
"""

from __future__ import annotations

import pytest
from tests.conftest import CONFIGS_DIR

from kleos_models.config import (
    DType,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    LoRAConfig,
    ModelConfig,
    Precision,
    QuantizationMode,
    ReasoningConfig,
    ReasoningMode,
    TrainingConfig,
    apply_overrides,
    deep_merge,
    load_config,
    load_model_config,
    parse_override,
)
from kleos_models.errors import ConfigError


class TestDefaults:
    def test_training_defaults_are_usable(self):
        config = TrainingConfig()
        assert config.learning_rate > 0
        assert config.gradient_checkpointing is True
        assert config.effective_batch_size == 8

    def test_lora_defaults(self):
        lora = LoRAConfig()
        assert lora.target_modules == "auto"
        assert lora.scaling == pytest.approx(2.0)

    def test_quantization_defaults_to_nf4(self):
        model = ModelConfig(name="m", family="qwen", base_model="Qwen/Qwen3-8B")
        assert model.quantization.mode is QuantizationMode.NF4
        assert model.quantization.enabled

    def test_tokenizer_defaults_to_the_base_model(self):
        model = ModelConfig(name="m", family="qwen", base_model="Qwen/Qwen3-8B")
        assert model.tokenizer_id == "Qwen/Qwen3-8B"

    def test_explicit_tokenizer_is_used(self):
        model = ModelConfig(
            name="m", family="qwen", base_model="Qwen/Qwen3-8B", tokenizer="other/tok"
        )
        assert model.tokenizer_id == "other/tok"


class TestInvalidHyperparameters:
    def test_negative_learning_rate_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1e-4)

    def test_learning_rate_above_one_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=5.0)

    def test_zero_batch_size_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(per_device_train_batch_size=0)

    def test_zero_gradient_accumulation_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_warmup_ratio_at_or_above_one_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(warmup_ratio=1.0)

    def test_lora_rank_must_be_positive(self):
        with pytest.raises(ValueError):
            LoRAConfig(r=0)

    def test_lora_dropout_must_be_below_one(self):
        with pytest.raises(ValueError):
            LoRAConfig(dropout=1.0)

    def test_empty_target_modules_list_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            LoRAConfig(target_modules=[])

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(lernign_rate=1e-4)

    def test_max_seq_length_beyond_context_limit_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            ModelConfig(
                name="m",
                family="qwen",
                base_model="x",
                context_limit=2048,
                max_seq_length=4096,
            )

    def test_load_best_requires_evaluation(self):
        with pytest.raises(ValueError, match="requires eval_strategy"):
            TrainingConfig(load_best_model_at_end=True, eval_strategy="no")

    def test_load_best_requires_matching_strategies(self):
        with pytest.raises(ValueError, match="save_strategy == eval_strategy"):
            TrainingConfig(
                load_best_model_at_end=True, eval_strategy="steps", save_strategy="epoch"
            )


class TestReasoningConfig:
    def test_thinking_default_requires_support(self):
        with pytest.raises(ValueError, match=r"requires reasoning\.supported"):
            ReasoningConfig(supported=False, default_mode=ReasoningMode.THINKING)

    def test_supported_model_may_default_to_thinking(self):
        config = ReasoningConfig(supported=True, default_mode=ReasoningMode.THINKING)
        assert config.default_mode is ReasoningMode.THINKING

    def test_reasoning_spans_are_stripped_by_default(self):
        config = ReasoningConfig()
        assert config.strip_thinking_from_targets is True
        assert config.strip_thinking_from_history is True


class TestGenerationConfig:
    def test_greedy_is_the_default(self):
        config = GenerationConfig()
        assert config.do_sample is False
        assert config.temperature == 0.0

    def test_sampling_at_zero_temperature_is_rejected(self):
        with pytest.raises(ValueError, match="contradictory"):
            GenerationConfig(do_sample=True, temperature=0.0)


class TestEvaluationConfig:
    def test_unknown_arm_is_rejected(self):
        with pytest.raises(ValueError, match="unknown research arm"):
            EvaluationConfig(arms=["arm9_imaginary"])

    def test_empty_arms_are_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EvaluationConfig(arms=[])

    def test_duplicate_seeds_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            EvaluationConfig(seeds=[42, 42])

    def test_empty_seeds_are_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EvaluationConfig(seeds=[])


class TestDatasetFilters:
    def test_unknown_task_filter_is_rejected(self):
        from kleos_models.config import DatasetFilters

        with pytest.raises(ValueError, match=r"unknown task"):
            DatasetFilters(tasks=["not_a_task"])

    def test_unknown_quality_status_is_rejected(self):
        from kleos_models.config import DatasetFilters

        with pytest.raises(ValueError, match="unknown quality status"):
            DatasetFilters(quality_statuses=["excellent"])

    def test_reviewed_only_is_the_default_gate(self):
        from kleos_models.config import DatasetFilters

        assert DatasetFilters().quality_statuses == ["reviewed"]


class TestConfigComposition:
    def test_deep_merge_merges_nested_dicts(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}
        merged = deep_merge(base, override)
        assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}

    def test_deep_merge_replaces_lists_outright(self):
        # Merging lists element-wise would silently produce a target set nobody
        # wrote down.
        merged = deep_merge({"targets": ["a", "b"]}, {"targets": ["c"]})
        assert merged["targets"] == ["c"]

    def test_deep_merge_does_not_mutate_its_inputs(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1

    def test_parse_override_types_values(self):
        assert parse_override("training.learning_rate=1e-4") == (
            ["training", "learning_rate"],
            0.0001,
        )
        assert parse_override("training.gradient_checkpointing=true")[1] is True
        assert parse_override("model.lora.r=32")[1] == 32

    def test_parse_override_rejects_a_missing_equals(self):
        with pytest.raises(ConfigError, match="Malformed override"):
            parse_override("no_equals_sign")

    def test_apply_overrides_creates_missing_paths(self):
        result = apply_overrides({}, ["a.b.c=1"])
        assert result == {"a": {"b": {"c": 1}}}


class TestRealConfigFiles:
    def test_qlora_small_loads(self):
        config = load_config(CONFIGS_DIR / "training" / "qlora_small.yaml")
        assert config.model.name == "qwen3_8b"
        assert config.dataset is not None
        assert config.training.optim == "paged_adamw_8bit"

    def test_debug_config_is_tiny(self):
        config = load_config(CONFIGS_DIR / "training" / "debug.yaml")
        assert config.training.max_steps == 10
        assert config.model.max_seq_length <= 512

    @pytest.mark.parametrize(
        "name",
        ["qwen3_8b", "ministral_8b", "mistral_small_3_2", "qwen3_30b_a3b_thinking"],
    )
    def test_every_model_config_loads(self, name):
        config = load_model_config(CONFIGS_DIR / "models" / f"{name}.yaml")
        assert config.name == name
        assert config.base_model

    def test_moe_config_records_active_parameters(self):
        config = load_model_config(CONFIGS_DIR / "models" / "qwen3_30b_a3b_thinking.yaml")
        assert config.is_moe
        assert config.active_parameter_count is not None
        assert config.active_parameter_count < config.parameter_count

    def test_multimodal_config_is_flagged(self):
        config = load_model_config(CONFIGS_DIR / "models" / "mistral_small_3_2.yaml")
        assert config.is_multimodal
        assert config.model_type == "mistral3"

    def test_thinking_model_defaults_to_thinking(self):
        config = load_model_config(CONFIGS_DIR / "models" / "qwen3_30b_a3b_thinking.yaml")
        assert config.reasoning.default_mode is ReasoningMode.THINKING

    def test_missing_file_raises_an_actionable_error(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config(CONFIGS_DIR / "training" / "does_not_exist.yaml")

    def test_cli_overrides_apply(self):
        config = load_config(
            CONFIGS_DIR / "training" / "qlora_small.yaml",
            overrides=["training.learning_rate=5e-5", "model.lora.r=64"],
        )
        assert config.training.learning_rate == 5e-5
        assert config.model.lora.r == 64

    def test_dataset_path_override_applies(self, tmp_path):
        config = load_config(CONFIGS_DIR / "training" / "qlora_small.yaml", dataset_path=tmp_path)
        assert config.dataset is not None
        assert config.dataset.path == tmp_path


class TestConfigHash:
    def _config(self, **overrides) -> ExperimentConfig:
        return load_config(
            CONFIGS_DIR / "training" / "qlora_small.yaml",
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )

    def test_hash_is_stable_across_loads(self):
        assert self._config().config_hash == self._config().config_hash

    def test_hash_changes_with_a_hyperparameter(self):
        assert (
            self._config().config_hash
            != self._config(**{"training.learning_rate": "9e-5"}).config_hash
        )

    def test_hash_changes_with_the_model(self):
        assert (
            self._config().config_hash
            != self._config(**{"model.base_model": "other/model"}).config_hash
        )

    def test_short_hash_is_a_prefix(self):
        config = self._config()
        assert config.config_hash.startswith(config.short_hash())

    def test_yaml_round_trip_preserves_the_hash(self, tmp_path):
        import yaml

        config = self._config()
        path = config.save(tmp_path / "effective.yaml")
        reloaded = ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert reloaded.config_hash == config.config_hash


class TestSeedPropagation:
    def test_top_level_seed_reaches_training(self):
        config = load_config(CONFIGS_DIR / "training" / "qlora_small.yaml", overrides=["seed=1234"])
        assert config.training.seed == 1234

    def test_explicit_training_seed_wins(self):
        config = load_config(
            CONFIGS_DIR / "training" / "qlora_small.yaml",
            overrides=["seed=1234", "training.seed=999"],
        )
        assert config.training.seed == 999


class TestEnums:
    def test_quantization_bit_widths(self):
        assert QuantizationMode.NF4.bits == 4
        assert QuantizationMode.NF4.is_4bit
        assert QuantizationMode.INT8.bits == 8
        assert not QuantizationMode.INT8.is_4bit
        assert QuantizationMode.NONE.bits == 16

    def test_precision_and_dtype_values(self):
        assert Precision.AUTO.value == "auto"
        assert DType.BFLOAT16.value == "bfloat16"
