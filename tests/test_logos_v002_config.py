"""KLEOS Logos v0.0.2's run configuration, pre-registered under H9."""

from __future__ import annotations

import pytest
import yaml
from tests.conftest import CONFIGS_DIR

from kleos_models.config import ReasoningMode, load_config
from kleos_models.models.adapters import Ministral3ReasoningTextAdapter, get_adapter

TRAINING = CONFIGS_DIR / "training"

#: The Kaggle notebook's paths, which H9's hash uses. Kaggle never saves /tmp,
#: so the private release never reaches a notebook's saved output.
KAGGLE_DATASET = "/tmp/kleos-data/kleos-policy-v0.0.7"
KAGGLE_OUTPUTS = "/kaggle/working/outputs"
#: The Colab paths v0.0.1's hash was computed with (tests/test_logos_config.py).
DRIVE_DATASET = "/content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6"
DRIVE_OUTPUTS = "/content/drive/MyDrive/kleos-private/outputs"

#: Pre-registered under H9 in docs/experiments.md, before any Logos v0.0.2 training.
LOGOS_V002_HASH = "d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9"

EXPERIMENT_ID = "kleos-v007-ministral314breasoning-run1"


@pytest.fixture
def clean_env(monkeypatch):
    """The environment variables that feed ``${env:...}`` in the configs."""
    for name in ("KLEOS_BENCHMARK_PATH", "KLEOS_DATASET_PATH", "KLEOS_ADAPTER_PATH"):
        monkeypatch.delenv(name, raising=False)


def kaggle_config(name: str):
    return load_config(TRAINING / name, dataset_path=KAGGLE_DATASET, output_dir=KAGGLE_OUTPUTS)


def v001():
    return load_config(
        TRAINING / "kleos_logos_v001.yaml", dataset_path=DRIVE_DATASET, output_dir=DRIVE_OUTPUTS
    )


class TestPreRegisteredHash:
    def test_the_hash_is_the_pre_registered_one(self, clean_env):
        assert kaggle_config("kleos_logos_v002.yaml").config_hash == LOGOS_V002_HASH

    def test_an_environment_variable_changes_it(self, clean_env, monkeypatch):
        # Why the notebook refuses to run with any KLEOS_* variable set.
        monkeypatch.setenv("KLEOS_BENCHMARK_PATH", "/kaggle/working/benchmark/benchmark.jsonl")
        assert kaggle_config("kleos_logos_v002.yaml").config_hash != LOGOS_V002_HASH

    def test_another_dataset_path_changes_it(self, clean_env):
        moved = load_config(
            TRAINING / "kleos_logos_v002.yaml",
            dataset_path="/kaggle/input/kleos-policy-v0-0-7",
            output_dir=KAGGLE_OUTPUTS,
        )
        assert moved.config_hash != LOGOS_V002_HASH


class TestAgainstLogosV001:
    def test_it_differs_from_v001_only_where_declared(self, clean_env):
        old = v001().canonical_dict()
        new = kaggle_config("kleos_logos_v002.yaml").canonical_dict()

        def differing(section: str) -> set[str]:
            left, right = old[section], new[section]
            return {k for k in set(left) | set(right) if left.get(k) != right.get(k)}

        # Training: only where the run writes, which the Kaggle paths move.
        assert differing("training") == {"output_dir"}
        assert differing("model") == {
            "base_model",
            "model_type",
            "name",
            "notes",
            "reasoning",
            "revision",
        }
        assert differing("dataset") == {"path", "version"}
        assert differing("evaluation") == {"generation"}
        assert {
            k
            for k in set(old["evaluation"]["generation"]) | set(new["evaluation"]["generation"])
            if old["evaluation"]["generation"].get(k) != new["evaluation"]["generation"].get(k)
        } == {"max_new_tokens"}
        top = {k for k in old if k not in ("model", "training", "dataset", "evaluation")}
        assert {k for k in top if old[k] != new.get(k)} == {
            "description",
            "hypothesis",
            "name",
            "tags",
        }

    def test_the_declared_values(self, clean_env):
        config = kaggle_config("kleos_logos_v002.yaml")
        assert config.hypothesis == "H9"
        assert config.dataset.version == "kleos-policy-v0.0.7"
        assert config.evaluation.generation.max_new_tokens == 1024
        assert config.evaluation.generation.do_sample is False
        assert config.evaluation.generation.temperature == 0.0
        assert config.evaluation.seeds == [42]
        assert config.model.max_seq_length == 1024

    def test_it_is_trained_to_think(self, clean_env):
        model = kaggle_config("kleos_logos_v002.yaml").model
        adapter = get_adapter(model)
        assert isinstance(adapter, Ministral3ReasoningTextAdapter)
        assert adapter.resolve_reasoning_mode(model.reasoning.default_mode) is (
            ReasoningMode.THINKING
        )
        assert model.reasoning.strip_thinking_from_targets is False

    def test_the_evaluation_config_overrides_only_the_token_budget(self):
        raw = yaml.safe_load(
            (CONFIGS_DIR / "evaluation" / "kleos_logos_v002.yaml").read_text(encoding="utf-8")
        )
        assert raw == {
            "extends": "kleos_logos_v001.yaml",
            "evaluation": {"generation": {"max_new_tokens": 1024}},
        }

    def test_the_header_states_the_pre_registered_command(self):
        header = (TRAINING / "kleos_logos_v002.yaml").read_text(encoding="utf-8")
        for fragment in (KAGGLE_DATASET, KAGGLE_OUTPUTS, EXPERIMENT_ID):
            assert fragment in header


class TestSmokeConfig:
    def test_the_smoke_config_keeps_the_real_model_and_data(self, clean_env):
        smoke = kaggle_config("debug_logos_v002.yaml")
        real = kaggle_config("kleos_logos_v002.yaml")
        assert smoke.model == real.model
        assert smoke.dataset == real.dataset
        assert smoke.training.strict_config is True
        assert smoke.training.max_steps == 10
        assert smoke.training.save_steps == 5

    def test_it_measures_at_the_real_settings(self, clean_env):
        smoke = kaggle_config("debug_logos_v002.yaml").training
        real = kaggle_config("kleos_logos_v002.yaml").training
        for field in (
            "per_device_train_batch_size",
            "gradient_checkpointing",
            "optim",
            "precision",
            "learning_rate",
        ):
            assert getattr(smoke, field) == getattr(real, field), field


class TestTruncationTripwireIsWired:
    """Review Focus 2: the trainer, not only the formatter, refuses a cut trace."""

    @staticmethod
    def bundle():
        from tests.conftest import make_example

        from kleos_models.data.loaders import DatasetBundle
        from kleos_models.data.schemas import TrainingExample

        payload = make_example("trace-0001", assistant="1. alpha")
        payload["version"] = "1.1"
        payload["messages"][-1]["reasoning"] = "step " * 40
        return DatasetBundle(train=[TrainingExample.model_validate(payload)])

    def test_a_reasoning_run_refuses_before_loading_weights(self, clean_env):
        from tests.conftest import FakeTokenizer

        from kleos_models.errors import DataValidationError
        from kleos_models.training.trainer import measure_sequence_lengths

        config = load_config(
            TRAINING / "kleos_logos_v002.yaml",
            ["model.max_seq_length=30"],
            dataset_path=KAGGLE_DATASET,
            output_dir=KAGGLE_OUTPUTS,
        )
        with pytest.raises(DataValidationError, match="max_seq_length"):
            measure_sequence_lengths(
                config, self.bundle(), tokenizer=FakeTokenizer(renders_reasoning=True)
            )

    def test_a_run_not_trained_to_think_drops_the_trace_instead(self, clean_env):
        from tests.conftest import FakeTokenizer

        from kleos_models.training.trainer import measure_sequence_lengths

        measured = measure_sequence_lengths(
            v001(), self.bundle(), tokenizer=FakeTokenizer(renders_reasoning=True)
        )
        assert measured["max_tokens"] < 40
