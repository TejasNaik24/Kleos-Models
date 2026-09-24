"""KLEOS Logos v0.0.1: configuration, the text-only view, and the tokenizer flag.

Three things are pinned here:

* **Hermes is untouched.** Its ``config_hash`` still reproduces, and the field added
  for Logos (``fix_mistral_regex``) is invisible to every existing config.
* **Logos differs from Hermes only where declared**, so H8 compares base models,
  not recipes.
* **The text-only view is refused unless it is exactly right**, and every other
  model still loads the way it did.

All of it runs without torch: checkpoints are represented by their config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import yaml
from tests.conftest import CONFIGS_DIR

from kleos_models.config import ModelConfig, load_config, load_model_config, load_raw_config
from kleos_models.errors import ConfigError, ModelCompatibilityError
from kleos_models.models import adapters
from kleos_models.models.adapters import (
    MISTRAL3_NON_TEXT_PREFIXES,
    MISTRAL3_TEXT_KEY_MAPPING,
    Ministral3TextAdapter,
    Mistral3VLMAdapter,
    MistralDenseAdapter,
    get_adapter,
    resolve_adapter_from_hf_config,
    resolve_load_plan,
)
from kleos_models.models.loading import resolve_fix_mistral_regex

TRAINING = CONFIGS_DIR / "training"
MODELS = CONFIGS_DIR / "models"

#: The paths every pinned hash is computed with (the Colab runbook's).
DRIVE_DATASET = "/content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6"
DRIVE_OUTPUTS = "/content/drive/MyDrive/kleos-private/outputs"

HERMES_HASH = "b2328857c6026dd7c904a6acadd00ec55a6ce5289bf6210e743615fce6d8d3e8"
#: Pre-registered under H8 in docs/experiments.md before any Logos training.
LOGOS_HASH = "18008c6716a58afc284c64eb7e1e96c9bfce34b8da2b8c489b6d59c0c45b6f71"

LOGOS_BASE = "mistralai/Ministral-3-14B-Instruct-2512-BF16"


@pytest.fixture
def clean_env(monkeypatch):
    """The environment variables that feed ``${env:...}`` in the configs."""
    for name in ("KLEOS_BENCHMARK_PATH", "KLEOS_DATASET_PATH", "KLEOS_ADAPTER_PATH"):
        monkeypatch.delenv(name, raising=False)


def drive_config(name: str):
    return load_config(TRAINING / name, dataset_path=DRIVE_DATASET, output_dir=DRIVE_OUTPUTS)


def logos_model() -> ModelConfig:
    return load_model_config(MODELS / "ministral3_14b.yaml")


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------


class TestFrozenConfigHashes:
    def test_hermes_hash_still_reproduces(self, clean_env):
        # The hash recorded in Hermes' manifest, deployment record and report.
        assert drive_config("kleos_hermes_v006.yaml").config_hash == HERMES_HASH

    def test_logos_hash_is_the_pre_registered_one(self, clean_env):
        # docs/experiments.md (H8) and docs/logos.md quote it; the runbook
        # tells the user to check the printed prefix against it.
        assert drive_config("kleos_logos_v001.yaml").config_hash == LOGOS_HASH

    def test_an_environment_variable_changes_the_hash(self, clean_env, monkeypatch):
        # Why the runbook insists no KLEOS_* variable is set.
        monkeypatch.setenv("KLEOS_BENCHMARK_PATH", "/content/benchmark/benchmark.jsonl")
        assert drive_config("kleos_hermes_v006.yaml").config_hash != HERMES_HASH


class TestHashNeutralField:
    """Design rule 4: a field added after runs were recorded moves no old hash."""

    @pytest.mark.parametrize("path", sorted(MODELS.glob("*.yaml")), ids=lambda p: p.stem)
    def test_existing_model_configs_do_not_serialize_the_flag(self, path):
        config = load_model_config(path)
        if path.stem == "ministral3_14b":
            assert config.model_dump()["fix_mistral_regex"] is True
            return
        assert "fix_mistral_regex" not in config.model_dump()
        assert "fix_mistral_regex" not in config.model_dump(mode="json")

    def test_unset_and_explicit_null_hash_identically(self, clean_env, tmp_path):
        explicit = _write_training_config(tmp_path, "null")
        assert (
            load_config(explicit, dataset_path=DRIVE_DATASET, output_dir=DRIVE_OUTPUTS).config_hash
            == drive_config("kleos_hermes_v006.yaml").config_hash
        )

    @pytest.mark.parametrize("value", ["true", "false"])
    def test_a_stated_value_is_hashed(self, clean_env, tmp_path, value):
        stated = _write_training_config(tmp_path, value)
        config = load_config(stated, dataset_path=DRIVE_DATASET, output_dir=DRIVE_OUTPUTS)
        assert config.model.model_dump()["fix_mistral_regex"] is (value == "true")
        assert config.config_hash != HERMES_HASH

    def test_true_and_false_hash_differently(self, clean_env, tmp_path):
        hashes = {
            load_config(
                _write_training_config(tmp_path / value, value),
                dataset_path=DRIVE_DATASET,
                output_dir=DRIVE_OUTPUTS,
            ).config_hash
            for value in ("true", "false")
        }
        assert len(hashes) == 2

    def test_the_flag_is_refused_outside_the_mistral_family(self):
        with pytest.raises(ValueError, match="Mistral tokenizers only"):
            ModelConfig(name="q", family="qwen", base_model="Qwen/Qwen3-8B", fix_mistral_regex=True)


def _write_training_config(directory: Path, value: str) -> Path:
    """Hermes' training config with ``model.fix_mistral_regex`` stated."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "hermes_with_flag.yaml"
    path.write_text(
        f"extends: {TRAINING / 'kleos_hermes_v006.yaml'}\nmodel:\n  fix_mistral_regex: {value}\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Logos against Hermes
# ---------------------------------------------------------------------------


class TestLogosAgainstHermes:
    def test_logos_differs_from_hermes_only_where_declared(self, clean_env):
        hermes = drive_config("kleos_hermes_v006.yaml").canonical_dict()
        logos = drive_config("kleos_logos_v001.yaml").canonical_dict()

        def differing(section: str) -> set[str]:
            left, right = hermes[section], logos[section]
            return {k for k in set(left) | set(right) if left.get(k) != right.get(k)}

        assert differing("training") == {"eval_steps", "save_steps", "strict_config"}
        assert differing("model") == {
            "architecture",
            "base_model",
            "context_limit",
            "fix_mistral_regex",
            "model_type",
            "name",
            "notes",
            "parameter_count",
            "revision",
        }
        assert hermes["dataset"] == logos["dataset"]
        assert hermes["evaluation"] == logos["evaluation"]
        top = {k for k in hermes if k not in ("model", "training", "dataset", "evaluation")}
        assert {k for k in top if hermes[k] != logos.get(k)} == {
            "description",
            "hypothesis",
            "name",
            "tags",
        }

    def test_declared_training_differences_have_the_declared_values(self, clean_env):
        training = drive_config("kleos_logos_v001.yaml").training
        assert (training.save_steps, training.eval_steps) == (25, 25)
        assert training.strict_config is True
        assert training.load_best_model_at_end is True

    def test_hermes_quantization_and_lora_carry_over_unchanged(self, clean_env):
        hermes = drive_config("kleos_hermes_v006.yaml").model
        logos = drive_config("kleos_logos_v001.yaml").model
        assert logos.quantization == hermes.quantization
        assert logos.lora == hermes.lora
        assert logos.max_seq_length == hermes.max_seq_length == 1024

    def test_logos_evaluation_config_is_hermes_verbatim(self):
        raw = yaml.safe_load(
            (CONFIGS_DIR / "evaluation" / "kleos_logos_v001.yaml").read_text(encoding="utf-8")
        )
        assert raw == {"extends": "kleos_policy_v006.yaml"}

    def test_the_smoke_config_keeps_the_real_model_settings(self, clean_env):
        smoke = drive_config("debug_logos.yaml")
        real = drive_config("kleos_logos_v001.yaml")
        assert smoke.model == real.model
        assert smoke.training.strict_config is True
        assert smoke.training.max_steps == 10
        assert smoke.training.save_steps == 5  # a checkpoint the resume drill can use

    def test_logos_selects_the_text_view_adapter(self):
        assert isinstance(get_adapter(logos_model()), Ministral3TextAdapter)

    def test_other_mistral_configs_keep_their_adapters(self):
        assert isinstance(
            get_adapter(load_model_config(MODELS / "mistral_nemo_12b.yaml")), MistralDenseAdapter
        )
        assert isinstance(
            get_adapter(load_model_config(MODELS / "ministral_8b.yaml")), MistralDenseAdapter
        )
        assert isinstance(
            get_adapter(load_model_config(MODELS / "mistral_small_3_2.yaml")), Mistral3VLMAdapter
        )

    def test_the_checkpoint_name_implies_the_text_tower(self):
        config = ModelConfig(name="m", family="mistral", base_model=LOGOS_BASE)
        assert adapters._infer_model_type(config) == "ministral3"


class TestSetModel:
    """``--set-model``: plan another model inside an existing recipe."""

    def test_the_model_include_is_replaced(self, clean_env):
        config = load_config(
            TRAINING / "qlora_small.yaml", model_path=MODELS / "ministral3_14b.yaml"
        )
        assert config.model.base_model == LOGOS_BASE

    def test_the_training_configs_own_model_keys_still_apply(self, clean_env):
        # Hermes' config restates max_seq_length; that survives the swap.
        raw = load_raw_config(
            TRAINING / "kleos_hermes_v006.yaml", model_path=MODELS / "ministral3_14b.yaml"
        )
        assert raw["model"]["base_model"] == LOGOS_BASE
        assert raw["model"]["max_seq_length"] == 1024

    def test_the_override_reaches_through_extends(self, clean_env, tmp_path):
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            f"extends: {CONFIGS_DIR / 'base.yaml'}\nincludes:\n  model: {MODELS / 'qwen3_8b.yaml'}\n",
            encoding="utf-8",
        )
        child = tmp_path / "child.yaml"
        child.write_text("extends: parent.yaml\nname: child\n", encoding="utf-8")
        raw = load_raw_config(child, model_path=MODELS / "ministral3_14b.yaml")
        assert raw["model"]["base_model"] == LOGOS_BASE

    def test_a_config_without_a_model_include_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to replace"):
            load_raw_config(
                CONFIGS_DIR / "evaluation" / "kleos_logos_v001.yaml",
                model_path=MODELS / "ministral3_14b.yaml",
            )


# ---------------------------------------------------------------------------
# The text-only view
# ---------------------------------------------------------------------------


def mistral3_container(**overrides: Any) -> SimpleNamespace:
    """What ``AutoConfig`` reports for the Ministral 3 checkpoint, in miniature."""
    text = SimpleNamespace(model_type="ministral3", dtype=None, hidden_size=5120)
    fields: dict[str, Any] = {
        "model_type": "mistral3",
        "architectures": ["Mistral3ForConditionalGeneration"],
        "text_config": text,
        "dtype": "bfloat16",
        "quantization_config": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestCheckpointView:
    def test_the_view_maps_keys_and_keeps_the_weights_dtype(self):
        container = mistral3_container()
        view = Ministral3TextAdapter(logos_model()).checkpoint_view(container)
        assert view is not None
        assert view.kind == "text_only"
        assert dict(view.key_mapping) == MISTRAL3_TEXT_KEY_MAPPING
        assert view.ignored_prefixes == MISTRAL3_NON_TEXT_PREFIXES
        assert view.text_config.dtype == "bfloat16"
        assert view.dtype_inherited == "bfloat16"

    def test_the_view_copies_the_text_config(self):
        container = mistral3_container()
        view = Ministral3TextAdapter(logos_model()).checkpoint_view(container)
        assert view is not None
        assert view.text_config is not container.text_config
        assert container.text_config.dtype is None  # the checkpoint's config is untouched

    def test_load_kwargs_request_loading_info(self):
        view = Ministral3TextAdapter(logos_model()).checkpoint_view(mistral3_container())
        assert view is not None
        kwargs = view.load_kwargs()
        assert kwargs["config"] is view.text_config
        assert kwargs["key_mapping"] == MISTRAL3_TEXT_KEY_MAPPING
        assert kwargs["output_loading_info"] is True

    def test_a_plain_ministral3_checkpoint_needs_no_view(self):
        container = mistral3_container(model_type="ministral3")
        assert Ministral3TextAdapter(logos_model()).checkpoint_view(container) is None

    @pytest.mark.parametrize(
        ("overrides", "problem"),
        [
            ({"text_config": SimpleNamespace(model_type="mistral", dtype=None)}, "text tower"),
            ({"architectures": ["LlavaForConditionalGeneration"]}, "declares"),
            ({"quantization_config": {"quant_method": "fp8"}}, "pre-quantized"),
        ],
        ids=["mistral-small-tower", "wrong-architecture", "fp8-release"],
    )
    def test_the_view_refuses_what_it_cannot_open(self, overrides, problem):
        with pytest.raises(ModelCompatibilityError, match="text-only view can open") as info:
            Ministral3TextAdapter(logos_model()).checkpoint_view(mistral3_container(**overrides))
        assert any(problem in item for item in info.value.details["problems"])


class TestLoadingInfo:
    def _view(self):
        view = Ministral3TextAdapter(logos_model()).checkpoint_view(mistral3_container())
        assert view is not None
        return view

    def test_a_clean_load_reports_what_was_left_on_disk(self):
        counts = self._view().validate_loading_info(
            {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": [
                    "vision_tower.transformer.layers.0.attention.q_proj.weight",
                    "vision_tower.patch_conv.weight",
                    "multi_modal_projector.linear_1.weight",
                ],
            }
        )
        assert counts["missing_keys"] == counts["mismatched_keys"] == 0
        assert counts["unexpected_by_prefix"] == {"vision_tower.": 2, "multi_modal_projector.": 1}

    @pytest.mark.parametrize(
        "info",
        [
            {"missing_keys": ["model.layers.0.mlp.up_proj.weight"]},
            {"mismatched_keys": [("lm_head.weight", (8, 4), (4, 8))]},
            {"unexpected_keys": ["language_model.model.layers.0.mlp.up_proj.weight"]},
        ],
        ids=["missing", "mismatched", "unmapped-text-key"],
    )
    def test_anything_but_the_vision_tower_left_over_is_refused(self, info):
        with pytest.raises(ModelCompatibilityError, match="did not load cleanly"):
            self._view().validate_loading_info(info)

    def test_vision_parameters_in_the_loaded_model_are_refused(self):
        model = SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.w", None), ("vision_tower.w", None)]
        )
        with pytest.raises(ModelCompatibilityError, match="vision parameters"):
            Ministral3TextAdapter(logos_model()).prepare_model_for_training(model)


class TestLoadPlan:
    def test_logos_gets_the_view_and_keeps_its_config(self):
        config = logos_model()
        plan = resolve_load_plan(config, mistral3_container())
        assert isinstance(plan.adapter, Ministral3TextAdapter)
        assert plan.view is not None
        assert plan.config is config
        assert plan.flipped_from is None

    def test_a_disagreeing_checkpoint_is_still_trusted_otherwise(self, caplog):
        # Ministral-8B's config says 'ministral'; transformers 4.x reports 'mistral'.
        config = load_model_config(MODELS / "ministral_8b.yaml")
        with caplog.at_level(logging.WARNING, logger="kleos_models.models.adapters"):
            plan = resolve_load_plan(config, SimpleNamespace(model_type="mistral"))
        assert plan.view is None
        assert plan.config.model_type == "mistral"
        assert plan.flipped_from == "ministral"
        assert isinstance(plan.adapter, MistralDenseAdapter)
        assert "Trusting the checkpoint" in caplog.text

    def test_hermes_loads_exactly_as_before(self):
        config = load_model_config(MODELS / "mistral_nemo_12b.yaml")
        plan = resolve_load_plan(config, SimpleNamespace(model_type="mistral"))
        assert plan.config is config
        assert plan.view is None and plan.flipped_from is None

    def test_mistral_small_keeps_the_full_vision_language_path(self):
        config = load_model_config(MODELS / "mistral_small_3_2.yaml")
        plan = resolve_load_plan(
            config,
            mistral3_container(text_config=SimpleNamespace(model_type="mistral", dtype=None)),
        )
        assert isinstance(plan.adapter, Mistral3VLMAdapter)
        assert plan.view is None

    def test_adapter_resolution_keeps_the_view_adapter(self):
        adapter = resolve_adapter_from_hf_config(mistral3_container(), logos_model())
        assert isinstance(adapter, Ministral3TextAdapter)


# ---------------------------------------------------------------------------
# The tokenizer regex flag
# ---------------------------------------------------------------------------


class TestRegexFlag:
    def _config(self, value: bool | None) -> ModelConfig:
        return logos_model().model_copy(update={"fix_mistral_regex": value})

    @pytest.mark.parametrize(
        ("configured", "requested", "expected"),
        [
            (None, None, (None, "unset")),
            (True, None, (True, "config")),
            (None, False, (False, "caller")),
            (True, True, (True, "both")),
        ],
    )
    def test_resolution(self, configured, requested, expected):
        assert resolve_fix_mistral_regex(self._config(configured), requested) == expected

    def test_a_disagreement_is_an_error(self):
        # Training and serving must tokenize identically.
        with pytest.raises(ModelCompatibilityError, match="disagree"):
            resolve_fix_mistral_regex(self._config(True), False)

    def test_the_flag_reaches_the_tokenizer(self, monkeypatch):
        from kleos_models.models import loading

        captured: dict[str, Any] = {}

        class FakeTokenizer:
            chat_template = "{{ messages }}"
            pad_token_id = 11
            padding_side = "left"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: Any) -> FakeTokenizer:
                captured.update(kwargs, model_id=model_id)
                return FakeTokenizer()

        monkeypatch.setattr(
            loading,
            "require_transformers",
            lambda: SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        monkeypatch.setattr(
            loading, "mistral_regex_kwarg", lambda flag: {"fix_mistral_regex": flag}
        )

        loading.load_tokenizer(self._config(True))
        assert captured["fix_mistral_regex"] is True
        assert captured["revision"] == "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"

        captured.clear()
        loading.load_tokenizer(load_model_config(MODELS / "mistral_nemo_12b.yaml"))
        assert "fix_mistral_regex" not in captured  # Hermes: nothing passed, as before


# ---------------------------------------------------------------------------
# load_model, with the heavy parts faked
# ---------------------------------------------------------------------------


class _FakeAuto:
    """Records what ``from_pretrained`` was asked for."""

    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    loading_info: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
        cls.calls.append((model_id, kwargs))
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            generation_config=SimpleNamespace(eos_token_id=2, pad_token_id=None),
            named_parameters=lambda: [("model.layers.0.self_attn.q_proj.weight", None)],
            parameters=lambda: [],
        )
        if kwargs.get("output_loading_info"):
            return model, cls.loading_info
        return model


@pytest.fixture
def fake_loading(monkeypatch):
    from kleos_models import compat
    from kleos_models.models import loading

    _FakeAuto.calls = []
    _FakeAuto.loading_info = {
        "missing_keys": [],
        "mismatched_keys": [],
        "unexpected_keys": ["vision_tower.a.weight", "multi_modal_projector.b.weight"],
    }
    containers: dict[str, Any] = {}

    monkeypatch.setattr(compat, "check_transformers_version", lambda: None)
    monkeypatch.setattr(loading, "load_hf_config", lambda config: containers[config.base_model])
    monkeypatch.setattr(
        loading, "quantization_support", lambda: SimpleNamespace(cuda_available=False)
    )
    monkeypatch.setattr(
        loading, "build_quantization_config", lambda *a, **k: (None, {"mode": "none"})
    )
    monkeypatch.setattr(loading, "resolve_compute_dtype", lambda *a, **k: (None, "fake"))
    monkeypatch.setattr(loading, "load_tokenizer", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(loading, "_hf_token", lambda: None)
    monkeypatch.setattr(adapters.ModelFamilyAdapter, "auto_model_class", lambda self: _FakeAuto)
    return loading, containers


class TestLoadModelKwargs:
    def test_hermes_load_kwargs_carry_nothing_new(self, fake_loading):
        loading, containers = fake_loading
        config = load_model_config(MODELS / "mistral_nemo_12b.yaml")
        containers[config.base_model] = SimpleNamespace(
            model_type="mistral", architectures=["MistralForCausalLM"]
        )
        loaded = loading.load_model(config)
        ((model_id, kwargs),) = _FakeAuto.calls
        assert model_id == config.base_model
        assert not {"config", "key_mapping", "output_loading_info"} & set(kwargs)
        assert kwargs["revision"] == "04d8a90549d23fc6bd7f642064003592df51e9b3"
        assert "view" not in loaded.load_metadata
        assert "fix_mistral_regex" not in loaded.load_metadata

    def test_logos_loads_through_the_view_and_records_it(self, fake_loading):
        loading, containers = fake_loading
        config = logos_model()
        containers[config.base_model] = mistral3_container()
        loaded = loading.load_model(config)
        ((_, kwargs),) = _FakeAuto.calls
        assert kwargs["key_mapping"] == MISTRAL3_TEXT_KEY_MAPPING
        assert kwargs["output_loading_info"] is True
        assert kwargs["config"].model_type == "ministral3"
        view = loaded.load_metadata["view"]
        assert view["kind"] == "text_only"
        assert view["loading_info"]["missing_keys"] == 0
        assert view["loading_info"]["unexpected_by_prefix"] == {
            "vision_tower.": 1,
            "multi_modal_projector.": 1,
        }
        assert loaded.load_metadata["fix_mistral_regex"] is True
        assert loaded.load_metadata["fix_mistral_regex_source"] == "config"
        assert isinstance(loaded.adapter, Ministral3TextAdapter)

    def test_an_incomplete_view_load_is_refused(self, fake_loading):
        loading, containers = fake_loading
        config = logos_model()
        containers[config.base_model] = mistral3_container()
        _FakeAuto.loading_info = {"missing_keys": ["model.norm.weight"]}
        with pytest.raises(ModelCompatibilityError, match="did not load cleanly"):
            loading.load_model(config)

    def test_a_conflicting_regex_flag_fails_before_any_download(self, fake_loading):
        loading, _containers = fake_loading
        with pytest.raises(ModelCompatibilityError, match="disagree"):
            loading.load_model(logos_model(), fix_mistral_regex=False)
        assert _FakeAuto.calls == []
