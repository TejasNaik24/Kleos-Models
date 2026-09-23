"""Loading a deployment, and refusing to.

The checks here are the ones that prevent a silent failure: an adapter attached
to the wrong base revision produces no error, just worse answers. Everything a
laptop can check runs here; the parts that need real weights are marked and skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import CONFIGS_DIR
from tests.test_serving_api import build_manifest

from kleos_models.config import DType, GenerationConfig, load_model_config
from kleos_models.errors import ConfigError
from kleos_models.inference.backends import GenerationOutput
from kleos_models.serving.loader import (
    LoadedDeployment,
    check_config_matches_manifest,
    check_loaded_adapter,
    check_loaded_tokenizer,
    load_packaged_model_config,
    verify_base_revision,
)
from kleos_models.serving.manifest import ServingLimits

PINNED = "04d8a90549d23fc6bd7f642064003592df51e9b3"
OTHER_SHA = "a" * 40


@dataclass
class RecordingBackend:
    """Captures the GenerationConfig it is asked for."""

    seen: list[GenerationConfig]

    def generate(self, messages, config):
        self.seen.append(config)
        return GenerationOutput(text="", completion_tokens=len(self.seen))


def build_loaded(manifest=None) -> tuple[LoadedDeployment, RecordingBackend]:
    backend = RecordingBackend(seen=[])
    loaded = LoadedDeployment(
        manifest=manifest or build_manifest(),
        model_config=None,  # not consulted by generate()
        backend=backend,
        package_dir=Path("."),
    )
    return loaded, backend


class TestUnpinnedBaseIsRefused:
    """Finding H-F1: the deployment path must fail closed on a moving pointer."""

    @pytest.mark.parametrize("revision", ["main", "master", "v1.0", "", PINNED[:12], None])
    def test_an_unpinned_revision_is_refused_before_any_download(self, revision):
        with pytest.raises(ConfigError, match="Refusing to serve"):
            verify_base_revision("mistralai/Mistral-Nemo-Instruct-2407", revision)

    def test_the_refusal_explains_the_silent_failure_it_prevents(self):
        with pytest.raises(ConfigError) as error:
            verify_base_revision("mistralai/Mistral-Nemo-Instruct-2407", "main")
        text = " ".join(error.value.suggestions)
        assert "40-character" in text
        assert "no error raised" in text or "H-F1" in text


class TestPackageSelfConsistency:
    def test_a_config_disagreeing_about_the_revision_is_refused(self):
        manifest = build_manifest()
        config = load_model_config(CONFIGS_DIR / "models" / "mistral_nemo_12b.yaml")
        config.revision = OTHER_SHA
        with pytest.raises(ConfigError, match="contradicts itself"):
            check_config_matches_manifest(config, manifest)

    def test_a_config_disagreeing_about_the_base_model_is_refused(self):
        manifest = build_manifest()
        config = load_model_config(CONFIGS_DIR / "models" / "mistral_nemo_12b.yaml")
        config.base_model = "mistralai/Ministral-8B-Instruct-2410"
        with pytest.raises(ConfigError, match="contradicts itself"):
            check_config_matches_manifest(config, manifest)

    def test_the_raw_training_config_is_not_accepted_as_the_served_model(self):
        # The training config says compute_dtype: auto, which resolved to
        # float16 on the T4 that trained Hermes and would resolve to bfloat16 on
        # any newer GPU. Served as-is it would not be the measured model.
        config = load_model_config(CONFIGS_DIR / "models" / "mistral_nemo_12b.yaml")
        with pytest.raises(ConfigError, match="contradicts itself") as error:
            check_config_matches_manifest(config, build_manifest())
        assert "compute_dtype is 'auto'" in " ".join(error.value.details["problems"])

    def test_the_training_config_with_the_runtime_contract_applied_is_accepted(self):
        # What build_deployment_package.py packages: same base and revision,
        # with the runtime contract stated.
        config = load_model_config(CONFIGS_DIR / "models" / "mistral_nemo_12b.yaml")
        config.quantization.compute_dtype = DType.FLOAT16
        check_config_matches_manifest(config, build_manifest())

    def test_a_package_without_a_model_config_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="No packaged model config"):
            load_packaged_model_config(tmp_path)


class TestGenerationContractIsEnforcedByTheLoader:
    def test_defaults_are_the_frozen_greedy_settings(self):
        loaded, _ = build_loaded()
        config = loaded.generation_defaults
        assert config.do_sample is False
        assert config.temperature == 0.0
        assert config.max_new_tokens == 512

    def test_a_caller_may_lower_the_token_budget(self):
        loaded, backend = build_loaded()
        loaded.generate([], max_new_tokens=32)
        assert backend.seen[-1].max_new_tokens == 32

    def test_a_caller_cannot_raise_the_budget_above_the_manifest_ceiling(self):
        manifest = build_manifest(
            limits=ServingLimits(max_new_tokens=128, max_input_chars=200, max_messages=4)
        )
        loaded, backend = build_loaded(manifest)
        loaded.generate([], max_new_tokens=100_000)
        assert backend.seen[-1].max_new_tokens == 128

    def test_omitting_the_budget_uses_the_frozen_default(self):
        loaded, backend = build_loaded()
        loaded.generate([])
        assert backend.seen[-1].max_new_tokens == 512

    def test_a_completion_that_fills_the_budget_is_reported_as_truncated(self):
        loaded, _ = build_loaded()
        # RecordingBackend returns one completion token per call so far.
        assert loaded.generate([], max_new_tokens=1).finish_reason == "length"
        assert loaded.generate([], max_new_tokens=512).finish_reason == "stop"

    def test_describe_reports_identity_without_secrets_or_paths(self):
        loaded, _ = build_loaded()
        described = loaded.describe()
        assert described["base_revision"] == PINNED
        assert described["tokenizer_fix_mistral_regex"] is False
        assert not any("/" in str(v) for k, v in described.items() if k.endswith("path"))


class FakeTokenizer:
    def __init__(self, **attributes: Any) -> None:
        self.padding_side = "right"
        self.pad_token = "</s>"
        self.chat_template = "{{ messages }}"
        for key, value in attributes.items():
            setattr(self, key, value)


class TestTokenizerContractIsChecked:
    def test_a_matching_tokenizer_passes(self):
        manifest = build_manifest()
        manifest.tokenizer.pad_token = "</s>"
        check_loaded_tokenizer(FakeTokenizer(), manifest)

    def test_a_different_padding_side_is_refused(self):
        manifest = build_manifest()
        with pytest.raises(Exception, match="padding_side"):
            check_loaded_tokenizer(FakeTokenizer(padding_side="left"), manifest)

    def test_a_different_pad_token_is_refused(self):
        manifest = build_manifest()
        manifest.tokenizer.pad_token = "</s>"
        with pytest.raises(Exception, match="pad_token"):
            check_loaded_tokenizer(FakeTokenizer(pad_token="<pad>"), manifest)

    def test_a_tokenizer_without_a_chat_template_is_refused(self):
        with pytest.raises(Exception, match="chat template"):
            check_loaded_tokenizer(FakeTokenizer(chat_template=None), build_manifest())


class FakePeftConfig:
    def __init__(self, **attributes: Any) -> None:
        self.r = 16
        self.lora_alpha = 32
        self.task_type = "CAUSAL_LM"
        self.base_model_name_or_path = "mistralai/Mistral-Nemo-Instruct-2407"
        for key, value in attributes.items():
            setattr(self, key, value)


class FakePeftModel:
    def __init__(self, config: Any | None) -> None:
        self.peft_config = {"default": config} if config is not None else {}


class TestAttachedAdapterIsChecked:
    def test_a_matching_adapter_passes(self):
        check_loaded_adapter(FakePeftModel(FakePeftConfig()), build_manifest())

    def test_a_base_model_with_no_adapter_is_refused(self):
        with pytest.raises(Exception, match="No PEFT adapter"):
            check_loaded_adapter(FakePeftModel(None), build_manifest())

    def test_a_different_rank_is_refused(self):
        with pytest.raises(Exception, match="r is 8"):
            check_loaded_adapter(FakePeftModel(FakePeftConfig(r=8)), build_manifest())

    def test_a_different_base_model_is_refused(self):
        config = FakePeftConfig(base_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")
        with pytest.raises(Exception, match="base_model_name_or_path"):
            check_loaded_adapter(FakePeftModel(config), build_manifest())
