"""A model's serving names and options come from its deployment record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from kleos_models.errors import ConfigError
from kleos_models.serving import space, zerogpu
from kleos_models.serving.profile import (
    HERMES_PROFILE,
    GpuDefaults,
    load_profile,
    profile_from_record,
)

ROOT = Path(__file__).resolve().parents[1]
HERMES_RECORD = ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def logos_spec(**serving: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "display_name": "Logos v0.0.2",
        "env_prefix": "LOGOS",
        "key_header": "x-logos-key",
        "space_dir": "deploy/zerogpu-space-logos",
        "record_file": "logos_record.yaml",
        "base_files": ["config.json"],
        "space_only": True,
        "reply": {"reasoning": True, "require_system_message": True},
        **serving,
    }
    return {"name": "kleos-logos", "version": "v0.0.2", "serving": block}


class TestHermesDefaults:
    def test_the_hermes_record_resolves_to_todays_constants(self):
        profile = load_profile(HERMES_RECORD)
        assert profile == HERMES_PROFILE
        assert profile.short_name == "Hermes"
        assert profile.display_name == "Hermes v0.0.6"
        assert profile.key_header == zerogpu.KEY_HEADER
        assert profile.env("API_KEY") == zerogpu.ENV_API_KEY
        assert profile.env("PACKAGE_REPO") == zerogpu.ENV_PACKAGE_REPO
        assert profile.env("PACKAGE_REVISION") == zerogpu.ENV_PACKAGE_REVISION
        assert profile.env("GPU_MAX_SECONDS") == zerogpu.ENV_GPU_MAX_SECONDS
        assert profile.space_dir == str(space.SPACE_SOURCE_DIR)
        assert profile.record_file == space.RECORD_NAME
        assert frozenset(profile.base_files) == space.BASE_PRELOAD_FILES
        assert profile.reasoning is False
        assert profile.require_system_message is False
        assert profile.space_only is False
        assert profile.contract_version == 1
        assert profile.gpu == GpuDefaults()
        assert profile.research_report == (
            "docs/experiments/kleos-v006-mistralnemo12b-run1-report.md"
        )


class TestLogosProfile:
    def test_a_serving_block_sets_every_name(self):
        profile = profile_from_record(logos_spec())
        assert profile.short_name == "Logos"
        assert profile.display_name == "Logos v0.0.2"
        assert profile.env("API_KEY") == "LOGOS_API_KEY"
        assert profile.key_header == "x-logos-key"
        assert profile.space_dir == "deploy/zerogpu-space-logos"
        assert profile.record_file == "logos_record.yaml"
        assert profile.base_files == ("config.json",)
        assert profile.reasoning and profile.require_system_message and profile.space_only
        assert profile.contract_version == 2

    def test_gpu_defaults_can_be_set(self):
        profile = profile_from_record(logos_spec(gpu={"tokens_per_second": 20}))
        assert profile.gpu.tokens_per_second == 20
        assert profile.gpu.max_seconds == 60

    @pytest.mark.parametrize(
        ("serving", "message"),
        [
            ({"colour": "blue"}, "colour"),
            ({"reply": {"reasoning": True, "stream": True}}, "stream"),
            ({"gpu": {"seconds": 5}}, "seconds"),
            ({"env_prefix": "logos"}, "env_prefix"),
            ({"key_header": "X-Logos"}, "key_header"),
            ({"record_file": "../x.yaml"}, "record_file"),
            ({"base_files": []}, "base_files"),
            ({"base_files": ["a/b.json"]}, "base_files"),
            ({"reply": {"reasoning": "yes"}}, "reasoning"),
        ],
    )
    def test_a_bad_serving_block_is_refused(self, serving, message):
        with pytest.raises(ConfigError, match=message):
            profile_from_record(logos_spec(**serving))

    def test_load_profile_reads_the_deployment_mapping(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump({"deployment": logos_spec()}), encoding="utf-8")
        assert load_profile(path).short_name == "Logos"

    def test_a_record_without_a_deployment_mapping_is_refused(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="deployment"):
            load_profile(path)
