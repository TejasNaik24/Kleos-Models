"""The Logos v0.0.2 deployment record, and the script that completes it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.test_logos_v002_config import LOGOS_V002_HASH

from kleos_models.errors import ConfigError
from kleos_models.serving.manifest import load_expected_identity
from kleos_models.serving.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "configs" / "deployment" / "kleos_logos_v002.yaml"
MODEL = ROOT / "configs" / "models" / "ministral3_14b_reasoning.yaml"
SCRIPT = ROOT / "scripts" / "fill_deployment_record.py"


def spec() -> dict:
    return yaml.safe_load(RECORD.read_text(encoding="utf-8"))["deployment"]


class TestTheRecord:
    def test_it_serves_logos_through_its_own_profile(self):
        profile = load_profile(RECORD)
        assert profile.short_name == "Logos" and profile.display_name == "Logos v0.0.2"
        assert profile.env_prefix == "LOGOS" and profile.key_header == "x-logos-key"
        assert profile.reasoning and profile.require_system_message and profile.space_only
        assert profile.record_file == "logos_record.yaml"
        assert len(profile.base_files) == 9

    def test_it_serves_as_evaluated(self):
        record = spec()
        assert record["generation"] == {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.0,
        }
        assert record["limits"]["max_new_tokens"] == 1024
        assert record["runtime"]["compute_dtype"] == "float16"
        assert record["tokenizer"]["fix_mistral_regex"] is True

    def test_it_names_the_run_and_base_h9_recorded(self):
        record = spec()
        model = yaml.safe_load(MODEL.read_text(encoding="utf-8"))["model"]
        assert record["base_model"] == model["base_model"]
        assert record["revision"] == model["revision"]
        assert record["training_config_hash"] == LOGOS_V002_HASH
        assert record["source_checkpoint"] == "checkpoint-175"
        assert record["experiment_id"] == "kleos-v007-ministral314breasoning-run1"

    def test_an_incomplete_record_cannot_be_used(self):
        if spec()["adapter_sha256"] is not None:
            pytest.skip("the record has been completed from the run")
        with pytest.raises(ConfigError, match="adapter_sha256"):
            load_expected_identity(RECORD)


def fake_run(
    root: Path,
    *,
    best: str = "checkpoint-175",
    config_hash: str = LOGOS_V002_HASH,
    pad: object = "<pad>",
) -> Path:
    run = root / "kleos-v007-ministral314breasoning-run1"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (run / "tokenizer").mkdir()
    (run / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (run / "tokenizer" / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": pad}), encoding="utf-8"
    )
    (run / "tokenizer" / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
    for step in (175, 300):
        state = {
            "best_model_checkpoint": f"/kaggle/working/outputs/x/{best}",
            "best_metric": 0.0293,
        }
        (run / f"checkpoint-{step}").mkdir()
        (run / f"checkpoint-{step}" / "trainer_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    record = spec()
    manifest = {
        "experiment_id": record["experiment_id"],
        "config_hash": config_hash,
        "dataset_hash": record["dataset_sha256"],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def fill(run: Path, record: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run", str(run), "--record", str(record), *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


@pytest.fixture
def record_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "kleos_logos_v002.yaml"
    shutil.copyfile(RECORD, copy)
    if yaml.safe_load(copy.read_text())["deployment"]["adapter_sha256"] is not None:
        pytest.skip("the record has been completed from the run")
    return copy


class TestFillScript:
    def test_a_dry_run_prints_the_values_and_writes_nothing(self, tmp_path, record_copy):
        before = record_copy.read_text()
        result = fill(fake_run(tmp_path), record_copy)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "adapter_sha256" in result.stdout and "<pad>" in result.stdout
        assert record_copy.read_text() == before

    def test_write_completes_the_record_and_keeps_its_comments(self, tmp_path, record_copy):
        result = fill(fake_run(tmp_path), record_copy, "--write")
        assert result.returncode == 0, result.stdout + result.stderr
        text = record_copy.read_text()
        assert text.startswith("# ----") and "fill_deployment_record.py" in text
        identity = load_expected_identity(record_copy)
        assert len(identity["adapter_sha256"]) == 64
        filled = yaml.safe_load(text)["deployment"]
        assert filled["selection_value"] == 0.0293
        assert filled["tokenizer"]["pad_token"] == "<pad>"
        assert all(len(v) == 64 for v in filled["tokenizer"]["files_sha256"].values())

    def test_writing_the_same_values_again_changes_nothing(self, tmp_path, record_copy):
        run = fake_run(tmp_path)
        assert fill(run, record_copy, "--write").returncode == 0
        once = record_copy.read_text()
        assert fill(run, record_copy, "--write").returncode == 0
        assert record_copy.read_text() == once

    def test_a_value_already_different_is_never_overwritten(self, tmp_path, record_copy):
        run = fake_run(tmp_path)
        assert fill(run, record_copy, "--write").returncode == 0
        (run / "adapter" / "adapter_model.safetensors").write_bytes(b"other weights")
        result = fill(run, record_copy, "--write")
        assert result.returncode != 0 and "adapter_sha256" in result.stdout + result.stderr

    def test_another_best_checkpoint_is_refused(self, tmp_path, record_copy):
        result = fill(fake_run(tmp_path, best="checkpoint-300"), record_copy)
        assert result.returncode != 0 and "checkpoint-300" in result.stdout + result.stderr

    def test_another_runs_config_hash_is_refused(self, tmp_path, record_copy):
        result = fill(fake_run(tmp_path, config_hash="0" * 64), record_copy)
        assert result.returncode != 0 and "config_hash" in result.stdout + result.stderr

    def test_a_pad_token_given_as_a_dict_is_read(self, tmp_path, record_copy):
        result = fill(fake_run(tmp_path, pad={"content": "<pad>"}), record_copy, "--write")
        assert result.returncode == 0, result.stdout + result.stderr
        assert yaml.safe_load(record_copy.read_text())["deployment"]["tokenizer"]["pad_token"] == (
            "<pad>"
        )

    def test_a_missing_file_is_named(self, tmp_path, record_copy):
        run = fake_run(tmp_path)
        (run / "tokenizer" / "chat_template.jinja").unlink()
        result = fill(run, record_copy)
        assert result.returncode != 0 and "chat_template.jinja" in result.stdout + result.stderr
