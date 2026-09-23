"""Deployment artifact identity and integrity.

These tests guard the failures that are silent in production: an adapter served
against the wrong base revision, a corrupted weight file, a manifest that
disagrees with the artifact it describes. None of them raises on its own, so
each one gets an explicit check here.

Everything runs without torch, transformers or a GPU: verifying an artifact must
be possible on the host that is about to serve it, before anything is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kleos_models.errors import ConfigError
from kleos_models.serving.manifest import (
    DEPLOYMENT_ARTIFACT_VERSION,
    DeploymentManifest,
    FileRecord,
    RuntimeContract,
    TokenizerContract,
    build_deployment_manifest,
    verify_package,
)

PINNED = "04d8a90549d23fc6bd7f642064003592df51e9b3"
ADAPTER_CONFIG = {
    "base_model_name_or_path": "mistralai/Mistral-Nemo-Instruct-2407",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
    "exclude_modules": ["embed_tokens", "lm_head"],
    "peft_version": "0.20.0",
    "revision": PINNED,
}


#: The runtime contract the real Hermes record pins: NF4, float16 compute.
RUNTIME = RuntimeContract(
    quantization_mode="nf4",
    double_quant=True,
    compute_dtype="float16",
    attn_implementation="sdpa",
    max_seq_length=1024,
)


def packaged_model_config(runtime: RuntimeContract = RUNTIME, revision: str = PINNED) -> dict:
    """The packaged model config, as build_deployment_package.py writes it."""
    return {
        "model": {
            "name": "mistral_nemo_12b",
            "family": "mistral",
            "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
            "revision": revision,
            "max_seq_length": runtime.max_seq_length,
            "attn_implementation": runtime.attn_implementation,
            "quantization": {
                "mode": runtime.quantization_mode,
                "double_quant": runtime.double_quant,
                "compute_dtype": runtime.compute_dtype,
            },
        }
    }


def build_package(
    tmp_path: Path,
    *,
    revision: str = PINNED,
    adapter_config=None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """A miniature but structurally real deployment package.

    ``extra_files`` (relative path -> bytes) are written before the manifest
    is built, so they are recorded in it like any other package file.
    """
    package = tmp_path / "hermes-test"
    (package / "adapter").mkdir(parents=True)
    (package / "tokenizer").mkdir(parents=True)
    (package / "deployment").mkdir(parents=True)
    for relative, content in (extra_files or {}).items():
        (package / relative).write_bytes(content)

    (package / "adapter" / "adapter_model.safetensors").write_bytes(b"fake weights" * 8)
    config = dict(ADAPTER_CONFIG if adapter_config is None else adapter_config)
    config.setdefault("revision", revision)
    (package / "adapter" / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (package / "tokenizer" / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
    (package / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (package / "deployment" / "model_config.yaml").write_text(
        yaml.safe_dump(packaged_model_config(revision=revision)), encoding="utf-8"
    )

    manifest = build_deployment_manifest(
        package,
        model_name="kleos-hermes",
        model_version="v0.0.6",
        base_model="mistralai/Mistral-Nemo-Instruct-2407",
        base_revision=revision,
        experiment_id="kleos-v006-mistralnemo12b-run1",
        source_checkpoint="checkpoint-200",
        trainable_parameters=57_016_320,
        dataset_version="kleos-policy-v0.0.6",
        dataset_sha256="c" * 64,
        training_config_hash="b" * 64,
        adapter_config=config,
        fix_mistral_regex=False,
        runtime=RUNTIME,
    )
    manifest.save(package)
    return package


class TestManifestRecordsIdentity:
    def test_a_built_package_verifies(self, tmp_path):
        package = build_package(tmp_path)
        manifest = verify_package(package)
        assert manifest.model_name == "kleos-hermes"
        assert manifest.base_revision == PINNED
        assert manifest.adapter.trainable_parameters == 57_016_320

    def test_the_weights_hash_is_recorded_and_matches_the_file_record(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        weights = next(
            r for r in manifest.adapter.files if r.path.endswith("adapter_model.safetensors")
        )
        assert weights.sha256 == manifest.adapter.weights_sha256

    def test_every_package_file_is_hashed(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        paths = {record.path for record in manifest.all_files()}
        assert "adapter/adapter_model.safetensors" in paths
        assert "adapter/adapter_config.json" in paths
        assert "tokenizer/tokenizer.json" in paths
        assert "tokenizer/tokenizer_config.json" in paths

    def test_the_manifest_carries_no_filesystem_paths_from_the_build_host(self, tmp_path):
        package = build_package(tmp_path)
        text = (package / "manifest.json").read_text(encoding="utf-8")
        # Recorded paths are relative to the package; the build directory must
        # not leak into an artifact that travels.
        assert str(tmp_path) not in text
        assert "MyDrive" not in text


class TestCorruptionIsDetected:
    def test_modified_weights_are_rejected(self, tmp_path):
        package = build_package(tmp_path)
        (package / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert "adapter_model.safetensors" in str(error.value.details["problems"])

    def test_a_missing_file_is_rejected(self, tmp_path):
        package = build_package(tmp_path)
        (package / "tokenizer" / "tokenizer.json").unlink()
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("missing file" in p for p in error.value.details["problems"])

    def test_a_truncated_file_is_rejected_on_size_before_hashing(self, tmp_path):
        package = build_package(tmp_path)
        weights = package / "adapter" / "adapter_model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-4])
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("size" in p for p in error.value.details["problems"])

    def test_a_package_with_no_manifest_is_refused(self, tmp_path):
        package = build_package(tmp_path)
        (package / "manifest.json").unlink()
        with pytest.raises(ConfigError, match="No deployment manifest"):
            verify_package(package)

    def test_a_corrupt_manifest_is_refused(self, tmp_path):
        package = build_package(tmp_path)
        (package / "manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            verify_package(package)


class TestBaseRevisionMustBePinned:
    """Finding H-F1. This is the check the research artifact could not pass."""

    @pytest.mark.parametrize("revision", ["main", "", "refs/heads/main", "v1.0", PINNED[:12]])
    def test_an_unpinned_revision_cannot_be_recorded(self, tmp_path, revision):
        with pytest.raises(Exception, match=r"40-character|revision"):
            build_package(tmp_path, revision=revision)

    def test_a_null_revision_in_the_adapter_config_is_rejected(self, tmp_path):
        package = build_package(tmp_path)
        config = json.loads((package / "adapter" / "adapter_config.json").read_text())
        config["revision"] = None
        (package / "adapter" / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        problems = " ".join(error.value.details["problems"])
        assert "revision=null" in problems or "H-F1" in problems

    def test_a_revision_disagreeing_with_the_manifest_is_rejected(self, tmp_path):
        package = build_package(tmp_path)
        config = json.loads((package / "adapter" / "adapter_config.json").read_text())
        config["revision"] = "a" * 40
        (package / "adapter" / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("revision" in p for p in error.value.details["problems"])

    def test_a_different_base_model_is_rejected(self, tmp_path):
        package = build_package(tmp_path)
        config = json.loads((package / "adapter" / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
        (package / "adapter" / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("base_model_name_or_path" in p for p in error.value.details["problems"])

    def test_a_changed_lora_shape_is_rejected(self, tmp_path):
        package = build_package(tmp_path)
        config = json.loads((package / "adapter" / "adapter_config.json").read_text())
        config["r"] = 8
        (package / "adapter" / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("r is 8" in p for p in error.value.details["problems"])


class TestTokenizerContract:
    def test_the_serving_tokenizer_flag_is_explicit(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        # Not None, not absent: the value v0.0.6 trained under, stated.
        assert manifest.tokenizer.fix_mistral_regex is False

    def test_the_tokenizer_must_come_from_the_package(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        assert manifest.tokenizer.source == "frozen_package"

    def test_resolving_the_tokenizer_anywhere_else_is_refused(self):
        # Loading from the Hub would put tokenization back at the mercy of a
        # library default, which is the drift the pin exists to prevent.
        with pytest.raises(Exception, match="frozen_package"):
            TokenizerContract(source="hub", fix_mistral_regex=False)

    def test_tokenizer_files_are_hashed_so_behaviour_cannot_drift(self, tmp_path):
        package = build_package(tmp_path)
        (package / "tokenizer" / "tokenizer.json").write_text('{"model": {"changed": 1}}')
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        assert any("tokenizer/tokenizer.json" in p for p in error.value.details["problems"])


class TestSchemaSafety:
    def test_a_file_record_cannot_escape_the_package(self):
        for path in ("../secrets.txt", "/etc/passwd", "a/../../b"):
            with pytest.raises(Exception, match="inside the package"):
                FileRecord(path=path, size_bytes=1, sha256="a" * 64)

    def test_a_file_record_requires_a_real_sha256(self):
        with pytest.raises(Exception, match="sha256"):
            FileRecord(path="adapter/x", size_bytes=1, sha256="nope")

    def test_an_unknown_schema_version_is_refused(self, tmp_path):
        package = build_package(tmp_path)
        payload = json.loads((package / "manifest.json").read_text())
        payload["deployment_artifact_version"] = DEPLOYMENT_ARTIFACT_VERSION + 1
        (package / "manifest.json").write_text(json.dumps(payload))
        with pytest.raises(ConfigError, match="schema"):
            DeploymentManifest.load(package)

    def test_unknown_fields_are_rejected(self, tmp_path):
        package = build_package(tmp_path)
        payload = json.loads((package / "manifest.json").read_text())
        payload["surprise"] = True
        (package / "manifest.json").write_text(json.dumps(payload))
        with pytest.raises(ConfigError):
            DeploymentManifest.load(package)


class TestGenerationContract:
    def test_decoding_defaults_are_greedy(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        assert manifest.generation.do_sample is False
        assert manifest.generation.temperature == 0.0

    def test_limits_are_recorded_for_review(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        assert manifest.limits.max_new_tokens > 0
        assert manifest.limits.max_input_chars > 0
        assert manifest.limits.request_timeout_seconds > 0


class TestRuntimeContract:
    """Schema v2: how the base is instantiated is part of the identity.

    v0.0.6 was evaluated with 4-bit matmuls in float16. `auto` would resolve to
    bfloat16 on any GPU newer than the T4 it trained on — including the ZeroGPU
    Blackwell — and nothing would raise.
    """

    @pytest.mark.parametrize("dtype", ["auto", "AUTO", "default", ""])
    def test_a_hardware_dependent_dtype_is_not_a_contract(self, dtype):
        with pytest.raises(Exception, match="compute_dtype"):
            RuntimeContract(
                quantization_mode="nf4",
                double_quant=True,
                compute_dtype=dtype,
                attn_implementation="sdpa",
                max_seq_length=1024,
            )

    def test_the_contract_is_recorded(self, tmp_path):
        manifest = verify_package(build_package(tmp_path))
        assert manifest.runtime.compute_dtype == "float16"
        assert manifest.runtime.quantization_mode == "nf4"
        assert "deployment/model_config.yaml" in {r.path for r in manifest.all_files()}

    def test_an_edited_model_config_is_caught_by_hash_and_by_content(self, tmp_path):
        package = build_package(tmp_path)
        edited = packaged_model_config(RUNTIME.model_copy(update={"compute_dtype": "bfloat16"}))
        (package / "deployment" / "model_config.yaml").write_text(yaml.safe_dump(edited))
        with pytest.raises(ConfigError) as error:
            verify_package(package)
        problems = " ".join(error.value.details["problems"])
        assert "deployment/model_config.yaml" in problems
        assert "compute_dtype is 'bfloat16'" in problems

    def test_a_package_without_a_model_config_is_refused(self, tmp_path):
        package = build_package(tmp_path)
        (package / "deployment" / "model_config.yaml").unlink()
        with pytest.raises(ConfigError):
            verify_package(package)

    def test_a_version_one_package_is_refused(self, tmp_path):
        # v1 packages never pinned the compute dtype; serving one would reopen
        # exactly the drift schema v2 closes.
        package = build_package(tmp_path)
        payload = json.loads((package / "manifest.json").read_text())
        payload["deployment_artifact_version"] = 1
        (package / "manifest.json").write_text(json.dumps(payload))
        with pytest.raises(ConfigError, match="schema"):
            DeploymentManifest.load(package)
