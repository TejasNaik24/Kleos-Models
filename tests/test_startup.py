"""The container startup sequence, without Docker or a GPU.

`python -m kleos_models.serving.startup` is the image's entrypoint. Everything
it checks before the 24.5 GB download is exercised here against a synthetic
package: secrets, package integrity, package *identity*, the model cache, and
the GPU check (with a stand-in torch). One test runs the entrypoint as a real
subprocess, the way the container does.

What these tests cannot show is that the model loads and generates on a GPU
inside the image. That is the first-start step in docs/deployment.md.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from tests.conftest import REPO_ROOT
from tests.test_deployment_manifest import PINNED, build_package

from kleos_models.errors import ConfigError
from kleos_models.serving import startup
from kleos_models.serving.manifest import DeploymentManifest, load_expected_identity

# Placeholders, asserted absent from every output. Kept under 16 characters so
# scripts/check_no_private_data.py cannot mistake them for real credentials.
SECRET = "unit-bearer"
HF_SECRET = "unit-hub"


def write_record(tmp_path: Path, manifest: DeploymentManifest, **overrides) -> Path:
    """A serving record describing exactly this package, then optionally edited."""
    tokenizer_files = {Path(r.path).name: r.sha256 for r in manifest.tokenizer.files}
    deployment = {
        "name": manifest.model_name,
        "version": manifest.model_version,
        "experiment_id": manifest.adapter.experiment_id,
        "base_model": manifest.base_model,
        "revision": manifest.base_revision,
        "source_checkpoint": manifest.adapter.source_checkpoint,
        "adapter_sha256": manifest.adapter.weights_sha256,
        "dataset_version": manifest.dataset.version,
        "dataset_sha256": manifest.dataset.sha256,
        "training_config_hash": manifest.training_config_hash,
        "tokenizer": {
            "source": "frozen_package",
            "fix_mistral_regex": manifest.tokenizer.fix_mistral_regex,
            "files_sha256": tokenizer_files,
        },
        "generation": manifest.generation.model_dump(),
        "runtime": manifest.runtime.model_dump(),
    }
    for key, value in overrides.items():
        if key.startswith("tokenizer__"):
            deployment["tokenizer"][key.removeprefix("tokenizer__")] = value
        elif key.startswith("generation__"):
            deployment["generation"][key.removeprefix("generation__")] = value
        elif key.startswith("runtime__"):
            deployment["runtime"][key.removeprefix("runtime__")] = value
        else:
            deployment[key] = value
    path = tmp_path / "serving_record.yaml"
    path.write_text(yaml.safe_dump({"deployment": deployment}), encoding="utf-8")
    return path


@pytest.fixture
def package(tmp_path) -> Path:
    return build_package(tmp_path)


@pytest.fixture
def env(tmp_path, package) -> dict[str, str]:
    manifest = DeploymentManifest.load(package)
    return {
        "HERMES_PACKAGE_DIR": str(package),
        "HERMES_API_KEY": SECRET,
        "HERMES_EXPECTED_DEPLOYMENT_CONFIG": str(write_record(tmp_path, manifest)),
        "HF_HOME": str(tmp_path / "cache"),
    }


def record_with(tmp_path, package, **overrides) -> str:
    return str(write_record(tmp_path, DeploymentManifest.load(package), **overrides))


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


class TestSecrets:
    def test_a_key_from_the_environment(self, env):
        result = startup.preflight(env, check_gpu_device=False)
        assert result.api_keys == (SECRET,)
        assert result.api_key_source == "env"

    def test_a_key_from_a_mounted_secret_file(self, env, tmp_path):
        secret_file = tmp_path / "hermes_api_key"
        secret_file.write_text(SECRET + "\n", encoding="utf-8")
        env.pop("HERMES_API_KEY")
        env["HERMES_API_KEY_FILE"] = str(secret_file)
        result = startup.preflight(env, check_gpu_device=False)
        assert result.api_keys == (SECRET,)
        assert result.api_key_source == "file"

    def test_setting_both_forms_is_refused(self, env, tmp_path):
        secret_file = tmp_path / "hermes_api_key"
        secret_file.write_text("other", encoding="utf-8")
        env["HERMES_API_KEY_FILE"] = str(secret_file)
        with pytest.raises(startup.StartupError, match="exactly one") as error:
            startup.preflight(env, check_gpu_device=False)
        assert SECRET not in str(error.value)

    def test_a_missing_secret_file_is_refused(self, env, tmp_path):
        env.pop("HERMES_API_KEY")
        env["HERMES_API_KEY_FILE"] = str(tmp_path / "absent")
        with pytest.raises(startup.StartupError, match="not a readable file"):
            startup.preflight(env, check_gpu_device=False)

    def test_an_empty_secret_file_is_refused(self, env, tmp_path):
        empty = tmp_path / "empty"
        empty.write_text("   \n", encoding="utf-8")
        env.pop("HERMES_API_KEY")
        env["HERMES_API_KEY_FILE"] = str(empty)
        with pytest.raises(startup.StartupError, match="empty"):
            startup.preflight(env, check_gpu_device=False)

    def test_no_api_key_means_no_start(self, env):
        env.pop("HERMES_API_KEY")
        with pytest.raises(startup.StartupError, match="HERMES_API_KEY"):
            startup.preflight(env, check_gpu_device=False)

    def test_an_empty_api_key_means_no_start(self, env):
        # docker/hermes.env.example ships the key empty; forgetting to fill it
        # must stop the container, not start it unauthenticated.
        env["HERMES_API_KEY"] = ""
        with pytest.raises(startup.StartupError, match="HERMES_API_KEY"):
            startup.preflight(env, check_gpu_device=False)

    def test_unauthenticated_mode_must_be_explicit(self, env):
        env.pop("HERMES_API_KEY")
        env["HERMES_ALLOW_UNAUTHENTICATED"] = "1"
        assert startup.preflight(env, check_gpu_device=False).api_keys == ()

    def test_several_keys_allow_rotation(self, env):
        env["HERMES_API_KEY"] = "old-key,new-key"
        assert startup.preflight(env, check_gpu_device=False).api_keys == ("old-key", "new-key")

    def test_a_missing_hf_token_is_a_warning_not_a_failure(self, env):
        result = startup.preflight(env, check_gpu_device=False)
        assert result.hf_token_source is None
        assert any("HF_TOKEN" in warning for warning in result.warnings)

    def test_an_hf_token_file_is_accepted(self, env, tmp_path):
        token = tmp_path / "hf_token"
        token.write_text(HF_SECRET, encoding="utf-8")
        env["HF_TOKEN_FILE"] = str(token)
        assert startup.preflight(env, check_gpu_device=False).hf_token_source == "file"


# ---------------------------------------------------------------------------
# Package integrity and identity — the fail-closed core
# ---------------------------------------------------------------------------


class TestPackageMustBeTheFrozenArtifact:
    def test_the_matching_package_passes(self, env):
        result = startup.preflight(env, check_gpu_device=False)
        assert result.manifest.base_revision == PINNED

    def test_a_missing_package_is_refused(self, env, tmp_path):
        env["HERMES_PACKAGE_DIR"] = str(tmp_path / "nowhere")
        with pytest.raises(startup.StartupError, match="No deployment package"):
            startup.preflight(env, check_gpu_device=False)

    def test_without_an_identity_record_nothing_is_served(self, env):
        env.pop("HERMES_EXPECTED_DEPLOYMENT_CONFIG")
        with pytest.raises(startup.StartupError, match="HERMES_EXPECTED_DEPLOYMENT_CONFIG"):
            startup.preflight(env, check_gpu_device=False)

    def test_a_tampered_package_is_refused(self, env, package):
        (package / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        with pytest.raises(startup.StartupError, match=r"adapter_model\.safetensors"):
            startup.preflight(env, check_gpu_device=False)

    @pytest.mark.parametrize(
        ("override", "value", "field"),
        [
            ("adapter_sha256", "e" * 64, "adapter_sha256"),
            ("revision", "a" * 40, "base_revision"),
            ("base_model", "mistralai/Ministral-8B-Instruct-2410", "base_model"),
            ("source_checkpoint", "checkpoint-309", "source_checkpoint"),
            ("tokenizer__fix_mistral_regex", True, "tokenizer_fix_mistral_regex"),
            ("generation__do_sample", True, "generation.do_sample"),
            ("generation__max_new_tokens", 2048, "generation.max_new_tokens"),
            ("runtime__compute_dtype", "bfloat16", "runtime.compute_dtype"),
            ("runtime__quantization_mode", "int8", "runtime.quantization_mode"),
        ],
    )
    def test_an_intact_but_different_package_is_refused(
        self, env, tmp_path, package, override, value, field
    ):
        # The package verifies against its own manifest — it is intact. It is
        # just not the artifact this image was built to serve.
        env["HERMES_EXPECTED_DEPLOYMENT_CONFIG"] = record_with(
            tmp_path, package, **{override: value}
        )
        with pytest.raises(startup.StartupError) as error:
            startup.preflight(env, check_gpu_device=False)
        assert field in str(error.value)

    def test_a_different_tokenizer_file_is_refused(self, env, tmp_path, package):
        record = record_with(
            tmp_path,
            package,
            tokenizer__files_sha256={"tokenizer.json": "f" * 64, "tokenizer_config.json": "f" * 64},
        )
        env["HERMES_EXPECTED_DEPLOYMENT_CONFIG"] = record
        with pytest.raises(startup.StartupError, match=r"tokenizer/tokenizer\.json"):
            startup.preflight(env, check_gpu_device=False)

    def test_a_record_that_pins_nothing_is_refused(self, env, tmp_path):
        weak = tmp_path / "weak.yaml"
        weak.write_text(yaml.safe_dump({"deployment": {"name": "kleos-hermes"}}), encoding="utf-8")
        env["HERMES_EXPECTED_DEPLOYMENT_CONFIG"] = str(weak)
        with pytest.raises(startup.StartupError, match="does not pin"):
            startup.preflight(env, check_gpu_device=False)

    def test_the_tokenizer_flag_must_be_a_real_boolean(self, env, tmp_path, package):
        # YAML `0` is falsy but is not a statement about tokenization.
        env["HERMES_EXPECTED_DEPLOYMENT_CONFIG"] = record_with(
            tmp_path, package, tokenizer__fix_mistral_regex=0
        )
        with pytest.raises(startup.StartupError, match="fix_mistral_regex"):
            startup.preflight(env, check_gpu_device=False)


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


class TestModelCache:
    def test_hf_home_is_required(self, env):
        env.pop("HF_HOME")
        with pytest.raises(startup.StartupError, match="HF_HOME"):
            startup.preflight(env, check_gpu_device=False)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
    def test_an_unwritable_cache_is_refused_with_the_fix(self, env, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        env["HF_HOME"] = str(locked)
        try:
            with pytest.raises(startup.StartupError, match="10001"):
                startup.preflight(env, check_gpu_device=False)
        finally:
            locked.chmod(0o700)

    def test_a_cache_outside_a_volume_is_warned_about(self, env, monkeypatch):
        monkeypatch.setattr(startup, "_on_mounted_volume", lambda path: False)
        result = startup.preflight(env, check_gpu_device=False)
        assert result.cache_persistent is False
        assert any("downloaded again" in warning for warning in result.warnings)

    def test_a_mounted_cache_is_not_warned_about(self, env, monkeypatch):
        monkeypatch.setattr(startup, "_on_mounted_volume", lambda path: True)
        result = startup.preflight(env, check_gpu_device=False)
        assert not any("downloaded again" in warning for warning in result.warnings)

    def test_first_start_is_reported_as_a_download(self, env):
        assert startup.preflight(env, check_gpu_device=False).base_cached is False

    def test_a_cached_base_is_reported_as_reused(self, env):
        snapshot = (
            Path(env["HF_HOME"])
            / "hub"
            / "models--mistralai--Mistral-Nemo-Instruct-2407"
            / "snapshots"
            / PINNED
        )
        snapshot.mkdir(parents=True)
        assert startup.preflight(env, check_gpu_device=False).base_cached is True

    def test_a_snapshot_of_another_revision_does_not_count(self, env):
        other = (
            Path(env["HF_HOME"])
            / "hub"
            / "models--mistralai--Mistral-Nemo-Instruct-2407"
            / "snapshots"
            / ("a" * 40)
        )
        other.mkdir(parents=True)
        assert startup.preflight(env, check_gpu_device=False).base_cached is False


# ---------------------------------------------------------------------------
# GPU check, with a stand-in torch
# ---------------------------------------------------------------------------


def fake_torch(*, available: bool, total_gib: float = 15.0) -> SimpleNamespace:
    properties = SimpleNamespace(
        name="Tesla T4", total_memory=int(total_gib * 1024**3), major=7, minor=5
    )
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda index: properties,
        device_count=lambda: 1,
    )
    return SimpleNamespace(cuda=cuda)


class TestGpuCheck:
    def test_no_visible_gpu_stops_the_start_with_the_fix(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False))
        with pytest.raises(startup.StartupError, match="--gpus all"):
            startup.check_gpu()

    def test_missing_torch_stops_the_start(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(startup.StartupError, match="torch is not installed"):
            startup.check_gpu()

    def test_a_t4_is_described_without_warning(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, total_gib=14.6))
        info = startup.check_gpu()
        assert info["device"] == "Tesla T4"
        assert info["compute_capability"] == "7.5"
        assert "warning" not in info

    def test_a_small_gpu_is_warned_about(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, total_gib=8.0))
        assert "warning" in startup.check_gpu()

    def test_preflight_includes_the_gpu_when_asked(self, env, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True))
        assert startup.preflight(env, check_gpu_device=True).gpu["device"] == "Tesla T4"


# ---------------------------------------------------------------------------
# The entrypoint
# ---------------------------------------------------------------------------


class TestEntrypoint:
    def test_check_only_passes_and_prints_identity(self, env, capsys):
        assert startup.main(["--check-only", "--no-gpu-check"], env=env) == 0
        output = capsys.readouterr().out
        assert "preflight passed" in output
        assert PINNED in output
        assert SECRET not in output

    def test_a_failed_check_exits_1_without_serving(self, env, package, capsys, monkeypatch):
        served = []
        monkeypatch.setattr(startup, "serve", lambda *a, **k: served.append(a) or 0)
        (package / "tokenizer" / "tokenizer.json").write_text("changed", encoding="utf-8")
        assert startup.main([], env=env) == 1
        captured = capsys.readouterr()
        assert "will not start" in captured.err
        assert served == []

    def test_a_passing_check_hands_over_to_the_server(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(
            startup, "serve", lambda result, e, *, host, port: calls.append((host, port)) or 0
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True))
        assert startup.main([], env={**env, "HERMES_PORT": "9001"}) == 0
        assert calls == [("0.0.0.0", 9001)]

    def test_the_banner_never_contains_secret_values(self, env, tmp_path):
        token = tmp_path / "hf_token"
        token.write_text(HF_SECRET, encoding="utf-8")
        env["HF_TOKEN_FILE"] = str(token)
        text = startup.banner(startup.preflight(env, check_gpu_device=False))
        assert SECRET not in text and HF_SECRET not in text
        assert "1 key(s)" in text

    def test_runs_as_a_real_process_the_way_the_container_does(self, env, package):
        process_env = {
            **{k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "HF_"))},
            **env,
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
        command = [
            sys.executable,
            "-m",
            "kleos_models.serving.startup",
            "--check-only",
            "--no-gpu-check",
        ]
        ok = subprocess.run(command, env=process_env, capture_output=True, text=True, timeout=120)
        assert ok.returncode == 0, ok.stderr
        assert "preflight passed" in ok.stdout

        (package / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        refused = subprocess.run(
            command, env=process_env, capture_output=True, text=True, timeout=120
        )
        assert refused.returncode == 1
        assert "will not start" in refused.stderr
        assert SECRET not in ok.stdout + ok.stderr + refused.stdout + refused.stderr


# ---------------------------------------------------------------------------
# Healthcheck probe
# ---------------------------------------------------------------------------


def serve_health(body: dict) -> tuple[http.server.HTTPServer, int]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class TestHealthcheck:
    def test_ready_is_healthy(self):
        server, port = serve_health({"status": "ok", "ready": True})
        try:
            assert startup.healthcheck({"HERMES_PORT": str(port)}) == 0
        finally:
            server.shutdown()

    def test_alive_but_not_ready_is_unhealthy(self):
        server, port = serve_health({"status": "ok", "ready": False, "error": "loading"})
        try:
            assert startup.healthcheck({"HERMES_PORT": str(port)}) == 1
        finally:
            server.shutdown()

    def test_nothing_listening_is_unhealthy(self):
        server, port = serve_health({"ready": True})
        server.shutdown()
        server.server_close()
        assert startup.healthcheck({"HERMES_PORT": str(port)}, timeout=1) == 1

    def test_the_probe_is_available_on_the_command_line(self):
        server, port = serve_health({"ready": True})
        try:
            assert startup.main(["--healthcheck"], env={"HERMES_PORT": str(port)}) == 0
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# The service itself: exit on load failure, identity inside the app
# ---------------------------------------------------------------------------


class TestServiceStartup:
    @pytest.fixture(autouse=True)
    def _needs_fastapi(self):
        pytest.importorskip("fastapi", reason="needs the serve extra")

    def test_with_exit_on_load_failure_a_bad_package_stops_the_process(self, tmp_path):
        from fastapi.testclient import TestClient

        from kleos_models.serving.app import ServingSettings, create_app

        settings = ServingSettings(
            package_dir=tmp_path / "missing", api_keys=("k",), exit_on_load_failure=True
        )
        with pytest.raises(ConfigError), TestClient(create_app(settings)):
            pass

    def test_without_it_the_process_stays_up_and_says_why(self, tmp_path):
        from fastapi.testclient import TestClient

        from kleos_models.serving.app import ServingSettings, create_app

        settings = ServingSettings(package_dir=tmp_path / "missing", api_keys=("k",))
        with TestClient(create_app(settings)) as client:
            body = client.get("/health").json()
        assert body["ready"] is False
        assert "No deployment manifest" in body["error"]

    def test_the_app_enforces_identity_even_without_the_entrypoint(self, tmp_path, package):
        # Someone overriding the container entrypoint to run serve_hermes.py
        # still gets the identity check, because the image sets the variable.
        from fastapi.testclient import TestClient

        from kleos_models.serving.app import ServingSettings, create_app

        settings = ServingSettings.from_env(
            {
                "HERMES_PACKAGE_DIR": str(package),
                "HERMES_API_KEY": "k",
                "HERMES_EXPECTED_DEPLOYMENT_CONFIG": record_with(
                    tmp_path, package, adapter_sha256="e" * 64
                ),
                "HERMES_EXIT_ON_LOAD_FAILURE": "1",
            }
        )
        with (
            pytest.raises(ConfigError, match="is not kleos-hermes"),
            TestClient(create_app(settings)),
        ):
            pass


# ---------------------------------------------------------------------------
# The real serving record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def identity() -> dict:
    return load_expected_identity(REPO_ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml")


class TestShippedServingRecord:
    def test_it_pins_the_frozen_adapter_and_base(self, identity):
        assert identity["adapter_sha256"] == (
            "dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32"
        )
        assert identity["base_revision"] == PINNED

    def test_it_pins_the_tokenizer_by_content_and_by_flag(self, identity):
        assert identity["tokenizer_fix_mistral_regex"] is False
        assert set(identity["tokenizer_files_sha256"]) == {
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }

    def test_it_pins_greedy_decoding(self, identity):
        assert identity["generation"]["do_sample"] is False
        assert identity["generation"]["temperature"] == 0.0
