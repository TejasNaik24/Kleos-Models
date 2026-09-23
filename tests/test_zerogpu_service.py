"""The ZeroGPU request path, without ZeroGPU, a GPU, Gradio or weights.

`ZeroGPUService` is where a request becomes either an answer or a status KLEOS
can fall back on. It runs here over the real `LoadedDeployment` (so the token
ceiling and decoding contract are the production ones) with a fake backend, and
a fake GPU call standing in for `@spaces.GPU`. What is under test:

* nothing reaches the GPU without the shared secret, a valid body and in-bounds
  input — GPU time is the scarce, per-account resource;
* every way the GPU call can fail becomes a contract status, never an exception;
* prompts, responses and keys stay out of the logs;
* startup fails closed before downloading anything it would refuse to serve.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.test_deployment_manifest import RUNTIME

from kleos_models.config import GenerationConfig
from kleos_models.errors import ConfigError
from kleos_models.inference.backends import GenerationOutput, PreparedPrompt
from kleos_models.serving import zerogpu
from kleos_models.serving.loader import LoadedDeployment
from kleos_models.serving.manifest import (
    AdapterRecord,
    DatasetRecord,
    DeploymentManifest,
    PeftRecord,
    ServingLimits,
    TokenizerContract,
)
from kleos_models.serving.zerogpu import (
    KEY_HEADER,
    ZeroGPUService,
    ZeroGPUSettings,
    gpu_duration,
    load_space_deployment,
    run_gpu_step,
)

KEY = "unit-zerogpu"
PINNED = "04d8a90549d23fc6bd7f642064003592df51e9b3"
#: Planted in prompts and outputs; must never appear in a log line.
MARKER = "PRIVATE-MARKER-7f3a"
RECORD = Path(__file__).resolve().parents[1] / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def build_manifest(**limits: Any) -> DeploymentManifest:
    return DeploymentManifest(
        model_name="kleos-hermes",
        model_version="v0.0.6",
        base_model="mistralai/Mistral-Nemo-Instruct-2407",
        base_revision=PINNED,
        adapter=AdapterRecord(
            experiment_id="kleos-v006-mistralnemo12b-run1",
            source_checkpoint="checkpoint-200",
            weights_sha256="d" * 64,
            trainable_parameters=57_016_320,
        ),
        peft=PeftRecord(r=16, lora_alpha=32, lora_dropout=0.05),
        tokenizer=TokenizerContract(fix_mistral_regex=False),
        runtime=RUNTIME,
        dataset=DatasetRecord(version="kleos-policy-v0.0.6", sha256="c" * 64),
        training_config_hash="b" * 64,
        limits=ServingLimits(
            **{"max_new_tokens": 512, "max_input_chars": 400, "max_messages": 4, **limits}
        ),
    )


class FakeBackend:
    """prepare / generate_ids / finish, recording what reached each step."""

    def __init__(
        self,
        *,
        prompt_tokens: int = 40,
        text: str = "Do the migration first; the deadline decides it.",
        fail_prepare: Exception | None = None,
        fail_finish: Exception | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.text = text
        self.fail_prepare = fail_prepare
        self.fail_finish = fail_finish
        self.prepared: list[Any] = []
        self.finished: list[list[int]] = []

    def prepare(self, messages):
        if self.fail_prepare is not None:
            raise self.fail_prepare
        self.prepared.append(messages)
        return PreparedPrompt(
            inputs={"input_ids": [[1] * self.prompt_tokens]}, prompt_length=self.prompt_tokens
        )

    def generate_ids(self, prepared, config):  # pragma: no cover - the GPU call is faked
        raise AssertionError("the service must go through gpu_call, not the backend")

    def finish(self, prepared, completion_ids):
        if self.fail_finish is not None:
            raise self.fail_finish
        self.finished.append(list(completion_ids))
        return GenerationOutput(
            text=self.text,
            prompt_tokens=prepared.prompt_length,
            completion_tokens=len(completion_ids),
        )


class FakeGPU:
    """Stands in for the @spaces.GPU-decorated call."""

    def __init__(self, *, fail: Exception | None = None, completion: int = 9) -> None:
        self.fail = fail
        self.completion = completion
        self.configs: list[GenerationConfig] = []
        self.calls = 0

    def __call__(self, prepared, config):
        self.calls += 1
        self.configs.append(config)
        if self.fail is not None:
            raise self.fail
        return {
            "completion_ids": list(range(min(self.completion, config.max_new_tokens))),
            "gpu_generate_s": 0.8,
            "worker_call_index": self.calls,
            "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "compute_capability": "12.0",
            "peak_vram_bytes": 9 * 1024**3,
            "total_vram_bytes": 48 * 1024**3,
            "cuda_runtime": "12.8",
        }


class FakeGradioError(Exception):
    def __init__(self, message: str, title: str) -> None:
        super().__init__(message)
        self.message = message
        self.title = title


def build_service(
    *,
    backend: FakeBackend | None = None,
    gpu: FakeGPU | None = None,
    settings: ZeroGPUSettings | None = None,
    **limits: Any,
) -> tuple[ZeroGPUService, FakeBackend, FakeGPU]:
    backend = backend or FakeBackend()
    gpu = gpu or FakeGPU()
    deployment = LoadedDeployment(
        manifest=build_manifest(**limits),
        model_config=None,  # type: ignore[arg-type]  # not consulted
        backend=backend,
        package_dir=Path("."),
    )
    service = ZeroGPUService(deployment, settings or ZeroGPUSettings(api_keys=(KEY,)), gpu_call=gpu)
    return service, backend, gpu


AUTH = {KEY_HEADER: KEY}
ONE = {"messages": [{"role": "user", "content": "Which task first?"}]}


class TestSettings:
    def test_no_key_fails_closed(self):
        with pytest.raises(ConfigError, match="HERMES_API_KEY"):
            ZeroGPUSettings.from_env({})

    def test_unauthenticated_mode_must_be_explicit(self):
        settings = ZeroGPUSettings.from_env({"HERMES_ALLOW_UNAUTHENTICATED": "1"})
        assert settings.api_keys == () and settings.allow_unauthenticated

    def test_keys_and_bounds_are_read(self):
        settings = ZeroGPUSettings.from_env(
            {
                "HERMES_API_KEY": " one , two ,",
                "HERMES_MAX_INPUT_TOKENS": "1024",
                "HERMES_MAX_NEW_TOKENS": "256",
            }
        )
        assert settings.api_keys == ("one", "two")
        assert settings.max_input_tokens == 1024
        assert settings.max_new_tokens == 256


class TestAuthentication:
    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {KEY_HEADER: "wrong"},
            {KEY_HEADER: ""},
            # Hugging Face's own header is not a substitute for the Hermes key.
            {"authorization": f"Bearer {KEY}"},
        ],
    )
    def test_a_request_without_the_key_never_reaches_the_model(self, headers):
        service, backend, gpu = build_service()
        response = service.generate(ONE, headers)
        assert response["status"] == "unauthorized"
        assert response["ok"] is False
        assert backend.prepared == [] and gpu.calls == 0

    def test_the_header_name_is_case_insensitive(self):
        service, _, _ = build_service()
        assert service.generate(ONE, {"X-Hermes-Key": KEY})["status"] == "ready"

    def test_any_configured_key_is_accepted(self):
        service, _, _ = build_service(settings=ZeroGPUSettings(api_keys=("old", KEY)))
        assert service.generate(ONE, AUTH)["ok"] is True

    def test_unauthenticated_mode_serves_without_a_key(self):
        settings = ZeroGPUSettings(api_keys=(), allow_unauthenticated=True)
        service, _, _ = build_service(settings=settings)
        assert service.generate(ONE, {})["ok"] is True

    def test_status_requires_the_key(self):
        service, _, _ = build_service()
        assert service.status({})["status"] == "unauthorized"

    def test_status_reports_identity_runtime_and_limits(self):
        service, _, _ = build_service()
        status = service.status(AUTH)
        assert status["status"] == "ready"
        assert status["model"]["base_revision"] == PINNED
        assert status["model"]["adapter_sha256"] == "d" * 64
        assert status["runtime"]["compute_dtype"] == "float16"
        assert status["limits"]["max_new_tokens"] == 512
        json.dumps(status)


class TestValidation:
    @pytest.mark.parametrize(
        ("payload", "field"),
        [
            ({**ONE, "workspace_id": "w1"}, "workspace_id"),
            ({**ONE, "tools": []}, "tools"),
            ({"messages": [{"role": "user", "content": "x", "user_id": "u"}]}, "user_id"),
            ({"messages": [{"role": "tool", "content": "x"}]}, "role"),
            ({"messages": []}, "messages"),
            ({**ONE, "max_new_tokens": 0}, "max_new_tokens"),
        ],
    )
    def test_malformed_requests_are_refused_before_any_gpu_work(self, payload, field):
        service, backend, gpu = build_service()
        response = service.generate(payload, AUTH)
        assert response["status"] == "invalid_request"
        assert field in response["message"]
        assert response["retryable"] is False
        assert backend.prepared == [] and gpu.calls == 0

    @pytest.mark.parametrize("payload", ["just a string", None, 42, ["messages"]])
    def test_a_body_that_is_not_an_object_is_refused(self, payload):
        service, _, gpu = build_service()
        assert service.generate(payload, AUTH)["status"] == "invalid_request"
        assert gpu.calls == 0

    def test_too_many_messages_are_refused(self):
        service, _, gpu = build_service(max_messages=2)
        payload = {"messages": [{"role": "user", "content": "x"}] * 3}
        response = service.generate(payload, AUTH)
        assert response["status"] == "invalid_request"
        assert "too many messages" in response["message"]
        assert gpu.calls == 0

    def test_too_many_characters_are_refused(self):
        service, _, gpu = build_service(max_input_chars=50)
        payload = {"messages": [{"role": "user", "content": "x" * 51}]}
        assert service.generate(payload, AUTH)["status"] == "invalid_request"
        assert gpu.calls == 0

    def test_too_many_tokens_are_refused_after_tokenizing_but_before_the_gpu(self):
        settings = ZeroGPUSettings(api_keys=(KEY,), max_input_tokens=100)
        service, backend, gpu = build_service(
            settings=settings, backend=FakeBackend(prompt_tokens=101)
        )
        response = service.generate(ONE, AUTH)
        assert response["status"] == "invalid_request"
        assert "101 tokens > 100" in response["message"]
        assert len(backend.prepared) == 1 and gpu.calls == 0

    def test_a_refusal_never_echoes_the_request_content(self):
        service, _, _ = build_service()
        payload = {"messages": [{"role": "user", "content": MARKER}], "extra": MARKER}
        assert MARKER not in json.dumps(service.generate(payload, AUTH))


class TestTokenBudget:
    @pytest.mark.parametrize(("requested", "expected"), [(32, 32), (512, 512), (100_000, 512)])
    def test_a_caller_may_lower_but_never_raise_the_budget(self, requested, expected):
        service, _, gpu = build_service()
        service.generate({**ONE, "max_new_tokens": requested}, AUTH)
        assert gpu.configs[-1].max_new_tokens == expected

    def test_omitting_the_budget_uses_the_frozen_default(self):
        service, _, gpu = build_service()
        service.generate(ONE, AUTH)
        assert gpu.configs[-1].max_new_tokens == 512

    def test_the_space_may_tighten_the_ceiling(self):
        settings = ZeroGPUSettings(api_keys=(KEY,), max_new_tokens=256)
        service, _, gpu = build_service(settings=settings)
        service.generate({**ONE, "max_new_tokens": 400}, AUTH)
        assert gpu.configs[-1].max_new_tokens == 256
        assert service.status(AUTH)["limits"]["max_new_tokens"] == 256

    def test_decoding_is_the_frozen_greedy_contract(self):
        service, _, gpu = build_service()
        service.generate(ONE, AUTH)
        config = gpu.configs[-1]
        assert config.do_sample is False
        assert config.temperature == 0.0


class TestFailuresBecomeStatuses:
    @pytest.mark.parametrize(
        ("error", "status", "retryable", "retry_after"),
        [
            (
                FakeGradioError(
                    "You have exceeded your free ZeroGPU quota (80s requested vs. 20s left). "
                    "Try again in 1:00:00. ",
                    "ZeroGPU quota exceeded",
                ),
                "quota_exhausted",
                True,
                3600,
            ),
            (
                FakeGradioError("No GPU was available after 60s", "ZeroGPU queue timeout"),
                "queue_unavailable",
                True,
                None,
            ),
            (
                FakeGradioError("GPU task aborted", "ZeroGPU worker error"),
                "model_error",
                False,
                None,
            ),
            (RuntimeError("CUDA error: an illegal memory access"), "model_error", False, None),
        ],
    )
    def test_a_failed_gpu_call_is_a_response_not_an_exception(
        self, error, status, retryable, retry_after
    ):
        service, _, _ = build_service(gpu=FakeGPU(fail=error))
        response = service.generate({**ONE, "request_id": "req-1"}, AUTH)
        assert response["ok"] is False
        assert response["status"] == status
        assert response["retryable"] is retryable
        assert response["retry_after_seconds"] == retry_after
        assert response["request_id"] == "req-1"
        json.dumps(response)

    def test_a_tokenizer_failure_is_a_model_error(self):
        backend = FakeBackend(fail_prepare=ValueError("bad template"))
        service, _, gpu = build_service(backend=backend)
        assert service.generate(ONE, AUTH)["status"] == "model_error"
        assert gpu.calls == 0

    def test_a_decode_failure_is_a_model_error(self):
        service, _, _ = build_service(backend=FakeBackend(fail_finish=ValueError("bad ids")))
        assert service.generate(ONE, AUTH)["status"] == "model_error"


class TestSuccess:
    def test_the_answer_carries_text_usage_identity_and_timings(self):
        service, backend, _ = build_service()
        response = service.generate({**ONE, "request_id": "req-9"}, AUTH)
        assert response["ok"] is True and response["status"] == "ready"
        assert response["text"] == backend.text
        assert response["usage"] == {"prompt_tokens": 40, "completion_tokens": 9}
        assert response["finish_reason"] == "stop"
        assert response["request_id"] == "req-9"
        assert response["model"] == {
            "name": "kleos-hermes",
            "version": "v0.0.6",
            "adapter_sha256": "d" * 64,
            "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
            "base_revision": PINNED,
        }
        assert set(response["timings"]) == {
            "prepare_s",
            "gpu_call_s",
            "gpu_generate_s",
            "gpu_acquire_s",
            "finish_s",
            "total_s",
        }
        assert response["timings"]["gpu_acquire_s"] >= 0
        assert backend.finished == [list(range(9))]
        json.dumps(response)

    def test_the_first_call_in_a_worker_is_reported_cold_and_later_ones_warm(self):
        service, _, _ = build_service()
        first = service.generate(ONE, AUTH)["diagnostics"]
        second = service.generate(ONE, AUTH)["diagnostics"]
        assert first["cold_start"] is True and first["worker_call_index"] == 1
        assert second["cold_start"] is False and second["worker_call_index"] == 2
        assert first["compute_capability"] == "12.0"
        assert first["peak_vram_gib"] == 9.0

    def test_a_completion_that_fills_the_budget_is_marked_truncated(self):
        service, _, _ = build_service(gpu=FakeGPU(completion=64))
        response = service.generate({**ONE, "max_new_tokens": 16}, AUTH)
        assert response["finish_reason"] == "length"

    @pytest.mark.parametrize("request_id", [None, 7, "", "x" * 129])
    def test_a_missing_or_unusable_request_id_is_replaced(self, request_id):
        service, _, _ = build_service()
        payload = {**ONE} if request_id is None else {**ONE, "request_id": request_id}
        response = service.generate(payload, AUTH)
        assert response["request_id"] != request_id
        assert len(response["request_id"]) == 32


class TestLogsCarryNoContent:
    def _log_text(self, caplog) -> str:
        return "\n".join(record.getMessage() for record in caplog.records)

    def test_a_successful_request_logs_counts_not_content(self, caplog):
        caplog.set_level(logging.DEBUG, logger="kleos_models")
        service, _, _ = build_service(backend=FakeBackend(text=f"answer {MARKER}"))
        payload = {"messages": [{"role": "user", "content": f"secret plan {MARKER}"}]}
        assert service.generate(payload, AUTH)["ok"] is True
        logged = self._log_text(caplog)
        assert "status=ready" in logged
        assert MARKER not in logged
        assert KEY not in logged

    def test_a_failure_logs_the_error_type_not_its_text(self, caplog):
        caplog.set_level(logging.DEBUG, logger="kleos_models")
        service, _, _ = build_service(gpu=FakeGPU(fail=RuntimeError(f"echo {MARKER}")))
        service.generate(ONE, AUTH)
        logged = self._log_text(caplog)
        assert "error=RuntimeError" in logged
        assert MARKER not in logged

    def test_a_quota_failure_logs_the_infrastructure_title(self, caplog):
        caplog.set_level(logging.DEBUG, logger="kleos_models")
        error = FakeGradioError("Try again in 0:10:00.", "ZeroGPU quota exceeded")
        service, _, _ = build_service(gpu=FakeGPU(fail=error))
        service.generate(ONE, AUTH)
        assert "title=ZeroGPU quota exceeded" in self._log_text(caplog)

    def test_a_refused_request_logs_no_key_material(self, caplog):
        caplog.set_level(logging.DEBUG, logger="kleos_models")
        service, _, _ = build_service()
        service.generate(ONE, {KEY_HEADER: f"guess-{MARKER}"})
        assert MARKER not in self._log_text(caplog)


# ---------------------------------------------------------------------------
# The GPU step, with a fake torch
# ---------------------------------------------------------------------------


def fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    properties = types.SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        major=12,
        minor=0,
        total_memory=48 * 1024**3,
    )
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        get_device_properties=lambda index: properties,
        max_memory_allocated=lambda: 8 * 1024**3,
    )
    torch.version = types.SimpleNamespace(cuda="12.8")  # type: ignore[attr-defined]
    return torch


class IdsDeployment:
    def generate_ids(self, prepared, config):
        return [5, 6, 7]


class TestGPUStep:
    @pytest.fixture(autouse=True)
    def _torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch())
        monkeypatch.setattr(zerogpu, "_WORKER", {"pid": None, "calls": 0})

    def test_the_result_is_plain_and_picklable(self):
        prepared = PreparedPrompt(inputs={}, prompt_length=3)
        result = run_gpu_step(IdsDeployment(), prepared, GenerationConfig(max_new_tokens=8))
        assert result["completion_ids"] == [5, 6, 7]
        assert result["compute_capability"] == "12.0"
        assert result["peak_vram_bytes"] == 8 * 1024**3
        assert result["cuda_runtime"] == "12.8"
        # ZeroGPU pickles the worker's return value back to the Space process.
        # Round-tripping our own freshly built dict is the point; nothing
        # untrusted is ever unpickled.
        assert pickle.loads(pickle.dumps(result)) == result

    def test_calls_are_counted_per_worker_process(self, monkeypatch):
        prepared = PreparedPrompt(inputs={}, prompt_length=3)
        config = GenerationConfig(max_new_tokens=8)
        assert run_gpu_step(IdsDeployment(), prepared, config)["worker_call_index"] == 1
        assert run_gpu_step(IdsDeployment(), prepared, config)["worker_call_index"] == 2
        # A newly forked worker has a new pid and starts cold.
        monkeypatch.setattr(zerogpu.os, "getpid", lambda: -1)
        assert run_gpu_step(IdsDeployment(), prepared, config)["worker_call_index"] == 1


class TestGPUDuration:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in (
            "HERMES_GPU_BASE_SECONDS",
            "HERMES_GPU_TOKENS_PER_SECOND",
            "HERMES_GPU_MIN_SECONDS",
            "HERMES_GPU_MAX_SECONDS",
        ):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize(("budget", "seconds"), [(512, 53), (256, 32), (32, 15)])
    def test_the_request_scales_with_the_token_budget(self, budget, seconds):
        prepared = PreparedPrompt(inputs={}, prompt_length=40)
        assert gpu_duration(prepared, GenerationConfig(max_new_tokens=budget)) == seconds

    def test_measured_throughput_tightens_it(self, monkeypatch):
        monkeypatch.setenv("HERMES_GPU_TOKENS_PER_SECOND", "40")
        prepared = PreparedPrompt(inputs={}, prompt_length=40)
        assert gpu_duration(prepared, GenerationConfig(max_new_tokens=512)) == 23

    def test_it_never_exceeds_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("HERMES_GPU_TOKENS_PER_SECOND", "1")
        prepared = PreparedPrompt(inputs={}, prompt_length=40)
        assert gpu_duration(prepared, GenerationConfig(max_new_tokens=512)) == 60


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


class FakeHub:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, Any]] = []

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        return str(self.path)


STARTUP_ENV = {
    "HERMES_PACKAGE_REPO": "owner/kleos-hermes-v006-package",
    "HERMES_PACKAGE_REVISION": "e" * 40,
    "HF_TOKEN": "hf_placeholder",
}


class TestStartup:
    @pytest.fixture
    def hub(self, monkeypatch, tmp_path) -> FakeHub:
        fake = FakeHub(tmp_path)
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = fake.snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return fake

    @pytest.fixture
    def loads(self, monkeypatch) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        def fake_load(path, **kwargs):
            seen.append({"path": path, **kwargs})
            raise ConfigError("stop here")

        monkeypatch.setattr("kleos_models.serving.loader.load_deployment", fake_load)
        return seen

    @pytest.mark.parametrize(
        ("env", "message"),
        [
            ({}, "HERMES_PACKAGE_REPO"),
            ({"HERMES_PACKAGE_REPO": "owner/pkg"}, "40-character"),
            (
                {"HERMES_PACKAGE_REPO": "owner/pkg", "HERMES_PACKAGE_REVISION": "main"},
                "40-character",
            ),
        ],
    )
    def test_an_unpinned_package_is_refused_before_download(self, hub, env, message):
        with pytest.raises(ConfigError, match=message):
            load_space_deployment(RECORD, env=env)
        assert hub.calls == []

    def test_a_bad_serving_record_is_refused_before_download(self, hub, tmp_path):
        record = tmp_path / "record.yaml"
        record.write_text(yaml.safe_dump({"deployment": {"model_name": "x"}}), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_space_deployment(record, env=STARTUP_ENV)
        assert hub.calls == []

    def test_a_foreign_but_self_consistent_package_is_refused_before_loading(
        self, monkeypatch, tmp_path
    ):
        # The Docker regression, on the Space's startup path, through the real
        # loader: a package that verifies against its own manifest but is not
        # the frozen Hermes artifact must never reach model loading.
        from tests.test_deployment_manifest import build_package

        package = build_package(tmp_path)
        fake = FakeHub(package)
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = fake.snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        monkeypatch.setattr(
            "kleos_models.serving.loader.load_packaged_model_config",
            lambda *_: pytest.fail("identity must be refused before the model config is read"),
        )
        with pytest.raises(ConfigError, match=r"not kleos-hermes v0\.0\.6") as error:
            load_space_deployment(RECORD, env=STARTUP_ENV)
        assert any("adapter_sha256" in p for p in error.value.details["problems"])

    def test_the_pinned_package_is_fetched_and_verified_against_the_record(self, hub, loads):
        with pytest.raises(ConfigError, match="stop here"):
            load_space_deployment(RECORD, env=STARTUP_ENV)
        assert hub.calls == [
            {
                "repo_id": "owner/kleos-hermes-v006-package",
                "revision": "e" * 40,
                "repo_type": "model",
                "token": "hf_placeholder",
                "local_dir": None,
            }
        ]
        (load,) = loads
        assert load["verify"] is True
        assert load["device_map"] == "cuda:0"
        expected = load["expected_identity"]
        assert expected["base_revision"] == PINNED
        assert expected["runtime"]["compute_dtype"] == "float16"
