"""The Hermes inference service boundary.

Run against a stub deployment: no GPU, no weights, no network. What is under
test is the boundary itself — authentication, bounds, error shapes and the
isolation guarantees — not the model.

The isolation tests matter as much as the functional ones. Hermes is a model
service, and KLEOS remains the source of truth for users, workspaces, memories
and conversations. If this service ever starts accepting or keeping that state,
these tests should fail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from tests.test_deployment_manifest import RUNTIME

from kleos_models.serving.app import (
    GenerateMessage,
    GenerateRequest,
    ServingSettings,
    _authorized,
    create_app,
    validate_request,
)
from kleos_models.serving.manifest import (
    AdapterRecord,
    DatasetRecord,
    DeploymentManifest,
    PeftRecord,
    ServingLimits,
    TokenizerContract,
)

fastapi = pytest.importorskip("fastapi", reason="needs the serve extra")
httpx = pytest.importorskip("httpx", reason="needs httpx for the in-process client")

from fastapi.testclient import TestClient  # noqa: E402

# Deliberately short: a placeholder, and too short to be mistaken for a real
# credential by scripts/check_no_private_data.py.
API_KEY = "unit-test"
PINNED = "04d8a90549d23fc6bd7f642064003592df51e9b3"


def build_manifest(**overrides: Any) -> DeploymentManifest:
    payload: dict[str, Any] = {
        "model_name": "kleos-hermes",
        "model_version": "v0.0.6",
        "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
        "base_revision": PINNED,
        "adapter": AdapterRecord(
            experiment_id="kleos-v006-mistralnemo12b-run1",
            source_checkpoint="checkpoint-200",
            weights_sha256="d" * 64,
            trainable_parameters=57_016_320,
        ),
        "peft": PeftRecord(r=16, lora_alpha=32, lora_dropout=0.05),
        "tokenizer": TokenizerContract(fix_mistral_regex=False),
        "runtime": RUNTIME,
        "dataset": DatasetRecord(version="kleos-policy-v0.0.6", sha256="c" * 64),
        "training_config_hash": "b" * 64,
        "limits": ServingLimits(
            max_new_tokens=512, max_input_chars=200, max_messages=4, request_timeout_seconds=5
        ),
    }
    payload.update(overrides)
    return DeploymentManifest(**payload)


@dataclass
class StubOutput:
    text: str = "Prioritise the migration; the deadline is the deciding factor."
    finish_reason: str = "stop"
    prompt_tokens: int = 42
    completion_tokens: int = 12


class StubDeployment:
    """Stands in for a loaded model. Records what it was asked, nothing else."""

    def __init__(self, manifest: DeploymentManifest, *, fail: Exception | None = None) -> None:
        self.manifest = manifest
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def generate(self, messages, *, max_new_tokens=None):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        if self.fail is not None:
            raise self.fail
        return StubOutput()

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.manifest.model_name,
            "model_version": self.manifest.model_version,
            "base_model": self.manifest.base_model,
            "base_revision": self.manifest.base_revision,
            "adapter_sha256": self.manifest.adapter.weights_sha256,
        }


@pytest.fixture
def deployment() -> StubDeployment:
    return StubDeployment(build_manifest())


@pytest.fixture
def settings(tmp_path) -> ServingSettings:
    return ServingSettings(package_dir=tmp_path, api_keys=(API_KEY,))


@pytest.fixture
def client(settings, deployment) -> Any:
    with TestClient(create_app(settings, deployment=deployment)) as test_client:
        yield test_client


def post(client, payload, *, key: str | None = API_KEY):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/v1/generate", json=payload, headers=headers)


ONE_MESSAGE = {"messages": [{"role": "user", "content": "Which task first?"}]}


class TestHealthAndReadiness:
    def test_health_needs_no_credentials(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_reports_readiness_separately(self, client):
        assert client.get("/health").json()["ready"] is True

    def test_readiness_reports_the_loaded_identity(self, client):
        body = client.get("/ready", headers={"Authorization": f"Bearer {API_KEY}"}).json()
        assert body["status"] == "ready"
        assert body["base_revision"] == PINNED
        assert body["adapter_sha256"] == "d" * 64

    def test_readiness_requires_credentials(self, client):
        assert client.get("/ready").status_code == 401

    def test_an_unloaded_model_is_not_ready_and_cannot_generate(self, settings):
        app = create_app(settings, deployment=None)
        app.router.lifespan_context = _no_startup(app.router.lifespan_context)
        with TestClient(app) as unloaded:
            assert unloaded.get("/health").json()["ready"] is False
            assert (
                unloaded.get("/ready", headers={"Authorization": f"Bearer {API_KEY}"}).status_code
                == 503
            )
            assert post(unloaded, ONE_MESSAGE).status_code == 503


def _no_startup(_original):
    """Skip model loading so the 'not loaded' path can be exercised."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        yield

    return lifespan


class TestAuthentication:
    def test_a_valid_key_is_accepted(self, client):
        assert post(client, ONE_MESSAGE).status_code == 200

    def test_a_missing_header_is_rejected(self, client):
        assert post(client, ONE_MESSAGE, key=None).status_code == 401

    def test_a_wrong_key_is_rejected(self, client):
        assert post(client, ONE_MESSAGE, key="wrong").status_code == 401

    def test_a_malformed_scheme_is_rejected(self, client):
        response = client.post("/v1/generate", json=ONE_MESSAGE, headers={"Authorization": API_KEY})
        assert response.status_code == 401

    def test_the_rejection_does_not_reveal_which_part_failed(self, client):
        detail = post(client, ONE_MESSAGE, key="wrong").json()["detail"]
        assert detail == "unauthorized"
        assert API_KEY not in json.dumps(detail)

    def test_multiple_keys_are_supported_for_rotation(self, tmp_path, deployment):
        settings = ServingSettings(package_dir=tmp_path, api_keys=("old", "new"))
        with TestClient(create_app(settings, deployment=deployment)) as rotating:
            assert post(rotating, ONE_MESSAGE, key="old").status_code == 200
            assert post(rotating, ONE_MESSAGE, key="new").status_code == 200
            assert post(rotating, ONE_MESSAGE, key="other").status_code == 401

    def test_authorization_helper_rejects_empty_and_malformed_headers(self):
        keys = ("k",)
        for header in (None, "", "Bearer", "Bearer ", "Basic k", "k"):
            assert not _authorized(header, keys), header
        assert _authorized("Bearer k", keys)

    def test_a_service_without_keys_refuses_to_start(self, tmp_path):
        from kleos_models.errors import ConfigError

        env = {"HERMES_PACKAGE_DIR": str(tmp_path)}
        with pytest.raises(ConfigError, match="HERMES_API_KEY"):
            ServingSettings.from_env(env)

    def test_unauthenticated_mode_must_be_asked_for_explicitly(self, tmp_path):
        env = {"HERMES_PACKAGE_DIR": str(tmp_path), "HERMES_ALLOW_UNAUTHENTICATED": "1"}
        assert ServingSettings.from_env(env).api_keys == ()


class TestRequestValidation:
    def test_a_well_formed_request_returns_the_generated_text(self, client):
        body = post(client, ONE_MESSAGE).json()
        assert body["text"].startswith("Prioritise")
        assert body["prompt_tokens"] == 42
        assert body["completion_tokens"] == 12

    def test_the_response_identifies_exactly_what_answered(self, client):
        model = post(client, ONE_MESSAGE).json()["model"]
        assert model["base_revision"] == PINNED
        assert model["adapter_sha256"] == "d" * 64
        assert model["version"] == "v0.0.6"

    def test_empty_messages_are_rejected(self, client):
        assert post(client, {"messages": []}).status_code == 422

    def test_empty_content_is_rejected(self, client):
        assert post(client, {"messages": [{"role": "user", "content": ""}]}).status_code == 422

    def test_an_unknown_role_is_rejected(self, client):
        payload = {"messages": [{"role": "root", "content": "hi"}]}
        assert post(client, payload).status_code == 422

    def test_unknown_fields_are_rejected(self, client):
        # Guards against a caller quietly passing workspace state or tool
        # definitions that this service must not receive.
        payload = {**ONE_MESSAGE, "workspace_id": "w1", "tools": [{"name": "shell"}]}
        assert post(client, payload).status_code == 422

    def test_a_non_positive_token_budget_is_rejected(self, client):
        assert post(client, {**ONE_MESSAGE, "max_new_tokens": 0}).status_code == 422

    def test_too_many_messages_are_rejected(self, client):
        payload = {"messages": [{"role": "user", "content": "x"}] * 5}
        assert post(client, payload).status_code == 413

    def test_oversized_input_is_rejected(self, client):
        payload = {"messages": [{"role": "user", "content": "x" * 5000}]}
        response = post(client, payload)
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_bounds_are_checked_before_any_generation(self, client, deployment):
        post(client, {"messages": [{"role": "user", "content": "x" * 5000}]})
        assert deployment.calls == []

    def test_a_request_id_is_echoed_and_generated_when_absent(self, client):
        assert post(client, {**ONE_MESSAGE, "request_id": "abc"}).json()["request_id"] == "abc"
        assert post(client, ONE_MESSAGE).json()["request_id"]

    def test_validate_request_helper_enforces_both_bounds(self):
        limits = ServingLimits(max_messages=2, max_input_chars=10)
        one = GenerateMessage(role="user", content="hello")
        validate_request(GenerateRequest(messages=[one]), limits)
        with pytest.raises(ValueError, match="too many messages"):
            validate_request(GenerateRequest(messages=[one, one, one]), limits)
        with pytest.raises(ValueError, match="input too large"):
            validate_request(
                GenerateRequest(messages=[GenerateMessage(role="user", content="x" * 11)]), limits
            )


class TestGenerationLimits:
    def test_a_caller_may_lower_the_token_budget(self, client, deployment):
        post(client, {**ONE_MESSAGE, "max_new_tokens": 16})
        assert deployment.calls[-1]["max_new_tokens"] == 16

    def test_the_api_forwards_the_budget_and_the_loader_clamps_it(self, client, deployment):
        # The clamp lives in LoadedDeployment.generate, where a request cannot
        # bypass it; see tests/test_serving_loader.py. The API's job is only to
        # pass the request through unchanged.
        post(client, {**ONE_MESSAGE, "max_new_tokens": 100_000})
        assert deployment.calls[-1]["max_new_tokens"] == 100_000

    def test_environment_overrides_may_only_tighten_limits(self, tmp_path):
        settings = ServingSettings(
            package_dir=tmp_path, api_keys=("k",), max_input_chars=50, request_timeout_seconds=1
        )
        limits = settings.effective_limits(
            ServingLimits(max_input_chars=200, request_timeout_seconds=5)
        )
        assert limits.max_input_chars == 50
        assert limits.request_timeout_seconds == 1

    def test_environment_overrides_cannot_widen_limits(self, tmp_path):
        settings = ServingSettings(
            package_dir=tmp_path, api_keys=("k",), max_input_chars=9999, request_timeout_seconds=999
        )
        limits = settings.effective_limits(
            ServingLimits(max_input_chars=200, request_timeout_seconds=5)
        )
        assert limits.max_input_chars == 200
        assert limits.request_timeout_seconds == 5


class TestFailureBehaviour:
    def test_a_generation_error_returns_500_without_internal_detail(self, settings):
        broken = StubDeployment(build_manifest(), fail=RuntimeError("CUDA out of memory at 0x7f"))
        with TestClient(create_app(settings, deployment=broken)) as client:
            response = post(client, ONE_MESSAGE)
        assert response.status_code == 500
        assert response.json()["detail"] == "generation failed"
        assert "0x7f" not in response.text

    def test_a_timeout_returns_504(self, settings):
        import time

        class Slow(StubDeployment):
            def generate(self, messages, *, max_new_tokens=None):
                time.sleep(1.5)
                return StubOutput()

        manifest = build_manifest(
            limits=ServingLimits(max_input_chars=200, max_messages=4, request_timeout_seconds=0.2)
        )
        with TestClient(create_app(settings, deployment=Slow(manifest))) as client:
            assert post(client, ONE_MESSAGE).status_code == 504


class TestIsolationAndSecrets:
    def test_no_prompt_or_response_text_is_logged(self, client, caplog):
        secret_prompt = "the-user-said-something-private"
        with caplog.at_level(logging.DEBUG):
            response = post(client, {"messages": [{"role": "user", "content": secret_prompt}]})
        assert response.status_code == 200
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert secret_prompt not in logged
        assert "Prioritise" not in logged

    def test_no_api_key_is_logged(self, client, caplog):
        with caplog.at_level(logging.DEBUG):
            post(client, ONE_MESSAGE)
            post(client, ONE_MESSAGE, key="wrong")
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert API_KEY not in logged

    def test_the_service_keeps_no_state_between_requests(self, client, deployment):
        first = post(client, {"messages": [{"role": "user", "content": "remember alpha"}]})
        second = post(client, {"messages": [{"role": "user", "content": "what did I say?"}]})
        assert first.status_code == second.status_code == 200
        # Each call receives only its own messages: nothing is carried forward.
        assert len(deployment.calls[-1]["messages"]) == 1
        assert "alpha" not in deployment.calls[-1]["messages"][0].content

    def test_there_is_no_endpoint_beyond_the_narrow_contract(self, client):
        paths = {route.path for route in client.app.routes if hasattr(route, "path")}
        assert paths <= {"/health", "/ready", "/v1/generate"} | {
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
        }
        # No schema or docs surface is published.
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404

    def test_the_service_exposes_no_way_to_run_tools_or_fetch_urls(self):
        fields = set(GenerateRequest.model_fields)
        assert fields == {"messages", "max_new_tokens", "request_id"}
