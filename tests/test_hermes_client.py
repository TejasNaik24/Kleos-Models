"""The reference KLEOS-side client: every outcome becomes the status contract.

"Quota must never become a KLEOS error." This is where that is enforced for the
caller: a sleeping Space, a spent quota, a full queue, a dead network and a
malformed reply all come back as a status the KLEOS backend can branch on, and
none of them raises. The transports are fakes shaped like gradio_client and
httpx; nothing here touches a network.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest

from kleos_models.serving.client import (
    HermesClient,
    HermesClientSettings,
    classify_call_error,
    classify_connect_error,
    validate_contract,
)
from kleos_models.serving.status import HermesStatus, error_response, ok_response

KEY = "unit-client"
HF = "hf_unit_placeholder"
MESSAGES = [{"role": "user", "content": "Which task first?"}]


def answer(**overrides: Any) -> dict[str, Any]:
    response = ok_response(
        text="Do the migration first.",
        finish_reason="stop",
        prompt_tokens=40,
        completion_tokens=6,
        model={"name": "kleos-hermes", "version": "v0.0.6"},
        request_id="r1",
    )
    response.update(overrides)
    return response


class FakeJob:
    def __init__(self, result: Any = None, error: BaseException | None = None) -> None:
        self._result = result
        self._error = error
        self.cancelled = False
        self.timeout: float | None = None

    def result(self, timeout=None):
        self.timeout = timeout
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancelled = True
        return True


class FakeSpace:
    """Shaped like gradio_client.Client: submit(*args, api_name=...) -> Job."""

    def __init__(self, job: FakeJob) -> None:
        self.job = job
        self.submitted: list[tuple[tuple[Any, ...], str]] = []

    def submit(self, *args, api_name=None):
        self.submitted.append((args, api_name))
        return self.job


class Factory:
    """Counts connections; can fail or stall like a waking Space."""

    def __init__(self, space: FakeSpace | None = None, error: BaseException | None = None) -> None:
        self.space = space
        self.error = error
        self.calls = 0
        self.gate: threading.Event | None = None

    def __call__(self, settings):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.space


def space_settings(**overrides: Any) -> HermesClientSettings:
    values: dict[str, Any] = {
        "enabled": True,
        "provider": "zerogpu",
        "space": "owner/kleos-hermes",
        "api_key": KEY,
        "hf_token": HF,
        "timeout": 30.0,
    }
    values.update(overrides)
    return HermesClientSettings(**values)


def space_client(job: FakeJob, **overrides: Any) -> tuple[HermesClient, Factory, FakeSpace]:
    space = FakeSpace(job)
    factory = Factory(space)
    return HermesClient(space_settings(**overrides), space_client_factory=factory), factory, space


class TransportError(Exception):
    """Named like httpx's base class, which is what the client matches."""


class ConnectError(TransportError):
    pass


class TestSettings:
    def test_hermes_is_off_unless_enabled(self):
        settings = HermesClientSettings.from_env({})
        assert settings.enabled is False
        assert settings.provider == "zerogpu"
        assert settings.timeout == 120.0

    def test_settings_are_read_from_the_environment(self):
        settings = HermesClientSettings.from_env(
            {
                "HERMES_ENABLED": "true",
                "HERMES_PROVIDER": "http",
                "HERMES_BASE_URL": "https://hermes.internal/",
                "HERMES_TIMEOUT": "45",
                "HERMES_MAX_OUTPUT_TOKENS": "256",
                "HERMES_API_KEY": KEY,
            }
        )
        assert settings.enabled and settings.provider == "http"
        assert settings.base_url == "https://hermes.internal"
        assert settings.timeout == 45.0
        assert settings.max_output_tokens == 256

    def test_a_malformed_number_falls_back_to_the_default(self):
        assert HermesClientSettings.from_env({"HERMES_TIMEOUT": "soon"}).timeout == 120.0

    def test_problems_name_settings_without_revealing_values(self):
        problems = HermesClientSettings(enabled=True, provider="zerogpu", hf_token=HF).problems()
        text = " ".join(problems)
        assert "HERMES_SPACE" in text and "HERMES_API_KEY" in text
        assert HF not in text


class TestDisabled:
    def test_a_disabled_client_answers_disabled_without_any_network(self):
        factory = Factory(FakeSpace(FakeJob(answer())))
        client = HermesClient(HermesClientSettings(enabled=False), space_client_factory=factory)
        response = client.generate(MESSAGES)
        assert response["status"] == "disabled" and response["ok"] is False
        assert client.status()["status"] == "disabled"
        assert factory.calls == 0

    def test_a_misconfigured_client_is_disabled_rather_than_broken(self):
        factory = Factory(FakeSpace(FakeJob(answer())))
        client = HermesClient(space_settings(space=None), space_client_factory=factory)
        assert client.generate(MESSAGES)["status"] == "disabled"
        assert factory.calls == 0


class TestZeroGPUProvider:
    def test_an_answer_passes_through(self):
        client, _, space = space_client(FakeJob(answer()))
        response = client.generate(MESSAGES, request_id="r1")
        assert response["ok"] is True and response["text"] == "Do the migration first."
        ((payload,), api_name) = space.submitted[0]
        assert api_name == "/generate"
        assert payload == {"messages": MESSAGES, "request_id": "r1"}

    @pytest.mark.parametrize(
        ("ceiling", "requested", "sent"),
        [(256, 512, 256), (256, None, 256), (256, 64, 64), (None, 64, 64), (None, None, None)],
    )
    def test_the_kleos_token_ceiling_applies(self, ceiling, requested, sent):
        client, _, space = space_client(FakeJob(answer()), max_output_tokens=ceiling)
        client.generate(MESSAGES, max_new_tokens=requested)
        ((payload,), _) = space.submitted[0]
        assert payload.get("max_new_tokens") == sent

    def test_the_call_is_bounded_by_the_timeout(self):
        job = FakeJob(answer())
        client, _, _ = space_client(job, timeout=12.5)
        client.generate(MESSAGES)
        assert job.timeout == 12.5

    def test_a_status_from_the_space_passes_through_unchanged(self):
        quota = error_response(
            HermesStatus.QUOTA_EXHAUSTED, request_id="r1", retry_after_seconds=3600
        )
        client, _, _ = space_client(FakeJob(quota))
        assert client.generate(MESSAGES) == quota

    def test_a_timed_out_call_is_cancelled_and_reported_as_busy(self):
        job = FakeJob(error=TimeoutError())
        client, factory, _ = space_client(job)
        response = client.generate(MESSAGES)
        assert response["status"] == "queue_unavailable" and response["retryable"] is True
        assert job.cancelled is True
        client.generate(MESSAGES)
        assert factory.calls == 2  # reconnects after a queue failure

    @pytest.mark.parametrize(
        ("error", "status", "retry_after"),
        [
            (
                type("AppError", (Exception,), {})(
                    "You have exceeded your free ZeroGPU quota "
                    "(80s requested vs. 5s left). Try again in 2:00:00. "
                ),
                "quota_exhausted",
                7200,
            ),
            (Exception("Queue is full! Please try again."), "queue_unavailable", None),
            (ConnectError("connection reset"), "queue_unavailable", None),
            (ValueError("unexpected"), "model_error", None),
        ],
    )
    def test_call_failures_become_statuses(self, error, status, retry_after):
        client, _, _ = space_client(FakeJob(error=error))
        response = client.generate(MESSAGES)
        assert response["status"] == status
        assert response["retry_after_seconds"] == retry_after

    @pytest.mark.parametrize(
        "result",
        [
            "plain text",
            None,
            {"text": "no contract"},
            answer(contract_version=99),
            answer(status="thinking"),
            answer(text=None),
            answer(status="quota_exhausted"),  # ok=True with a non-ready status
        ],
    )
    def test_a_reply_outside_the_contract_is_not_passed_on(self, result):
        client, _, _ = space_client(FakeJob(result))
        assert client.generate(MESSAGES)["status"] == "model_error"

    def test_status_uses_no_gpu_and_needs_no_text(self):
        status = {"contract_version": 1, "ok": True, "status": "ready", "model": {}}
        client, _, space = space_client(FakeJob(status))
        assert client.status() == status
        assert space.submitted == [((), "/status")]


class TestConnectingToTheSpace:
    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (
                ValueError(
                    "The current space is in the invalid state: PAUSED. Please contact "
                    "the owner to fix this."
                ),
                "disabled",
            ),
            (
                ValueError(
                    "Could not find Space: owner/x. If it is a private Space, please "
                    "provide a Hugging Face token."
                ),
                "disabled",
            ),
            (OSError("Connection refused"), "starting"),
            (ConnectError("waking"), "starting"),
        ],
    )
    def test_connection_failures_become_statuses(self, error, status):
        factory = Factory(error=error)
        client = HermesClient(space_settings(), space_client_factory=factory)
        assert client.generate(MESSAGES)["status"] == status

    def test_a_failed_connection_is_retried_on_the_next_call(self):
        factory = Factory(FakeSpace(FakeJob(answer())), error=OSError("asleep"))
        client = HermesClient(space_settings(), space_client_factory=factory)
        assert client.generate(MESSAGES)["status"] == "starting"
        factory.error = None
        assert client.generate(MESSAGES)["status"] == "ready"
        assert factory.calls == 2

    def test_a_slow_wake_is_reported_as_starting_and_finishes_in_the_background(self):
        factory = Factory(FakeSpace(FakeJob(answer())))
        factory.gate = threading.Event()
        client = HermesClient(space_settings(connect_wait=0.05), space_client_factory=factory)

        started = time.perf_counter()
        assert client.generate(MESSAGES)["status"] == "starting"
        assert time.perf_counter() - started < 2  # the caller was not held hostage

        factory.gate.set()
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            response = client.generate(MESSAGES)
            if response["status"] == "ready":
                break
            time.sleep(0.02)
        assert response["status"] == "ready"
        assert factory.calls == 1  # one connection, not one per call


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHTTP:
    def __init__(self, response: FakeResponse | None = None, error: BaseException | None = None):
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def _send(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    def post(self, url, **kwargs):
        return self._send("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._send("GET", url, **kwargs)


def http_client(http: FakeHTTP) -> HermesClient:
    settings = HermesClientSettings(
        enabled=True, provider="http", base_url="https://hermes.internal", api_key=KEY
    )
    return HermesClient(settings, http_client=http)


SERVICE_BODY = {
    "text": "Do the migration first.",
    "finish_reason": "stop",
    "prompt_tokens": 40,
    "completion_tokens": 6,
    "request_id": "r7",
    "model": {"name": "kleos-hermes", "version": "v0.0.6"},
}


class TestHTTPProvider:
    def test_a_service_answer_is_translated_into_the_contract(self):
        http = FakeHTTP(FakeResponse(200, SERVICE_BODY))
        response = http_client(http).generate(MESSAGES, request_id="r7")
        assert response["ok"] is True and response["status"] == "ready"
        assert response["usage"] == {"prompt_tokens": 40, "completion_tokens": 6}
        (request,) = http.requests
        assert request["url"] == "https://hermes.internal/v1/generate"
        assert request["headers"] == {"Authorization": f"Bearer {KEY}"}

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (401, "unauthorized"),
            (413, "invalid_request"),
            (422, "invalid_request"),
            (500, "model_error"),
            (503, "starting"),
            (504, "queue_unavailable"),
        ],
    )
    def test_service_error_codes_become_statuses(self, code, status):
        http = FakeHTTP(FakeResponse(code, {"detail": "x"}))
        assert http_client(http).generate(MESSAGES)["status"] == status

    def test_an_unreadable_body_is_a_model_error(self):
        for body in (ValueError("not json"), {"text": "missing fields"}, ["list"]):
            http = FakeHTTP(FakeResponse(200, body))
            assert http_client(http).generate(MESSAGES)["status"] == "model_error"

    def test_a_timeout_is_busy_and_an_unreachable_service_is_disabled(self):
        assert http_client(FakeHTTP(error=ConnectError("x"))).generate(MESSAGES)["status"] == (
            "queue_unavailable"
        )
        assert http_client(FakeHTTP(error=RuntimeError("dns"))).generate(MESSAGES)["status"] == (
            "disabled"
        )

    def test_readiness_reports_identity(self):
        body = {"model_name": "kleos-hermes", "model_version": "v0.0.6", "base_revision": "a" * 40}
        status = http_client(FakeHTTP(FakeResponse(200, body))).status()
        assert status["status"] == "ready"
        assert status["model"]["base_revision"] == "a" * 40
        assert http_client(FakeHTTP(FakeResponse(503, {}))).status()["status"] == "starting"


class TestNeverRaisesNeverLeaks:
    @pytest.mark.parametrize(
        "error",
        [TimeoutError(), KeyError("x"), RuntimeError("boom"), ConnectError("x"), MemoryError()],
    )
    def test_no_failure_escapes_as_an_exception(self, error):
        client, _, _ = space_client(FakeJob(error=error))
        response = client.generate(MESSAGES)
        assert response["ok"] is False
        assert response["status"] in {s.value for s in HermesStatus}

    def test_credentials_stay_out_of_the_logs(self, caplog):
        caplog.set_level(logging.DEBUG, logger="kleos_models")
        client, _, _ = space_client(FakeJob(error=RuntimeError("boom")))
        client.generate(MESSAGES)
        HermesClient(space_settings(space=None), space_client_factory=Factory())
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert KEY not in logged and HF not in logged
        assert "Which task first?" not in logged


class TestClassifiers:
    def test_validate_contract_accepts_an_error_status(self):
        response = error_response(HermesStatus.STARTING, request_id="r")
        assert validate_contract(response, "r") == response

    def test_classify_helpers_cover_timeouts_and_invalid_states(self):
        assert classify_call_error(TimeoutError()) == (HermesStatus.QUEUE_UNAVAILABLE, None)
        assert classify_connect_error(ValueError("invalid state: BUILD_ERROR")) is (
            HermesStatus.DISABLED
        )
