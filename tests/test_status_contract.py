"""The Hermes status contract: what KLEOS branches on when Hermes does not answer."""

from __future__ import annotations

import json

import pytest

from kleos_models.serving.status import (
    CONTRACT_VERSION,
    PUBLIC_MESSAGES,
    RETRYABLE,
    SUPPORTED_CONTRACT_VERSIONS,
    HermesStatus,
    classify_exception,
    classify_zerogpu_error,
    error_response,
    finish_reason,
    ok_response,
    parse_retry_after,
    public_messages,
    status_for_http,
)


class FakeGradioError(Exception):
    """Shaped like gr.Error: a message, and a title on Gradio >= 4.39."""

    def __init__(self, message: str, title: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if title is not None:
            self.title = title


# (title, message, expected status, expected retry seconds) — spaces 0.51.3.
ZEROGPU_CASES = [
    pytest.param(
        "ZeroGPU quota exceeded",
        "You have exceeded your free ZeroGPU quota (90s requested vs. 41s left). "
        "Try again in 13:42:07. Subscribe to Hugging Face PRO to get 40 min of ZeroGPU "
        "quota a day - https://huggingface.co/subscribe/pro?from=ZeroGPU",
        HermesStatus.QUOTA_EXHAUSTED,
        13 * 3600 + 42 * 60 + 7,
        id="free-account-quota",
    ),
    pytest.param(
        "ZeroGPU quota exceeded",
        "You have exceeded your ZeroGPU quota (90s requested vs. 12s left). "
        "Try again in 0:47:10. Authenticate with a Hugging Face token for more quota - "
        "https://huggingface.co/settings/tokens",
        HermesStatus.QUOTA_EXHAUSTED,
        47 * 60 + 10,
        id="unauthenticated-quota",
    ),
    pytest.param(
        "ZeroGPU quota exceeded",
        "You have exceeded your ZeroGPU runs limit. ",
        HermesStatus.QUOTA_EXHAUSTED,
        None,
        id="runs-limit",
    ),
    pytest.param(
        "ZeroGPU quota exceeded",
        "Space app has reached its GPU limit. "
        "Try re-running outside of examples if it happened after clicking one",
        HermesStatus.QUOTA_EXHAUSTED,
        None,
        id="space-limit",
    ),
    pytest.param(
        "ZeroGPU queue timeout",
        "No GPU was available after 60s Retry later",
        HermesStatus.QUEUE_UNAVAILABLE,
        None,
        id="queue-timeout",
    ),
    pytest.param(
        # Queue timeout without a quota token: spaces uses the quota title, but none was used.
        "ZeroGPU quota exceeded",
        "No GPU was available after 60s. "
        "Try re-running outside of examples if it happened after clicking one",
        HermesStatus.QUEUE_UNAVAILABLE,
        None,
        id="queue-timeout-under-quota-title",
    ),
    pytest.param(
        "ZeroGPU client error",
        "No GPU was available",
        HermesStatus.QUEUE_UNAVAILABLE,
        None,
        id="scheduler-503",
    ),
    pytest.param(
        "ZeroGPU client error",
        "Expired ZeroGPU proxy token",
        HermesStatus.QUEUE_UNAVAILABLE,
        None,
        id="expired-proxy-token",
    ),
    pytest.param(
        "ZeroGPU pending credits exceeded",
        "You have too many ZeroGPU credits allocated to running tasks. "
        "Try again once some of those tasks have completed.",
        HermesStatus.QUEUE_UNAVAILABLE,
        None,
        id="pending-credits",
    ),
    pytest.param(
        "ZeroGPU illegal duration",
        "The requested GPU duration (180s) is larger than the maximum allowed",
        HermesStatus.MODEL_ERROR,
        None,
        id="illegal-duration",
    ),
    pytest.param(
        "ZeroGPU worker error",
        "GPU task aborted",
        HermesStatus.MODEL_ERROR,
        None,
        id="worker-aborted",
    ),
    pytest.param(
        "ZeroGPU worker error",
        "OutOfMemoryError",
        HermesStatus.MODEL_ERROR,
        None,
        id="worker-exception",
    ),
    pytest.param(
        "ZeroGPU client error",
        "Internal Gradio error",
        HermesStatus.MODEL_ERROR,
        None,
        id="internal-gradio-error",
    ),
]


class TestStatusValuesAreAnApi:
    def test_the_status_values_are_pinned(self):
        # KLEOS switches on these strings. Renaming one is a breaking change.
        assert {s.value for s in HermesStatus} == {
            "ready",
            "starting",
            "quota_exhausted",
            "queue_unavailable",
            "model_error",
            "invalid_request",
            "unauthorized",
            "disabled",
        }

    def test_only_transient_states_are_retryable(self):
        assert {s.value for s in RETRYABLE} == {"starting", "quota_exhausted", "queue_unavailable"}

    def test_every_status_has_a_public_message(self):
        assert set(PUBLIC_MESSAGES) == set(HermesStatus)

    @pytest.mark.parametrize("status", list(HermesStatus))
    def test_public_messages_reveal_no_infrastructure(self, status):
        text = PUBLIC_MESSAGES[status].lower()
        for leak in ("zerogpu", "hugging", "space", "token", "credit", "http", "$", " pro "):
            assert leak not in text, f"{status.value}: {leak!r} in {text!r}"


class TestResponseShapes:
    def test_a_success_carries_text_usage_and_identity(self):
        response = ok_response(
            text="Do the migration first.",
            finish_reason="stop",
            prompt_tokens=40,
            completion_tokens=9,
            model={"name": "kleos-hermes", "version": "v0.0.6"},
            request_id="r1",
        )
        assert response["ok"] is True
        assert response["status"] == "ready"
        assert response["contract_version"] == CONTRACT_VERSION
        assert response["usage"] == {"prompt_tokens": 40, "completion_tokens": 9}
        json.dumps(response)

    @pytest.mark.parametrize("status", [s for s in HermesStatus if s is not HermesStatus.READY])
    def test_a_non_answer_is_a_response_not_an_exception(self, status):
        response = error_response(status, request_id="r2")
        assert response["ok"] is False
        assert response["status"] == status.value
        assert response["message"] == PUBLIC_MESSAGES[status]
        assert response["retryable"] is (status in RETRYABLE)
        assert "text" not in response
        json.dumps(response)

    def test_an_error_response_cannot_claim_ready(self):
        with pytest.raises(ValueError, match="ready"):
            error_response(HermesStatus.READY, request_id=None)

    def test_no_countdown_is_invented(self):
        assert (
            error_response(HermesStatus.QUOTA_EXHAUSTED, request_id=None)["retry_after_seconds"]
            is None
        )


class TestZeroGPUErrorsAreClassified:
    @pytest.mark.parametrize(("title", "message", "expected", "retry"), ZEROGPU_CASES)
    def test_with_the_title(self, title, message, expected, retry):
        assert classify_zerogpu_error(title, message) is expected
        assert classify_exception(FakeGradioError(message, title)) == (expected, retry)

    @pytest.mark.parametrize(("title", "message", "expected", "retry"), ZEROGPU_CASES)
    def test_without_the_title(self, title, message, expected, retry):
        # Gradio < 4.39 has no title parameter: the message alone must suffice,
        # except for the worker and duration errors, which are model errors anyway.
        assert classify_exception(FakeGradioError(message)) == (expected, retry)

    def test_a_scheduler_failure_is_a_queue_problem(self):
        error = RuntimeError("ZeroGPU API /schedule error: 500 (Internal Server Error)")
        assert classify_exception(error) == (HermesStatus.QUEUE_UNAVAILABLE, None)

    def test_a_full_request_queue_is_a_queue_problem(self):
        # gradio_client.utils.QueueError, raised on the caller's side.
        error = Exception("Queue is full! Please try again.")
        assert classify_exception(error) == (HermesStatus.QUEUE_UNAVAILABLE, None)

    def test_an_unrecognised_exception_is_a_model_error(self):
        assert classify_exception(ValueError("shape mismatch")) == (HermesStatus.MODEL_ERROR, None)

    def test_a_retry_time_is_only_read_from_a_quota_error(self):
        error = FakeGradioError("GPU task aborted. Try again in 0:01:00", "ZeroGPU worker error")
        assert classify_exception(error) == (HermesStatus.MODEL_ERROR, None)


class TestRetryAfter:
    @pytest.mark.parametrize(
        ("message", "seconds"),
        [
            ("Try again in 13:42:07.", 49_327),
            ("Try again in 0:00:59.500000.", 59),
            ("Try again in 1 day, 2:03:04.", 93_784),
            ("Try again in 2 days, 0:00:00.", 172_800),
        ],
    )
    def test_the_provider_stated_wait_is_parsed(self, message, seconds):
        assert parse_retry_after(message) == seconds

    @pytest.mark.parametrize("message", ["", "Space app has reached its GPU limit.", "later"])
    def test_nothing_is_parsed_when_nothing_is_stated(self, message):
        assert parse_retry_after(message) is None


class TestDockerServiceStatusMapping:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (200, HermesStatus.READY),
            (401, HermesStatus.UNAUTHORIZED),
            (413, HermesStatus.INVALID_REQUEST),
            (422, HermesStatus.INVALID_REQUEST),
            (500, HermesStatus.MODEL_ERROR),
            (503, HermesStatus.STARTING),
            (504, HermesStatus.QUEUE_UNAVAILABLE),
            (418, HermesStatus.MODEL_ERROR),
        ],
    )
    def test_http_codes_map_onto_the_same_vocabulary(self, code, expected):
        assert status_for_http(code) is expected


class TestPerModelAndV2:
    """Logos v0.0.2 speaks the same contract under its own name, with its trace."""

    def test_hermes_messages_are_exactly_todays(self):
        assert public_messages("Hermes", "Hermes v0.0.6") == {
            HermesStatus.READY: "Hermes v0.0.6 is available.",
            HermesStatus.STARTING: "Hermes is starting a free GPU worker.",
            HermesStatus.QUOTA_EXHAUSTED: (
                "Hermes' free GPU quota is currently exhausted. Please try again later."
            ),
            HermesStatus.QUEUE_UNAVAILABLE: (
                "Hermes is temporarily busy. Try again later or continue with the default model."
            ),
            HermesStatus.MODEL_ERROR: "Hermes could not generate a response.",
            HermesStatus.INVALID_REQUEST: "The request to Hermes was not valid.",
            HermesStatus.UNAUTHORIZED: "unauthorized",
            HermesStatus.DISABLED: "Hermes is currently unavailable.",
        }
        assert public_messages("Hermes", "Hermes v0.0.6") == PUBLIC_MESSAGES

    def test_logos_messages_name_logos(self):
        messages = public_messages("Logos", "Logos v0.0.2")
        assert messages[HermesStatus.READY] == "Logos v0.0.2 is available."
        assert messages[HermesStatus.QUOTA_EXHAUSTED].startswith("Logos' free GPU quota")
        assert all("Hermes" not in text for text in messages.values())

    def test_a_name_not_ending_in_s_takes_an_apostrophe_s(self):
        assert public_messages("Iris", "Iris v1")[HermesStatus.QUOTA_EXHAUSTED].startswith("Iris'")
        assert public_messages("Atlas", "Atlas v1")[HermesStatus.QUOTA_EXHAUSTED].startswith(
            "Atlas'"
        )
        assert public_messages("Orion", "Orion v1")[HermesStatus.QUOTA_EXHAUSTED].startswith(
            "Orion's"
        )

    def test_a_v1_reply_has_no_reasoning_key(self):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
        )
        assert reply["contract_version"] == CONTRACT_VERSION == 1
        assert "reasoning" not in reply

    def test_a_v2_reply_carries_the_trace(self):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning="t",
            contract_version=2,
        )
        assert reply["contract_version"] == 2 and reply["reasoning"] == "t"
        empty = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning=None,
            contract_version=2,
        )
        assert "reasoning" in empty and empty["reasoning"] is None
        assert SUPPORTED_CONTRACT_VERSIONS == (1, 2)

    @pytest.mark.parametrize(
        ("tokens", "budget", "backend", "expected"),
        [
            (10, 1024, "stop", "stop"),
            (1024, 1024, "stop", "length"),
            (200, 1024, "length", "length"),
        ],
    )
    def test_finish_reason_honours_the_backend(self, tokens, budget, backend, expected):
        assert finish_reason(tokens, budget, backend) == expected

    def test_finish_reason_without_a_backend_reason_is_unchanged(self):
        assert finish_reason(10, 512) == "stop" and finish_reason(512, 512) == "length"

    def test_error_responses_take_per_model_messages_and_version(self):
        messages = public_messages("Logos", "Logos v0.0.2")
        reply = error_response(
            HermesStatus.DISABLED, request_id="r", messages=messages, contract_version=2
        )
        assert reply["message"] == "Logos is currently unavailable."
        assert reply["contract_version"] == 2
        hermes = error_response(HermesStatus.DISABLED, request_id="r")
        assert hermes["message"] == "Hermes is currently unavailable."
        assert hermes["contract_version"] == 1
