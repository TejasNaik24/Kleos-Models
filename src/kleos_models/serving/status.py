# ---------------------------------------------------------------------------
# The Hermes status contract.
#
# One small, provider-neutral vocabulary for "did Hermes answer, and if not,
# why". The KLEOS backend branches on `status`; the frontend renders from it.
# Neither needs to know whether Hermes runs on ZeroGPU or behind the Docker
# service, and neither ever sees an infrastructure message verbatim.
#
# The values are an API. Renaming one breaks KLEOS; tests pin them.
#
# Hermes is optional. Every status other than `ready` means "use the fallback
# model", and none of them is an error KLEOS should surface as a failure.
# ---------------------------------------------------------------------------

from __future__ import annotations

import re
from enum import Enum
from typing import Any

#: Bumped only on a breaking change to the response shape.
CONTRACT_VERSION = 1


class HermesStatus(str, Enum):
    """Why Hermes did or did not answer."""

    #: Answered. `text` is present.
    READY = "ready"
    #: The host is booting, waking from sleep, or loading the model.
    STARTING = "starting"
    #: The free daily GPU quota is used up. Resets on the provider's schedule.
    QUOTA_EXHAUSTED = "quota_exhausted"
    #: No GPU could be obtained in time, or the request queue is full.
    QUEUE_UNAVAILABLE = "queue_unavailable"
    #: The model or its host failed while generating.
    MODEL_ERROR = "model_error"
    #: The request was malformed or over a size limit. Retrying will not help.
    INVALID_REQUEST = "invalid_request"
    #: Missing or wrong credentials.
    UNAUTHORIZED = "unauthorized"
    #: Hermes is switched off, paused, or unreachable.
    DISABLED = "disabled"


#: Messages safe to show a KLEOS user. No provider names, quotas, paths or ids.
PUBLIC_MESSAGES: dict[HermesStatus, str] = {
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

#: Statuses where the same request may succeed later without changes.
RETRYABLE = frozenset(
    {HermesStatus.STARTING, HermesStatus.QUOTA_EXHAUSTED, HermesStatus.QUEUE_UNAVAILABLE}
)


def ok_response(
    *,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: dict[str, Any],
    request_id: str,
    timings: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A successful generation, in the contract's shape."""
    return {
        "contract_version": CONTRACT_VERSION,
        "ok": True,
        "status": HermesStatus.READY.value,
        "text": text,
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "model": model,
        "request_id": request_id,
        "timings": timings or {},
        "diagnostics": diagnostics or {},
    }


def finish_reason(completion_tokens: int, max_new_tokens: int) -> str:
    """``"length"`` when the completion used its whole token budget, else ``"stop"``.

    Decoding is greedy with no stop strings, so generation ends either on the
    end-of-sequence token or at the budget. A completion that fills the budget
    was almost certainly cut off — for Hermes that usually means unfinished
    JSON, which KLEOS should treat as incomplete rather than parse.
    """
    return "length" if completion_tokens >= max_new_tokens else "stop"


def error_response(
    status: HermesStatus,
    *,
    request_id: str | None,
    message: str | None = None,
    retry_after_seconds: int | None = None,
) -> dict[str, Any]:
    """A non-answer, in the contract's shape.

    ``retry_after_seconds`` is only ever set from a figure the provider itself
    supplied. The contract never invents a countdown.
    """
    if status is HermesStatus.READY:
        raise ValueError("error_response cannot carry the ready status")
    return {
        "contract_version": CONTRACT_VERSION,
        "ok": False,
        "status": status.value,
        "message": message or PUBLIC_MESSAGES[status],
        "retryable": status in RETRYABLE,
        "retry_after_seconds": retry_after_seconds,
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# ZeroGPU error classification
#
# The `spaces` package raises gradio errors with a title and a message
# (spaces/zero/client.py and wrappers.py, 0.51.3). Both are matched, because
# the title parameter only exists on newer Gradio versions. Queue conditions
# are checked first: `spaces` files a queue timeout for a caller without a quota
# token under a "quota exceeded" title.
# ---------------------------------------------------------------------------

_QUOTA_TITLES = ("zerogpu quota exceeded",)
_QUOTA_PHRASES = (
    "exceeded your",  # "You have exceeded your free ZeroGPU quota ..."
    "reached its gpu limit",  # "Space app has reached its GPU limit."
    "zerogpu runs limit",
)
_QUEUE_TITLES = ("zerogpu queue timeout", "zerogpu pending credits exceeded")
_QUEUE_PHRASES = (
    # "You have too many ZeroGPU credits allocated to running tasks. Try again
    # once some of those tasks have completed." Quota is reserved, not spent.
    "too many zerogpu credits",
    # Checked before the quota titles: a caller without a quota token who times
    # out in the GPU queue gets this message under a "quota exceeded" title.
    "no gpu was available",
    # The per-request proxy token outlived its validity while the request
    # waited in the Space's queue: congestion, and a later retry gets a new one.
    "expired zerogpu proxy token",
    # The ZeroGPU scheduler itself failed (a RuntimeError, no title).
    "zerogpu api /schedule error",
    # gradio_client's QueueError when the Space's request queue is full.
    "queue is full",
)

# str(timedelta): "0:23:45", "1 day, 2:03:04", optionally with microseconds.
_RETRY_AFTER = re.compile(
    r"try again in (?:(?P<days>\d+) days?, )?(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})(?:\.\d+)?",
    re.IGNORECASE,
)


def parse_retry_after(message: str) -> int | None:
    """Seconds until retry, only when the provider's message states it."""
    match = _RETRY_AFTER.search(message or "")
    if not match:
        return None
    days = int(match.group("days") or 0)
    return (
        days * 86_400
        + int(match.group("h")) * 3_600
        + int(match.group("m")) * 60
        + int(match.group("s"))
    )


def classify_zerogpu_error(title: str | None, message: str | None) -> HermesStatus:
    """Map a ZeroGPU scheduling or worker failure to a contract status."""
    t = (title or "").strip().lower()
    m = (message or "").strip().lower()

    if t in _QUEUE_TITLES or any(phrase in m for phrase in _QUEUE_PHRASES):
        return HermesStatus.QUEUE_UNAVAILABLE
    if t in _QUOTA_TITLES or any(phrase in m for phrase in _QUOTA_PHRASES):
        return HermesStatus.QUOTA_EXHAUSTED
    # "ZeroGPU worker error", "ZeroGPU illegal duration" (a configuration
    # problem on our side), "ZeroGPU client error" without the no-GPU phrase,
    # and anything unrecognised: the model did not produce an answer.
    return HermesStatus.MODEL_ERROR


def classify_exception(error: BaseException) -> tuple[HermesStatus, int | None]:
    """Status and provider-stated retry delay for an exception from a GPU call."""
    title = getattr(error, "title", None)
    message = getattr(error, "message", None) or str(error)
    status = classify_zerogpu_error(title if isinstance(title, str) else None, message)
    retry = parse_retry_after(message) if status is HermesStatus.QUOTA_EXHAUSTED else None
    return status, retry


# ---------------------------------------------------------------------------
# The Docker/FastAPI service reports through HTTP status codes; this is how a
# client maps them onto the same contract, so KLEOS has one vocabulary.
# ---------------------------------------------------------------------------

HTTP_STATUS_MAP: dict[int, HermesStatus] = {
    401: HermesStatus.UNAUTHORIZED,
    413: HermesStatus.INVALID_REQUEST,
    422: HermesStatus.INVALID_REQUEST,
    500: HermesStatus.MODEL_ERROR,
    503: HermesStatus.STARTING,
    504: HermesStatus.QUEUE_UNAVAILABLE,
}


def status_for_http(code: int) -> HermesStatus:
    """Contract status for a Docker-service HTTP response code."""
    if 200 <= code < 300:
        return HermesStatus.READY
    return HTTP_STATUS_MAP.get(code, HermesStatus.MODEL_ERROR)
