# The Hermes status contract: a provider-neutral vocabulary for "did Hermes answer,
# and if not, why". KLEOS never sees an infrastructure message verbatim.
# The values are an API: renaming one breaks KLEOS (tests pin them). Every status
# other than `ready` means "use the fallback model", not a failure to surface.

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

#: Bumped only on a breaking change to the response shape.
CONTRACT_VERSION = 1
#: Replies that also carry the thinking trace (`reasoning`); version 1 never does.
REASONING_CONTRACT_VERSION = 2
#: What a client accepts.
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, REASONING_CONTRACT_VERSION)


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


def public_messages(short_name: str, display_name: str) -> dict[HermesStatus, str]:
    """Messages safe to show a KLEOS user, for one model."""
    possessive = short_name + ("'" if short_name.endswith("s") else "'s")
    return {
        HermesStatus.READY: f"{display_name} is available.",
        HermesStatus.STARTING: f"{short_name} is starting a free GPU worker.",
        HermesStatus.QUOTA_EXHAUSTED: (
            f"{possessive} free GPU quota is currently exhausted. Please try again later."
        ),
        HermesStatus.QUEUE_UNAVAILABLE: (
            f"{short_name} is temporarily busy. Try again later or continue with the default model."
        ),
        HermesStatus.MODEL_ERROR: f"{short_name} could not generate a response.",
        HermesStatus.INVALID_REQUEST: f"The request to {short_name} was not valid.",
        HermesStatus.UNAUTHORIZED: "unauthorized",
        HermesStatus.DISABLED: f"{short_name} is currently unavailable.",
    }


#: Hermes v0.0.6's messages, unchanged since release.
PUBLIC_MESSAGES: dict[HermesStatus, str] = public_messages("Hermes", "Hermes v0.0.6")

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
    reasoning: str | None = None,
    contract_version: int = CONTRACT_VERSION,
) -> dict[str, Any]:
    """A successful generation, in the contract's shape."""
    reply: dict[str, Any] = {
        "contract_version": contract_version,
        "ok": True,
        "status": HermesStatus.READY.value,
        "text": text,
    }
    if contract_version == REASONING_CONTRACT_VERSION:
        reply["reasoning"] = reasoning
    reply.update(
        {
            "finish_reason": finish_reason,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model,
            "request_id": request_id,
            "timings": timings or {},
            "diagnostics": diagnostics or {},
        }
    )
    return reply


def finish_reason(completion_tokens: int, max_new_tokens: int, backend_reason: str = "stop") -> str:
    """``"length"`` when the completion was cut off, else ``"stop"``."""
    if completion_tokens >= max_new_tokens or backend_reason == "length":
        return "length"
    return "stop"


def error_response(
    status: HermesStatus,
    *,
    request_id: str | None,
    message: str | None = None,
    retry_after_seconds: int | None = None,
    messages: Mapping[HermesStatus, str] | None = None,
    contract_version: int = CONTRACT_VERSION,
) -> dict[str, Any]:
    """A non-answer, in the contract's shape."""
    if status is HermesStatus.READY:
        raise ValueError("error_response cannot carry the ready status")
    return {
        "contract_version": contract_version,
        "ok": False,
        "status": status.value,
        "message": message or (messages or PUBLIC_MESSAGES)[status],
        "retryable": status in RETRYABLE,
        "retry_after_seconds": retry_after_seconds,
        "request_id": request_id,
    }


# `spaces` (0.51.3) raises gradio errors with a title and a message. Both are matched:
# older Gradio has no title. Queue phrases win, as some arrive under a quota title.

_QUOTA_TITLES = ("zerogpu quota exceeded",)
_QUOTA_PHRASES = (
    "exceeded your",  # "You have exceeded your free ZeroGPU quota ..."
    "reached its gpu limit",  # "Space app has reached its GPU limit."
    "zerogpu runs limit",
)
_QUEUE_TITLES = ("zerogpu queue timeout", "zerogpu pending credits exceeded")
_QUEUE_PHRASES = (
    # Quota reserved by running tasks, not spent.
    "too many zerogpu credits",
    # A queue timeout; arrives under a "quota exceeded" title without a quota token.
    "no gpu was available",
    # Token expired while queued (congestion); a retry gets a new one.
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
    # Worker errors, "illegal duration" (our config) and anything unrecognised.
    return HermesStatus.MODEL_ERROR


def classify_exception(error: BaseException) -> tuple[HermesStatus, int | None]:
    """Status and provider-stated retry delay for an exception from a GPU call."""
    title = getattr(error, "title", None)
    message = getattr(error, "message", None) or str(error)
    status = classify_zerogpu_error(title if isinstance(title, str) else None, message)
    retry = parse_retry_after(message) if status is HermesStatus.QUOTA_EXHAUSTED else None
    return status, retry


# The Docker/FastAPI service reports through HTTP codes; map them onto the same contract.

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
