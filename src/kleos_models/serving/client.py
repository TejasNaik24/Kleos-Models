# ---------------------------------------------------------------------------
# The reference Hermes client for the KLEOS backend.
#
# KLEOS calls Hermes from its backend only, never from a browser. This client
# turns every outcome — an answer, a sleeping Space, a spent quota, a full
# queue, a network failure — into the status contract in `status.py`, and it
# never raises for any of them. Hermes is optional: whatever this returns, KLEOS
# either shows the answer or quietly uses its default model.
#
# Two providers behind one interface:
#   zerogpu  a Gradio Space on free ZeroGPU, through gradio_client
#   http     the Docker/FastAPI service (`serving/app.py`), through httpx
#
# KLEOS can import this module or port it. What matters is the mapping, and
# the mapping is what tests/test_hermes_client.py pins.
#
# gradio_client and httpx are imported lazily: neither is a dependency of this
# package, and only the provider in use needs its library.
# ---------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from kleos_models.logging_utils import get_logger
from kleos_models.serving.status import (
    CONTRACT_VERSION,
    HermesStatus,
    classify_exception,
    error_response,
    ok_response,
    status_for_http,
)
from kleos_models.serving.zerogpu import KEY_HEADER

logger = get_logger("kleos_models.serving.client")

ENV_ENABLED = "HERMES_ENABLED"
ENV_PROVIDER = "HERMES_PROVIDER"
ENV_SPACE = "HERMES_SPACE"
ENV_BASE_URL = "HERMES_BASE_URL"
ENV_TIMEOUT = "HERMES_TIMEOUT"
ENV_MAX_OUTPUT_TOKENS = "HERMES_MAX_OUTPUT_TOKENS"
ENV_HF_TOKEN = "HERMES_HF_TOKEN"
ENV_API_KEY = "HERMES_API_KEY"

PROVIDERS = ("zerogpu", "http")
DEFAULT_TIMEOUT_SECONDS = 120.0
#: How long one call waits for a connection to a waking Space before saying
#: `starting`. The connection keeps going in the background either way.
DEFAULT_CONNECT_WAIT_SECONDS = 10.0

_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class HermesClientSettings:
    """What the KLEOS backend needs to reach Hermes. Secrets from the environment."""

    enabled: bool = False
    provider: str = "zerogpu"
    #: zerogpu: the Space id, "owner/name".
    space: str | None = None
    #: http: the Docker service's base URL.
    base_url: str | None = None
    #: Upper bound on one generate call, queueing included.
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    #: KLEOS-side ceiling on max_new_tokens. The service enforces its own too.
    max_output_tokens: int | None = None
    #: zerogpu: opens a private Space, and ZeroGPU charges GPU time to this account.
    hf_token: str | None = None
    #: The shared secret the Space (X-Hermes-Key) or service (Bearer) checks.
    api_key: str | None = None
    connect_wait: float = DEFAULT_CONNECT_WAIT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HermesClientSettings:
        source = os.environ if env is None else env

        def number(name: str, cast: Callable[[str], Any], default: Any) -> Any:
            raw = (source.get(name) or "").strip()
            try:
                return cast(raw) if raw else default
            except ValueError:
                logger.error("%s=%r is not a number; using %r.", name, raw, default)
                return default

        return cls(
            enabled=(source.get(ENV_ENABLED) or "").strip().lower() in _TRUTHY,
            provider=(source.get(ENV_PROVIDER) or "zerogpu").strip().lower(),
            space=(source.get(ENV_SPACE) or "").strip() or None,
            base_url=(source.get(ENV_BASE_URL) or "").strip().rstrip("/") or None,
            timeout=number(ENV_TIMEOUT, float, DEFAULT_TIMEOUT_SECONDS),
            max_output_tokens=number(ENV_MAX_OUTPUT_TOKENS, int, None),
            hf_token=(source.get(ENV_HF_TOKEN) or "").strip() or None,
            api_key=(source.get(ENV_API_KEY) or "").strip() or None,
        )

    def problems(self) -> list[str]:
        """Configuration errors. Names only — never values."""
        found: list[str] = []
        if self.provider not in PROVIDERS:
            found.append(f"{ENV_PROVIDER} must be one of {PROVIDERS}")
        if self.provider == "zerogpu" and not self.space:
            found.append(f"{ENV_SPACE} is not set")
        if self.provider == "http" and not self.base_url:
            found.append(f"{ENV_BASE_URL} is not set")
        if not self.api_key:
            found.append(f"{ENV_API_KEY} is not set")
        if self.timeout <= 0:
            found.append(f"{ENV_TIMEOUT} must be positive")
        return found


# ---------------------------------------------------------------------------
# Error mapping on the caller's side
# ---------------------------------------------------------------------------


def classify_connect_error(error: BaseException) -> HermesStatus:
    """Why a connection to the Space could not be made.

    `disabled` only when an operator has to act (the Space is paused, failed
    to build, crashed, or cannot be found with this token). Anything else is
    most likely a Space waking from sleep, and is worth retrying.
    """
    text = str(error).lower()
    if "invalid state" in text or "could not find space" in text:
        return HermesStatus.DISABLED
    return HermesStatus.STARTING


def _is_transport_error(error: BaseException) -> bool:
    # httpx.TransportError covers connect/read/write failures and timeouts.
    # Matched by name so this module does not need httpx to import.
    return any(cls.__name__ == "TransportError" for cls in type(error).__mro__)


def classify_call_error(error: BaseException) -> tuple[HermesStatus, int | None]:
    """A failure during a call that had connected."""
    if isinstance(error, TimeoutError) or _is_transport_error(error):
        return HermesStatus.QUEUE_UNAVAILABLE, None
    return classify_exception(error)


def validate_contract(result: Any, request_id: str, *, answer: bool = True) -> dict[str, Any]:
    """Pass a contract response through; replace anything else with model_error.

    A response KLEOS cannot interpret must not reach it as if it were one.
    ``answer`` requires a successful response to carry text (generate, not status).
    """
    statuses = {status.value for status in HermesStatus}
    if (
        isinstance(result, dict)
        and result.get("contract_version") == CONTRACT_VERSION
        and result.get("status") in statuses
        and isinstance(result.get("ok"), bool)
        and result["ok"] is (result["status"] == HermesStatus.READY.value)
        and (not (answer and result["ok"]) or isinstance(result.get("text"), str))
    ):
        return result
    logger.error("Hermes returned a response outside the contract (request_id=%s).", request_id)
    return error_response(HermesStatus.MODEL_ERROR, request_id=request_id)


# ---------------------------------------------------------------------------
# Default transports
# ---------------------------------------------------------------------------


def _gradio_client(settings: HermesClientSettings) -> Any:
    from gradio_client import Client

    return Client(
        settings.space,
        # False, not None: None would fall back to a token cached on the host
        # by `hf auth login`, silently charging someone else's quota.
        token=settings.hf_token or False,
        headers={KEY_HEADER: settings.api_key or ""},
        verbose=False,
        download_files=False,
    )


class HermesClient:
    """Calls Hermes and always answers with the status contract."""

    def __init__(
        self,
        settings: HermesClientSettings | None = None,
        *,
        space_client_factory: Callable[[HermesClientSettings], Any] | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.settings = settings or HermesClientSettings.from_env()
        self._factory = space_client_factory or _gradio_client
        self._http = http_client
        self._lock = threading.Lock()
        self._space: Any | None = None
        self._connecting: Future[Any] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._problems = self.settings.problems() if self.settings.enabled else []
        if self._problems:
            logger.error("Hermes is enabled but misconfigured: %s", "; ".join(self._problems))

    # -- public --------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """One generation. Returns the contract; never raises for availability."""
        request_id = request_id or uuid.uuid4().hex
        unavailable = self._unavailable(request_id)
        if unavailable is not None:
            return unavailable

        payload: dict[str, Any] = {"messages": messages, "request_id": request_id}
        budget = max_new_tokens
        if self.settings.max_output_tokens is not None:
            budget = min(budget or self.settings.max_output_tokens, self.settings.max_output_tokens)
        if budget is not None:
            payload["max_new_tokens"] = budget

        if self.settings.provider == "zerogpu":
            return self._call_space("/generate", payload, request_id)
        return self._post_http(payload, request_id)

    def status(self) -> dict[str, Any]:
        """Readiness and identity, without generating (no GPU time is used)."""
        request_id = uuid.uuid4().hex
        unavailable = self._unavailable(request_id)
        if unavailable is not None:
            return unavailable
        if self.settings.provider == "zerogpu":
            return self._call_space("/status", None, request_id)
        return self._get_http_ready(request_id)

    # -- shared --------------------------------------------------------------

    def _unavailable(self, request_id: str) -> dict[str, Any] | None:
        if not self.settings.enabled or self._problems:
            return error_response(HermesStatus.DISABLED, request_id=request_id)
        return None

    # -- zerogpu -------------------------------------------------------------

    def _connect(self) -> tuple[Any | None, HermesStatus]:
        """The Space client, connecting in the background if needed.

        Connecting wakes a sleeping Space and can take minutes. It runs on a
        worker thread; a call waits at most ``connect_wait`` for it and reports
        `starting` otherwise, so a KLEOS request is never held hostage by a
        cold boot.
        """
        with self._lock:
            if self._space is not None:
                return self._space, HermesStatus.READY
            if self._connecting is None:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(1, thread_name_prefix="hermes-connect")
                self._connecting = self._executor.submit(self._factory, self.settings)
            pending = self._connecting
        try:
            client = pending.result(timeout=self.settings.connect_wait)
        except FutureTimeout:
            return None, HermesStatus.STARTING
        except Exception as exc:
            with self._lock:
                self._connecting = None
            status = classify_connect_error(exc)
            logger.warning(
                "Hermes Space connection failed (%s): %s", type(exc).__name__, status.value
            )
            return None, status
        with self._lock:
            self._space, self._connecting = client, None
        return client, HermesStatus.READY

    def _call_space(
        self, api_name: str, payload: dict[str, Any] | None, request_id: str
    ) -> dict[str, Any]:
        client, status = self._connect()
        if client is None:
            return error_response(status, request_id=request_id)
        args = () if payload is None else (payload,)
        job = None
        try:
            job = client.submit(*args, api_name=api_name)
            result = job.result(timeout=self.settings.timeout)
        except Exception as exc:
            if job is not None and isinstance(exc, TimeoutError):
                with contextlib.suppress(Exception):  # best effort
                    job.cancel()  # free the Space's queue slot
            status, retry_after = classify_call_error(exc)
            if status is HermesStatus.QUEUE_UNAVAILABLE:
                with self._lock:
                    self._space = None  # reconnect next time
            logger.warning(
                "Hermes call failed request_id=%s error=%s status=%s",
                request_id,
                type(exc).__name__,
                status.value,
            )
            return error_response(status, request_id=request_id, retry_after_seconds=retry_after)
        return validate_contract(result, request_id, answer=payload is not None)

    # -- http ----------------------------------------------------------------

    def _http_client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=self.settings.timeout)
        return self._http

    def _post_http(self, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        try:
            response = self._http_client().post(
                f"{self.settings.base_url}/v1/generate",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        except Exception as exc:
            status, _ = classify_call_error(exc)
            if status is HermesStatus.MODEL_ERROR:
                status = HermesStatus.DISABLED  # unreachable service
            logger.warning(
                "Hermes service unreachable request_id=%s (%s)", request_id, type(exc).__name__
            )
            return error_response(status, request_id=request_id)

        if response.status_code != 200:
            return error_response(status_for_http(response.status_code), request_id=request_id)
        try:
            body = response.json()
            return ok_response(
                text=str(body["text"]),
                finish_reason=str(body["finish_reason"]),
                prompt_tokens=int(body["prompt_tokens"]),
                completion_tokens=int(body["completion_tokens"]),
                model=dict(body["model"]),
                request_id=str(body.get("request_id") or request_id),
            )
        except (ValueError, KeyError, TypeError):
            logger.error("Hermes service returned an unreadable body (request_id=%s).", request_id)
            return error_response(HermesStatus.MODEL_ERROR, request_id=request_id)

    def _get_http_ready(self, request_id: str) -> dict[str, Any]:
        try:
            response = self._http_client().get(
                f"{self.settings.base_url}/ready",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        except Exception:
            return error_response(HermesStatus.DISABLED, request_id=request_id)
        if response.status_code != 200:
            return error_response(status_for_http(response.status_code), request_id=request_id)
        try:
            body = dict(response.json())
        except (ValueError, TypeError):
            return error_response(HermesStatus.MODEL_ERROR, request_id=request_id)
        return {
            "contract_version": CONTRACT_VERSION,
            "ok": True,
            "status": HermesStatus.READY.value,
            "model": {
                "name": body.get("model_name"),
                "version": body.get("model_version"),
                "adapter_sha256": body.get("adapter_sha256"),
                "base_model": body.get("base_model"),
                "base_revision": body.get("base_revision"),
            },
            "runtime": body.get("runtime"),
        }
