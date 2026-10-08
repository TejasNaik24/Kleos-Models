# The reference Hermes client for the KLEOS backend (never called from a browser).
#
# Every outcome, from an answer to a sleeping Space or a network failure, maps to the
# status contract in `status.py`; nothing raises. Hermes is optional: KLEOS shows the
# answer or falls back to its default model.
#
# Providers: zerogpu (a Gradio Space, via gradio_client) and http (the FastAPI service,
# via httpx), both imported lazily since neither is a package dependency.
# tests/test_hermes_client.py pins the mapping.

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
    REASONING_CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    HermesStatus,
    classify_exception,
    error_response,
    ok_response,
    public_messages,
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
#: How long a call waits on a waking Space before returning `starting`; it keeps connecting.
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
    #: KLEOS-side ceiling on max_new_tokens; the service enforces its own too.
    max_output_tokens: int | None = None
    #: zerogpu: opens a private Space, and ZeroGPU charges GPU time to this account.
    hf_token: str | None = None
    #: The shared secret the Space (X-Hermes-Key) or service (Bearer) checks.
    api_key: str | None = None
    connect_wait: float = DEFAULT_CONNECT_WAIT_SECONDS
    #: The header the Space reads the secret from (``x-hermes-key``, ``x-logos-key``).
    key_header: str = KEY_HEADER
    #: The prefix of this model's settings, for naming them in problems().
    env_prefix: str = "HERMES"
    #: Reply contract: 1, or 2 for a model that returns its trace (Logos). Refusals carry it too.
    contract_version: int = CONTRACT_VERSION

    @property
    def model_name(self) -> str:
        """The name this client's own refusals use: Hermes, Logos."""
        return self.env_prefix.capitalize()

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        prefix: str = "HERMES",
        key_header: str = KEY_HEADER,
        contract_version: int = CONTRACT_VERSION,
    ) -> HermesClientSettings:
        """Read one model's settings, ``<prefix>_ENABLED`` and so on."""
        source = os.environ if env is None else env

        def name(suffix: str) -> str:
            return f"{prefix}_{suffix}"

        def number(name: str, cast: Callable[[str], Any], default: Any) -> Any:
            raw = (source.get(name) or "").strip()
            try:
                return cast(raw) if raw else default
            except ValueError:
                logger.error("%s=%r is not a number; using %r.", name, raw, default)
                return default

        return cls(
            enabled=(source.get(name("ENABLED")) or "").strip().lower() in _TRUTHY,
            provider=(source.get(name("PROVIDER")) or "zerogpu").strip().lower(),
            space=(source.get(name("SPACE")) or "").strip() or None,
            base_url=(source.get(name("BASE_URL")) or "").strip().rstrip("/") or None,
            timeout=number(name("TIMEOUT"), float, DEFAULT_TIMEOUT_SECONDS),
            max_output_tokens=number(name("MAX_OUTPUT_TOKENS"), int, None),
            hf_token=(source.get(name("HF_TOKEN")) or "").strip() or None,
            api_key=(source.get(name("API_KEY")) or "").strip() or None,
            key_header=key_header,
            env_prefix=prefix,
            contract_version=contract_version,
        )

    def problems(self) -> list[str]:
        """Configuration errors. Names only — never values."""
        prefix = self.env_prefix
        found: list[str] = []
        if self.provider not in PROVIDERS:
            found.append(f"{prefix}_PROVIDER must be one of {PROVIDERS}")
        if self.provider == "zerogpu" and not self.space:
            found.append(f"{prefix}_SPACE is not set")
        if self.provider == "http" and not self.base_url:
            found.append(f"{prefix}_BASE_URL is not set")
        if not self.api_key:
            found.append(f"{prefix}_API_KEY is not set")
        if self.timeout <= 0:
            found.append(f"{prefix}_TIMEOUT must be positive")
        if self.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            found.append(f"the contract version must be one of {SUPPORTED_CONTRACT_VERSIONS}")
        return found


def classify_connect_error(error: BaseException) -> HermesStatus:
    """Why a connection to the Space could not be made."""
    text = str(error).lower()
    if "invalid state" in text or "could not find space" in text:
        return HermesStatus.DISABLED
    return HermesStatus.STARTING


def _is_transport_error(error: BaseException) -> bool:
    # httpx.TransportError, matched by name so importing needs no httpx.
    return any(cls.__name__ == "TransportError" for cls in type(error).__mro__)


def classify_call_error(error: BaseException) -> tuple[HermesStatus, int | None]:
    """A failure during a call that had connected."""
    if isinstance(error, TimeoutError) or _is_transport_error(error):
        return HermesStatus.QUEUE_UNAVAILABLE, None
    return classify_exception(error)


def _carries_its_reasoning(result: dict[str, Any]) -> bool:
    """A version 2 answer states its trace, even if there was none (``None``)."""
    if result.get("contract_version") != REASONING_CONTRACT_VERSION:
        return True
    return "reasoning" in result and (
        result["reasoning"] is None or isinstance(result["reasoning"], str)
    )


def validate_contract(
    result: Any,
    request_id: str,
    *,
    answer: bool = True,
    messages: Mapping[HermesStatus, str] | None = None,
    contract_version: int = CONTRACT_VERSION,
) -> dict[str, Any]:
    """Pass a contract response through; replace anything else with model_error."""
    statuses = {status.value for status in HermesStatus}
    if (
        isinstance(result, dict)
        and result.get("contract_version") in SUPPORTED_CONTRACT_VERSIONS
        and result.get("status") in statuses
        and isinstance(result.get("ok"), bool)
        and result["ok"] is (result["status"] == HermesStatus.READY.value)
        and (not (answer and result["ok"]) or isinstance(result.get("text"), str))
        and (not (answer and result["ok"]) or _carries_its_reasoning(result))
    ):
        return result
    logger.error("Hermes returned a response outside the contract (request_id=%s).", request_id)
    return error_response(
        HermesStatus.MODEL_ERROR,
        request_id=request_id,
        messages=messages,
        contract_version=contract_version,
    )


def _gradio_client(settings: HermesClientSettings) -> Any:
    from gradio_client import Client

    return Client(
        settings.space,
        # False, not None: None falls back to a host-cached `hf auth login` token and its quota.
        token=settings.hf_token or False,
        headers={settings.key_header: settings.api_key or ""},
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
        name = self.settings.model_name
        # Only the refusals use these; "ready" replies come from the model's own server.
        self._messages = public_messages(name, name)
        self._problems = self.settings.problems() if self.settings.enabled else []
        if self._problems:
            logger.error("Hermes is enabled but misconfigured: %s", "; ".join(self._problems))

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

    def _refuse(self, status: HermesStatus, request_id: str, **kwargs: Any) -> dict[str, Any]:
        """A refusal the client makes itself, in its model's words and contract version."""
        return error_response(
            status,
            request_id=request_id,
            messages=self._messages,
            contract_version=self.settings.contract_version,
            **kwargs,
        )

    def _unavailable(self, request_id: str) -> dict[str, Any] | None:
        if not self.settings.enabled or self._problems:
            return self._refuse(HermesStatus.DISABLED, request_id)
        return None

    def _connect(self) -> tuple[Any | None, HermesStatus]:
        """The Space client, connecting in the background if needed."""
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
            return self._refuse(status, request_id)
        args = () if payload is None else (payload,)
        job = None
        try:
            job = client.submit(*args, api_name=api_name)
            result = job.result(timeout=self.settings.timeout)
        except Exception as exc:
            if job is not None and isinstance(exc, TimeoutError):
                with contextlib.suppress(Exception):
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
            return self._refuse(status, request_id, retry_after_seconds=retry_after)
        return validate_contract(
            result,
            request_id,
            answer=payload is not None,
            messages=self._messages,
            contract_version=self.settings.contract_version,
        )

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
            return self._refuse(status, request_id)

        if response.status_code != 200:
            return self._refuse(status_for_http(response.status_code), request_id)
        try:
            body = response.json()
            # A `reasoning` field is a thinking model's trace (contract version 2). A non-object
            # body fails below, as a TypeError.
            traced = isinstance(body, dict) and "reasoning" in body
            reasoning = body.get("reasoning") if traced else None
            return ok_response(
                text=str(body["text"]),
                finish_reason=str(body["finish_reason"]),
                prompt_tokens=int(body["prompt_tokens"]),
                completion_tokens=int(body["completion_tokens"]),
                model=dict(body["model"]),
                request_id=str(body.get("request_id") or request_id),
                reasoning=None if reasoning is None else str(reasoning),
                contract_version=REASONING_CONTRACT_VERSION if traced else CONTRACT_VERSION,
            )
        except (ValueError, KeyError, TypeError):
            logger.error("Hermes service returned an unreadable body (request_id=%s).", request_id)
            return self._refuse(HermesStatus.MODEL_ERROR, request_id)

    def _get_http_ready(self, request_id: str) -> dict[str, Any]:
        try:
            response = self._http_client().get(
                f"{self.settings.base_url}/ready",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        except Exception:
            return self._refuse(HermesStatus.DISABLED, request_id)
        if response.status_code != 200:
            return self._refuse(status_for_http(response.status_code), request_id)
        try:
            body = dict(response.json())
        except (ValueError, TypeError):
            return self._refuse(HermesStatus.MODEL_ERROR, request_id)
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
