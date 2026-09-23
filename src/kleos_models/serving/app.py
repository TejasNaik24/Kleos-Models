# ---------------------------------------------------------------------------
# The Hermes inference service.
#
#   KLEOS Web → KLEOS FastAPI backend → HTTPS → this service
#                                                   ↓
#                              Mistral-Nemo base @ pinned revision + Hermes LoRA
#
# Hermes is a MODEL SERVICE, not a second KLEOS data store. It holds no users,
# workspaces, memories, conversations or documents, and it keeps nothing between
# requests. KLEOS remains the source of truth for all of that; this process turns
# messages into one response and forgets them.
#
# Consequences that are enforced here rather than left to convention:
#   * no persistence of any kind, so there is no cross-user or cross-workspace state
#   * no tool execution and no outbound fetching — the only egress is the Hub, at
#     startup, for the pinned base weights
#   * prompts and responses are never logged by default
#   * the service refuses to start without an API key
#   * generation is serialized: one GPU cannot do two generates at once, and a
#     queue is safer than an OOM
#
# fastapi is imported inside create_app so this module, and the manifest tooling
# that shares the package, stay importable without web dependencies installed.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kleos_models.data.schemas import Message
from kleos_models.errors import ConfigError
from kleos_models.logging_utils import get_logger
from kleos_models.serving.manifest import ServingLimits

logger = get_logger("kleos_models.serving.app")

#: Environment variables the service reads. Secrets come from the environment
#: and never from the manifest, the package or this repository.
ENV_PACKAGE_DIR = "HERMES_PACKAGE_DIR"
ENV_API_KEY = "HERMES_API_KEY"
ENV_DEVICE_MAP = "HERMES_DEVICE_MAP"
ENV_REQUIRE_REMOTE_REVISION = "HERMES_REQUIRE_REMOTE_REVISION"
ENV_REQUEST_TIMEOUT = "HERMES_REQUEST_TIMEOUT_SECONDS"
ENV_MAX_INPUT_CHARS = "HERMES_MAX_INPUT_CHARS"
ENV_ALLOW_UNAUTHENTICATED = "HERMES_ALLOW_UNAUTHENTICATED"
#: Serving record whose identity the package must match (set by the container).
ENV_EXPECTED_DEPLOYMENT_CONFIG = "HERMES_EXPECTED_DEPLOYMENT_CONFIG"
#: Exit the process when the model cannot be loaded, instead of staying up with
#: /v1/generate returning 503. The container sets it so an orchestrator sees the
#: failure rather than a live process that will never be ready.
ENV_EXIT_ON_LOAD_FAILURE = "HERMES_EXIT_ON_LOAD_FAILURE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() in _TRUTHY


@dataclass
class ServingSettings:
    """Everything the process needs, resolved once at startup."""

    package_dir: Path
    api_keys: tuple[str, ...] = ()
    device_map: str | None = None
    require_remote_revision: bool = False
    #: Overrides may only *tighten* the manifest's limits, never widen them.
    max_input_chars: int | None = None
    request_timeout_seconds: float | None = None
    allow_unauthenticated: bool = False
    #: When set, a package that is intact but not this artifact is refused.
    expected_deployment_config: Path | None = None
    exit_on_load_failure: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServingSettings:
        source = os.environ if env is None else env
        package = source.get(ENV_PACKAGE_DIR, "").strip()
        if not package:
            raise ConfigError(
                f"{ENV_PACKAGE_DIR} is not set.",
                suggestions=[
                    "Point it at a deployment package built by "
                    "scripts/build_deployment_package.py.",
                ],
            )

        raw_keys = source.get(ENV_API_KEY, "")
        keys = tuple(part.strip() for part in raw_keys.split(",") if part.strip())
        allow_unauthenticated = _flag(source.get(ENV_ALLOW_UNAUTHENTICATED))
        if not keys and not allow_unauthenticated:
            raise ConfigError(
                f"{ENV_API_KEY} is not set.",
                suggestions=[
                    "Set a shared secret that the KLEOS backend sends as "
                    "'Authorization: Bearer <key>'.",
                    f"For a local experiment only, set {ENV_ALLOW_UNAUTHENTICATED}=1 "
                    "and bind to localhost.",
                ],
            )
        if not keys and allow_unauthenticated:
            logger.warning(
                "%s is set: this process serves unauthenticated requests. Never do "
                "this on a reachable network interface.",
                ENV_ALLOW_UNAUTHENTICATED,
            )

        timeout = source.get(ENV_REQUEST_TIMEOUT)
        max_chars = source.get(ENV_MAX_INPUT_CHARS)
        expected = source.get(ENV_EXPECTED_DEPLOYMENT_CONFIG, "").strip()
        return cls(
            package_dir=Path(package),
            api_keys=keys,
            device_map=source.get(ENV_DEVICE_MAP) or None,
            require_remote_revision=_flag(source.get(ENV_REQUIRE_REMOTE_REVISION)),
            max_input_chars=int(max_chars) if max_chars else None,
            request_timeout_seconds=float(timeout) if timeout else None,
            allow_unauthenticated=allow_unauthenticated,
            expected_deployment_config=Path(expected) if expected else None,
            exit_on_load_failure=_flag(source.get(ENV_EXIT_ON_LOAD_FAILURE)),
        )

    def effective_limits(self, manifest_limits: ServingLimits) -> ServingLimits:
        """Combine manifest limits with environment overrides, taking the tighter."""
        max_chars = manifest_limits.max_input_chars
        if self.max_input_chars is not None:
            max_chars = min(max_chars, self.max_input_chars)
        timeout = manifest_limits.request_timeout_seconds
        if self.request_timeout_seconds is not None:
            timeout = min(timeout, self.request_timeout_seconds)
        return manifest_limits.model_copy(
            update={"max_input_chars": max_chars, "request_timeout_seconds": timeout}
        )


# ---------------------------------------------------------------------------
# Wire contract
# ---------------------------------------------------------------------------


class GenerateMessage(BaseModel):
    """One conversation turn. Only what the model needs to answer."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    """A generation request.

    Deliberately narrow. There is no workspace id, user id, memory handle or
    tool list, because Hermes needs none of them to produce a response, and
    anything sent here is data KLEOS has chosen to expose to the model process.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[GenerateMessage] = Field(min_length=1)
    #: May lower the ceiling, never raise it above the manifest's limit.
    max_new_tokens: int | None = Field(default=None, gt=0)
    #: Echoed back and used in logs, so one request can be traced across services.
    request_id: str | None = Field(default=None, max_length=128)


class ModelIdentity(BaseModel):
    name: str
    version: str
    adapter_sha256: str
    base_model: str
    base_revision: str


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    request_id: str
    model: ModelIdentity


def key_matches(presented: str | None, keys: tuple[str, ...]) -> bool:
    """Constant-time comparison of a presented key against every configured key.

    Shared by the Docker service (bearer header) and the ZeroGPU Space (custom
    header). Compares against every key rather than stopping at the first
    match: short-circuiting would leak through timing which key matched.
    """
    if not presented:
        return False
    matched = False
    for key in keys:
        if hmac.compare_digest(presented, key):
            matched = True
    return matched


def _authorized(header: str | None, keys: tuple[str, ...]) -> bool:
    """Constant-time bearer-token check."""
    if not keys:
        return True  # only reachable when allow_unauthenticated is set
    if not header:
        return False
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return key_matches(presented, keys)


def validate_request(request: GenerateRequest, limits: ServingLimits) -> None:
    """Enforce bounds before any GPU work. Raises ValueError with a safe message."""
    if len(request.messages) > limits.max_messages:
        raise ValueError(f"too many messages: {len(request.messages)} > {limits.max_messages}")
    total = sum(len(message.content) for message in request.messages)
    if total > limits.max_input_chars:
        raise ValueError(f"input too large: {total} characters > {limits.max_input_chars}")


def create_app(
    settings: ServingSettings | None = None,
    *,
    deployment: Any | None = None,
) -> Any:
    """Build the FastAPI application.

    Args:
        settings: Resolved configuration. Read from the environment when omitted.
        deployment: A pre-loaded (or stubbed) deployment. When given, the model
            is not loaded from disk — this is how the API tests run without a GPU.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised by the extras install
        from kleos_models.errors import MissingDependencyError

        raise MissingDependencyError(
            "fastapi", extra="serve", purpose="run the Hermes inference service"
        ) from exc

    from contextlib import asynccontextmanager

    import anyio
    import anyio.to_thread

    settings = settings or ServingSettings.from_env()

    state: dict[str, Any] = {"deployment": deployment, "error": None}
    if deployment is not None:
        state["limits"] = settings.effective_limits(deployment.manifest.limits)

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        if state["deployment"] is None:
            from kleos_models.serving.loader import load_deployment
            from kleos_models.serving.manifest import load_expected_identity

            try:
                expected = (
                    load_expected_identity(settings.expected_deployment_config)
                    if settings.expected_deployment_config is not None
                    else None
                )
                loaded = load_deployment(
                    settings.package_dir,
                    verify=True,
                    require_remote_revision=settings.require_remote_revision,
                    device_map=settings.device_map,
                    expected_identity=expected,
                )
            except Exception as exc:
                # Never answer /v1/generate with an artifact that failed
                # verification — that is the one outcome this service exists to
                # prevent. By default the process stays up so /health can say
                # why; with exit_on_load_failure it exits non-zero instead, so an
                # orchestrator marks the deploy failed rather than waiting on a
                # process that will never become ready.
                state["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("Startup verification failed: %s", state["error"])
                if settings.exit_on_load_failure:
                    raise
            else:
                state["deployment"] = loaded
                state["limits"] = settings.effective_limits(loaded.manifest.limits)
                logger.info("Model verified and loaded: %s", loaded.describe())
        yield

    app = FastAPI(
        title="KLEOS Hermes inference service",
        version=deployment.manifest.model_version if deployment is not None else "unloaded",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # One generate at a time. A single GPU cannot overlap them, and queueing is
    # preferable to an out-of-memory kill mid-request.
    #
    # A *threading* lock, not an async one, and held inside the worker thread on
    # purpose. `model.generate` is not interruptible, so a timed-out request is
    # abandoned by the event loop while its thread runs on. Holding the lock in
    # the thread means the next request still waits for the GPU to be free
    # instead of starting a second generation on top of the first.
    gpu_lock = threading.Semaphore(1)

    def _require_auth(authorization: str | None = Header(default=None)) -> None:
        if not _authorized(authorization, settings.api_keys):
            # 401 with no detail about which part failed.
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness. Always answers while the process is alive, and says why not ready."""
        loaded = state["deployment"]
        return {
            "status": "ok",
            "ready": loaded is not None,
            "error": state["error"],
            "model_name": loaded.manifest.model_name if loaded else None,
        }

    @app.get("/ready", dependencies=[Depends(_require_auth)])
    async def ready() -> Any:
        """Readiness, with the identity of exactly what is loaded."""
        loaded = state["deployment"]
        if loaded is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "error": state["error"]},
            )
        return {"status": "ready", **loaded.describe()}

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        dependencies=[Depends(_require_auth)],
    )
    async def generate(request: GenerateRequest) -> Any:
        loaded = state["deployment"]
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        limits: ServingLimits = state["limits"]
        request_id = request.request_id or uuid.uuid4().hex

        try:
            validate_request(request, limits)
        except ValueError as exc:
            # 413 rather than 422: the request is well-formed, just too big.
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        messages = [Message(role=m.role, content=m.content) for m in request.messages]
        started = time.perf_counter()

        def _blocking_generate() -> Any:
            with gpu_lock:
                return loaded.generate(messages, max_new_tokens=request.max_new_tokens)

        try:
            with anyio.fail_after(limits.request_timeout_seconds):
                output = await anyio.to_thread.run_sync(_blocking_generate, abandon_on_cancel=True)
        except TimeoutError as exc:
            logger.warning(
                "generate timeout request_id=%s messages=%d elapsed=%.1fs",
                request_id,
                len(messages),
                time.perf_counter() - started,
            )
            raise HTTPException(status_code=504, detail="generation timed out") from exc
        except Exception as exc:
            # Never surface an internal traceback or prompt content to a caller.
            logger.exception("generate failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="generation failed") from exc

        elapsed = time.perf_counter() - started
        # Structured, and free of prompt or response text by design.
        logger.info(
            "generate ok request_id=%s messages=%d prompt_tokens=%d completion_tokens=%d "
            "elapsed=%.2fs",
            request_id,
            len(messages),
            output.prompt_tokens,
            output.completion_tokens,
            elapsed,
        )

        manifest = loaded.manifest
        return GenerateResponse(
            text=output.text,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            request_id=request_id,
            model=ModelIdentity(
                name=manifest.model_name,
                version=manifest.model_version,
                adapter_sha256=manifest.adapter.weights_sha256,
                base_model=manifest.base_model,
                base_revision=manifest.base_revision,
            ),
        )

    return app
