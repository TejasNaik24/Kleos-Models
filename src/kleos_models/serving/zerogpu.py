# ---------------------------------------------------------------------------
# Serving Hermes from a free Hugging Face ZeroGPU Space.
#
# An infrastructure adapter, not a second model implementation. It loads the
# same frozen package through the same `load_deployment`, and generates through
# the same `HuggingFaceBackend` as the Docker service — only split so that the
# GPU is held for the one step that needs it.
#
#   request ──► auth ─► validate ─► prepare (render + tokenize) ─┐   CPU
#                                                                 ▼
#                                          @spaces.GPU  generate_ids   GPU (quota)
#                                                                 │
#   response ◄── contract JSON ◄── finish (decode) ◄─────────────┘   CPU
#
# ZeroGPU facts this relies on (verified against the HF docs and `spaces`
# 0.51.3, 2026-09-23):
#   * the model is loaded at import time under "CUDA emulation"; weights move to
#     a real GPU when a decorated call runs;
#   * quota and queue decisions happen in the caller's process *before* any GPU
#     work, and surface as gradio errors this module classifies;
#   * a worker process that stays assigned and idle is reused, with the weights
#     still on the GPU (warm); otherwise a new worker is forked (cold);
#   * GPU time is charged to the calling Hugging Face account.
#
# This module imports neither gradio nor spaces, so all of it is testable
# without either. `deploy/zerogpu-space/app.py` is the thin wiring.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
import os
import platform
import re
import resource
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kleos_models.compat import package_version
from kleos_models.config import GenerationConfig
from kleos_models.data.schemas import Message
from kleos_models.errors import ConfigError, KleosError
from kleos_models.logging_utils import get_logger
from kleos_models.serving.app import GenerateRequest, key_matches, validate_request
from kleos_models.serving.manifest import load_expected_identity
from kleos_models.serving.status import (
    CONTRACT_VERSION,
    HermesStatus,
    classify_exception,
    error_response,
    finish_reason,
    ok_response,
)

logger = get_logger("kleos_models.serving.zerogpu")

#: The header carrying the shared secret. A custom header rather than
#: `Authorization`, which Hugging Face's proxy uses for its own token.
KEY_HEADER = "x-hermes-key"

ENV_API_KEY = "HERMES_API_KEY"
ENV_ALLOW_UNAUTHENTICATED = "HERMES_ALLOW_UNAUTHENTICATED"
ENV_MAX_INPUT_TOKENS = "HERMES_MAX_INPUT_TOKENS"
ENV_MAX_NEW_TOKENS = "HERMES_MAX_NEW_TOKENS"
ENV_PACKAGE_REPO = "HERMES_PACKAGE_REPO"
ENV_PACKAGE_REVISION = "HERMES_PACKAGE_REVISION"
ENV_GPU_BASE_SECONDS = "HERMES_GPU_BASE_SECONDS"
ENV_GPU_TOKENS_PER_SECOND = "HERMES_GPU_TOKENS_PER_SECOND"
ENV_GPU_MIN_SECONDS = "HERMES_GPU_MIN_SECONDS"
ENV_GPU_MAX_SECONDS = "HERMES_GPU_MAX_SECONDS"

#: The v0.0.6 dataset's longest prompt is ~440 tokens. 2048 is generous for
#: real KLEOS prompts and still bounds what one request can cost.
DEFAULT_MAX_INPUT_TOKENS = 2048

_PINNED = re.compile(r"^[0-9a-f]{40}$")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class ZeroGPUSettings:
    """What the Space needs beyond the package. Read once, at startup."""

    api_keys: tuple[str, ...]
    allow_unauthenticated: bool = False
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    #: May only tighten the manifest's ceiling.
    max_new_tokens: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ZeroGPUSettings:
        source = os.environ if env is None else env
        keys = tuple(
            part.strip() for part in (source.get(ENV_API_KEY) or "").split(",") if part.strip()
        )
        allow = _flag(source.get(ENV_ALLOW_UNAUTHENTICATED))
        if not keys and not allow:
            raise ConfigError(
                f"{ENV_API_KEY} is not set.",
                suggestions=[
                    "Add it as a Space secret. The KLEOS backend sends it in the "
                    f"'{KEY_HEADER}' header.",
                    "Without it, anyone who can reach the Space could use Hermes.",
                ],
            )
        max_input = source.get(ENV_MAX_INPUT_TOKENS)
        max_new = source.get(ENV_MAX_NEW_TOKENS)
        return cls(
            api_keys=keys,
            allow_unauthenticated=allow,
            max_input_tokens=int(max_input) if max_input else DEFAULT_MAX_INPUT_TOKENS,
            max_new_tokens=int(max_new) if max_new else None,
        )


# ---------------------------------------------------------------------------
# GPU side
# ---------------------------------------------------------------------------

#: Per-process call counter. A forked ZeroGPU worker inherits the parent's
#: values, sees a different pid, and restarts from zero: call 1 is cold.
_WORKER: dict[str, int | None] = {"pid": None, "calls": 0}


def run_gpu_step(deployment: Any, prepared: Any, config: GenerationConfig) -> dict[str, Any]:
    """The only work done while a GPU is held: generate token ids, and measure.

    Returns plain values, because ZeroGPU pickles the result back to the
    calling process.
    """
    import torch

    pid = os.getpid()
    if _WORKER["pid"] != pid:
        _WORKER["pid"] = pid
        _WORKER["calls"] = 0
    _WORKER["calls"] = int(_WORKER["calls"] or 0) + 1

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    completion_ids = deployment.generate_ids(prepared, config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    properties = torch.cuda.get_device_properties(0)
    return {
        "completion_ids": [int(token) for token in completion_ids],
        "gpu_generate_s": round(elapsed, 3),
        "worker_call_index": _WORKER["calls"],
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "total_vram_bytes": int(properties.total_memory),
        "cuda_runtime": torch.version.cuda,
    }


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def gpu_duration(prepared: Any, config: GenerationConfig) -> int:
    """GPU seconds to request for one call.

    ZeroGPU admits a call only if the account has at least the *requested*
    duration left (x1.5 on the default GPU size), and charges the time actually
    used. An over-generous request therefore makes the daily quota appear
    exhausted while most of it is unused — so this stays tight, scaled by the
    token budget, and is calibrated from measured throughput.
    """
    base = _float_env(ENV_GPU_BASE_SECONDS, 10.0)
    tokens_per_second = _float_env(ENV_GPU_TOKENS_PER_SECOND, 12.0)
    minimum = _float_env(ENV_GPU_MIN_SECONDS, 15.0)
    maximum = _float_env(ENV_GPU_MAX_SECONDS, 60.0)
    estimate = base + config.max_new_tokens / max(tokens_per_second, 0.1)
    return int(min(max(math.ceil(estimate), minimum), maximum))


# ---------------------------------------------------------------------------
# The request path
# ---------------------------------------------------------------------------


class ZeroGPUService:
    """Authenticated, validated, quota-aware generation over a loaded deployment."""

    def __init__(
        self,
        deployment: Any,
        settings: ZeroGPUSettings,
        *,
        gpu_call: Callable[[Any, GenerationConfig], dict[str, Any]],
    ) -> None:
        self.deployment = deployment
        self.settings = settings
        self.gpu_call = gpu_call
        manifest = deployment.manifest
        self.model_identity = {
            "name": manifest.model_name,
            "version": manifest.model_version,
            "adapter_sha256": manifest.adapter.weights_sha256,
            "base_model": manifest.base_model,
            "base_revision": manifest.base_revision,
        }

    # -- helpers -------------------------------------------------------------

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        if not self.settings.api_keys:
            return self.settings.allow_unauthenticated
        presented = {k.lower(): v for k, v in headers.items()}.get(KEY_HEADER)
        return key_matches(presented, self.settings.api_keys)

    def _ceiling(self) -> int:
        ceiling = self.deployment.manifest.limits.max_new_tokens
        if self.settings.max_new_tokens is not None:
            ceiling = min(ceiling, self.settings.max_new_tokens)
        return ceiling

    @staticmethod
    def _request_id(payload: Any) -> str:
        if isinstance(payload, dict):
            candidate = payload.get("request_id")
            if isinstance(candidate, str) and 0 < len(candidate) <= 128:
                return candidate
        return uuid.uuid4().hex

    def _log(self, request_id: str, status: HermesStatus, **fields: Any) -> None:
        # Identifiers, counts and timings only. Never prompt or response text,
        # never headers, never keys.
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("generate request_id=%s status=%s %s", request_id, status.value, details)

    # -- endpoints -----------------------------------------------------------

    def status(self, headers: Mapping[str, str]) -> dict[str, Any]:
        """Readiness and identity. The model is loaded before the app serves, so
        answering at all means ready; `starting` is observed by the caller as
        the Space not responding yet."""
        if not self._authorized(headers):
            return error_response(HermesStatus.UNAUTHORIZED, request_id=None)
        manifest = self.deployment.manifest
        return {
            "contract_version": CONTRACT_VERSION,
            "ok": True,
            "status": HermesStatus.READY.value,
            "model": self.model_identity,
            "runtime": manifest.runtime.model_dump(mode="json"),
            "limits": {
                "max_new_tokens": self._ceiling(),
                "max_input_tokens": self.settings.max_input_tokens,
                "max_input_chars": manifest.limits.max_input_chars,
                "max_messages": manifest.limits.max_messages,
            },
            "versions": runtime_versions(),
        }

    def generate(self, payload: Any, headers: Mapping[str, str]) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = self._request_id(payload)

        if not self._authorized(headers):
            self._log(request_id, HermesStatus.UNAUTHORIZED)
            return error_response(HermesStatus.UNAUTHORIZED, request_id=request_id)

        # Validate: schema first (unknown fields such as workspace ids or tool
        # definitions are refused), then size bounds — all before any GPU work.
        try:
            request = GenerateRequest.model_validate(payload)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="schema")
            return error_response(
                HermesStatus.INVALID_REQUEST,
                request_id=request_id,
                message=f"Invalid request fields: {', '.join(fields) or 'body'}.",
            )
        try:
            validate_request(request, self.deployment.manifest.limits)
        except ValueError as exc:
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="size")
            return error_response(
                HermesStatus.INVALID_REQUEST, request_id=request_id, message=str(exc)
            )

        ceiling = self._ceiling()
        budget = min(request.max_new_tokens or ceiling, ceiling)
        config = self.deployment.generation_config(budget)
        messages = [Message(role=m.role, content=m.content) for m in request.messages]

        try:
            prepared = self.deployment.prepare(messages)
        except Exception as exc:
            self._log(
                request_id, HermesStatus.MODEL_ERROR, stage="prepare", error=type(exc).__name__
            )
            return error_response(HermesStatus.MODEL_ERROR, request_id=request_id)
        prepared_at = time.perf_counter()

        if prepared.prompt_length > self.settings.max_input_tokens:
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="tokens")
            return error_response(
                HermesStatus.INVALID_REQUEST,
                request_id=request_id,
                message=(
                    f"input too long: {prepared.prompt_length} tokens > "
                    f"{self.settings.max_input_tokens}"
                ),
            )

        try:
            result = self.gpu_call(prepared, config)
        except Exception as exc:
            status, retry_after = classify_exception(exc)
            title = getattr(exc, "title", None)
            infra = isinstance(title, str) and title.startswith("ZeroGPU")
            # A ZeroGPU message describes quota or queue state and is safe to
            # log; any other exception text could echo input, so only its type is.
            self._log(
                request_id,
                status,
                stage="gpu",
                error=type(exc).__name__,
                title=title if infra else None,
                retry_after_s=retry_after,
            )
            return error_response(status, request_id=request_id, retry_after_seconds=retry_after)
        gpu_returned_at = time.perf_counter()

        try:
            output = self.deployment.finish(prepared, result["completion_ids"])
        except Exception as exc:
            self._log(
                request_id, HermesStatus.MODEL_ERROR, stage="finish", error=type(exc).__name__
            )
            return error_response(HermesStatus.MODEL_ERROR, request_id=request_id)
        finished_at = time.perf_counter()

        gpu_call_s = gpu_returned_at - prepared_at
        gpu_generate_s = float(result.get("gpu_generate_s") or 0.0)
        timings = {
            "prepare_s": round(prepared_at - started, 3),
            "gpu_call_s": round(gpu_call_s, 3),
            "gpu_generate_s": round(gpu_generate_s, 3),
            # Scheduling, queueing, worker start and weight transfer.
            "gpu_acquire_s": round(max(gpu_call_s - gpu_generate_s, 0.0), 3),
            "finish_s": round(finished_at - gpu_returned_at, 3),
            "total_s": round(finished_at - started, 3),
        }
        call_index = result.get("worker_call_index")
        diagnostics = {
            "cold_start": call_index == 1,
            "worker_call_index": call_index,
            "device": result.get("device"),
            "compute_capability": result.get("compute_capability"),
            "peak_vram_gib": round(int(result.get("peak_vram_bytes") or 0) / 1024**3, 2),
            "cuda_runtime": result.get("cuda_runtime"),
        }
        self._log(
            request_id,
            HermesStatus.READY,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            cold=diagnostics["cold_start"],
            gpu_generate_s=timings["gpu_generate_s"],
            total_s=timings["total_s"],
        )
        return ok_response(
            text=output.text,
            finish_reason=finish_reason(output.completion_tokens, config.max_new_tokens),
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            model=self.model_identity,
            request_id=request_id,
            timings=timings,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_VERSION_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "gradio",
    "spaces",
)


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    versions.update({name: package_version(name) for name in _VERSION_PACKAGES})
    return versions


def _memory_report() -> dict[str, float | None]:
    """Host RAM and this process's peak, for the startup log. Linux only."""
    report: dict[str, float | None] = {"total_gib": None, "available_gib": None}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                gib = int(value.split()[0]) / 1024**2
                report["total_gib" if key == "MemTotal" else "available_gib"] = round(gib, 1)
    except OSError:
        pass
    # ru_maxrss is KiB on Linux.
    report["process_peak_gib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1
    )
    return report


def load_space_deployment(
    record: Path | str,
    *,
    env: Mapping[str, str] | None = None,
    local_dir: Path | str | None = None,
) -> Any:
    """Fetch the private package at a pinned revision, verify it, and load it.

    Runs once, at import time in the Space, on the CPU under ZeroGPU's CUDA
    emulation: nothing here consumes GPU quota. Fails closed — an exception
    here stops the Space from starting at all.
    """
    source = os.environ if env is None else env
    repo = (source.get(ENV_PACKAGE_REPO) or "").strip()
    revision = (source.get(ENV_PACKAGE_REVISION) or "").strip()
    if not repo:
        raise ConfigError(
            f"{ENV_PACKAGE_REPO} is not set.",
            suggestions=["Set it as a Space secret: the private repo holding the package."],
        )
    if not _PINNED.match(revision):
        raise ConfigError(
            f"{ENV_PACKAGE_REVISION} must be a 40-character commit sha, got {revision!r}.",
            suggestions=[
                "Pin the exact package commit that upload_deployment_package.py printed.",
                "A branch name would let the served artifact change underneath the Space.",
            ],
        )

    # Read the identity first: a bad record should fail before any download.
    expected = load_expected_identity(record)

    from huggingface_hub import snapshot_download

    from kleos_models.serving.loader import load_deployment

    started = time.perf_counter()
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        repo_type="model",
        token=(source.get("HF_TOKEN") or None),
        local_dir=str(local_dir) if local_dir else None,
    )
    downloaded = time.perf_counter()
    try:
        deployment = load_deployment(
            path,
            verify=True,
            expected_identity=expected,
            # One device, explicitly. Under emulation 'auto' has nothing to
            # balance, and a fixed placement is easier to reason about.
            device_map="cuda:0",
            # Read the adapter file on the CPU. Left to itself PEFT sees the
            # emulated CUDA and reads straight onto it, which needs a real GPU
            # and failed the first ZeroGPU start ("No CUDA GPUs are
            # available"). The weights are then copied into the emulated model
            # like the rest of it; their values are the same either way.
            adapter_device="cpu",
        )
    except KleosError:
        logger.error("Hermes package failed verification or load; refusing to serve.")
        raise
    loaded = time.perf_counter()

    logger.info(
        "Hermes ready: %s %s adapter=%s… base=%s@%s compute_dtype=%s "
        "download_s=%.1f load_s=%.1f memory=%s versions=%s",
        deployment.manifest.model_name,
        deployment.manifest.model_version,
        deployment.manifest.adapter.weights_sha256[:16],
        deployment.manifest.base_model,
        deployment.manifest.base_revision[:12],
        deployment.manifest.runtime.compute_dtype,
        downloaded - started,
        loaded - downloaded,
        _memory_report(),
        runtime_versions(),
    )
    return deployment
