# Serves Hermes from a Hugging Face ZeroGPU Space: the same `load_deployment` and
# `HuggingFaceBackend` as the Docker service, split so the GPU is held only for
# `generate_ids` (prepare and finish run on the CPU).
#
# ZeroGPU behaviour relied on (`spaces` 0.51.3):
#   * the model loads at import under CUDA emulation; weights reach a real GPU
#     only inside a decorated call;
#   * quota and queue checks run in the caller's process before any GPU work and
#     surface as gradio errors, classified here;
#   * an idle assigned worker is reused with weights on the GPU (warm); otherwise
#     a new worker is forked (cold);
#   * GPU time is charged to the calling Hugging Face account.
#
# Imports neither gradio nor spaces; `deploy/zerogpu-space/app.py` is the wiring.

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
from kleos_models.serving.profile import HERMES_PROFILE, ServingProfile, load_profile
from kleos_models.serving.status import (
    HermesStatus,
    classify_exception,
    error_response,
    finish_reason,
    ok_response,
    public_messages,
)

logger = get_logger("kleos_models.serving.zerogpu")

#: Shared-secret header; not `Authorization`, which the HF proxy uses for its own token.
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

#: The longest v0.0.6 dataset prompt is ~440 tokens; 2048 leaves headroom yet bounds cost.
DEFAULT_MAX_INPUT_TOKENS = 2048

_PINNED = re.compile(r"^[0-9a-f]{40}$")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() in _TRUTHY


@dataclass
class ZeroGPUSettings:
    """What the Space needs beyond the package. Read once, at startup."""

    api_keys: tuple[str, ...]
    allow_unauthenticated: bool = False
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    #: May only tighten the manifest's ceiling.
    max_new_tokens: int | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        profile: ServingProfile = HERMES_PROFILE,
    ) -> ZeroGPUSettings:
        """Read the model's settings; names come from its serving profile."""
        source = os.environ if env is None else env
        api_key_name = profile.env("API_KEY")
        keys = tuple(
            part.strip() for part in (source.get(api_key_name) or "").split(",") if part.strip()
        )
        allow = _flag(source.get(profile.env("ALLOW_UNAUTHENTICATED")))
        if not keys and not allow:
            raise ConfigError(
                f"{api_key_name} is not set.",
                suggestions=[
                    "Add it as a Space secret. The KLEOS backend sends it in the "
                    f"'{profile.key_header}' header.",
                    f"Without it, anyone who can reach the Space could use {profile.short_name}.",
                ],
            )
        max_input = source.get(profile.env("MAX_INPUT_TOKENS"))
        max_new = source.get(profile.env("MAX_NEW_TOKENS"))
        return cls(
            api_keys=keys,
            allow_unauthenticated=allow,
            max_input_tokens=int(max_input) if max_input else DEFAULT_MAX_INPUT_TOKENS,
            max_new_tokens=int(max_new) if max_new else None,
        )


#: Per-process call counter; a forked worker sees a new pid and restarts at 0 (call 1 is cold).
_WORKER: dict[str, int | None] = {"pid": None, "calls": 0}


def run_gpu_step(deployment: Any, prepared: Any, config: GenerationConfig) -> dict[str, Any]:
    """The only work done while a GPU is held: generate token ids, and measure."""
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


def gpu_duration_for(profile: ServingProfile) -> Callable[[Any, GenerationConfig], int]:
    """The GPU-duration rule for one model, read from its ``<PREFIX>_GPU_*`` secrets."""
    defaults = profile.gpu

    def duration(prepared: Any, config: GenerationConfig) -> int:
        base = _float_env(profile.env("GPU_BASE_SECONDS"), defaults.base_seconds)
        tokens_per_second = _float_env(
            profile.env("GPU_TOKENS_PER_SECOND"), defaults.tokens_per_second
        )
        minimum = _float_env(profile.env("GPU_MIN_SECONDS"), defaults.min_seconds)
        maximum = _float_env(profile.env("GPU_MAX_SECONDS"), defaults.max_seconds)
        estimate = base + config.max_new_tokens / max(tokens_per_second, 0.1)
        return int(min(max(math.ceil(estimate), minimum), maximum))

    return duration


#: Hermes' rule: the HERMES_GPU_* secrets, else the defaults.
gpu_duration = gpu_duration_for(HERMES_PROFILE)


class ZeroGPUService:
    """Authenticated, validated, quota-aware generation over a loaded deployment."""

    def __init__(
        self,
        deployment: Any,
        settings: ZeroGPUSettings,
        *,
        gpu_call: Callable[[Any, GenerationConfig], dict[str, Any]],
        profile: ServingProfile = HERMES_PROFILE,
    ) -> None:
        self.deployment = deployment
        self.settings = settings
        self.gpu_call = gpu_call
        self.profile = profile
        self.messages = public_messages(profile.short_name, profile.display_name)
        manifest = deployment.manifest
        self.model_identity = {
            "name": manifest.model_name,
            "version": manifest.model_version,
            "adapter_sha256": manifest.adapter.weights_sha256,
            "base_model": manifest.base_model,
            "base_revision": manifest.base_revision,
        }

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        if not self.settings.api_keys:
            return self.settings.allow_unauthenticated
        presented = {k.lower(): v for k, v in headers.items()}.get(self.profile.key_header)
        return key_matches(presented, self.settings.api_keys)

    def _error(self, status: HermesStatus, **kwargs: Any) -> dict[str, Any]:
        """A non-answer in this model's words and contract version."""
        return error_response(
            status,
            messages=self.messages,
            contract_version=self.profile.contract_version,
            **kwargs,
        )

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
        # Identifiers, counts and timings only: never prompt/response text, headers or keys.
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("generate request_id=%s status=%s %s", request_id, status.value, details)

    def status(self, headers: Mapping[str, str]) -> dict[str, Any]:
        """Readiness and identity. The model is loaded before the app serves, so
        answering at all means ready; `starting` is observed by the caller as
        the Space not responding yet."""
        if not self._authorized(headers):
            return self._error(HermesStatus.UNAUTHORIZED, request_id=None)
        manifest = self.deployment.manifest
        report: dict[str, Any] = {
            "contract_version": self.profile.contract_version,
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
        if self.profile.reasoning:
            report["reasoning"] = True
        return report

    def generate(self, payload: Any, headers: Mapping[str, str]) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = self._request_id(payload)

        if not self._authorized(headers):
            self._log(request_id, HermesStatus.UNAUTHORIZED)
            return self._error(HermesStatus.UNAUTHORIZED, request_id=request_id)

        # Schema (refuses unknown fields), then size bounds, all before any GPU work.
        try:
            request = GenerateRequest.model_validate(payload)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="schema")
            return self._error(
                HermesStatus.INVALID_REQUEST,
                request_id=request_id,
                message=f"Invalid request fields: {', '.join(fields) or 'body'}.",
            )
        try:
            validate_request(request, self.deployment.manifest.limits)
        except ValueError as exc:
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="size")
            return self._error(
                HermesStatus.INVALID_REQUEST, request_id=request_id, message=str(exc)
            )

        if self.profile.require_system_message and request.messages[0].role != "system":
            # Every evaluated prompt had a system message; without one a thinking model's
            # template may inject default instructions the model was never evaluated with.
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="system")
            return self._error(
                HermesStatus.INVALID_REQUEST,
                request_id=request_id,
                message="The first message must be a system message.",
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
            return self._error(HermesStatus.MODEL_ERROR, request_id=request_id)
        prepared_at = time.perf_counter()

        if prepared.prompt_length > self.settings.max_input_tokens:
            self._log(request_id, HermesStatus.INVALID_REQUEST, reason="tokens")
            return self._error(
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
            # ZeroGPU messages are quota/queue state, safe to log; other text may echo input.
            self._log(
                request_id,
                status,
                stage="gpu",
                error=type(exc).__name__,
                title=title if infra else None,
                retry_after_s=retry_after,
            )
            return self._error(status, request_id=request_id, retry_after_seconds=retry_after)
        gpu_returned_at = time.perf_counter()

        try:
            output = self.deployment.finish(prepared, result["completion_ids"])
        except Exception as exc:
            self._log(
                request_id, HermesStatus.MODEL_ERROR, stage="finish", error=type(exc).__name__
            )
            return self._error(HermesStatus.MODEL_ERROR, request_id=request_id)
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
            finish_reason=finish_reason(
                output.completion_tokens, config.max_new_tokens, output.finish_reason
            ),
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            model=self.model_identity,
            request_id=request_id,
            timings=timings,
            diagnostics=diagnostics,
            reasoning=output.reasoning if self.profile.reasoning else None,
            contract_version=self.profile.contract_version,
        )


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
    """Fetch the private package at a pinned revision, verify it, and load it."""
    source = os.environ if env is None else env
    profile = load_profile(record)
    repo_name, revision_name = profile.env("PACKAGE_REPO"), profile.env("PACKAGE_REVISION")
    repo = (source.get(repo_name) or "").strip()
    revision = (source.get(revision_name) or "").strip()
    if not repo:
        raise ConfigError(
            f"{repo_name} is not set.",
            suggestions=["Set it as a Space secret: the private repo holding the package."],
        )
    if not _PINNED.match(revision):
        raise ConfigError(
            f"{revision_name} must be a 40-character commit sha, got {revision!r}.",
            suggestions=[
                "Pin the exact package commit that upload_deployment_package.py printed.",
                "A branch name would let the served artifact change underneath the Space.",
            ],
        )

    # Fail on a bad record before any download.
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
            # Under emulation 'auto' has nothing to balance; pin one device.
            device_map="cuda:0",
            # Otherwise PEFT reads the adapter onto the emulated CUDA, which needs a real GPU
            # ("No CUDA GPUs are available"). The values are the same either way.
            adapter_device="cpu",
        )
    except KleosError:
        logger.error(
            "%s package failed verification or load; refusing to serve.", profile.short_name
        )
        raise
    loaded = time.perf_counter()

    logger.info(
        "%s ready: %s %s adapter=%s… base=%s@%s compute_dtype=%s "
        "download_s=%.1f load_s=%.1f memory=%s versions=%s",
        profile.short_name,
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
