# ---------------------------------------------------------------------------
# Container entrypoint for the Hermes inference service.
#
#     python -m kleos_models.serving.startup                # verify, then serve
#     python -m kleos_models.serving.startup --check-only   # verify, then exit
#     python -m kleos_models.serving.startup --healthcheck  # Docker HEALTHCHECK probe
#
# Startup sequence. Each step fails closed — exit 1, nothing served — before the
# next one runs, and every check that can fail cheaply runs before the 24.5 GB
# base-model download:
#
#   1. Secrets. HERMES_API_KEY (or HERMES_API_KEY_FILE) is required; HF_TOKEN
#      (or HF_TOKEN_FILE) is optional. Values are never printed.
#   2. Package integrity. Every file in the mounted package is re-hashed against
#      the package's own manifest, and adapter_config.json must pin the base.
#   3. Package identity. The package must be *the* frozen artifact: adapter
#      sha256, base revision, tokenizer files and flag, decoding settings — all
#      compared against the serving record baked into the image. An intact but
#      different package is refused here.
#   4. Model cache. HF_HOME must be writable; a warning if it is not on a mounted
#      volume, because the base model would then be re-downloaded whenever the
#      container is replaced.
#   5. GPU. A CUDA device must be visible (skippable with --no-gpu-check, for
#      verification on a machine without one).
#   6. Serve. uvicorn starts the app, which loads the base at the pinned
#      revision, the frozen tokenizer and the adapter, re-verifying as it goes.
#      If loading fails the process exits non-zero instead of idling.
#
# This module imports torch only inside the GPU check and uvicorn only when
# serving, so the preflight runs — and is tested — without either.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.compat import package_version
from kleos_models.errors import KleosError
from kleos_models.logging_utils import configure_logging, get_logger
from kleos_models.serving.app import (
    ENV_ALLOW_UNAUTHENTICATED,
    ENV_API_KEY,
    ENV_EXPECTED_DEPLOYMENT_CONFIG,
    ENV_PACKAGE_DIR,
    ServingSettings,
    create_app,
)
from kleos_models.serving.manifest import (
    DeploymentManifest,
    load_expected_identity,
    verify_identity,
    verify_package,
)

logger = get_logger("kleos_models.serving.startup")

ENV_API_KEY_FILE = f"{ENV_API_KEY}_FILE"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_TOKEN_FILE = "HF_TOKEN_FILE"
ENV_HOST = "HERMES_HOST"
ENV_PORT = "HERMES_PORT"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

#: Below this, a warning. Evaluation and the 9/9 smoke test ran on a 14.6 GiB
#: T4, so that much is known to be enough; the container is specified for
#: >= 16 GB hosts.
MIN_VRAM_GIB = 14.0

#: Packages whose versions decide whether the served model reproduces the
#: measured one. Printed at every start so a log shows exactly what ran.
REPORTED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "fastapi",
    "uvicorn",
)


class StartupError(Exception):
    """A preflight check failed. The message is safe to print: no secret values."""


@dataclass
class Preflight:
    """What the checks established. Safe to print; carries no secret values."""

    manifest: DeploymentManifest
    package_dir: Path
    expected_config: Path
    api_keys: tuple[str, ...] = field(repr=False)
    api_key_source: str
    hf_token_source: str | None
    cache_dir: Path
    cache_persistent: bool
    base_cached: bool
    gpu: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def resolve_secret(env: Mapping[str, str], name: str) -> tuple[str | None, str | None]:
    """Read a secret from ``NAME`` or from the file named by ``NAME_FILE``.

    ``_FILE`` is the usual container convention for secrets mounted by an
    orchestrator (e.g. ``/run/secrets/...``), which keeps the value out of
    ``docker inspect``. Setting both is refused: which one wins would be a guess.

    Returns:
        ``(value, source)`` where source is ``"env"``, ``"file"`` or ``None``.
    """
    file_name = f"{name}_FILE"
    direct = (env.get(name) or "").strip()
    path = (env.get(file_name) or "").strip()

    if direct and path:
        raise StartupError(f"Both {name} and {file_name} are set. Set exactly one.")
    if path:
        source = Path(path)
        if not source.is_file():
            raise StartupError(f"{file_name} points at {path}, which is not a readable file.")
        try:
            value = source.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise StartupError(f"{file_name} ({path}) could not be read: {exc.strerror}.") from exc
        if not value:
            raise StartupError(f"{file_name} ({path}) is empty.")
        return value, "file"
    if direct:
        return direct, "env"
    return None, None


def _on_mounted_volume(path: Path) -> bool:
    """Whether ``path`` or one of its parents (other than ``/``) is a mount point."""
    current = path.resolve()
    while current != current.parent:
        if os.path.ismount(current):
            return True
        current = current.parent
    return False


def check_cache(env: Mapping[str, str]) -> tuple[Path, bool]:
    """Require a writable Hugging Face cache; report whether it will persist."""
    raw = (env.get("HF_HOME") or "").strip()
    if not raw:
        raise StartupError(
            "HF_HOME is not set. The image sets it to /cache/huggingface; mount a "
            "persistent volume there so the base model is downloaded once."
        )
    cache = Path(raw)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".hermes-write-probe-"):
            pass
    except OSError as exc:
        raise StartupError(
            f"The model cache at {cache} is not writable ({exc.strerror}). The "
            "container runs as uid 10001: give that user write access to the "
            "mounted volume (e.g. `chown -R 10001:10001 <host dir>`)."
        ) from exc
    return cache, _on_mounted_volume(cache)


def base_snapshot_present(env: Mapping[str, str], base_model: str, revision: str) -> bool:
    """Whether the pinned base revision already has a snapshot in the cache.

    Presence of the directory means a previous start got at least part-way; the
    loader then skips or resumes the download. It is not a completeness check.
    """
    hub = Path(
        (env.get("HF_HUB_CACHE") or "").strip()
        or Path((env.get("HF_HOME") or "").strip() or "~/.cache/huggingface").expanduser() / "hub"
    )
    snapshot = hub / f"models--{base_model.replace('/', '--')}" / "snapshots" / revision
    return snapshot.is_dir()


def check_gpu(min_gib: float = MIN_VRAM_GIB) -> dict[str, Any]:
    """Require a visible CUDA device, and describe it."""
    try:
        import torch
    except ImportError as exc:
        raise StartupError(
            "torch is not installed in this environment, so no GPU can be used."
        ) from exc

    if not torch.cuda.is_available():
        raise StartupError(
            "No CUDA device is visible to this container. Hermes loads in 4-bit "
            "through bitsandbytes, which needs one. Run with the NVIDIA container "
            "runtime: `docker run --gpus all ...`, or a GPU reservation in compose "
            "(docker/compose.yaml does this)."
        )
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    info: dict[str, Any] = {
        "device": props.name,
        "count": torch.cuda.device_count(),
        "total_gib": round(total_gib, 1),
        "compute_capability": f"{props.major}.{props.minor}",
    }
    if total_gib < min_gib:
        info["warning"] = (
            f"{props.name} has {total_gib:.1f} GiB; Hermes was verified on 14.6 GiB "
            f"and this container is specified for >= 16 GB. Loading may run out of memory."
        )
    return info


def preflight(env: Mapping[str, str], *, check_gpu_device: bool = True) -> Preflight:
    """Run every check that can fail before the model is downloaded or loaded."""
    warnings: list[str] = []

    # 1. Secrets.
    api_key, api_key_source = resolve_secret(env, ENV_API_KEY)
    keys = tuple(part.strip() for part in (api_key or "").split(",") if part.strip())
    allow_unauthenticated = (env.get(ENV_ALLOW_UNAUTHENTICATED) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not keys and not allow_unauthenticated:
        raise StartupError(
            f"{ENV_API_KEY} (or {ENV_API_KEY_FILE}) is not set. The KLEOS backend "
            "authenticates with it; the service will not start without one."
        )
    _, hf_token_source = resolve_secret(env, ENV_HF_TOKEN)
    if hf_token_source is None:
        warnings.append(
            f"{ENV_HF_TOKEN} is not set. The base model is ungated, so this works, but "
            "anonymous downloads are rate-limited and the first start is slower."
        )

    # 2 & 3. Package integrity, then identity.
    package = Path((env.get(ENV_PACKAGE_DIR) or "").strip() or "/models/hermes-v0.0.6")
    if not package.is_dir():
        raise StartupError(
            f"No deployment package at {package}. Mount the package built by "
            "scripts/build_deployment_package.py there, read-only."
        )
    expected_raw = (env.get(ENV_EXPECTED_DEPLOYMENT_CONFIG) or "").strip()
    if not expected_raw:
        raise StartupError(
            f"{ENV_EXPECTED_DEPLOYMENT_CONFIG} is not set, so the package cannot be "
            "checked against the artifact this deployment was built to serve."
        )
    expected_config = Path(expected_raw)
    try:
        manifest = verify_package(package)
        verify_identity(manifest, load_expected_identity(expected_config))
    except KleosError as exc:
        raise StartupError(_describe(exc)) from exc

    # 4. Model cache.
    cache, persistent = check_cache(env)
    if not persistent:
        warnings.append(
            f"The model cache at {cache} is not on a mounted volume. The base model "
            "(~24.5 GB) will be downloaded again whenever this container is replaced."
        )
    cached = base_snapshot_present(env, manifest.base_model, manifest.base_revision)

    # 5. GPU.
    gpu = check_gpu() if check_gpu_device else None
    if gpu and gpu.get("warning"):
        warnings.append(gpu["warning"])

    return Preflight(
        manifest=manifest,
        package_dir=package,
        expected_config=expected_config,
        api_keys=keys,
        api_key_source=api_key_source or "none (unauthenticated)",
        hf_token_source=hf_token_source,
        cache_dir=cache,
        cache_persistent=persistent,
        base_cached=cached,
        gpu=gpu,
        warnings=warnings,
    )


def _describe(error: KleosError) -> str:
    """Flatten a KleosError, including its itemised problems, into one message."""
    lines = [str(error).splitlines()[0]]
    details = getattr(error, "details", None) or {}
    for problem in details.get("problems", []) or []:
        lines.append(f"  - {problem}")
    return "\n".join(lines)


def banner(result: Preflight) -> str:
    """Human-readable startup record. Contains identity, never secret values."""
    m = result.manifest
    lines = [
        "=" * 72,
        f"KLEOS {m.model_name} {m.model_version} — preflight passed",
        "=" * 72,
        f"  base          : {m.base_model}@{m.base_revision}",
        f"  adapter       : {m.adapter.weights_sha256}",
        f"  checkpoint    : {m.adapter.source_checkpoint} ({m.adapter.experiment_id})",
        f"  tokenizer     : {m.tokenizer.source}, fix_mistral_regex={m.tokenizer.fix_mistral_regex}",
        f"  dataset       : {m.dataset.version}",
        f"  package       : {result.package_dir} (verified against its manifest and "
        f"{result.expected_config.name})",
        f"  api key       : {len(result.api_keys)} key(s), from {result.api_key_source}",
        f"  HF token      : {'set, from ' + result.hf_token_source if result.hf_token_source else 'not set'}",
        f"  model cache   : {result.cache_dir} "
        f"({'mounted volume' if result.cache_persistent else 'NOT a mounted volume'})",
        f"  base in cache : {'yes — reusing it' if result.base_cached else 'no — first start will download ~24.5 GB'}",
    ]
    if result.gpu:
        gpu = result.gpu
        lines.append(
            f"  gpu           : {gpu['device']} x{gpu['count']}, {gpu['total_gib']} GiB, "
            f"cc {gpu['compute_capability']}"
        )
    else:
        lines.append("  gpu           : not checked (--no-gpu-check)")
    versions = ", ".join(f"{name} {package_version(name) or '-'}" for name in REPORTED_PACKAGES)
    lines.append(f"  libraries     : {versions}")
    for warning in result.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def healthcheck(env: Mapping[str, str], *, timeout: float = 4.0) -> int:
    """Docker HEALTHCHECK probe: 0 only when the service reports ready.

    Uses the unauthenticated /health endpoint, so the probe never needs the API
    key. Ready means the model is loaded and verified.
    """
    port = int((env.get(ENV_PORT) or "").strip() or DEFAULT_PORT)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as reply:
            body = json.loads(reply.read().decode("utf-8"))
    except Exception:
        return 1
    return 0 if body.get("ready") is True else 1


def serve(result: Preflight, env: Mapping[str, str], *, host: str, port: int) -> int:
    """Hand over to uvicorn. The app loads and re-verifies the model itself."""
    import uvicorn

    # The API key may have come from a file; pass it on without exporting it.
    # Serve exactly what preflight verified: same package, same identity record.
    merged = dict(env)
    merged[ENV_API_KEY] = ",".join(result.api_keys)
    merged[ENV_PACKAGE_DIR] = str(result.package_dir)
    merged[ENV_EXPECTED_DEPLOYMENT_CONFIG] = str(result.expected_config)
    settings = ServingSettings.from_env(merged)
    settings.exit_on_load_failure = True

    # huggingface_hub reads HF_TOKEN from the process environment.
    token, source = resolve_secret(env, ENV_HF_TOKEN)
    if source == "file" and token:
        os.environ[ENV_HF_TOKEN] = token

    app = create_app(settings)
    # One worker: each loads its own copy of the model, and one GPU holds one.
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=30,
    )
    return 0


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kleos_models.serving.startup",
        description="Verify the Hermes deployment package, then serve it.",
    )
    parser.add_argument("--check-only", action="store_true", help="Run every check, then exit.")
    parser.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip the CUDA check (verification on a machine without a GPU).",
    )
    parser.add_argument("--healthcheck", action="store_true", help="Probe /health and exit.")
    parser.add_argument("--host", help=f"Bind address (default ${ENV_HOST} or {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, help=f"Port (default ${ENV_PORT} or {DEFAULT_PORT}).")
    args = parser.parse_args(argv)
    source = os.environ if env is None else env

    if args.healthcheck:
        return healthcheck(source)

    configure_logging()
    try:
        result = preflight(source, check_gpu_device=not args.no_gpu_check)
    except StartupError as exc:
        print(f"\n✗ Hermes will not start.\n{exc}\n", file=sys.stderr)
        return 1

    print(banner(result), flush=True)
    if args.check_only:
        print("\n✓ Preflight passed. Not serving (--check-only).\n", flush=True)
        return 0

    host = args.host or (source.get(ENV_HOST) or "").strip() or DEFAULT_HOST
    port = args.port or int((source.get(ENV_PORT) or "").strip() or DEFAULT_PORT)
    print(f"\n  Serving on {host}:{port}. Loading the model now…\n", flush=True)
    return serve(result, source, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
