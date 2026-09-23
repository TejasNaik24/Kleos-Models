# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# KLEOS Hermes v0.0.6 — inference service for a generic NVIDIA GPU host.
#
#   docker build -f docker/hermes.Dockerfile -t kleos-hermes:v0.0.6 .
#
# WHAT IS IN THE IMAGE
#   Code, pinned dependencies, and the serving record
#   (configs/deployment/kleos_hermes_v006.yaml) that says which artifact this
#   image is allowed to serve.
#
# WHAT IS NOT
#   * The deployment package (adapter + frozen tokenizer + manifest, ~245 MB).
#     It is mounted read-only at /models/hermes-v0.0.6 at run time, so private
#     weights never enter the build context or the image.
#   * The base model (~24.5 GB). Downloaded at the pinned revision on first
#     start into a persistent volume at /cache/huggingface, reused afterwards.
#   * Secrets. HERMES_API_KEY and HF_TOKEN arrive at run time, as environment
#     variables or as *_FILE paths to mounted secret files.
#
# STARTUP (python -m kleos_models.serving.startup) fails closed, before any
# download, if the mounted package does not hash-match its manifest or is not
# the frozen Hermes artifact recorded in the serving record. See
# docs/deployment.md, "Container".
#
# STATUS (2026-09-23): built for linux/amd64 (under emulation, on Apple
# Silicon), and its preflight exercised with no GPU: it refuses a foreign
# package, accepts a matching one, and stops cleanly when no GPU is visible.
# NOT GPU-TESTED: the model has not been loaded or served inside this image.
# On the first GPU start, run scripts/hermes_smoke.py in the container to
# re-establish the 9/9 exact-match result (docs/deployment.md).
# ---------------------------------------------------------------------------

# CUDA 12.8 runtime on Ubuntu 24.04 (system Python 3.12), matching the
# torch 2.11.0+cu128 build the research run used. The digest pins the exact
# image, for linux/amd64 and linux/arm64 alike; the tag is for humans.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867

FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

LABEL org.opencontainers.image.title="kleos-hermes" \
      org.opencontainers.image.description="KLEOS Hermes v0.0.6 inference service (Mistral-Nemo-Instruct-2407 + frozen LoRA adapter)" \
      org.opencontainers.image.version="v0.0.6" \
      ai.kleos.base.model="mistralai/Mistral-Nemo-Instruct-2407" \
      ai.kleos.base.revision="04d8a90549d23fc6bd7f642064003592df51e9b3" \
      ai.kleos.adapter.sha256="dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668), so
# everything goes into a virtual environment.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

WORKDIR /app

# --- dependencies: rebuilt only when the pins change -------------------------
# torch first, from the CUDA 12.8 index, so the PyPI install below cannot swap
# in a different CUDA build. pip treats 2.11.0+cu128 as satisfying the
# torch==2.11.0 its dependents require, and leaves it alone.
COPY docker/requirements-hermes.txt /app/docker/requirements-hermes.txt
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==2.11.0+cu128" \
 && pip install -r /app/docker/requirements-hermes.txt \
 && pip freeze > /app/requirements.lock

# --- code ---------------------------------------------------------------------
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY scripts/_cli.py scripts/serve_hermes.py scripts/verify_deployment_package.py scripts/hermes_smoke.py /app/scripts/
COPY configs/deployment/kleos_hermes_v006.yaml /app/configs/deployment/kleos_hermes_v006.yaml
RUN pip install --no-deps /app

# --- runtime ------------------------------------------------------------------
# Non-root. The cache directory is created here with the right owner, so a
# fresh named volume mounted over it inherits that ownership.
RUN useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin hermes \
 && mkdir -p /cache/huggingface /models \
 && chown -R hermes:hermes /cache

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HERMES_PACKAGE_DIR=/models/hermes-v0.0.6 \
    HERMES_EXPECTED_DEPLOYMENT_CONFIG=/app/configs/deployment/kleos_hermes_v006.yaml \
    HERMES_EXIT_ON_LOAD_FAILURE=1 \
    HERMES_HOST=0.0.0.0 \
    HERMES_PORT=8000

USER hermes
EXPOSE 8000

# Healthy only once the model is loaded and verified. /health needs no
# credentials, so the probe never handles the API key. The long start period
# covers the first start, which downloads ~24.5 GB before the model can load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45m --retries=3 \
    CMD ["python", "-m", "kleos_models.serving.startup", "--healthcheck"]

ENTRYPOINT ["python", "-m", "kleos_models.serving.startup"]
CMD []
