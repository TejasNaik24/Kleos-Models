---
title: KLEOS Hermes v0.0.6
emoji: 🧭
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Frozen KLEOS Hermes v0.0.6 inference API on ZeroGPU
# Startup downloads the verified package and quantizes 12B parameters on the
# CPU before serving; the default 30 minutes is too tight to rely on.
startup_duration_timeout: 1h
# The base model's Hugging Face shards and configs at the pinned revision, baked
# into the image at build time. Not consolidated.safetensors (a second 24.5 GB
# copy of the same weights) and not the tokenizer (Hermes uses its own frozen
# copy from the deployment package).
preload_from_hub:
  - mistralai/Mistral-Nemo-Instruct-2407 config.json,generation_config.json,model.safetensors.index.json,model-00001-of-00005.safetensors,model-00002-of-00005.safetensors,model-00003-of-00005.safetensors,model-00004-of-00005.safetensors,model-00005-of-00005.safetensors 04d8a90549d23fc6bd7f642064003592df51e9b3
---

# KLEOS Hermes v0.0.6

An inference API for the KLEOS backend. There is no chat interface here.

It serves the frozen Hermes v0.0.6 artifact: a LoRA adapter on
`mistralai/Mistral-Nemo-Instruct-2407` at a pinned revision, in 4-bit NF4 with
float16 compute and greedy decoding, exactly as it was evaluated. At startup
the Space downloads the deployment package at a pinned commit, verifies every
file hash and the full model identity, and refuses to start if anything
differs.

Endpoints (Gradio API, via `gradio_client`):

| Endpoint | Input | Output |
|---|---|---|
| `/generate` | `{"messages": [...], "max_new_tokens"?: int, "request_id"?: str}` | Hermes status contract |
| `/status` | none | identity, runtime and limits (no GPU used) |

Both require the `X-Hermes-Key` header. Requests carry no user, workspace or
tool data, and neither prompts nor responses are logged.

Source, tests and the full contract: the `kleos-models` repository,
`docs/deployment.md` and `docs/kleos-hermes-integration.md`.
