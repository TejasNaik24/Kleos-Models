---
title: KLEOS Logos v0.0.2
emoji: 🧭
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Frozen KLEOS Logos v0.0.2 inference API on ZeroGPU
# Startup downloads the verified package and quantizes the 14B text tower on the
# CPU before serving; the default 30 minutes is too tight to rely on.
startup_duration_timeout: 1h
# The base model's Hugging Face shards and configs at the pinned revision, baked
# into the image at build time. Not the tokenizer: Logos uses its own frozen
# copy from the deployment package. The shards also hold the vision tower,
# which the text-only view leaves on disk.
preload_from_hub:
  - mistralai/Ministral-3-14B-Reasoning-2512 config.json,generation_config.json,model.safetensors.index.json,model-00001-of-00006.safetensors,model-00002-of-00006.safetensors,model-00003-of-00006.safetensors,model-00004-of-00006.safetensors,model-00005-of-00006.safetensors,model-00006-of-00006.safetensors 51f9210f3cd20f3452a80d5819d15dc61cc50630
---

# KLEOS Logos v0.0.2

An inference API for the KLEOS backend. There is no chat interface here.

It serves the frozen Logos v0.0.2 artifact: a LoRA adapter on the text tower of
`mistralai/Ministral-3-14B-Reasoning-2512` at a pinned revision, in 4-bit NF4
with float16 compute and greedy decoding, exactly as it was evaluated. At
startup the Space downloads the deployment package at a pinned commit, verifies
every file hash and the full model identity, and refuses to start if anything
differs.

Logos always thinks before it answers. A `/generate` reply carries the answer in
`text` and the thinking in `reasoning` (contract version 2). `finish_reason` is
`length` when the token budget filled or the thinking never closed; treat such a
reply as cut. A request must start with a system message, as every prompt Logos
was trained and evaluated on did.

Endpoints (Gradio API, via `gradio_client`):

| Endpoint | Input | Output |
|---|---|---|
| `/generate` | `{"messages": [...], "max_new_tokens"?: int, "request_id"?: str}` | status contract, version 2 |
| `/status` | none | identity, runtime and limits (no GPU used) |

Both require the `X-Logos-Key` header. Requests carry no user, workspace or
tool data, and neither prompts, answers nor thinking traces are logged.

Source, tests and the full contract: the `kleos-models` repository,
`docs/deployment.md` and `docs/kleos-hermes-integration.md`.
