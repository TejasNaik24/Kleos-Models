# ---------------------------------------------------------------------------
# KLEOS Hermes v0.0.6 on a free Hugging Face ZeroGPU Space.
#
# Thin wiring, deliberately. Everything that decides behaviour — package and
# identity verification, authentication, request bounds, the quota-aware status
# contract — lives in kleos_models.serving.zerogpu, installed from the pinned
# kleos-models commit in requirements.txt and tested there. This file only
# connects it to Gradio and ZeroGPU.
#
# No chat UI: this Space is an API for the KLEOS backend, not a demo.
#
# No `from __future__ import annotations` here: gr.api builds the endpoint
# schema from real type hints, and string annotations break it.
# ---------------------------------------------------------------------------

import spaces  # first: ZeroGPU must patch CUDA before torch is imported

from pathlib import Path

import gradio as gr

from kleos_models.serving.zerogpu import (
    ZeroGPUService,
    ZeroGPUSettings,
    gpu_duration,
    load_space_deployment,
    run_gpu_step,
)

# Both fail closed. Without HERMES_API_KEY, or with a package that is not the
# frozen artifact this record describes, the Space does not start.
SETTINGS = ZeroGPUSettings.from_env()
# Runs once, on the CPU, under ZeroGPU's CUDA emulation: consumes no GPU quota.
DEPLOYMENT = load_space_deployment(Path(__file__).with_name("hermes_record.yaml"))


@spaces.GPU(duration=gpu_duration)
def generate_on_gpu(prepared, config):
    # The only step that holds a GPU. Rendering, tokenizing, validation and
    # decoding all happen outside it.
    return run_gpu_step(DEPLOYMENT, prepared, config)


SERVICE = ZeroGPUService(DEPLOYMENT, SETTINGS, gpu_call=generate_on_gpu)


def generate(request_body: dict, request: gr.Request) -> dict:
    """Generate one Hermes response. Always returns the status contract."""
    return SERVICE.generate(request_body, dict(request.headers))


def status(request: gr.Request) -> dict:
    """Identity, runtime and limits. Uses no GPU."""
    return SERVICE.status(dict(request.headers))


with gr.Blocks(title="KLEOS Hermes v0.0.6") as demo:
    gr.Markdown("KLEOS Hermes v0.0.6 — API only. See the README for the endpoints.")
    gr.api(generate, api_name="generate")
    gr.api(status, api_name="status")

# Every call goes through the queue; with the queue on, Gradio refuses direct
# REST calls that would skip it. One generation at a time, and a short queue: a
# burst beyond it is refused at once (`queue_unavailable` for KLEOS) instead of
# waiting on a GPU it may not get. /status has its own concurrency slot, so it
# never waits behind a generation.
demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(show_error=False, ssr_mode=False)
