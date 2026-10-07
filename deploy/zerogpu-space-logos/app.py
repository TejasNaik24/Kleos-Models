# ---------------------------------------------------------------------------
# KLEOS Logos v0.0.2 on a free Hugging Face ZeroGPU Space.
#
# The same thin wiring as the Hermes Space: everything that decides behaviour
# lives in kleos_models.serving, installed from the pinned kleos-models commit.
# What differs is read from logos_record.yaml's `serving` profile: the LOGOS_*
# secrets, the x-logos-key header, the reasoning reply and the system-message
# rule.
#
# No `from __future__ import annotations` here: gr.api builds the endpoint
# schema from real type hints, and string annotations break it.
# ---------------------------------------------------------------------------

import spaces  # first: ZeroGPU must patch CUDA before torch is imported

from pathlib import Path

import gradio as gr

from kleos_models.logging_utils import configure_logging
from kleos_models.serving.profile import load_profile
from kleos_models.serving.zerogpu import (
    ZeroGPUService,
    ZeroGPUSettings,
    gpu_duration_for,
    load_space_deployment,
    run_gpu_step,
)

configure_logging()

RECORD = Path(__file__).with_name("logos_record.yaml")
PROFILE = load_profile(RECORD)
SETTINGS = ZeroGPUSettings.from_env(profile=PROFILE)
DEPLOYMENT = load_space_deployment(RECORD)


@spaces.GPU(duration=gpu_duration_for(PROFILE))
def generate_on_gpu(prepared, config):
    return run_gpu_step(DEPLOYMENT, prepared, config)


SERVICE = ZeroGPUService(DEPLOYMENT, SETTINGS, gpu_call=generate_on_gpu, profile=PROFILE)


def generate(request_body: dict, request: gr.Request) -> dict:
    """Generate one Logos response. Always returns the status contract."""
    return SERVICE.generate(request_body, dict(request.headers))


def status(request: gr.Request) -> dict:
    """Identity, runtime and limits. Uses no GPU."""
    return SERVICE.status(dict(request.headers))


with gr.Blocks(title=f"KLEOS {PROFILE.display_name}") as demo:
    gr.Markdown(f"KLEOS {PROFILE.display_name}: API only. See the README for the endpoints.")
    gr.api(generate, api_name="generate")
    gr.api(status, api_name="status")

demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(show_error=False, ssr_mode=False)
