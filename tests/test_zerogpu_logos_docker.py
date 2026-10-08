"""A thinking reply end to end on ZeroGPU's request path, with a real tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.test_backend_split import build_thinking_backend, ids_of
from tests.test_zerogpu_logos import AUTH, LOGOS, SYSTEM_FIRST
from tests.test_zerogpu_service import KEY, FakeGPU, build_manifest

from kleos_models.serving.loader import LoadedDeployment
from kleos_models.serving.zerogpu import ZeroGPUService, ZeroGPUSettings

pytestmark = [pytest.mark.requires_torch]


class ScriptedGPU(FakeGPU):
    """Returns the token ids of a fixed completion."""

    def __init__(self, completion_ids: list[int]) -> None:
        super().__init__()
        self.completion_ids = completion_ids

    def __call__(self, prepared, config):
        result = super().__call__(prepared, config)
        result["completion_ids"] = list(self.completion_ids)
        return result


def serve(completion: str) -> dict:
    backend = build_thinking_backend()
    backend.formatter = type(
        "Formatter",
        (),
        {"render_prompt": staticmethod(lambda messages: " ".join(m.content for m in messages))},
    )()
    deployment = LoadedDeployment(
        manifest=build_manifest(max_new_tokens=1024),
        model_config=None,  # type: ignore[arg-type]  # not consulted
        backend=backend,
        package_dir=Path("."),
    )
    service = ZeroGPUService(
        deployment,
        ZeroGPUSettings(api_keys=(KEY,)),
        gpu_call=ScriptedGPU(ids_of(backend, completion)),
        profile=LOGOS,
    )
    return service.generate(SYSTEM_FIRST, AUTH)


def test_the_trace_and_the_answer_arrive_separately():
    reply = serve("[THINK] the deadline decides it [/THINK] migration first </s>")
    assert reply["status"] == "ready" and reply["contract_version"] == 2
    assert reply["text"] == "migration first"
    assert reply["reasoning"] == "the deadline decides it"
    assert reply["finish_reason"] == "stop"


def test_thinking_that_never_closes_is_a_cut_reply_with_no_answer():
    reply = serve("[THINK] the deadline decides")
    assert reply["status"] == "ready"
    assert reply["text"] == ""
    assert reply["reasoning"] == "the deadline decides"
    assert reply["finish_reason"] == "length"
