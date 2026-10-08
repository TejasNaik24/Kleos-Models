"""Logos v0.0.2 on ZeroGPU: the same request path, under its own profile."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from tests.test_zerogpu_service import (
    KEY,
    MARKER,
    ONE,
    FakeBackend,
    FakeGPU,
    build_manifest,
    build_service,
)

from kleos_models.config import GenerationConfig
from kleos_models.errors import ConfigError
from kleos_models.inference.backends import GenerationOutput
from kleos_models.serving import zerogpu
from kleos_models.serving.loader import LoadedDeployment
from kleos_models.serving.profile import profile_from_record
from kleos_models.serving.zerogpu import (
    ZeroGPUService,
    ZeroGPUSettings,
    gpu_duration,
    gpu_duration_for,
)

LOGOS = profile_from_record(
    {
        "name": "kleos-logos",
        "version": "v0.0.2",
        "serving": {
            "env_prefix": "LOGOS",
            "key_header": "x-logos-key",
            "base_files": ["config.json"],
            "space_dir": "deploy/zerogpu-space-logos",
            "record_file": "logos_record.yaml",
            "reply": {"reasoning": True, "require_system_message": True},
        },
    }
)
AUTH = {"x-logos-key": KEY}
SYSTEM_FIRST = {
    "messages": [
        {"role": "system", "content": "Rank the items."},
        {"role": "user", "content": "Which first?"},
    ]
}


class ThinkingBackend(FakeBackend):
    """A backend whose completion had a thinking span."""

    def __init__(
        self,
        *,
        text: str = "1. Alpha. What decided it: impact.",
        finish_reason: str = "stop",
        **kwargs: Any,
    ) -> None:
        super().__init__(text=text, **kwargs)
        self.finish_reason = finish_reason

    def finish(self, prepared, completion_ids):
        output = super().finish(prepared, completion_ids)
        return GenerationOutput(
            text=output.text,
            reasoning=f"trace {MARKER}",
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=self.finish_reason,
        )


def logos_service(
    *, backend: FakeBackend | None = None, gpu: FakeGPU | None = None
) -> tuple[ZeroGPUService, FakeBackend, FakeGPU]:
    backend = backend or ThinkingBackend()
    gpu = gpu or FakeGPU()
    deployment = LoadedDeployment(
        manifest=build_manifest(max_new_tokens=1024),
        model_config=None,  # type: ignore[arg-type]  # not consulted
        backend=backend,
        package_dir=Path("."),
    )
    service = ZeroGPUService(
        deployment, ZeroGPUSettings(api_keys=(KEY,)), gpu_call=gpu, profile=LOGOS
    )
    return service, backend, gpu


class TestLogosReply:
    def test_the_reply_carries_the_trace_beside_the_answer(self):
        service, _, _ = logos_service()
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["status"] == "ready" and reply["contract_version"] == 2
        assert reply["reasoning"] == f"trace {MARKER}"
        assert reply["text"] == "1. Alpha. What decided it: impact."
        assert reply["finish_reason"] == "stop"

    def test_never_closed_thinking_is_marked_cut(self):
        service, _, _ = logos_service(backend=ThinkingBackend(text="", finish_reason="length"))
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["status"] == "ready"
        assert reply["finish_reason"] == "length" and reply["text"] == ""

    def test_a_full_budget_is_marked_cut(self):
        service, _, _ = logos_service(gpu=FakeGPU(completion=2000))
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["usage"]["completion_tokens"] == 1024
        assert reply["finish_reason"] == "length"

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": "Which first?"}],
            [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Rank."}],
            [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Rank."}],
        ],
    )
    def test_a_request_without_a_leading_system_message_never_reaches_the_gpu(self, messages):
        service, backend, gpu = logos_service()
        reply = service.generate({"messages": messages}, AUTH)
        assert reply["status"] == "invalid_request"
        assert "system message" in reply["message"]
        assert reply["contract_version"] == 2
        assert gpu.calls == 0 and backend.prepared == []

    def test_the_hermes_header_does_not_open_logos(self):
        service, _, gpu = logos_service()
        reply = service.generate(SYSTEM_FIRST, {"x-hermes-key": KEY})
        assert reply["status"] == "unauthorized"
        assert gpu.calls == 0

    def test_errors_speak_as_logos(self):
        service, _, _ = logos_service(gpu=FakeGPU(fail=RuntimeError("boom")))
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["status"] == "model_error"
        assert reply["message"] == "Logos could not generate a response."
        assert reply["contract_version"] == 2

    def test_status_reports_the_reasoning_reply_and_its_budget(self):
        service, _, _ = logos_service()
        status = service.status(AUTH)
        assert status["contract_version"] == 2 and status["reasoning"] is True
        assert status["limits"]["max_new_tokens"] == 1024

    def test_the_trace_never_reaches_the_logs(self, caplog):
        service, _, _ = logos_service()
        with caplog.at_level(logging.INFO, logger="kleos_models.serving.zerogpu"):
            reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["status"] == "ready"
        assert MARKER not in caplog.text


class TestLogosSettingsAndDuration:
    def test_settings_read_logos_names(self):
        settings = ZeroGPUSettings.from_env(
            {"LOGOS_API_KEY": "k", "LOGOS_MAX_NEW_TOKENS": "512"}, profile=LOGOS
        )
        assert settings.api_keys == ("k",) and settings.max_new_tokens == 512

    def test_a_hermes_key_does_not_configure_logos(self):
        with pytest.raises(ConfigError, match="LOGOS_API_KEY"):
            ZeroGPUSettings.from_env({"HERMES_API_KEY": "k"}, profile=LOGOS)

    def test_a_1024_token_budget_requests_the_60_second_cap(self, monkeypatch):
        for suffix in ("BASE_SECONDS", "TOKENS_PER_SECOND", "MIN_SECONDS", "MAX_SECONDS"):
            monkeypatch.delenv(f"LOGOS_GPU_{suffix}", raising=False)
        duration = gpu_duration_for(LOGOS)
        assert duration(None, GenerationConfig(max_new_tokens=1024)) == 60

    def test_logos_gpu_env_overrides_and_hermes_ignores_it(self, monkeypatch):
        for suffix in ("BASE_SECONDS", "TOKENS_PER_SECOND", "MIN_SECONDS", "MAX_SECONDS"):
            monkeypatch.delenv(f"HERMES_GPU_{suffix}", raising=False)
        monkeypatch.setenv("LOGOS_GPU_TOKENS_PER_SECOND", "100")
        assert gpu_duration_for(LOGOS)(None, GenerationConfig(max_new_tokens=1024)) == 21
        assert gpu_duration(None, GenerationConfig(max_new_tokens=512)) == 53


class TestHermesUnchanged:
    def test_hermes_replies_have_no_reasoning_and_version_1(self):
        service, _, _ = build_service()
        reply = service.generate(ONE, {zerogpu.KEY_HEADER: KEY})
        assert reply["status"] == "ready"
        assert reply["contract_version"] == 1 and "reasoning" not in reply
        status = service.status({zerogpu.KEY_HEADER: KEY})
        assert status["contract_version"] == 1 and "reasoning" not in status

    def test_hermes_still_accepts_a_user_only_request(self):
        service, _, _ = build_service()
        assert service.generate(ONE, {zerogpu.KEY_HEADER: KEY})["status"] == "ready"
