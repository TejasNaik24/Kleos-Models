# Logos v0.0.2 Serving (Sub-project 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (the owner chose Native) to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Kleos-Models can package, stage and smoke-test KLEOS Logos v0.0.2 on its own private ZeroGPU Space, served exactly as evaluated, while Hermes stays byte-for-byte unchanged.

**Architecture:**
- **Profile.** An optional `serving` block in each deployment record becomes a `ServingProfile`. When the block is absent, every default equals today's Hermes constant.
- **Who reads it:**
  - the ZeroGPU service, settings, startup, status messages, client, Space staging and scripts read names and options from the profile;
  - the Logos profile turns on the v2 reply (`reasoning`), the backend-aware `finish_reason` and the system-message rule.
- **The Logos Space** gets its own folder and record.

**Tech stack:**
- Python 3.12, pydantic 2, pytest, ruff, mypy, PyYAML; host `.venv/bin/*`.
- torch tests run in the Docker image `kleos-hermes:v0.0.6`.

**Spec:** `docs/superpowers/specs/2026-10-06-logos-v002-serving-design.md` (approved 2026-10-06). Read it with this plan.

## Global Constraints

- **Version control.** Never commit or push; the owner does. A "Checkpoint" step reports test results only.
- **Never create, upload to, or change** a Hugging Face repo or Space. The owner runs every remote step (Task 10).
- **No credentials** in any file, test, fixture or command line. Tokens come from the owner's `hf auth login` store or Space secrets.
- **Hermes is byte-for-byte unchanged.** Every existing test passes unmodified:
  - `deploy/zerogpu-space/*` and `configs/deployment/kleos_hermes_v006.yaml` are not edited;
  - Hermes' replies keep `contract_version: 1` and gain no `reasoning` key;
  - its env names stay `HERMES_*`, its header `x-hermes-key`, and its log lines and public messages identical.
- **What the model generates does not change:** greedy decoding, 1,024 tokens, repetition penalty 1.0, no stop strings.
- **No Qwen** anywhere in new text.
- **Repo style:** module docstrings and comments as in the surrounding serving code. Tests use `from __future__ import annotations` and `TestXxx` classes with sentence-style names.
- **Commands:**
  - `.venv/bin/pytest -q -p no:cacheprovider`
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format --check .`
  - `.venv/bin/mypy src`
  - Docker: `docker run --rm --platform linux/amd64 -u 0 -e HF_HUB_OFFLINE=1 -v "$PWD":/work:ro -w /work --entrypoint bash kleos-hermes:v0.0.6 -c 'pip install -q pytest==9.1.1 nbformat 2>/dev/null; pytest -p no:cacheprovider -q <tests>'`
- **Baseline before Task 1:** 1493 passed, 46 skipped, 1 xfailed (host).

## Review Focus

1. **A Hermes code path changes silently,** in a name, message, reply key, staged file, log line or GPU duration. Expected: identical. Pinned by Task 1's Hermes-default test, Task 2's `PUBLIC_MESSAGES` equality, and the untouched Hermes tests.
2. **The system message isn't first.** A Logos request whose system message is missing, second, or whose first message is a user turn. Expected: `invalid_request` with no GPU call. Pinned in Task 3.
3. **The trace reaches a log line or `text`.** Expected: logs carry counts only, and `text` is the answer alone. Pinned in Task 3 (log test with a planted marker in the trace) and its Docker test.
4. **A client refuses a valid v2 reply, or loses the trace.** On the HTTP path too. Expected: v1 and v2 both pass `validate_contract`, and `reasoning` survives both transports. Pinned in Task 4.
5. **The fill script writes a wrong or overwritten value.** For example, the best checkpoint is not `checkpoint-175`, or a field already holds a value. Expected: the script refuses. Pinned in Task 5.

---

### Task 1: `ServingProfile`, read from a deployment record

**Files:**
- Create: `src/kleos_models/serving/profile.py`
- Test: `tests/test_serving_profile.py`

**Interfaces (produced):**

```python
HERMES_BASE_FILES: tuple[str, ...]   # today's space.BASE_PRELOAD_FILES, in order
@dataclass(frozen=True) class GpuDefaults: base_seconds=10.0; tokens_per_second=12.0; min_seconds=15.0; max_seconds=60.0
@dataclass(frozen=True) class ServingProfile:
    short_name: str; display_name: str
    env_prefix: str = "HERMES"; key_header: str = "x-hermes-key"
    space_dir: str = "deploy/zerogpu-space"; record_file: str = "hermes_record.yaml"
    research_report: str = "docs/experiments/kleos-v006-mistralnemo12b-run1-report.md"
    base_files: tuple[str, ...] = HERMES_BASE_FILES
    reasoning: bool = False; require_system_message: bool = False; space_only: bool = False
    gpu: GpuDefaults = GpuDefaults()
    def env(self, suffix: str) -> str            # f"{env_prefix}_{suffix}"
    @property contract_version -> int            # 2 if reasoning else 1
def profile_from_record(spec: Mapping[str, Any]) -> ServingProfile   # spec = the record's `deployment:` mapping
def load_profile(record_path: Path | str) -> ServingProfile
HERMES_PROFILE: ServingProfile                    # profile_from_record of a record with no `serving` block, name kleos-hermes, version v0.0.6
```

**Rules:**
- With no `serving` block, `short_name` comes from `name` (strip `kleos-`, capitalise; `kleos-hermes` gives `Hermes`), and `display_name` is `f"{short_name} {version}"`, i.e. `Hermes v0.0.6`.
- **Allowed keys** in `serving`: `short_name`, `display_name`, `env_prefix`, `key_header`, `space_dir`, `record_file`, `research_report`, `base_files`, `space_only`, `reply` (with `reasoning` and `require_system_message`) and `gpu` (with the four fields).
- **Anything else is refused:**
  - an unknown key, including under `reply`/`gpu`, raises `ConfigError` naming it;
  - `env_prefix` must match `^[A-Z][A-Z0-9]*$`;
  - `key_header` must match `^x-[a-z0-9-]+$`;
  - `record_file` must end `.yaml` with no path separator;
  - `base_files` must be a non-empty list of bare file names.

- [ ] **Step 1: Write the failing tests**

```python
"""A model's serving names and options come from its deployment record."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kleos_models.errors import ConfigError
from kleos_models.serving import space, zerogpu
from kleos_models.serving.profile import (
    HERMES_PROFILE,
    GpuDefaults,
    load_profile,
    profile_from_record,
)

ROOT = Path(__file__).resolve().parents[1]
HERMES_RECORD = ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def logos_spec(**serving) -> dict:
    block = {
        "display_name": "Logos v0.0.2",
        "env_prefix": "LOGOS",
        "key_header": "x-logos-key",
        "space_dir": "deploy/zerogpu-space-logos",
        "record_file": "logos_record.yaml",
        "base_files": ["config.json"],
        "space_only": True,
        "reply": {"reasoning": True, "require_system_message": True},
        **serving,
    }
    return {"name": "kleos-logos", "version": "v0.0.2", "serving": block}


class TestHermesDefaults:
    def test_the_hermes_record_resolves_to_todays_constants(self):
        profile = load_profile(HERMES_RECORD)
        assert profile == HERMES_PROFILE
        assert profile.short_name == "Hermes" and profile.display_name == "Hermes v0.0.6"
        assert profile.key_header == zerogpu.KEY_HEADER
        assert profile.env("API_KEY") == zerogpu.ENV_API_KEY
        assert profile.env("PACKAGE_REPO") == zerogpu.ENV_PACKAGE_REPO
        assert profile.env("GPU_MAX_SECONDS") == zerogpu.ENV_GPU_MAX_SECONDS
        assert profile.space_dir == str(space.SPACE_SOURCE_DIR)
        assert profile.record_file == space.RECORD_NAME
        assert frozenset(profile.base_files) == space.BASE_PRELOAD_FILES
        assert profile.reasoning is False and profile.require_system_message is False
        assert profile.contract_version == 1
        assert profile.gpu == GpuDefaults()


class TestLogosProfile:
    def test_a_serving_block_sets_every_name(self):
        profile = profile_from_record(logos_spec())
        assert profile.short_name == "Logos"
        assert profile.display_name == "Logos v0.0.2"
        assert profile.env("API_KEY") == "LOGOS_API_KEY"
        assert profile.key_header == "x-logos-key"
        assert profile.reasoning and profile.require_system_message and profile.space_only
        assert profile.contract_version == 2

    def test_gpu_defaults_can_be_set(self):
        profile = profile_from_record(logos_spec(gpu={"tokens_per_second": 20}))
        assert profile.gpu.tokens_per_second == 20 and profile.gpu.max_seconds == 60

    @pytest.mark.parametrize(
        ("serving", "message"),
        [
            ({"colour": "blue"}, "colour"),
            ({"reply": {"reasoning": True, "stream": True}}, "stream"),
            ({"gpu": {"seconds": 5}}, "seconds"),
            ({"env_prefix": "logos"}, "env_prefix"),
            ({"key_header": "X-Logos"}, "key_header"),
            ({"record_file": "../x.yaml"}, "record_file"),
            ({"base_files": []}, "base_files"),
        ],
    )
    def test_a_bad_serving_block_is_refused(self, serving, message):
        with pytest.raises(ConfigError, match=message):
            profile_from_record(logos_spec(**serving))

    def test_load_profile_reads_the_deployment_mapping(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump({"deployment": logos_spec()}), encoding="utf-8")
        assert load_profile(path).short_name == "Logos"
```

- [ ] **Step 2: Run them.** `.venv/bin/pytest tests/test_serving_profile.py -q -p no:cacheprovider`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `profile.py`.** Use the interfaces and rules above, in the module style of `serving/space.py`: a docstring, `from __future__ import annotations`, and `ConfigError` with `suggestions`. `HERMES_BASE_FILES` keeps the same order as `space.BASE_PRELOAD_FILES`: the three config files, then `model-0000N-of-00005.safetensors` for N = 1–5. `HERMES_PROFILE = profile_from_record({"name": "kleos-hermes", "version": "v0.0.6"})`.
- [ ] **Step 4: Run.** `.venv/bin/pytest tests/test_serving_profile.py -q -p no:cacheprovider`, then the full suite. Expected: all pass (1493 + 13).
- [ ] **Step 5: Checkpoint.**

---

### Task 2: Status messages per model, and the v2 reply

**Files:**
- Modify: `src/kleos_models/serving/status.py`: `CONTRACT_VERSION` (:22), `PUBLIC_MESSAGES` (:47-59), `ok_response` (:67-91), `finish_reason` (:94-102), `error_response` (:105-127)
- Test: `tests/test_status_contract.py`, adding class `TestPerModelAndV2`

**Interfaces (produced):**

```python
CONTRACT_VERSION = 1                        # unchanged
REASONING_CONTRACT_VERSION = 2
SUPPORTED_CONTRACT_VERSIONS = (1, 2)
def public_messages(short_name: str, display_name: str) -> dict[HermesStatus, str]
PUBLIC_MESSAGES = public_messages("Hermes", "Hermes v0.0.6")       # must equal today's dict exactly
def ok_response(*, text, finish_reason, prompt_tokens, completion_tokens, model, request_id,
                timings=None, diagnostics=None, reasoning: str | None = None,
                contract_version: int = CONTRACT_VERSION) -> dict  # `reasoning` key only when contract_version == 2
def finish_reason(completion_tokens: int, max_new_tokens: int, backend_reason: str = "stop") -> str
    # "length" if completion_tokens >= max_new_tokens or backend_reason == "length" else "stop"
def error_response(status, *, request_id, message=None, retry_after_seconds=None,
                   messages: Mapping[HermesStatus, str] | None = None,
                   contract_version: int = CONTRACT_VERSION) -> dict
```

**Possessive:** `short_name + ("'" if short_name.endswith("s") else "'s")`. Both "Hermes'" and "Logos'" end with `s'`.

- [ ] **Step 1: Write the failing tests**

```python
class TestPerModelAndV2:
    def test_hermes_messages_are_exactly_todays(self):
        assert public_messages("Hermes", "Hermes v0.0.6") == {
            HermesStatus.READY: "Hermes v0.0.6 is available.",
            HermesStatus.STARTING: "Hermes is starting a free GPU worker.",
            HermesStatus.QUOTA_EXHAUSTED: "Hermes' free GPU quota is currently exhausted. Please try again later.",
            HermesStatus.QUEUE_UNAVAILABLE: "Hermes is temporarily busy. Try again later or continue with the default model.",
            HermesStatus.MODEL_ERROR: "Hermes could not generate a response.",
            HermesStatus.INVALID_REQUEST: "The request to Hermes was not valid.",
            HermesStatus.UNAUTHORIZED: "unauthorized",
            HermesStatus.DISABLED: "Hermes is currently unavailable.",
        }
        assert PUBLIC_MESSAGES == public_messages("Hermes", "Hermes v0.0.6")

    def test_logos_messages_name_logos(self):
        messages = public_messages("Logos", "Logos v0.0.2")
        assert messages[HermesStatus.READY] == "Logos v0.0.2 is available."
        assert all("Hermes" not in text for text in messages.values())

    def test_a_v1_reply_has_no_reasoning_key(self):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
        )
        assert reply["contract_version"] == 1 and "reasoning" not in reply

    def test_a_v2_reply_carries_the_trace(self):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning="t",
            contract_version=2,
        )
        assert reply["contract_version"] == 2 and reply["reasoning"] == "t"
        none = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning=None,
            contract_version=2,
        )
        assert "reasoning" in none and none["reasoning"] is None

    @pytest.mark.parametrize(
        ("tokens", "budget", "backend", "expected"),
        [
            (10, 1024, "stop", "stop"),
            (1024, 1024, "stop", "length"),
            (200, 1024, "length", "length"),
        ],
    )
    def test_finish_reason_honours_the_backend(self, tokens, budget, backend, expected):
        assert finish_reason(tokens, budget, backend) == expected

    def test_error_responses_take_per_model_messages_and_version(self):
        messages = public_messages("Logos", "Logos v0.0.2")
        reply = error_response(
            HermesStatus.DISABLED, request_id="r", messages=messages, contract_version=2
        )
        assert (
            reply["message"] == "Logos is currently unavailable." and reply["contract_version"] == 2
        )
```

  Import `public_messages` with the existing imports.

- [ ] **Step 2: Run.** `.venv/bin/pytest tests/test_status_contract.py -q -p no:cacheprovider`. Expected: the new tests FAIL; the existing ones pass.
- [ ] **Step 3: Implement** the interfaces above. Build `PUBLIC_MESSAGES` from `public_messages`. `ok_response` adds `"reasoning": reasoning` after `"text"` only when `contract_version == REASONING_CONTRACT_VERSION`. `error_response` uses `(messages or PUBLIC_MESSAGES)[status]`.
- [ ] **Step 4: Run.** `.venv/bin/pytest tests/test_status_contract.py -q -p no:cacheprovider`, then the full suite. Expected: all pass.
- [ ] **Step 5: Checkpoint.**

---

### Task 3: The ZeroGPU service, settings, GPU duration and startup take a profile

**Files:**
- Modify:
  - `src/kleos_models/serving/zerogpu.py`: settings :95-125, `gpu_duration` :178-192, `ZeroGPUService` :200-393, `load_space_deployment` :437-516
  - `src/kleos_models/serving/loader.py:89`: `finish_reason` with the backend's reason
- Test:
  - `tests/test_zerogpu_logos.py` (new, host)
  - `tests/test_zerogpu_logos_docker.py` (new, `requires_torch`)

**Interfaces:**
- Consumes: `ServingProfile`, `HERMES_PROFILE` and `load_profile` (Task 1); `public_messages`, `finish_reason(…, backend_reason)`, `ok_response(…, reasoning, contract_version)` and `error_response(…, messages, contract_version)` (Task 2).
- Produces:

```python
ZeroGPUSettings.from_env(env=None, *, profile: ServingProfile = HERMES_PROFILE)   # names from profile.env(...)
ZeroGPUService(deployment, settings, *, gpu_call, profile: ServingProfile = HERMES_PROFILE)
def gpu_duration_for(profile: ServingProfile) -> Callable[[Any, GenerationConfig], int]
gpu_duration = gpu_duration_for(HERMES_PROFILE)    # same name, same behaviour, same HERMES_GPU_* env
load_space_deployment(record, *, env=None, local_dir=None)   # profile = load_profile(record): env names and log lines from it
```

**Service behaviour:**
- **The key header** is `profile.key_header`.
- **Error replies** use `public_messages(profile.short_name, profile.display_name)` and `profile.contract_version`.
- **The system-message rule.** After the size validation and before `prepare`: when `profile.require_system_message` is set and the first message is not a system message, return `invalid_request` with the message "The first message must be a system message.". Log `reason="system"`. No GPU call is made.
- **Success:**
  - `finish_reason(output.completion_tokens, config.max_new_tokens, output.finish_reason)`;
  - `ok_response(..., reasoning=output.reasoning, contract_version=profile.contract_version)`, where `reasoning` is passed only for reasoning profiles;
  - the log line is unchanged: counts only, never the trace.
- **`/status`** uses `contract_version` from the profile. For reasoning profiles it adds `"reasoning": True`.
- **Startup:** `"%s ready: ..."` with `profile.short_name` (`Hermes ready:` unchanged), and `"%s package failed verification or load; refusing to serve."`.

- [ ] **Step 1: Write the failing host tests** (`tests/test_zerogpu_logos.py`). Reuse the helpers `build_manifest`, `FakeBackend`, `FakeGPU`, `KEY` and `MARKER` by importing them from `tests.test_zerogpu_service`. Add a `ThinkingBackend(FakeBackend)` whose `finish` returns `GenerationOutput(text=…, reasoning=f"trace {MARKER}", finish_reason=…)`.

```python
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
        {"role": "system", "content": "Rank."},
        {"role": "user", "content": "Which first?"},
    ]
}


def logos_service(*, backend=None, gpu=None, **limits):
    backend = backend or ThinkingBackend()
    gpu = gpu or FakeGPU()
    deployment = LoadedDeployment(
        manifest=build_manifest(**{"max_new_tokens": 1024, **limits}),
        model_config=None,
        backend=backend,
        package_dir=Path("."),
    )
    return (
        ZeroGPUService(deployment, ZeroGPUSettings(api_keys=(KEY,)), gpu_call=gpu, profile=LOGOS),
        backend,
        gpu,
    )


class TestLogosReply:
    def test_the_reply_carries_the_trace_beside_the_answer(self):
        service, _, _ = logos_service()
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["contract_version"] == 2 and reply["status"] == "ready"
        assert reply["reasoning"].startswith("trace") and MARKER not in reply["text"]

    def test_never_closed_thinking_is_marked_cut(self):
        service, _, _ = logos_service(backend=ThinkingBackend(text="", finish_reason="length"))
        reply = service.generate(SYSTEM_FIRST, AUTH)
        assert reply["finish_reason"] == "length" and reply["text"] == ""

    def test_a_full_budget_is_marked_cut(self):
        service, _, _ = logos_service(gpu=FakeGPU(completion=2000))
        assert service.generate(SYSTEM_FIRST, AUTH)["finish_reason"] == "length"

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": "Which first?"}],
            [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Rank."}],
        ],
    )
    def test_a_request_without_a_leading_system_message_never_reaches_the_gpu(self, messages):
        service, backend, gpu = logos_service()
        reply = service.generate({"messages": messages}, AUTH)
        assert reply["status"] == "invalid_request" and "system message" in reply["message"]
        assert gpu.calls == 0 and backend.prepared == []

    def test_the_hermes_header_is_not_the_logos_key(self):
        service, _, gpu = logos_service()
        assert service.generate(SYSTEM_FIRST, {"x-hermes-key": KEY})["status"] == "unauthorized"
        assert gpu.calls == 0

    def test_errors_and_status_speak_as_logos_v2(self):
        service, _, _ = logos_service()
        assert service.generate(SYSTEM_FIRST, {})["message"] == "unauthorized"
        status = service.status(AUTH)
        assert status["contract_version"] == 2 and status["reasoning"] is True
        assert status["limits"]["max_new_tokens"] == 1024

    def test_the_trace_never_reaches_the_logs(self, caplog):
        service, _, _ = logos_service()
        with caplog.at_level(logging.INFO, logger="kleos_models.serving.zerogpu"):
            service.generate(SYSTEM_FIRST, AUTH)
        assert MARKER not in caplog.text


class TestLogosSettingsAndDuration:
    def test_settings_read_logos_names(self):
        settings = ZeroGPUSettings.from_env(
            {"LOGOS_API_KEY": "k", "LOGOS_MAX_NEW_TOKENS": "512"}, profile=LOGOS
        )
        assert settings.api_keys == ("k",) and settings.max_new_tokens == 512
        with pytest.raises(ConfigError, match="LOGOS_API_KEY"):
            ZeroGPUSettings.from_env({"HERMES_API_KEY": "k"}, profile=LOGOS)

    def test_a_1024_token_budget_requests_the_60_second_cap(self, monkeypatch):
        for name in (
            "LOGOS_GPU_BASE_SECONDS",
            "LOGOS_GPU_TOKENS_PER_SECOND",
            "LOGOS_GPU_MIN_SECONDS",
            "LOGOS_GPU_MAX_SECONDS",
        ):
            monkeypatch.delenv(name, raising=False)
        assert gpu_duration_for(LOGOS)(None, GenerationConfig(max_new_tokens=1024)) == 60

    def test_logos_gpu_env_overrides_and_hermes_ignores_it(self, monkeypatch):
        monkeypatch.setenv("LOGOS_GPU_TOKENS_PER_SECOND", "100")
        assert gpu_duration_for(LOGOS)(None, GenerationConfig(max_new_tokens=1024)) == 21
        assert gpu_duration(None, GenerationConfig(max_new_tokens=512)) == 53


class TestHermesUnchanged:
    def test_hermes_replies_have_no_reasoning_and_version_1(self):
        service, _, _ = build_service()  # from tests.test_zerogpu_service
        reply = service.generate(ONE, {zerogpu.KEY_HEADER: KEY})
        assert reply["contract_version"] == 1 and "reasoning" not in reply
        assert "reasoning" not in service.status({zerogpu.KEY_HEADER: KEY})
```

  `ThinkingBackend(text=…, finish_reason=…)` extends `FakeBackend.__init__` with those two keywords; its default text is `"1. Alpha. What decided it: impact."`.

- [ ] **Step 2: Run.** `.venv/bin/pytest tests/test_zerogpu_logos.py -q -p no:cacheprovider`. Expected: FAIL (`profile` keyword, `gpu_duration_for`).
- [ ] **Step 3: Implement** the interfaces and behaviour above.
  - `ENV_*` and `KEY_HEADER` stay as module constants for Hermes compatibility.
  - `gpu_duration_for(profile)` returns a closure that reads `profile.env("GPU_…")`, with `profile.gpu` as the defaults, at call time.
  - In `loader.py:89`, pass the backend's `finish_reason` as the third argument.
- [ ] **Step 4: Write the Docker test** (`tests/test_zerogpu_logos_docker.py`, `pytestmark = [pytest.mark.requires_torch]`):
  - Build a thinking `HuggingFaceBackend`, as `build_thinking_backend()` does in `tests/test_backend_split.py` (import it), and wrap it in `LoadedDeployment` with a 1,024-token manifest.
  - A `FakeGPU` subclass returns the ids of `"[THINK] the deadline decides it [/THINK] migration first </s>"`.
  - Expected:
    - the reply's `text` is `"migration first"`, `reasoning` is `"the deadline decides it"`, and `finish_reason` is `"stop"`;
    - for `"[THINK] the deadline decides"`, `text` is `""` and `finish_reason` is `"length"`.
- [ ] **Step 5: Run.**
  - Host: `.venv/bin/pytest tests/test_zerogpu_logos.py tests/test_zerogpu_service.py tests/test_serving_loader.py tests/test_serving_api.py -q -p no:cacheprovider`, then the full suite.
  - Docker: `tests/test_zerogpu_logos_docker.py tests/test_zerogpu_service.py tests/test_serving_loader.py`.
  - Expected: all pass, and every Hermes test is unchanged.
- [ ] **Step 6: Checkpoint.**

---

### Task 4: The reference client carries v2 and per-model names

**Files:**
- Modify: `src/kleos_models/serving/client.py`: settings :63-125, `_gradio_client` :179-190, `validate_contract` :154-171, `_post_http` :331-357
- Test: `tests/test_hermes_client.py`, adding class `TestPerModelClient`

**Interfaces (produced):**

```python
HermesClientSettings(..., key_header: str = KEY_HEADER, env_prefix: str = "HERMES")
HermesClientSettings.from_env(env=None, *, prefix: str = "HERMES", key_header: str = KEY_HEADER)
validate_contract(result, request_id, *, answer=True)
    # accepts contract_version in SUPPORTED_CONTRACT_VERSIONS; a v2 ready reply must
    # carry "reasoning" as str or None
```

**Behaviour:**
- `problems()` names the settings with `env_prefix`.
- `_gradio_client` sends `{settings.key_header: api_key}`.
- `_post_http` passes `reasoning` and the version through: when the body has a `reasoning` key, it builds a v2 `ok_response` with `reasoning=body["reasoning"]`.

- [ ] **Step 1: Write the failing tests**

```python
class TestPerModelClient:
    def test_logos_settings_read_logos_names(self):
        s = HermesClientSettings.from_env(
            {"LOGOS_ENABLED": "true", "LOGOS_SPACE": "o/l", "LOGOS_API_KEY": "k"},
            prefix="LOGOS",
            key_header="x-logos-key",
        )
        assert s.enabled and s.space == "o/l" and s.api_key == "k" and s.key_header == "x-logos-key"
        broken = HermesClientSettings(enabled=True, provider="zerogpu", env_prefix="LOGOS")
        assert "LOGOS_SPACE is not set" in broken.problems()

    def test_the_space_client_sends_the_models_header(self, monkeypatch):
        seen = {}

        class FakeClient:
            def __init__(self, space, **kwargs):
                seen.update(kwargs)

        monkeypatch.setitem(sys.modules, "gradio_client", types.SimpleNamespace(Client=FakeClient))
        _gradio_client(HermesClientSettings(space="o/l", api_key="k", key_header="x-logos-key"))
        assert seen["headers"] == {"x-logos-key": "k"}

    @pytest.mark.parametrize("version", [1, 2])
    def test_both_contract_versions_pass(self, version):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning="t" if version == 2 else None,
            contract_version=version,
        )
        assert validate_contract(reply, "r") is reply

    def test_a_v2_reply_without_its_reasoning_key_is_refused(self):
        reply = ok_response(
            text="a",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            model={},
            request_id="r",
            reasoning="t",
            contract_version=2,
        )
        del reply["reasoning"]
        assert validate_contract(reply, "r")["status"] == "model_error"

    def test_the_http_path_keeps_the_trace(self):
        body = {
            "text": "a",
            "reasoning": "t",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "model": {},
            "request_id": "r",
        }
        http = FakeHTTP(200, body)  # existing helper in this file; see note below
        client = HermesClient(
            HermesClientSettings(enabled=True, provider="http", base_url="http://x", api_key="k"),
            http_client=http,
        )
        reply = client.generate([{"role": "user", "content": "q"}])
        assert reply["reasoning"] == "t" and reply["contract_version"] == 2
```

  Use the file's existing HTTP fake if there is one; otherwise define `FakeHTTP` in the test module with `post()` returning an object that has `status_code` and `json()`. Import `types`, `sys`, `_gradio_client`, `ok_response` and `validate_contract`.

- [ ] **Step 2: Run.** `.venv/bin/pytest tests/test_hermes_client.py -q -p no:cacheprovider`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** `.venv/bin/pytest tests/test_hermes_client.py -q -p no:cacheprovider`, then the full suite. Expected: all pass.
- [ ] **Step 5: Checkpoint.**

---

### Task 5: The Logos deployment record, and the script that completes it from the run

**Files:**
- Create:
  - `configs/deployment/kleos_logos_v002.yaml`
  - `scripts/fill_deployment_record.py`
  - `tests/test_logos_deployment_record.py`

**The record.** These values are copied verbatim from the spec. The run-derived lines are literally `null` until the owner runs the script.

```yaml
# ---------------------------------------------------------------------------
# KLEOS Logos v0.0.2: what the deployment package and the Space must match.
# Built from run kleos-v007-ministral314breasoning-run1 (H9: better than Hermes).
# The four `null` values are read from the downloaded training output by
#   python scripts/fill_deployment_record.py --run <run dir> --record <this file> --write
# Until then the package cannot be built: the identity check requires them.
# ---------------------------------------------------------------------------
deployment:
  name: kleos-logos
  version: v0.0.2
  experiment_id: kleos-v007-ministral314breasoning-run1
  base_model: mistralai/Ministral-3-14B-Reasoning-2512
  revision: 51f9210f3cd20f3452a80d5819d15dc61cc50630
  auto_class: AutoModelForCausalLM
  adapter_path: ${env:KLEOS_ADAPTER_PATH:}
  source_checkpoint: checkpoint-175
  selection_metric: eval_loss
  selection_value: null
  adapter_sha256: null
  trainable_parameters: 60948480
  dataset_version: kleos-policy-v0.0.7
  dataset_sha256: 5d02a5afd7be18e57f13b7d86fa8c31a1b008dd9e1927f4fa506356649699c48
  split_strategy: format_holdout
  training_config_hash: d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9
  tokenizer:
    source: frozen_package
    fix_mistral_regex: true
    padding_side: right
    pad_token: null
    files_sha256:
      tokenizer.json: null
      tokenizer_config.json: null
      chat_template.jinja: null
  runtime:
    quantization_mode: nf4
    double_quant: true
    compute_dtype: float16
    attn_implementation: sdpa
    max_seq_length: 1024
  generation:
    temperature: 0.0
    top_p: 1.0
    do_sample: false
    max_new_tokens: 1024
    repetition_penalty: 1.0
  limits:
    max_new_tokens: 1024
    max_input_chars: 24000
    max_messages: 64
    request_timeout_seconds: 120
  serving:
    display_name: "Logos v0.0.2"
    env_prefix: LOGOS
    key_header: x-logos-key
    space_dir: deploy/zerogpu-space-logos
    record_file: logos_record.yaml
    research_report: docs/experiments/kleos-v007-ministral314breasoning-run1-report.md
    space_only: true
    base_files:
      - config.json
      - generation_config.json
      - model.safetensors.index.json
      - model-00001-of-00006.safetensors
      - model-00002-of-00006.safetensors
      - model-00003-of-00006.safetensors
      - model-00004-of-00006.safetensors
      - model-00005-of-00006.safetensors
      - model-00006-of-00006.safetensors
    reply:
      reasoning: true
      require_system_message: true
    gpu:
      base_seconds: 10
      tokens_per_second: 12
      min_seconds: 15
      max_seconds: 60
  notes:
    - "H9: better than Hermes on the 271 answerable benchmark items (+0.0409, cluster
       95% CI +0.0040 to +0.0780); one training run; base and data changed together."
    - "Always thinks: completions are [THINK]trace[/THINK]answer. The reply's `text`
       is the answer and `reasoning` the trace; only the answer was graded."
    - "Greedy decoding can loop: 6 of 349 evaluation answers repeated a phrase until
       the 1,024-token budget. Served as evaluated; such a reply is marked `length`
       or aborted at the GPU-time cap, and KLEOS falls back."
    - "Outputs prose/bullets with a 'What decided it' line, NOT JSON (format_valid 0.0
       on all 349 held-out examples, as for Hermes)."
    - "Requests must start with a system message, as every training and benchmark
       prompt did; without one the Reasoning template injects its own thinking prompt."
    - "Base model is Apache-2.0. Attribution and a statement of modification are
       required when redistributing a derivative."
```

**The script** (`scripts/fill_deployment_record.py --run DIR --record PATH [--write]`):
- **Reads from `DIR`:**
  - the sha256 of `adapter/adapter_model.safetensors`;
  - the sha256 of `tokenizer/{tokenizer.json,tokenizer_config.json,chat_template.jinja}`;
  - `pad_token` from `tokenizer/tokenizer_config.json`, either a string or a dict with `content`;
  - `best_model_checkpoint` and `best_metric` from the newest `checkpoint-*/trainer_state.json`;
  - `config_hash`, `dataset_hash` and `experiment_id` from `manifest.json`.
- **Refuses (exit 1, naming the problem)** when:
  - any file is missing;
  - `basename(best_model_checkpoint) != record source_checkpoint`;
  - the manifest's `config_hash`, `dataset_hash` or `experiment_id` differ from the record's `training_config_hash`, `dataset_sha256` or `experiment_id`;
  - a target line already holds a different non-null value.
- **Prints** the values.
- **With `--write`,** it replaces exactly those `null` lines textually, keeping comments, then re-reads the record with `load_expected_identity` to prove it is now complete.

- [ ] **Step 1: Write the failing tests**
  - The record loads with `yaml.safe_load`.
  - Its `serving` block gives `load_profile(...).short_name == "Logos"` with `reasoning`, `require_system_message` and `space_only` all true.
  - `generation.max_new_tokens == limits.max_new_tokens == 1024`, and greedy.
  - `revision` is 40-hex and equals `configs/models/ministral3_14b_reasoning.yaml`'s revision; `training_config_hash` equals `LOGOS_V002_HASH` from `tests/test_logos_v002_config.py`.
  - `load_expected_identity` refuses the record while `adapter_sha256` is null.
  - **Fill-script tests** on a fake run directory in `tmp_path`. A helper writes the adapter, the three tokenizer files, `checkpoint-175/trainer_state.json` and `checkpoint-300/trainer_state.json`, the latter carrying `best_model_checkpoint: ".../checkpoint-175"`, `best_metric: 0.0293`, and a `manifest.json` with matching hashes. Then:
    - (a) a dry run prints the four values and writes nothing;
    - (b) `--write` fills them, and `load_expected_identity` then passes;
    - (c) a best checkpoint of `checkpoint-300` is refused;
    - (d) a manifest `config_hash` mismatch is refused;
    - (e) re-running `--write` with the same values is a no-op;
    - (f) a different already-filled value is refused.

    Run the script through `subprocess.run([sys.executable, "scripts/fill_deployment_record.py", ...])` on a copy of the record in `tmp_path`.
- [ ] **Step 2: Run.** `.venv/bin/pytest tests/test_logos_deployment_record.py -q -p no:cacheprovider`. Expected: FAIL (no record, no script).
- [ ] **Step 3: Write the record and the script** in the style of `scripts/verify_deployment_package.py`: `_cli` imports, `print_header`, `run(main)`.
- [ ] **Step 4: Run.** The new tests, `tests/test_revision_pinning.py`, `tests/test_deployment_manifest.py`, then the full suite and `.venv/bin/python scripts/validate_configs.py`.
  - Expected: all pass.
  - If any existing test enumerates `configs/deployment/*.yaml` and requires a complete identity, make it skip records whose `adapter_sha256` is null, and record a ruling.
- [ ] **Step 5: Checkpoint.**

---

### Task 6: The Logos Space folder and per-profile staging

**Files:**
- Modify:
  - `src/kleos_models/serving/space.py`: `check_space_readme` :184-212 and `stage_space` :230-280
  - `scripts/stage_zerogpu_space.py`
- Create:
  - `deploy/zerogpu-space-logos/README.md`
  - `deploy/zerogpu-space-logos/app.py`
  - `deploy/zerogpu-space-logos/requirements.txt.template`, a byte copy of Hermes'
- Test: `tests/test_zerogpu_space_logos.py`

**Interfaces:**
- `check_space_readme(readme, record)` compares the preload files with `set(profile_from_record(record).base_files)`, where `record` is the `deployment:` mapping. Hermes' result is unchanged, because its default base files equal `BASE_PRELOAD_FILES`.
- `stage_space(source_dir, record_path, out_dir, *, commit)` names the staged record `profile.record_file`, and checks the staged set against `("README.md", "app.py", "requirements.txt", profile.record_file)`.
- `stage_zerogpu_space.py`:
  - `source_dir = REPO_ROOT / load_profile(args.record).space_dir`;
  - the header and commit message use `profile.display_name`;
  - for `--push` the token is `os.environ.get("HF_TOKEN") or huggingface_hub.get_token()`. With neither, it stops with "Log in with `hf auth login`".
  - The public-Space warning names `profile.key_header`.

**`deploy/zerogpu-space-logos/README.md`:** the same front matter as Hermes', except:
- `title: KLEOS Logos v0.0.2`;
- `short_description: Frozen KLEOS Logos v0.0.2 inference API on ZeroGPU`;
- the comment says 14B;
- `preload_from_hub`:
  `  - mistralai/Ministral-3-14B-Reasoning-2512 config.json,generation_config.json,model.safetensors.index.json,model-00001-of-00006.safetensors,model-00002-of-00006.safetensors,model-00003-of-00006.safetensors,model-00004-of-00006.safetensors,model-00005-of-00006.safetensors,model-00006-of-00006.safetensors 51f9210f3cd20f3452a80d5819d15dc61cc50630`

The body describes:
- the Logos API (the `/generate` reply has `reasoning`);
- the system-message rule;
- the `x-logos-key` header and the `LOGOS_*` secrets.

**`deploy/zerogpu-space-logos/app.py`:**

```python
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
```

- [ ] **Step 1: Write the failing tests**
  - Staging the Logos record into `tmp_path` produces exactly `README.md`, `app.py`, `requirements.txt` and `logos_record.yaml`.
  - The README's `preload_from_hub` repo, files and commit equal the record's `base_model`, `base_files` and `revision`.
  - `app.py` contains `with_name("logos_record.yaml")`, `gpu_duration_for(PROFILE)` and `profile=PROFILE`, and has no `__future__` import.
  - The requirements template equals Hermes' byte for byte.
  - A README listing Nemo's shards is refused for the Logos record.
  - The Hermes Space folder is unchanged. Its existing test `tests/test_zerogpu_space.py` passes untouched.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `space.py`, the script and the three files.
- [ ] **Step 4: Run.** `.venv/bin/pytest tests/test_zerogpu_space_logos.py tests/test_zerogpu_space.py -q -p no:cacheprovider`, the full suite, and `.venv/bin/python scripts/stage_zerogpu_space.py --record configs/deployment/kleos_logos_v002.yaml --out <scratch>/logos-space --allow-dirty`.
  - Expected: all pass, and the dry stage succeeds offline. Staging does not load the identity, so the `null` hashes don't block it.
- [ ] **Step 5: Checkpoint.**

---

### Task 7: The build, upload and smoke scripts read the profile

**Files:**
- Modify:
  - `scripts/build_deployment_package.py`: the README template's Serve section, the error text :209, `research_report` :314 and the Next line :355
  - `scripts/upload_deployment_package.py`: the token, messages and printed names
  - `scripts/zerogpu_smoke.py`: the API key env, header, identity check, exit rules and copy
  - `src/kleos_models/serving/smoke.py`: `judge_response` and `summarize_run` compare `reasoning`
  - `scripts/verify_deployment_package.py`: help text only
- Test:
  - `tests/test_smoke_suite.py`, adding class `TestReasoningJudge`
  - `tests/test_logos_scripts.py` (new)

**Behaviour:**
- **The build script** reads `profile = load_profile(args.deployment_config)`:
  - `research_report=profile.research_report`;
  - the README's Serve section stays Hermes' text unless `profile.space_only`. Then it reads: "Served from its private ZeroGPU Space only; see docs/deployment.md, 'Logos v0.0.2 on ZeroGPU'.";
  - error text: "The adapter in this run is not the frozen {display_name} adapter.";
  - the Next line for `space_only`: `python scripts/upload_deployment_package.py --package {package} --repo <owner>/<name> --deployment-config {record} --dry-run`.
- **The upload script:**
  - token = `HF_TOKEN` env or `huggingface_hub.get_token()`; if neither, stop with "Log in with `hf auth login` (write access)";
  - it prints `{profile.env('PACKAGE_REPO')}` and `{profile.env('PACKAGE_REVISION')}`;
  - messages use `display_name`.
- **`smoke.judge_response`:**
  - when the reference record has a `reasoning` key, set `record["reasoning"]` to the reply's and `record["reasoning_exact"]` to `compare_output(reference["reasoning"] or "", reply.get("reasoning") or "")`;
  - `summarize_run` adds `reasoning_compared` and `reasoning_matches`.
- **`zerogpu_smoke.py`:**
  - The client is built with `key_header=profile.key_header` and `api_key=os.environ[profile.env("API_KEY")]`. `hf_token` is `HF_TOKEN` or `get_token()`: the owner's own quota, intended here.
  - `check_identity` also compares `status["limits"]["max_new_tokens"]` with `expected["generation"]["max_new_tokens"]`, and `bool(status.get("reasoning"))` with `profile.reasoning`.
  - A **match** means `exact.match` and, when compared, `reasoning_exact.match`.
  - **Empty answers:** for a non-reasoning profile they keep today's failure (exit 1). For a reasoning profile they are reported ("N answers empty: thinking never closed") and judged by the comparison, so an empty reference with an empty reply counts as a match.
  - Copy uses `display_name`.

- [ ] **Step 1: Write the failing tests**
  - **Smoke judge:**
    - identical text and trace give a match;
    - the same text with a different trace gives `reasoning_exact.match` False;
    - a reference without `reasoning` produces no `reasoning_exact`;
    - `summarize_run` counts `reasoning_matches`.
  - **Scripts** (import the modules; no network):
    - `zerogpu_smoke.check_identity` flags a `max_new_tokens` of 512 against a record of 1,024, and a missing `reasoning` flag for a reasoning profile;
    - the build script's README rendering for the Logos record has no `HERMES_` and no `serve_hermes.py`, while Hermes' rendering is unchanged (compare it with the template text as it stands before the edit);
    - the upload script's token helper returns the env token, falls back to a monkeypatched `get_token`, and raises with "hf auth login" when both are empty.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The scripts take their record from `--deployment-config` (build and upload) or `--record` (smoke, stage), as today.
- [ ] **Step 4: Run.** The new tests, `tests/test_smoke_suite.py`, `tests/test_deployment_manifest.py`, the full suite, and each script's `--help`. Expected: all pass.
- [ ] **Step 5: Checkpoint.**

---

### Task 8: Docs

**Files:**
- `docs/deployment.md`:
  - a new section "Logos v0.0.2 on ZeroGPU" after the Hermes ZeroGPU section;
  - the profile, the v2 reply, the system-message rule, the duration and quota arithmetic, the stop conditions, the owner's steps (Task 10's list) and an empty "Verification record — Logos" to be filled after the smoke test.
- `docs/kleos-hermes-integration.md`: an "Integrating Logos (contract v2)" section with:
  - `reasoning`, the `length` rules, the system-message rule, `x-logos-key`/`LOGOS_*`;
  - about 7 answers a day (ESTIMATED), the 60 s GPU cap, a client timeout of ≥120 s;
  - a note that the trace must stay outside the answer body.
- `docs/logos.md`:
  - §7: serving is built, with a link;
  - §11.7 L-F7: resolved in serving.
- `CHANGELOG.md`, under Unreleased:
  - Added: the serving profile, Logos v0.0.2 serving, the v2 reply and `fill_deployment_record.py`;
  - Changed: `finish_reason` honours the backend's never-closed signal (Hermes unaffected);
  - upload/stage/smoke use the stored `hf auth login` token when `HF_TOKEN` is unset.

- [ ] **Step 1:** Write the docs. Every figure must come from the spec, the run report or a test; ESTIMATED where it is not measured.
- [ ] **Step 2:** Run `.venv/bin/pytest -q -p no:cacheprovider`, ruff, mypy and `.venv/bin/python scripts/check_no_private_data.py`. Expected: all pass, and the scan flags only the pre-existing git-ignored files.
- [ ] **Step 3: Checkpoint.**

---

### Task 9: Verification and the final review

- [ ] **Step 1: Repo checks.**
  - Host: full suite, ruff, format and mypy.
  - Docker: the full suite, so every `requires_torch` test runs.
  - Expected: all pass. Hermes' test files are byte-identical to HEAD: `git diff --stat -- tests/test_zerogpu_service.py tests/test_zerogpu_space.py tests/test_serving_loader.py tests/test_serving_api.py tests/test_status_contract.py tests/test_hermes_client.py` shows only additions in the latter two.
- [ ] **Step 2: Offline rehearsal of the owner's steps** in a `*.noindex` scratch directory.
  1. Build a fake run directory (as in Task 5's tests) with a tiny `adapter_model.safetensors`.
  2. Run `fill_deployment_record.py --write` on a **copy** of the record.
  3. Run `build_deployment_package.py --deployment-config <copy> --model-config configs/models/ministral3_14b_reasoning.yaml --run <fake> --output <scratch>/pkg`.
  4. Run `verify_deployment_package.py`.
  5. Run `stage_zerogpu_space.py --record <copy> --allow-dirty`.

  Expected: each succeeds, the staged record carries the filled values, and nothing goes to the network.
- [ ] **Step 3: Final review.** One fresh reviewer on the most capable model, with this plan, the spec, the Review Focus and the ledger's rulings. Then re-grade the findings, fix the Critical and Important ones test-first in one pass, and record the Minor ones as deferred.

---

### Task 10: The owner's runbook (handed over, not executed)

Printed in the final message and written to `docs/deployment.md` (Task 8):

1. **Commit and push Kleos-Models.** The Space installs it at a commit.
2. **Download the training output.** On Kaggle, open the training notebook's finished version, then **Output → Download**, and unzip into `~/kleos-private/logos-v002/` (outside both repos).
3. **Fill and check the record:**
   - run `.venv/bin/python scripts/fill_deployment_record.py --run ~/kleos-private/logos-v002/outputs/kleos-v007-ministral314breasoning-run1 --record configs/deployment/kleos_logos_v002.yaml`;
   - read what it prints, then add `--write`;
   - commit and push the filled record.
4. **Build and verify:**
   - `.venv/bin/python scripts/build_deployment_package.py --run <run dir> --deployment-config configs/deployment/kleos_logos_v002.yaml --model-config configs/models/ministral3_14b_reasoning.yaml --output ~/kleos-private/logos-v0.0.2-package`
   - `.venv/bin/python scripts/verify_deployment_package.py --package ~/kleos-private/logos-v0.0.2-package`
5. **Upload:**
   - `.venv/bin/pip install huggingface_hub`, then `.venv/bin/hf auth login`, using a write token you create on huggingface.co and paste only into that prompt;
   - `upload_deployment_package.py --package … --repo <you>/kleos-logos-v002-package --deployment-config configs/deployment/kleos_logos_v002.yaml --dry-run`, then `--create`;
   - note the printed `LOGOS_PACKAGE_REVISION`.
6. **Create the Space.** On huggingface.co/new-space: Gradio, **ZeroGPU**, **Private**, named `<you>/kleos-logos`. Set four secrets:
   - `HF_TOKEN`: a new fine-grained token with read access to the package repo only;
   - `LOGOS_API_KEY`: generate one with `.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))"`;
   - `LOGOS_PACKAGE_REPO`;
   - `LOGOS_PACKAGE_REVISION`.
7. **Stage and push:** `.venv/bin/python scripts/stage_zerogpu_space.py --record configs/deployment/kleos_logos_v002.yaml --out /tmp/logos-space --push --space <you>/kleos-logos`.
8. **Watch the Space's logs** for `Logos ready: kleos-logos v0.0.2 …`. An out-of-memory crash is a stop: report it.
9. **Smoke test, day 1.**
   - Build the benchmark: `.venv/bin/python scripts/build_benchmark.py --dataset <Kleos-Training-Data>/releases/kleos-policy-v0.0.7 --output /tmp/bench`.
   - Run `read -s LOGOS_API_KEY && export LOGOS_API_KEY`, then `.venv/bin/python scripts/zerogpu_smoke.py --space <you>/kleos-logos --record configs/deployment/kleos_logos_v002.yaml --benchmark /tmp/bench/benchmark.jsonl --reference ~/Downloads/arm2_finetuned.txt --warm-repeats 0 --output ~/kleos-private/logos-smoke-day1.json`.
   - It stops when the day's quota is spent.
   - **Day 2:** the same command with `--only <remaining ids>`.
   - Any mismatch is a stop: send the output.

## Execution

**Native** (the owner's choice):
- `superpowers:executing-plans` in this session, with the ledger at `.superpowers/sdd/2026-10-06-logos-v002-serving/progress.md`;
- test-first per step;
- one fresh reviewer at the end;
- no commits.
