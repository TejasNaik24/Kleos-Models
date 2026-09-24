"""Resumable evaluation.

An arm takes two to three hours on a free T4; a disconnect must cost one example, not the
arm. The guarantees pinned here:

* a crash followed by a resume produces the result an uninterrupted run would;
* a resume under any other identity is refused;
* a torn last line is dropped and regenerated, a damaged file is refused;
* a finished result is never overwritten silently.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.conftest import CONFIGS_DIR, REPO_ROOT, make_eval_example

from kleos_models.config import EvaluationConfig
from kleos_models.data.schemas import EvaluationExample
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.resume import (
    PartialWriter,
    ResumingBackend,
    adapter_identity,
    build_identity,
    identity_sha256,
    load_partial,
    partial_path,
    start_partial,
)
from kleos_models.evaluation.runner import run_evaluation
from kleos_models.inference.backends import EchoBackend


def identity(**changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "arm": "arm2_finetuned",
        "seeds": [42],
        "example_ids": ["ex-0", "ex-1", "ex-2"],
        "benchmark_sha256": "b" * 64,
        "generation": {"do_sample": False, "max_new_tokens": 512},
        "orchestration_prompt": None,
        "model": {"base_model": "m", "revision": "r"},
        "adapter": {"adapter_model.safetensors": "a" * 64},
        "git_commit": "abc123",
        "libraries": {"transformers": "5.16.1"},
        "compute_capability": "7.5",
    }
    fields.update(changes)
    return build_identity(**fields)


def record(seed: int, example_id: str, text: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "example_id": example_id,
        "text": text,
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "finish_reason": "stop",
        "latency_seconds": 4.2,
    }


class TestPartialFile:
    def test_recorded_generations_are_read_back(self, tmp_path):
        path = tmp_path / "out.json.partial.jsonl"
        start_partial(path, identity())
        writer = PartialWriter(path)
        writer.append(record(42, "ex-0", "first"))
        writer.append(record(42, "ex-1", "second"))
        writer.close()
        state = load_partial(path, identity())
        assert set(state.generations) == {(42, "ex-0"), (42, "ex-1")}
        assert state.generations[(42, "ex-1")]["text"] == "second"
        assert not state.dropped_torn_line

    def test_a_partial_file_is_never_replaced(self, tmp_path):
        path = tmp_path / "p.jsonl"
        start_partial(path, identity())
        with pytest.raises(FileExistsError):
            start_partial(path, identity())

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            ({"arm": "arm1_base_orchestrated"}, "arm"),
            ({"adapter": {"adapter_model.safetensors": "c" * 64}}, "adapter"),
            ({"benchmark_sha256": "d" * 64}, "benchmark_sha256"),
            ({"example_ids": ["ex-0", "ex-1"]}, "example_ids_sha256"),
            ({"libraries": {"transformers": "5.17.0"}}, "libraries"),
            ({"compute_capability": "8.0"}, "compute_capability"),
            ({"git_commit": "def456"}, "git_commit"),
            ({"orchestration_prompt": "Be careful."}, "orchestration_prompt_sha256"),
        ],
    )
    def test_a_different_identity_is_refused(self, tmp_path, change, field):
        path = tmp_path / "p.jsonl"
        start_partial(path, identity())
        with pytest.raises(EvaluationError, match="different identity") as info:
            load_partial(path, identity(**change))
        assert field in info.value.details["differs_in"]

    def test_a_torn_last_line_is_dropped(self, tmp_path):
        path = tmp_path / "p.jsonl"
        start_partial(path, identity())
        writer = PartialWriter(path)
        writer.append(record(42, "ex-0", "kept"))
        writer.close()
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"seed": 42, "example_id": "ex-1", "te')  # killed mid-write
        state = load_partial(path, identity())
        assert state.dropped_torn_line
        assert set(state.generations) == {(42, "ex-0")}
        # The file is repaired, so appending continues a valid log.
        writer = PartialWriter(path)
        writer.append(record(42, "ex-1", "regenerated"))
        writer.close()
        assert set(load_partial(path, identity()).generations) == {(42, "ex-0"), (42, "ex-1")}

    def test_a_damaged_middle_line_is_refused(self, tmp_path):
        path = tmp_path / "p.jsonl"
        start_partial(path, identity())
        with path.open("a", encoding="utf-8") as handle:
            handle.write("not json\n")
            handle.write(json.dumps(record(42, "ex-1", "x")) + "\n")
        with pytest.raises(EvaluationError, match="damaged"):
            load_partial(path, identity())

    def test_a_file_without_a_header_is_refused(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(EvaluationError, match="header"):
            load_partial(path, identity())

    def test_the_partial_sits_beside_the_output(self):
        assert partial_path(Path("/x/eval.json")) == Path("/x/eval.json.partial.jsonl")

    def test_identity_digest_is_stable(self):
        assert identity_sha256(identity()) == identity_sha256(identity())


class TestAdapterIdentity:
    def test_the_adapter_is_identified_by_its_bytes(self, tmp_path):
        (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
        (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "README.md").write_text("ignored", encoding="utf-8")
        assert adapter_identity(tmp_path) == {
            "adapter_config.json": hashlib.sha256(b"{}").hexdigest(),
            "adapter_model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        }

    def test_a_directory_without_weights_is_refused(self, tmp_path):
        with pytest.raises(EvaluationError, match="No adapter weights"):
            adapter_identity(tmp_path)

    def test_no_adapter_has_no_identity(self):
        assert adapter_identity(None) is None


# ---------------------------------------------------------------------------
# Crash, then resume, through the real runner
# ---------------------------------------------------------------------------


def examples(count: int = 5) -> list[EvaluationExample]:
    rows = []
    for i in range(count):
        row = make_eval_example(
            f"ex-{i}",
            reference={
                "label": "memory_search",
                "options": ["memory_search", "none"],
                "confident": True,
            },
        )
        row["metadata"]["group_id"] = f"grp-{i // 2}"
        rows.append(EvaluationExample.model_validate(row))
    return rows


class Flaky(EchoBackend):
    """Answers until its budget runs out, then dies like a runtime."""

    def __init__(self, budget: int) -> None:
        super().__init__(responses={f"ex-{i}": f"memory_search {i}" for i in range(9)})
        self.budget = budget

    def generate(self, messages, config, **kwargs):
        if self.calls >= self.budget:
            raise RuntimeError("runtime disconnected")
        return super().generate(messages, config, **kwargs)


def comparable(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-example records without the fields a replay may legitimately change."""
    return [{k: v for k, v in r.items() if k != "latency_seconds"} for r in payload["results"]]


class TestCrashThenResume:
    def test_a_resumed_run_equals_an_uninterrupted_one(self, tmp_path):
        config = EvaluationConfig()
        path = tmp_path / "p.jsonl"
        start_partial(path, identity())

        # First session: dies after three generations.
        writer = PartialWriter(path)
        with pytest.raises(RuntimeError, match="disconnected"):
            run_evaluation(
                ResumingBackend(Flaky(budget=3), recorded={}, writer=writer),
                examples(),
                config,
                default_grader="classification",
            )
        writer.close()

        # Second session: a fresh backend, the recorded three replayed.
        state = load_partial(path, identity())
        writer = PartialWriter(path)
        fresh = Flaky(budget=99)
        resumed_backend = ResumingBackend(fresh, recorded=state.generations, writer=writer)
        resumed = run_evaluation(
            resumed_backend, examples(), config, default_grader="classification"
        )
        writer.close()
        assert (resumed_backend.replayed, resumed_backend.generated) == (3, 2)
        assert fresh.calls == 2

        uninterrupted = run_evaluation(
            Flaky(budget=99), examples(), config, default_grader="classification"
        )
        assert comparable(resumed.to_dict()) == comparable(uninterrupted.to_dict())
        assert resumed.to_dict()["metrics"] == uninterrupted.to_dict()["metrics"]


# ---------------------------------------------------------------------------
# scripts/evaluate.py
# ---------------------------------------------------------------------------


def load_evaluate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_under_test", REPO_ROOT / "scripts" / "evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(tmp_path, monkeypatch):
    for name in ("KLEOS_BENCHMARK_PATH", "KLEOS_DATASET_PATH", "KLEOS_ADAPTER_PATH"):
        monkeypatch.delenv(name, raising=False)
    bench = tmp_path / "benchmark.jsonl"
    with bench.open("w", encoding="utf-8") as handle:
        for example in examples(4):
            handle.write(example.model_dump_json() + "\n")
    evaluate = load_evaluate()
    output = tmp_path / "results" / "eval.json"

    def run(*extra: str) -> int:
        return evaluate.main(
            [
                "--config",
                str(CONFIGS_DIR / "training" / "debug.yaml"),
                "--benchmark",
                str(bench),
                "--output",
                str(output),
                "--echo",
                *extra,
            ]
        )

    return run, output, bench


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestEvaluateScript:
    def test_a_run_records_its_identity_and_benchmark(self, cli):
        run, output, bench = cli
        assert run() == 0
        payload = json.loads(output.read_text())
        assert payload["benchmark_sha256"] == sha(bench)
        assert len(payload["resume"]["identity_sha256"]) == 64
        assert payload["resume"]["new_generations"] == 4
        assert payload["resource_usage"]["compute_capability"]
        assert not partial_path(output).exists()  # the partial has done its job

    def test_a_finished_result_is_never_overwritten_silently(self, cli):
        run, output, _ = cli
        assert run() == 0
        before = sha(output)
        assert run() == 1
        assert sha(output) == before

    def test_resume_on_a_finished_result_does_nothing(self, cli):
        run, output, _ = cli
        assert run() == 0
        before = sha(output)
        assert run("--resume") == 0
        assert sha(output) == before

    def test_overwrite_starts_again(self, cli):
        run, output, _ = cli
        assert run() == 0
        assert run("--overwrite") == 0
        assert json.loads(output.read_text())["resume"]["replayed_generations"] == 0

    def test_resume_and_overwrite_are_exclusive(self, cli):
        run, _, _ = cli
        with pytest.raises(SystemExit):
            run("--resume", "--overwrite")

    def test_an_interrupted_run_resumes_to_the_same_result(self, cli, tmp_path, monkeypatch):
        run, output, _ = cli
        original = EchoBackend.generate
        budget = {"left": 2}

        def dying(self, messages, config, **kwargs):
            if budget["left"] == 0:
                raise RuntimeError("runtime disconnected")
            budget["left"] -= 1
            return original(self, messages, config, **kwargs)

        monkeypatch.setattr(EchoBackend, "generate", dying)
        with pytest.raises(RuntimeError):
            run()
        assert partial_path(output).exists() and not output.exists()

        monkeypatch.setattr(EchoBackend, "generate", original)
        assert run() == 1  # an interrupted run is not restarted by accident
        assert run("--resume") == 0
        resumed = json.loads(output.read_text())
        assert resumed["resume"]["replayed_generations"] == 2
        assert resumed["resume"]["new_generations"] == 2

        clean_output = output.with_name("clean.json")
        evaluate = load_evaluate()
        assert (
            evaluate.main(
                [
                    "--config",
                    str(CONFIGS_DIR / "training" / "debug.yaml"),
                    "--benchmark",
                    str(tmp_path / "benchmark.jsonl"),
                    "--output",
                    str(clean_output),
                    "--echo",
                ]
            )
            == 0
        )
        clean = json.loads(clean_output.read_text())
        assert comparable(resumed) == comparable(clean)
        assert resumed["resume"]["identity_sha256"] == clean["resume"]["identity_sha256"]

    def test_a_resume_under_another_identity_is_refused(self, cli, monkeypatch):
        run, _output, _ = cli
        original = EchoBackend.generate

        def dying(self, messages, config, **kwargs):
            raise RuntimeError("runtime disconnected")

        monkeypatch.setattr(EchoBackend, "generate", dying)
        with pytest.raises(RuntimeError):
            run()
        monkeypatch.setattr(EchoBackend, "generate", original)
        with pytest.raises(EvaluationError, match="different identity"):
            run("--resume", "--max-examples", "2")
