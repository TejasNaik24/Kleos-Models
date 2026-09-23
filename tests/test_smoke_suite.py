"""The shared smoke-test judging: one set of rules for every serving path.

`hermes_smoke.py` (in-process) and `zerogpu_smoke.py` (over the network) must
judge reproducibility identically, or a pass on one says nothing about the
other. These tests pin the judging and the run summary on synthetic examples.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_eval_example

from kleos_models.data.schemas import EvaluationExample
from kleos_models.errors import KleosError
from kleos_models.serving.smoke import (
    REQUIRED_TASKS,
    api_messages,
    compare_output,
    judge_response,
    select_suite,
    summarize_run,
)
from kleos_models.serving.status import HermesStatus, error_response, ok_response


def example(example_id: str = "eval-000001", **overrides) -> EvaluationExample:
    return EvaluationExample.model_validate(make_eval_example(example_id, **overrides))


def answer(text: str, *, prompt_tokens: int = 30, cold: bool = True, call: int = 1) -> dict:
    return ok_response(
        text=text,
        finish_reason="stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=4,
        model={},
        request_id="r",
        timings={
            "prepare_s": 0.01,
            "gpu_call_s": 9.0 if cold else 2.0,
            "gpu_generate_s": 2.0,
            "gpu_acquire_s": 7.0 if cold else 0.0,
            "finish_s": 0.01,
            "total_s": 9.1 if cold else 2.1,
        },
        diagnostics={
            "cold_start": cold,
            "worker_call_index": call,
            "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "compute_capability": "12.0",
            "peak_vram_gib": 8.5,
            "cuda_runtime": "12.8",
        },
    )


REFERENCE = {
    "example_id": "eval-000001",
    "response": "memory_search",
    "prompt_tokens": 30,
    "completion_tokens": 4,
    "decision": "memory_search",
}


class TestSuite:
    def test_the_suite_covers_every_family_and_adds_declines(self):
        examples = [example(f"eval-{i:06d}", task=task) for i, task in enumerate(REQUIRED_TASKS)]
        examples.append(
            example(
                "eval-999999",
                task="tool_routing",
                reference={"label": "none", "options": ["none"], "confident": False},
            )
        )
        suite = select_suite(examples, per_task=1, abstention=2)
        assert {e.task for e in suite} == set(REQUIRED_TASKS)
        assert "eval-999999" in {e.id for e in suite}
        assert select_suite(list(reversed(examples))) == suite  # order-independent

    def test_a_missing_family_is_an_error_not_a_smaller_suite(self):
        with pytest.raises(KleosError, match="missing required task families"):
            select_suite([example(task="tool_routing")])


class TestComparison:
    def test_identical_text_matches(self):
        assert compare_output(" same\n", "same") == {"match": True, "first_difference": None}

    def test_the_first_difference_is_located(self):
        result = compare_output("abcdef", "abcxef")
        assert result["match"] is False
        assert result["first_difference"] == 3
        assert result["shared_prefix_fraction"] == 0.5

    def test_the_api_carries_system_user_and_assistant_turns_only(self):
        assert api_messages(example()) == [{"role": "user", "content": "Which tool handles this?"}]
        payload = make_eval_example()
        payload["messages"].append({"role": "tool", "content": "{}", "name": "search"})
        assert api_messages(EvaluationExample.model_validate(payload)) is None


class TestJudging:
    def test_an_exact_reproduction(self):
        record = judge_response(example(), REFERENCE, answer("memory_search"))
        assert record["exact"]["match"] is True
        assert record["prompt_tokens_match"] is True
        assert record["completion_tokens_match"] is True
        assert record["decision_matches"] is True

    def test_a_difference_is_localised(self):
        # Same prompt tokens (tokenization agrees), different text and decision.
        record = judge_response(example(), REFERENCE, answer("none"))
        assert record["exact"]["match"] is False
        assert record["prompt_tokens_match"] is True
        assert record["decision_matches"] is False

    def test_a_tokenization_difference_shows_in_the_prompt_count(self):
        record = judge_response(example(), REFERENCE, answer("memory_search", prompt_tokens=31))
        assert record["prompt_tokens_match"] is False

    def test_a_non_answer_is_recorded_with_its_status(self):
        response = error_response(
            HermesStatus.QUOTA_EXHAUSTED, request_id="r", retry_after_seconds=60
        )
        record = judge_response(example(), REFERENCE, response)
        assert record == {
            "example_id": "eval-000001",
            "task": "tool_routing",
            "status": "quota_exhausted",
            "ok": False,
            "retry_after_seconds": 60,
        }


class TestSummary:
    def test_counts_and_observed_measurements(self):
        records = [
            judge_response(example(), REFERENCE, answer("memory_search", cold=True, call=1)),
            judge_response(example(), REFERENCE, answer("none", cold=False, call=2)),
            judge_response(
                example(), REFERENCE, error_response(HermesStatus.QUEUE_UNAVAILABLE, request_id="r")
            ),
        ]
        for record, wall in zip(records, (12.0, 3.0, 1.0), strict=True):
            record["client_wall_s"] = wall
        summary = summarize_run(records)
        assert summary["requests"] == 3 and summary["answered"] == 2
        assert summary["statuses"] == {"ready": 2, "queue_unavailable": 1}
        assert summary["compared"] == 2 and summary["exact_matches"] == 1
        assert summary["decision_matches"] == 1
        measured = summary["measurements"]
        assert measured["label"] == "Observed ZeroGPU smoke-test measurements"
        assert measured["cold"]["calls"] == 1 and measured["warm"]["calls"] == 1
        assert measured["cold"]["median_gpu_acquire_s"] == 7.0
        assert measured["warm"]["median_tokens_per_second"] == 2.0
        assert measured["gpu_call_seconds_total"] == 11.0
        assert measured["gpu_generate_seconds_total"] == 4.0
        assert measured["peak_vram_gib"] == 8.5
