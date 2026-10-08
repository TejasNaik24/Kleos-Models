"""A thinking model's trace is split off before grading (Logos v0.0.2)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kleos_models.config import ReasoningMode
from kleos_models.evaluation.corrections import generation_stats
from kleos_models.evaluation.runner import RESULTS_SCHEMA_VERSION, ExampleResult
from kleos_models.inference.backends import HuggingFaceBackend, PreparedPrompt, split_thinking

BEGIN, END, EOS, UNK = 34, 35, 2, 0


class TestSplitThinking:
    def test_a_closed_span_splits_at_the_close(self):
        assert split_thinking([BEGIN, 7, 8, END, 9, EOS], BEGIN, END) == ([7, 8], [9, EOS], True)

    def test_an_unclosed_span_leaves_no_answer(self):
        assert split_thinking([BEGIN, 7, 8, 9], BEGIN, END) == ([7, 8, 9], [], False)

    def test_no_span_is_all_answer(self):
        assert split_thinking([9, 10, EOS], BEGIN, END) == (None, [9, 10, EOS], True)

    def test_a_close_without_an_open_still_splits(self):
        assert split_thinking([7, END, 9], BEGIN, END) == ([7], [9], True)

    def test_only_the_first_close_splits(self):
        assert split_thinking([BEGIN, 7, END, 9, END, 10], BEGIN, END) == (
            [7],
            [9, END, 10],
            True,
        )

    def test_an_empty_completion(self):
        assert split_thinking([], BEGIN, END) == (None, [], True)


class FakeTokenizer:
    """Ids 34/35 are the thinking markers; every other id n decodes as 'wn'."""

    unk_token_id = UNK

    def __init__(self, *, has_markers: bool = True) -> None:
        self.has_markers = has_markers

    def convert_tokens_to_ids(self, token: str) -> int:
        markers = {"[THINK]": BEGIN, "[/THINK]": END}
        return markers.get(token, UNK) if self.has_markers else UNK

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        special = {BEGIN: "[THINK]", END: "[/THINK]", EOS: "</s>"}
        words = [special.get(i, f"w{i}") for i in ids if not (skip_special_tokens and i in special)]
        return " ".join(words)


def backend(mode: ReasoningMode, *, has_markers: bool = True) -> Any:
    instance = object.__new__(HuggingFaceBackend)
    instance.name = "split-test"
    instance.loaded = SimpleNamespace(
        tokenizer=FakeTokenizer(has_markers=has_markers), reasoning_mode=mode
    )
    return instance


PREPARED = PreparedPrompt(inputs={}, prompt_length=5)


class TestFinishInThinkingMode:
    def test_only_the_answer_is_graded(self):
        output = backend(ReasoningMode.THINKING).finish(PREPARED, [BEGIN, 7, 8, END, 9, EOS])
        assert output.text == "w9"
        assert output.reasoning == "w7 w8"
        assert output.finish_reason == "stop"
        assert output.completion_tokens == 6

    def test_a_trace_that_never_closes_leaves_an_empty_answer(self):
        output = backend(ReasoningMode.THINKING).finish(PREPARED, [BEGIN, 7, 8, 9])
        assert output.text == ""
        assert output.reasoning == "w7 w8 w9"
        assert output.finish_reason == "length"

    def test_an_answer_without_thinking_is_kept_whole(self):
        output = backend(ReasoningMode.THINKING).finish(PREPARED, [9, 10, EOS])
        assert output.text == "w9 w10"
        assert output.reasoning is None
        assert output.finish_reason == "stop"


class TestFinishOtherwiseUnchanged:
    def test_standard_mode_never_splits_on_tokens(self):
        output = backend(ReasoningMode.STANDARD).finish(PREPARED, [BEGIN, 7, END, 9])
        assert output.text == "w7 w9"
        assert output.reasoning is None

    def test_a_tokenizer_without_the_markers_keeps_the_text_split(self):
        output = backend(ReasoningMode.THINKING, has_markers=False).finish(PREPARED, [7, 9])
        assert output.text == "w7 w9"
        assert output.reasoning is None


def record(**fields: Any) -> dict[str, Any]:
    base = {"completion_tokens": 10, "prompt_tokens": 5, "latency_seconds": 1.0, "response": "a"}
    return {**base, **fields}


class TestPersisted:
    def result(self, **fields: Any) -> ExampleResult:
        values: dict[str, Any] = {
            "example_id": "e1",
            "task": "t",
            "arm": "arm2_finetuned",
            "seed": 42,
            "score": 1.0,
            "grader": "g",
            "response": "w9",
        }
        values.update(fields)
        return ExampleResult(**values)

    def test_the_schema_version_records_the_new_fields(self):
        assert RESULTS_SCHEMA_VERSION == 3

    def test_the_trace_is_saved_beside_the_answer(self):
        payload = self.result(reasoning="w7 w8", finish_reason="stop").to_dict()
        assert payload["reasoning"] == "w7 w8"
        assert payload["response"] == "w9"
        assert payload["finish_reason"] == "stop"

    def test_the_trace_is_withheld_with_the_response(self):
        payload = self.result(reasoning="w7 w8").to_dict(include_response=False)
        assert "reasoning" not in payload and "response" not in payload
        assert payload["finish_reason"] == "stop"

    def test_a_record_without_a_trace_has_no_reasoning_key(self):
        assert "reasoning" not in self.result().to_dict()


class TestGenerationStats:
    def test_unclosed_traces_are_counted(self):
        rows = [
            record(finish_reason="stop", reasoning="abc"),
            record(finish_reason="length", reasoning="abcdef"),
        ]
        stats = generation_stats(rows, max_new_tokens=1024)
        assert stats["thinking_truncated"] == 1
        assert stats["reasoning_responses"] == 2
        assert stats["reasoning_chars"]["max"] == 6

    @pytest.mark.parametrize(
        "rows",
        [
            [record(), record()],  # results written before schema 3
            [record(finish_reason="stop"), record()],
        ],
    )
    def test_older_records_report_nothing_new(self, rows):
        stats = generation_stats(rows, max_new_tokens=512)
        assert "thinking_truncated" not in stats
        assert "reasoning_chars" not in stats
        assert "reasoning_responses" not in stats

    def test_traces_are_summarized_over_the_answers_that_have_one(self):
        # One answer without a trace must not hide H9's trace length distribution.
        rows = [record(finish_reason="stop", reasoning="abcd"), record(finish_reason="stop")]
        stats = generation_stats(rows, max_new_tokens=1024)
        assert stats["thinking_truncated"] == 0
        assert stats["reasoning_responses"] == 1
        assert stats["reasoning_chars"]["max"] == 4
