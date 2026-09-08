"""Formatting and assistant-loss-masking tests (spec §30).

Masking correctness is the highest-consequence detail in the training pipeline.
If the mask is wrong the model trains on the wrong tokens, the loss still looks
plausible, and nothing downstream notices. These tests pin the exact spans.

They run against a fake tokenizer, so masking is verified without downloading a
model. The same logic is exercised against real Qwen and Mistral templates in
``test_training_tiny_model.py`` when transformers is installed.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_example

from kleos_models.constants import IGNORE_INDEX
from kleos_models.data.formatting import ConversationFormatter, messages_to_dicts
from kleos_models.data.schemas import Message, TrainingExample
from kleos_models.errors import DataValidationError


def build(example_id: str = "test-000001", **kwargs) -> TrainingExample:
    return TrainingExample.model_validate(make_example(example_id, **kwargs))


class TestMessageConversion:
    def test_messages_convert_to_dicts(self):
        messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
        assert messages_to_dicts(messages) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_tool_name_is_preserved(self):
        messages = [Message(role="tool", content="result", name="search")]
        assert messages_to_dicts(messages)[0]["name"] == "search"

    def test_names_can_be_omitted(self):
        messages = [Message(role="tool", content="result", name="search")]
        assert "name" not in messages_to_dicts(messages, include_names=False)[0]


class TestRendering:
    def test_conversation_renders_through_the_template(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        example = build()
        rendered = formatter.render(example.messages)
        assert "<|system|>" in rendered
        assert "<|user|>" in rendered
        assert "<|assistant|>" in rendered

    def test_generation_prompt_is_appended(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        example = build()
        prompt = formatter.render_prompt(example.messages[:2])
        assert prompt.rstrip().endswith("<|assistant|>")

    def test_unsupported_template_kwarg_raises_an_actionable_error(self, fake_tokenizer):
        # A Thinking-only checkpoint rejects enable_thinking; that must surface
        # as a clear message rather than a jinja traceback.
        formatter = ConversationFormatter(fake_tokenizer, template_kwargs={"enable_thinking": True})
        with pytest.raises(DataValidationError, match="chat template rejected"):
            formatter.render(build().messages)

    def test_supported_template_kwarg_is_accepted(self, thinking_tokenizer):
        formatter = ConversationFormatter(
            thinking_tokenizer, template_kwargs={"enable_thinking": True}
        )
        assert formatter.render(build().messages)


class TestMasking:
    def test_only_assistant_tokens_are_supervised(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None

        supervised = [
            token
            for token, label in zip(result.input_ids, result.labels, strict=True)
            if label != IGNORE_INDEX
        ]
        decoded = fake_tokenizer.decode(supervised)

        # The answer is supervised.
        assert "1." in decoded
        # The prompt is not.
        assert "Rank" not in decoded
        assert "<|user|>" not in decoded

    def test_labels_match_input_ids_where_supervised(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None
        for token, label in zip(result.input_ids, result.labels, strict=True):
            if label != IGNORE_INDEX:
                assert label == token

    def test_some_tokens_are_masked(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None
        assert IGNORE_INDEX in result.labels, "the prompt must be masked out"

    def test_target_token_count_is_accurate(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None
        assert result.target_token_count == sum(
            1 for label in result.labels if label != IGNORE_INDEX
        )
        assert result.target_token_count > 0

    def test_attention_mask_covers_every_token(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None
        assert result.attention_mask == [1] * len(result.input_ids)

    def test_multi_turn_supervises_every_assistant_turn(self, fake_tokenizer):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "ALPHA answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "BETA answer"},
        ]
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(TrainingExample.model_validate(payload))
        assert result is not None

        supervised = fake_tokenizer.decode(
            [
                t
                for t, label in zip(result.input_ids, result.labels, strict=True)
                if label != IGNORE_INDEX
            ]
        )
        assert "ALPHA" in supervised
        assert "BETA" in supervised
        assert "question" not in supervised

    def test_last_turn_only_mode_supervises_one_turn(self, fake_tokenizer):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "ALPHA answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "BETA answer"},
        ]
        formatter = ConversationFormatter(fake_tokenizer, train_on_last_turn_only=True)
        result = formatter.format_example(TrainingExample.model_validate(payload))
        assert result is not None

        supervised = fake_tokenizer.decode(
            [
                t
                for t, label in zip(result.input_ids, result.labels, strict=True)
                if label != IGNORE_INDEX
            ]
        )
        assert "BETA" in supervised
        assert "ALPHA" not in supervised


class TestReasoningStripping:
    def test_reasoning_is_removed_from_targets_by_default(self, fake_tokenizer):
        payload = make_example(
            assistant="<think>secret deliberation SENTINEL</think>The answer is alpha."
        )
        formatter = ConversationFormatter(fake_tokenizer, strip_reasoning=True)
        result = formatter.format_example(TrainingExample.model_validate(payload))
        assert result is not None

        supervised = fake_tokenizer.decode(
            [
                t
                for t, label in zip(result.input_ids, result.labels, strict=True)
                if label != IGNORE_INDEX
            ]
        )
        assert "SENTINEL" not in supervised, (
            "hidden chain-of-thought must never become a training target"
        )
        assert "alpha" in supervised

    def test_reasoning_can_be_retained_deliberately(self, fake_tokenizer):
        payload = make_example(assistant="<think>deliberation SENTINEL</think>The answer is alpha.")
        formatter = ConversationFormatter(fake_tokenizer, strip_reasoning=False)
        result = formatter.format_example(TrainingExample.model_validate(payload))
        assert result is not None
        supervised = fake_tokenizer.decode(
            [
                t
                for t, label in zip(result.input_ids, result.labels, strict=True)
                if label != IGNORE_INDEX
            ]
        )
        assert "SENTINEL" in supervised


class TestTruncation:
    def test_long_examples_are_truncated(self, fake_tokenizer):
        payload = make_example(user="word " * 500)
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=64)
        result = formatter.format_example(TrainingExample.model_validate(payload))
        if result is not None:
            assert len(result.input_ids) <= 64
            assert result.truncated

    def test_example_whose_answer_is_truncated_away_is_dropped(self, fake_tokenizer):
        # Training on an example with zero supervised tokens produces a loss over
        # nothing, so it is dropped rather than silently included.
        payload = make_example(user="word " * 500, assistant="answer")
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=20)
        assert formatter.format_example(TrainingExample.model_validate(payload)) is None

    def test_short_examples_are_untouched(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=4096)
        result = formatter.format_example(build())
        assert result is not None
        assert not result.truncated


class TestDatasetFormatting:
    def test_all_valid_examples_are_formatted(self, fake_tokenizer, training_examples):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=2048)
        formatted, stats = formatter.format_dataset(training_examples)
        assert len(formatted) == len(training_examples)
        assert stats.total == len(training_examples)
        assert stats.dropped == 0

    def test_stats_summarize_token_lengths(self, fake_tokenizer, training_examples):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=2048)
        _, stats = formatter.format_dataset(training_examples)
        summary = stats.summary()
        assert summary["max_tokens"] > 0
        assert summary["mean_target_tokens"] > 0

    def test_unusable_examples_are_counted_not_hidden(self, fake_tokenizer, training_examples):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=12)
        formatted, stats = formatter.format_dataset(training_examples)
        assert stats.dropped == len(training_examples) - len(formatted)
        assert stats.dropped > 0

    def test_committed_fixtures_all_format(self, fake_tokenizer, fixture_examples):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=4096)
        formatted, stats = formatter.format_dataset(fixture_examples)
        assert stats.dropped == 0, "every committed fixture must produce supervised tokens"
        assert all(f.target_token_count > 0 for f in formatted)

    def test_describe_masking_reports_counts_only(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        description = formatter.describe_masking(build("described-001"))
        assert "described-001" in description
        assert "supervised" in description
        # Must not leak example content into a debugging string.
        assert "Rank these" not in description


class TestSerialization:
    def test_formatted_example_converts_to_a_row(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer)
        result = formatter.format_example(build())
        assert result is not None
        row = result.to_dict()
        assert set(row) == {"input_ids", "labels", "attention_mask"}
        assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])


class MistralStyleTokenizer:
    """Reproduces Mistral's ``loop.last`` system-prompt handling.

    The real template injects the system message into the LAST message of the
    conversation. Generating, that is the user turn, so the system prompt shows
    up. Training, it is the assistant turn, so the system prompt is DROPPED —
    verified against mistralai/Ministral-8B-Instruct-2410 under transformers
    5.16:

        [system, user]            -> '<s>[INST]SYSTEM\n\nUSER[/INST]'
        [system, user, assistant] -> '<s>[INST]USER[/INST]ASSISTANT</s>'
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.padding_side = "right"
        self.chat_template = "mistral-like"
        self._vocab: dict[str, int] = {"<pad>": 0, "</s>": 1}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        system = None
        rest = list(conversation)
        if rest and rest[0]["role"] == "system":
            system = rest[0]["content"]
            rest = rest[1:]

        out = ["<s>"]
        for index, message in enumerate(rest):
            is_last = index == len(rest) - 1
            if message["role"] == "user":
                content = message["content"]
                if is_last and system is not None:
                    content = f"{system}\n\n{content}"
                out.append(f"[INST] {content} [/INST]")
            else:
                out.append(f" {message['content']} </s>")
        return " ".join(out)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 100
            ids.append(self._vocab[token])
        return ids


class TestSystemPromptSurvivesTemplate:
    """A template that drops the system turn must not silently train without it.

    Found on Colab against the real Ministral checkpoint: the policy instructions
    that KLEOS exists to teach were being deleted from every training example,
    while evaluation kept them.
    """

    @staticmethod
    def _formatter() -> ConversationFormatter:
        return ConversationFormatter(MistralStyleTokenizer(), max_seq_length=512)

    def test_the_dropping_template_is_detected(self):
        assert self._formatter()._template_drops_trailing_system() is True

    def test_a_well_behaved_template_is_left_alone(self, fake_tokenizer):
        formatter = ConversationFormatter(fake_tokenizer, max_seq_length=512)
        assert formatter._template_drops_trailing_system() is False
        messages = [
            Message(role="system", content="POLICY"),
            Message(role="user", content="Q"),
            Message(role="assistant", content="A"),
        ]
        assert formatter._normalize(messages) == messages

    def test_the_system_prompt_reaches_the_training_text(self):
        formatter = self._formatter()
        rendered = formatter.render(
            [
                Message(role="system", content="POLICY"),
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
            ]
        )
        assert "POLICY" in rendered, "system prompt was dropped from the training text"

    def test_training_and_inference_prompts_are_identical(self):
        # The whole point. If these diverge the model is trained on one prompt
        # and evaluated on another.
        formatter = self._formatter()
        system = Message(role="system", content="POLICY")
        user = Message(role="user", content="Q")
        inference = formatter.render([system, user], add_generation_prompt=True)
        training = formatter.render([system, user, Message(role="assistant", content="A")])
        assert training.startswith(inference)

    def test_only_the_assistant_turn_is_supervised(self):
        formatter = self._formatter()
        example = build(
            system="POLICY",
            user="Q",
            assistant="ANSWER",
        )
        formatted = formatter.format_example(example)
        assert formatted is not None
        supervised = [
            i
            for i, label in zip(formatted.input_ids, formatted.labels, strict=True)
            if label != IGNORE_INDEX
        ]
        # The answer plus its end-of-turn marker — not the whole sequence.
        assert 0 < len(supervised) < len(formatted.input_ids) / 2
        assert formatted.target_token_count == len(supervised)
