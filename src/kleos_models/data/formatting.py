"""Conversation formatting and assistant-only loss masking.

Two jobs:

1. Render a conversation through the model's **own** chat template. Qwen and
   Mistral templates differ in special tokens, role markers and whitespace, so
   the template always comes from the tokenizer (or the family adapter), never
   from a hard-coded f-string here.

2. Compute which tokens the loss is applied to. Training on the prompt as well
   as the answer teaches the model to predict user messages, which is not the
   behaviour we are trying to learn. Everything except assistant content is set
   to ``IGNORE_INDEX``.

Masking strategy
----------------
Assistant spans are located by **incremental template application**: render the
conversation up to (but excluding) each assistant turn with a generation prompt,
render it again including that turn, and take the token range between the two.
This is family-agnostic — it asks the template where the boundary is rather than
guessing from a delimiter string.

Prefix stability is verified per span. If a template turns out not to be
prefix-stable, the longest common prefix is used and a warning is emitted, rather
than silently mislabelling tokens.

This module imports no torch: it works on plain lists of ints. The tokenizer is
injected as a protocol so masking can be tested with a fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kleos_models.constants import IGNORE_INDEX
from kleos_models.data.schemas import Message, TrainingExample
from kleos_models.errors import DataValidationError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ChatTokenizer(Protocol):
    """Minimal tokenizer surface this module depends on.

    Satisfied by a Hugging Face tokenizer and by the test fake, which is the
    point: masking logic is verifiable without downloading a model.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Render a conversation to a string (``tokenize=False``)."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ids."""
        ...


@dataclass
class FormattedExample:
    """A tokenized example ready for the collator."""

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    example_id: str
    rendered_text: str = field(default="", repr=False)
    target_token_count: int = 0
    truncated: bool = False
    target_truncated: bool = False

    def __len__(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict[str, list[int]]:
        """Plain dict for a datasets.Dataset row."""
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
        }


@dataclass
class FormattingStats:
    """Aggregate outcome of formatting a dataset."""

    total: int = 0
    truncated: int = 0
    target_truncated: int = 0
    empty_targets: int = 0
    dropped: int = 0
    token_lengths: list[int] = field(default_factory=list)
    target_lengths: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        lengths = self.token_lengths or [0]
        targets = self.target_lengths or [0]
        return {
            "examples_formatted": self.total,
            "dropped": self.dropped,
            "truncated": self.truncated,
            "target_truncated": self.target_truncated,
            "empty_targets": self.empty_targets,
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 1),
            "mean_target_tokens": round(sum(targets) / len(targets), 1),
        }


def messages_to_dicts(
    messages: Sequence[Message], *, include_names: bool = True
) -> list[dict[str, str]]:
    """Convert typed messages to the dict form chat templates expect."""
    result: list[dict[str, str]] = []
    for message in messages:
        payload: dict[str, str] = {"role": message.role, "content": message.content}
        if include_names and message.name:
            payload["name"] = message.name
        result.append(payload)
    return result


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the shared leading run of two token sequences."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class ConversationFormatter:
    """Turns :class:`TrainingExample` objects into masked token sequences.

    Args:
        tokenizer: Object satisfying :class:`ChatTokenizer`.
        max_seq_length: Truncation limit.
        train_on_last_turn_only: Supervise only the final assistant message.
            Useful when earlier turns are context rather than target behaviour.
        template_kwargs: Extra kwargs forwarded to ``apply_chat_template`` — this
            is how reasoning-mode controls such as Qwen's ``enable_thinking``
            reach the template.
        strip_reasoning: Remove ``<think>`` spans from assistant targets before
            tokenizing. Keep this on: we train decision policy, not display
            chain-of-thought.
    """

    def __init__(
        self,
        tokenizer: ChatTokenizer,
        *,
        max_seq_length: int = 2048,
        train_on_last_turn_only: bool = False,
        template_kwargs: dict[str, Any] | None = None,
        strip_reasoning: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_last_turn_only = train_on_last_turn_only
        self.template_kwargs = dict(template_kwargs or {})
        self.strip_reasoning = strip_reasoning
        self._warned_unstable_template = False

    # -- template helpers ---------------------------------------------------

    def render(self, messages: Sequence[Message], *, add_generation_prompt: bool = False) -> str:
        """Render messages to text using the tokenizer's chat template."""
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages_to_dicts(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self.template_kwargs,
            )
        except TypeError as exc:
            # A template that does not accept one of our kwargs (for example a
            # Thinking-only checkpoint refusing enable_thinking) should say so
            # clearly rather than fail deep inside jinja.
            raise DataValidationError(
                "The tokenizer's chat template rejected the supplied arguments.",
                details={
                    "template_kwargs": self.template_kwargs,
                    "error": str(exc),
                },
                suggestions=[
                    "Reasoning-mode kwargs are model-specific: Qwen3-8B accepts "
                    "enable_thinking, Qwen3-30B-A3B-Thinking-2507 does not.",
                    "Check the model config's reasoning section against the checkpoint.",
                ],
            ) from exc
        if not isinstance(rendered, str):  # pragma: no cover - tokenize=False given
            raise DataValidationError(
                "apply_chat_template(tokenize=False) did not return a string.",
                details={"returned_type": type(rendered).__name__},
            )
        return rendered

    def _encode(self, text: str) -> list[int]:
        # The chat template already inserts BOS/special tokens; adding them again
        # produces a doubled BOS, which shifts every position by one.
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    # -- masking ------------------------------------------------------------

    def _assistant_spans(
        self, messages: Sequence[Message]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize the conversation and locate assistant token spans.

        Returns:
            ``(input_ids, spans)`` where each span is ``[start, end)`` over
            ``input_ids`` covering one assistant turn's content.
        """
        assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]
        if self.train_on_last_turn_only and assistant_indices:
            assistant_indices = assistant_indices[-1:]

        full_ids = self._encode(self.render(messages, add_generation_prompt=False))
        spans: list[tuple[int, int]] = []

        for index in assistant_indices:
            prefix_messages = messages[:index]
            # Render the conversation as the model would see it just before
            # producing this turn, including the generation prompt.
            prefix_text = self.render(prefix_messages, add_generation_prompt=True)
            prefix_ids = self._encode(prefix_text)

            through_text = self.render(messages[: index + 1], add_generation_prompt=False)
            through_ids = self._encode(through_text)

            shared = _common_prefix_length(prefix_ids, through_ids)
            if shared != len(prefix_ids) and not self._warned_unstable_template:
                self._warned_unstable_template = True
                logger.warning(
                    "Chat template is not prefix-stable (prefix len %d, shared %d). "
                    "Falling back to the longest common prefix for masking. Verify "
                    "the resulting spans with scripts/inspect_dataset.py --show-masking.",
                    len(prefix_ids),
                    shared,
                )

            start = shared
            end = min(len(through_ids), len(full_ids))
            # Guard against a degenerate span; an empty target teaches nothing.
            if end > start:
                spans.append((start, end))

        return full_ids, spans

    def format_example(self, example: TrainingExample) -> FormattedExample | None:
        """Format a single example, or return ``None`` if it cannot be supervised.

        Returns ``None`` when the example yields no assistant tokens at all,
        which would otherwise contribute a NaN loss.
        """
        source = example.strip_reasoning_spans() if self.strip_reasoning else example
        input_ids, spans = self._assistant_spans(source.messages)

        if not spans:
            logger.warning("Example %s produced no assistant token span; skipping it.", example.id)
            return None

        labels = [IGNORE_INDEX] * len(input_ids)
        for start, end in spans:
            for position in range(start, min(end, len(input_ids))):
                labels[position] = input_ids[position]

        truncated = False
        target_truncated = False
        if len(input_ids) > self.max_seq_length:
            truncated = True
            kept_targets = sum(
                1 for label in labels[: self.max_seq_length] if label != IGNORE_INDEX
            )
            total_targets = sum(1 for label in labels if label != IGNORE_INDEX)
            target_truncated = kept_targets < total_targets
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        target_count = sum(1 for label in labels if label != IGNORE_INDEX)
        if target_count == 0:
            # Truncation removed the entire answer. Training on this example would
            # produce a loss over nothing.
            logger.warning(
                "Example %s has no supervised tokens after truncation to %d; skipping. "
                "Increase model.max_seq_length or shorten the example.",
                example.id,
                self.max_seq_length,
            )
            return None

        return FormattedExample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=[1] * len(input_ids),
            example_id=example.id,
            target_token_count=target_count,
            truncated=truncated,
            target_truncated=target_truncated,
        )

    def format_dataset(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[FormattedExample], FormattingStats]:
        """Format every example, collecting statistics.

        Examples that cannot be supervised are dropped and counted, never
        silently included with an all-ignored label row.
        """
        stats = FormattingStats()
        formatted: list[FormattedExample] = []

        for example in examples:
            result = self.format_example(example)
            if result is None:
                stats.dropped += 1
                stats.empty_targets += 1
                continue
            formatted.append(result)
            stats.total += 1
            stats.token_lengths.append(len(result.input_ids))
            stats.target_lengths.append(result.target_token_count)
            if result.truncated:
                stats.truncated += 1
            if result.target_truncated:
                stats.target_truncated += 1

        if stats.truncated:
            logger.warning(
                "%d of %d example(s) exceeded max_seq_length=%d and were truncated "
                "(%d lost part of the assistant target).",
                stats.truncated,
                stats.total + stats.dropped,
                self.max_seq_length,
                stats.target_truncated,
            )
        if stats.dropped:
            logger.warning(
                "%d example(s) were dropped because they produced no supervised tokens.",
                stats.dropped,
            )
        return formatted, stats

    def render_prompt(self, messages: Sequence[Message]) -> str:
        """Render an inference prompt, ending with the generation prompt."""
        return self.render(messages, add_generation_prompt=True)

    def describe_masking(self, example: TrainingExample) -> str:
        """Human-readable dump of what is supervised, for debugging.

        Prints token counts rather than raw content, so it is safe to run against
        a private dataset without printing user data.
        """
        formatted = self.format_example(example)
        if formatted is None:
            return f"{example.id}: no supervised tokens"
        supervised = formatted.target_token_count
        total = len(formatted.input_ids)
        return (
            f"{example.id}: {supervised}/{total} token(s) supervised "
            f"({supervised / total:.1%}), truncated={formatted.truncated}"
        )
