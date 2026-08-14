"""Versioned data contract for KLEOS training and evaluation (spec section 7).

Every example is a conversation plus the metadata needed to reason about
*coverage* and *provenance*. The metadata is not decoration: variation axes are
what let us answer "how many examples cover each underlying situation type?"
rather than mistaking a large dataset for a diverse one.

These pydantic models are the source of truth. ``data/schema/*.schema.json`` is
generated from them (``scripts/prepare_dataset.py --emit-schemas``) and a test
asserts the two never drift apart.

This module imports no torch.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kleos_models.constants import (
    DATASET_SCHEMA_VERSION,
    MESSAGE_ROLES,
    PREPROCESSING_VERSION,
    QUALITY_STATUSES,
    SOURCE_TYPES,
    SUPPORTED_TASKS,
    VARIATION_AXES,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")

#: Reasoning spans are stripped before training. We never teach a model to emit
#: display chain-of-thought (spec section 39).
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_DANGLING_CLOSE = re.compile(r"^\s*.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning spans from assistant text.

    Handles both a well-formed ``<think>…</think>`` block and the Qwen Thinking
    case where the template pre-opens ``<think>`` so generated text contains only
    the closing tag.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = _DANGLING_CLOSE.sub("", cleaned, count=1)
    return cleaned.strip()


def contains_reasoning(text: str) -> bool:
    """Whether the text carries a reasoning span."""
    return "</think>" in text.lower()


class Message(BaseModel):
    """One conversation turn."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = Field(default=None, description="Tool name for role='tool'.")

    @field_validator("content")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def _tool_needs_name(self) -> Message:
        if self.role == "tool" and not self.name:
            raise ValueError("messages with role='tool' require a 'name'")
        return self


class VariationAxes(BaseModel):
    """Situation descriptors used for coverage reporting (spec section 10).

    Extra keys are permitted so new axes can be piloted in the private dataset
    before being promoted into :data:`kleos_models.constants.VARIATION_AXES`.
    Unknown axes are surfaced by the coverage report rather than silently kept.
    """

    model_config = ConfigDict(extra="allow")

    domain: str = Field(description="Life/work domain, e.g. 'career', 'research'.")
    entities: str | None = Field(
        default=None, description="'seen' | 'unseen' | free-form entity-set label."
    )
    urgency: str | None = None
    deadlines: str | None = None
    evidence_quality: str | None = None
    conflicting_evidence: str | None = None
    context_length: str | None = None
    presentation_order: str | None = None
    format: str | None = None
    source_type: str | None = None
    workspace: str | None = None
    difficulty: str | None = None
    ambiguity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """All axes with a value, including extras."""
        base = {k: v for k, v in self.model_dump().items() if v is not None}
        return base

    def known_axes(self) -> dict[str, Any]:
        """Only axes registered in ``VARIATION_AXES``."""
        return {k: v for k, v in self.as_dict().items() if k in VARIATION_AXES}

    def unknown_axes(self) -> dict[str, Any]:
        """Axes present on the example but not registered."""
        return {k: v for k, v in self.as_dict().items() if k not in VARIATION_AXES}


class ExampleMetadata(BaseModel):
    """Provenance and grouping information."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(
        default="synthetic",
        description="Provenance. Raw real user data must never appear in this repo.",
    )
    quality_status: str = Field(default="draft")
    scenario_family: str | None = Field(
        default=None,
        description=(
            "Groups logically related examples. Used by scenario-family-held-out "
            "splitting and by consistency testing, which needs perturbations of "
            "one scenario to stay together."
        ),
    )
    group_id: str | None = Field(default=None, description="Generic grouping key for splitting.")
    perturbation_of: str | None = Field(
        default=None, description="Id of the example this one perturbs."
    )
    perturbation_kind: str | None = None
    author: str | None = Field(default=None, description="Pipeline or role, never a person.")
    created_at: str | None = None
    notes: str | None = None

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"unknown source {value!r}; valid: {list(SOURCE_TYPES)}")
        return value

    @field_validator("quality_status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        if value not in QUALITY_STATUSES:
            raise ValueError(f"unknown quality_status {value!r}; valid: {list(QUALITY_STATUSES)}")
        return value


class TrainingExample(BaseModel):
    """One supervised conversational training example."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier, unique within a dataset version.")
    version: str = Field(default=DATASET_SCHEMA_VERSION, description="Schema version.")
    task: str = Field(description="Registered KLEOS task.")
    domain: str | None = Field(default=None, description="Convenience mirror of axes.domain.")
    messages: list[Message] = Field(min_length=2)
    variation_axes: VariationAxes
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"id {value!r} must be 3-128 chars of [A-Za-z0-9._:-] and start alphanumeric"
            )
        return value

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @model_validator(mode="after")
    def _conversation_shape(self) -> TrainingExample:
        roles = [m.role for m in self.messages]

        # A supervised example must have something to learn from.
        if "assistant" not in roles:
            raise ValueError("a training example needs at least one assistant message")
        if roles[-1] != "assistant":
            raise ValueError(
                f"the final message must be from the assistant, got role={roles[-1]!r}"
            )
        # System messages only lead.
        for index, role in enumerate(roles):
            if role == "system" and index != 0:
                raise ValueError(f"system message must be first, found one at index {index}")
        # There must be a user turn before the first assistant turn.
        first_assistant = roles.index("assistant")
        if "user" not in roles[:first_assistant]:
            raise ValueError("the first assistant message must be preceded by a user message")
        # No two consecutive assistant turns: that is a malformed conversation and
        # produces ambiguous supervision spans.
        for index in range(1, len(roles)):
            if roles[index] == "assistant" and roles[index - 1] == "assistant":
                raise ValueError(f"consecutive assistant messages at index {index - 1} and {index}")

        # Keep the convenience mirror consistent with the axes.
        if self.domain is None:
            self.domain = self.variation_axes.domain
        elif self.domain != self.variation_axes.domain:
            raise ValueError(
                f"domain={self.domain!r} contradicts variation_axes.domain="
                f"{self.variation_axes.domain!r}"
            )
        return self

    # -- derived views ------------------------------------------------------

    @property
    def system_prompt(self) -> str | None:
        """Leading system message, if any."""
        first = self.messages[0]
        return first.content if first.role == "system" else None

    @property
    def assistant_targets(self) -> list[str]:
        """Assistant message contents, in order."""
        return [m.content for m in self.messages if m.role == "assistant"]

    def conversation_text(self, *, include_assistant: bool = True) -> str:
        """Flat text view used for duplicate and leakage detection."""
        parts = [
            f"{m.role}: {m.content}"
            for m in self.messages
            if include_assistant or m.role != "assistant"
        ]
        return "\n".join(parts)

    def content_hash(self, *, include_assistant: bool = True) -> str:
        """Digest of the conversation content, ignoring id and metadata."""
        text = self.conversation_text(include_assistant=include_assistant)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def group_key(self, key: str | None = None) -> str:
        """Resolve the grouping key used by group-aware splitting.

        Falls back through ``group_id`` → ``scenario_family`` → the example's own
        id, so every example always belongs to exactly one group.
        """
        if key:
            value = getattr(self.metadata, key, None)
            if value is None:
                value = self.metadata.model_extra.get(key) if self.metadata.model_extra else None
            if value is None:
                value = self.variation_axes.as_dict().get(key)
            if value is not None:
                return str(value)
        return self.metadata.group_id or self.metadata.scenario_family or self.id

    def strip_reasoning_spans(self) -> TrainingExample:
        """Return a copy with reasoning removed from every assistant turn.

        Applied before tokenization so hidden chain-of-thought never becomes a
        training target.
        """
        messages = [
            m.model_copy(update={"content": strip_reasoning(m.content)})
            if m.role == "assistant" and contains_reasoning(m.content)
            else m
            for m in self.messages
        ]
        return self.model_copy(update={"messages": messages})


class EvaluationExample(BaseModel):
    """One benchmark item.

    Unlike a training example, the assistant turn is optional: what matters is
    the *reference* — the decision the grader checks against.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str = Field(default=DATASET_SCHEMA_VERSION)
    task: str
    messages: list[Message] = Field(min_length=1)
    variation_axes: VariationAxes
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)

    reference: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Ground truth for graders: {'label': ...}, {'ranking': [...]}, "
            "{'text': ...}, {'required_points': [...]}, {'evidence_ids': [...]}."
        ),
    )
    grader: str = Field(default="exact_match", description="Grader id from evaluation.graders.")
    split_tag: str | None = Field(
        default=None,
        description="'in_distribution' | 'ood' | 'capability'. Drives separate reporting.",
    )
    ood_shift: str | None = Field(default=None, description="OOD shift kind, when applicable.")

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(f"id {value!r} is not a valid example id")
        return value

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @model_validator(mode="after")
    def _prompt_shape(self) -> EvaluationExample:
        roles = [m.role for m in self.messages]
        if "user" not in roles:
            raise ValueError("an evaluation example needs at least one user message")
        for index, role in enumerate(roles):
            if role == "system" and index != 0:
                raise ValueError(f"system message must be first, found one at index {index}")
        if not self.reference:
            raise ValueError(
                "an evaluation example needs a non-empty 'reference'; "
                "without ground truth it cannot be graded"
            )
        return self

    @property
    def prompt_messages(self) -> list[Message]:
        """Messages fed to the model: everything up to the first assistant turn."""
        result: list[Message] = []
        for message in self.messages:
            if message.role == "assistant":
                break
            result.append(message)
        return result

    def conversation_text(self) -> str:
        """Flat prompt text, used for leakage checks against the training set."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.prompt_messages)

    def content_hash(self) -> str:
        return hashlib.sha256(self.conversation_text().encode("utf-8")).hexdigest()


class SplitCounts(BaseModel):
    """Example counts per partition."""

    model_config = ConfigDict(extra="forbid")

    train: int = 0
    validation: int = 0
    test: int = 0

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test


class DatasetManifest(BaseModel):
    """Dataset-level provenance (spec section 26).

    A dataset version is immutable. ``scripts/split_dataset.py`` refuses to
    overwrite an existing version directory, because silently rewriting a version
    breaks every comparison that referenced it.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="e.g. 'kleos-policy-v0.1.0'.")
    schema_version: str = Field(default=DATASET_SCHEMA_VERSION)
    preprocessing_version: str = Field(default=PREPROCESSING_VERSION)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: str = Field(
        default="synthetic",
        description="Where the data came from, at a level safe to publish.",
    )
    description: str = ""

    example_count: int = Field(default=0, ge=0)
    splits: SplitCounts = Field(default_factory=SplitCounts)
    task_distribution: dict[str, int] = Field(default_factory=dict)
    domain_distribution: dict[str, int] = Field(default_factory=dict)
    source_distribution: dict[str, int] = Field(default_factory=dict)
    quality_distribution: dict[str, int] = Field(default_factory=dict)

    split_strategy: str | None = None
    split_seed: int | None = None
    holdout_values: list[str] = Field(default_factory=list)

    file_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="sha256 per split file, so a manifest identifies exact content.",
    )
    content_hash: str | None = Field(
        default=None, description="Digest over all split hashes combined."
    )

    contains_private_data: bool = Field(
        default=False,
        description=(
            "Set true only in the PRIVATE repository. A dataset with this flag "
            "must never be committed here; loaders warn loudly when they see it."
        ),
    )
    notes: str | None = None

    def compute_content_hash(self) -> str:
        """Deterministic digest over the per-file hashes."""
        payload = json.dumps(self.file_hashes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def finalize(self) -> DatasetManifest:
        """Fill in the aggregate content hash."""
        self.content_hash = self.compute_content_hash()
        return self


#: Message roles as a set, for fast validation in hot loops.
VALID_ROLES = frozenset(MESSAGE_ROLES)
