"""Shared vocabulary for the KLEOS research pipeline.

Anything that must stay stable across dataset versions, experiment manifests and
evaluation reports lives here. Changing a value in this module is a *research*
change, not a refactor: it can invalidate comparisons between existing runs.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

#: Version of the training/evaluation example schema. Bump on any breaking
#: change to required fields. Datasets record the schema version they were
#: written against so old artifacts remain interpretable.
DATASET_SCHEMA_VERSION: Final[str] = "1.0"

#: Version of the preprocessing/formatting logic. Bump when formatting changes
#: in a way that would alter tokenized output for identical source examples.
PREPROCESSING_VERSION: Final[str] = "1.0"


# ---------------------------------------------------------------------------
# Task registry (spec section 8)
# ---------------------------------------------------------------------------

#: Tasks the pipeline knows how to train and evaluate.
#:
#: Registering a task here does NOT mean it should be trained. The research plan
#: explicitly rejects the assumption that a single monolithic "KLEOS personality"
#: fine-tune is the right approach, so each task must be trainable in isolation.
SUPPORTED_TASKS: Final[tuple[str, ...]] = (
    "notification_prioritization",
    "tool_routing",
    "mission_control_briefing",
    "context_prioritization",
    "recommendation_generation",
    "memory_conflict_resolution",
    "workspace_reasoning",
)

#: Human-readable descriptions, surfaced in dataset reports and model cards.
TASK_DESCRIPTIONS: Final[dict[str, str]] = {
    "notification_prioritization": (
        "Rank or triage competing notifications by urgency, evidence and impact."
    ),
    "tool_routing": ("Decide which tool (if any) should handle a request, given availability."),
    "mission_control_briefing": (
        "Produce a structured briefing that synthesizes multiple context sources."
    ),
    "context_prioritization": (
        "Select and order the context items most relevant to answering a request."
    ),
    "recommendation_generation": (
        "Recommend next actions grounded in supplied evidence, with justification."
    ),
    "memory_conflict_resolution": (
        "Resolve contradictions between memory records using recency and reliability."
    ),
    "workspace_reasoning": (
        "Reason within workspace scoping rules without leaking across workspaces."
    ),
}


# ---------------------------------------------------------------------------
# Variation axes (spec section 10)
# ---------------------------------------------------------------------------

#: Axes along which a dataset must demonstrate coverage.
#:
#: A dataset is not "diverse" because it is large. It is diverse when each
#: underlying situation type is represented. ``kleos_models.data.coverage``
#: reports counts per axis and per axis combination.
VARIATION_AXES: Final[tuple[str, ...]] = (
    "domain",
    "entities",
    "urgency",
    "deadlines",
    "evidence_quality",
    "conflicting_evidence",
    "context_length",
    "presentation_order",
    "format",
    "source_type",
    "workspace",
    "task",
    "difficulty",
    "ambiguity",
)

#: Axes that must be present on every example. The remainder are optional but
#: are reported as "missing" in coverage output so gaps stay visible.
REQUIRED_VARIATION_AXES: Final[tuple[str, ...]] = ("domain",)


# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------

#: Where an example came from. ``real_sanitized`` means it originated from real
#: usage and passed sanitization in the PRIVATE repository; raw real data must
#: never appear in this public repository.
SOURCE_TYPES: Final[tuple[str, ...]] = (
    "synthetic",
    "synthetic_seeded",
    "real_sanitized",
    "expert_authored",
    "development_fixture",
)

#: Human review state. Only ``reviewed`` examples should enter a research run.
QUALITY_STATUSES: Final[tuple[str, ...]] = (
    "draft",
    "auto_generated",
    "reviewed",
    "rejected",
)

#: Message roles permitted in a conversation.
MESSAGE_ROLES: Final[tuple[str, ...]] = ("system", "user", "assistant", "tool")


# ---------------------------------------------------------------------------
# Splitting and evaluation
# ---------------------------------------------------------------------------

#: Split strategies (spec section 12). ``random`` is for development only;
#: generalization claims require a held-out strategy.
SPLIT_STRATEGIES: Final[tuple[str, ...]] = (
    "random",
    "group",
    "entity_holdout",
    "domain_holdout",
    "format_holdout",
    "scenario_family_holdout",
)

#: Named split partitions.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Research arms (spec section 20). Each arm is a distinct experimental
#: condition; results must never be pooled across arms.
RESEARCH_ARMS: Final[tuple[str, ...]] = (
    "arm0_base",
    "arm1_base_orchestrated",
    "arm2_finetuned",
    "arm3_finetuned_orchestrated",
    "frontier_zero_shot",
    "frontier_orchestrated",
)

ARM_DESCRIPTIONS: Final[dict[str, str]] = {
    "arm0_base": "Base open-weight model, standard prompting, no orchestration.",
    "arm1_base_orchestrated": "Base model plus KLEOS retrieval/context/prompt scaffolding.",
    "arm2_finetuned": "Fine-tuned model with controlled task context, no full orchestration.",
    "arm3_finetuned_orchestrated": "Fine-tuned model plus KLEOS orchestration.",
    "frontier_zero_shot": "Frontier reference model, zero-shot (not implemented by default).",
    "frontier_orchestrated": "Frontier reference model with orchestration (not implemented).",
}

#: Default rubric dimensions for structured grading (spec section 21).
RUBRIC_DIMENSIONS: Final[tuple[str, ...]] = (
    "correctness",
    "evidence_usage",
    "prioritization_quality",
    "actionability",
    "relevance",
    "critical_omission",
    "unsupported_claims",
    "consistency",
)

#: Rubric dimensions where a HIGHER raw score is WORSE. These are inverted
#: before aggregation so that every reported score is "higher is better".
NEGATIVE_RUBRIC_DIMENSIONS: Final[frozenset[str]] = frozenset(
    {"critical_omission", "unsupported_claims"}
)

#: Perturbation kinds used by consistency testing (spec section 22).
PERTURBATION_KINDS: Final[tuple[str, ...]] = (
    "paraphrase",
    "evidence_order",
    "context_order",
    "irrelevant_context",
    "formatting",
    "schema",
    "length",
)

#: OOD shift kinds (spec section 23).
OOD_SHIFT_KINDS: Final[tuple[str, ...]] = (
    "unseen_entities",
    "unseen_domains",
    "unseen_formats",
    "unseen_source_types",
    "context_length_shift",
    "reordered_evidence",
    "conflicting_evidence",
)


# ---------------------------------------------------------------------------
# Filesystem conventions
# ---------------------------------------------------------------------------

MANIFEST_FILENAME: Final[str] = "manifest.json"
DATASET_MANIFEST_FILENAME: Final[str] = "manifest.json"
METRICS_FILENAME: Final[str] = "metrics.json"
EFFECTIVE_CONFIG_FILENAME: Final[str] = "config.yaml"
TRAINING_LOG_FILENAME: Final[str] = "training.log"
STRUCTURED_LOG_FILENAME: Final[str] = "events.jsonl"
ADAPTER_DIRNAME: Final[str] = "adapter"
TOKENIZER_DIRNAME: Final[str] = "tokenizer"
CHECKPOINT_PREFIX: Final[str] = "checkpoint-"

#: Label id that PyTorch cross-entropy ignores. Used for prompt masking.
IGNORE_INDEX: Final[int] = -100
