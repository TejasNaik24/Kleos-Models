"""Shared vocabulary for the KLEOS research pipeline."""

from __future__ import annotations

from typing import Final

#: Example schema version. Bump on any breaking change to required fields.
DATASET_SCHEMA_VERSION: Final[str] = "1.0"

#: Schema versions read. 1.1 adds an optional assistant ``reasoning``; writes stay 1.0.
SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("1.0", "1.1")

#: Bump when formatting would change tokenized output for identical examples.
PREPROCESSING_VERSION: Final[str] = "1.0"


#: Tasks the pipeline can train and evaluate. Registered is not "should be trained":
#: each task must be trainable in isolation.
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


#: Axes along which a dataset must demonstrate coverage (see ``data.coverage``).
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

#: Required on every example; other axes are reported as "missing" when absent.
REQUIRED_VARIATION_AXES: Final[tuple[str, ...]] = ("domain",)


#: ``real_sanitized`` was sanitized in the PRIVATE repository; raw real data must
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


#: Split strategies (spec section 12). ``random`` is development-only; claims need a holdout.
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

#: Research arms (spec section 20). Never pool results across arms.
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

#: Dimensions where a higher raw score is worse; inverted before aggregation.
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
