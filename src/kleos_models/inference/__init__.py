"""Inference layer: generation backends and orchestration prompt assembly."""

from kleos_models.inference.backends import (
    Backend,
    BaseBackend,
    EchoBackend,
    FrontierAPIBackend,
    GenerationOutput,
    HuggingFaceBackend,
    build_backend,
)
from kleos_models.inference.generate import (
    DEFAULT_ORCHESTRATION_PROMPT,
    OrchestrationConfig,
    build_prompt,
    collect_evidence_ids,
    extract_context_text,
    summarize_arm,
)

__all__ = [
    "DEFAULT_ORCHESTRATION_PROMPT",
    "Backend",
    "BaseBackend",
    "EchoBackend",
    "FrontierAPIBackend",
    "GenerationOutput",
    "HuggingFaceBackend",
    "OrchestrationConfig",
    "build_backend",
    "build_prompt",
    "collect_evidence_ids",
    "extract_context_text",
    "summarize_arm",
]
