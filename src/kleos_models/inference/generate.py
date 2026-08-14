"""Orchestration scaffolding and prompt assembly for evaluation arms.

Spec section 20 defines four arms. The distinction between them is *what context
the model receives*, not which model it is:

``arm0_base``                   base model, task prompt only
``arm1_base_orchestrated``      base model + KLEOS retrieval/context scaffolding
``arm2_finetuned``              fine-tuned model, controlled task context
``arm3_finetuned_orchestrated`` fine-tuned model + the same scaffolding

This module owns that difference so the runner does not have to. Scaffolding is
applied identically to base and fine-tuned arms, which is what makes
"orchestration helps" and "fine-tuning helps" separable questions rather than one
confounded one.

The orchestration layer here is a *prompt-assembly* layer. It does not query
Supabase or any production service (spec section 51): retrieval output arrives as
data in the benchmark example.

This module imports no torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kleos_models.data.schemas import EvaluationExample, Message
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Default scaffolding. Deliberately about *procedure*, not about any user.
DEFAULT_ORCHESTRATION_PROMPT = """\
You are the reasoning layer of KLEOS, an AI operating system for computer science students.

Deterministic systems have already retrieved, scored and ordered the context below.
Your job is judgment, not retrieval.

Follow these policies:
1. Ground every claim in the supplied context. If the context does not support a
   claim, do not make it.
2. When signals conflict, prefer the more recent and better-evidenced one, and say
   which you preferred and why.
3. When prioritizing, weigh deadline proximity, evidence strength and stated impact.
4. State the decision explicitly before the justification.
5. If the context is insufficient to decide, say so rather than guessing.
"""


@dataclass
class OrchestrationConfig:
    """How much scaffolding an arm applies."""

    enabled: bool = False
    system_prompt: str | None = None
    include_policy_reminder: bool = True
    context_header: str = "## Retrieved context"

    @classmethod
    def for_arm(cls, arm: str, *, prompt_path: Path | str | None = None) -> OrchestrationConfig:
        """Build the orchestration config for a research arm."""
        orchestrated = arm in ("arm1_base_orchestrated", "arm3_finetuned_orchestrated")
        prompt: str | None = None
        if orchestrated:
            if prompt_path and Path(prompt_path).exists():
                prompt = Path(prompt_path).read_text(encoding="utf-8")
                logger.info("Loaded orchestration prompt from %s", prompt_path)
            else:
                prompt = DEFAULT_ORCHESTRATION_PROMPT
        return cls(enabled=orchestrated, system_prompt=prompt)


def build_prompt(
    example: EvaluationExample,
    orchestration: OrchestrationConfig,
) -> list[Message]:
    """Assemble the message list a backend will receive.

    The example's own messages are always preserved verbatim. Orchestration only
    *prepends* a system prompt, so the task the model is asked to perform is
    identical across arms and the arms differ only in scaffolding.
    """
    messages = list(example.prompt_messages)

    if not orchestration.enabled or not orchestration.system_prompt:
        return messages

    if messages and messages[0].role == "system":
        # Compose rather than replace: dropping the task's own system prompt would
        # change the task, not just the scaffolding.
        combined = f"{orchestration.system_prompt.strip()}\n\n{messages[0].content}"
        return [messages[0].model_copy(update={"content": combined}), *messages[1:]]

    return [Message(role="system", content=orchestration.system_prompt.strip()), *messages]


def extract_context_text(example: EvaluationExample) -> str:
    """Concatenate everything the model was shown, for faithfulness checking."""
    return "\n\n".join(m.content for m in example.prompt_messages)


def collect_evidence_ids(example: EvaluationExample) -> list[str]:
    """Evidence identifiers the example supplied to the model."""
    provided = example.reference.get("provided_evidence_ids")
    if isinstance(provided, list):
        return [str(item) for item in provided]

    from kleos_models.evaluation.faithfulness import extract_citations

    return extract_citations(extract_context_text(example))


def summarize_arm(arm: str, orchestration: OrchestrationConfig) -> dict[str, Any]:
    """Manifest-ready description of an arm's configuration."""
    from kleos_models.constants import ARM_DESCRIPTIONS

    return {
        "arm": arm,
        "description": ARM_DESCRIPTIONS.get(arm, "unknown arm"),
        "orchestration_enabled": orchestration.enabled,
        "orchestration_prompt_chars": (
            len(orchestration.system_prompt) if orchestration.system_prompt else 0
        ),
    }


def render_conversation(messages: Sequence[Message], *, max_chars: int = 400) -> str:
    """Truncated conversation rendering for debugging.

    Truncates by design: evaluation prompts can contain sanitized-but-still-
    sensitive context, and full prompts do not belong in logs.
    """
    parts = []
    for message in messages:
        content = message.content
        if len(content) > max_chars:
            content = f"{content[:max_chars]}… (+{len(message.content) - max_chars} chars)"
        parts.append(f"[{message.role}] {content}")
    return "\n".join(parts)
