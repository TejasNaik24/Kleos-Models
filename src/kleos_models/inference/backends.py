"""Generation backends (spec sections 19, 20).

The evaluation harness talks to a :class:`Backend`, never to a model directly.
That indirection is what makes the research arms possible:

* ``arm0_base`` and ``arm2_finetuned`` differ only by whether a LoRA adapter is
  attached — same base weights, same decoding settings, same prompts.
* ``arm1`` / ``arm3`` add the orchestration scaffolding on top of either.
* A frontier reference can be added later by implementing this protocol, with no
  change to task definitions, graders or the runner.

Spec section 20 says not to implement proprietary APIs unless asked, so
:class:`FrontierAPIBackend` is a documented stub that raises rather than a
half-working client.

An ``EchoBackend`` is included for testing the harness itself without a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kleos_models.config import GenerationConfig, ModelConfig, ReasoningMode
from kleos_models.constants import RESEARCH_ARMS
from kleos_models.data.schemas import Message
from kleos_models.errors import EvaluationError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationOutput:
    """One generated response plus what it took to produce it."""

    text: str
    reasoning: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "has_reasoning": self.reasoning is not None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


@runtime_checkable
class Backend(Protocol):
    """What the evaluation runner needs from anything that generates text."""

    name: str

    def generate(
        self, messages: Sequence[Message], config: GenerationConfig, **kwargs: Any
    ) -> GenerationOutput:
        """Produce one response for a conversation."""
        ...

    def describe(self) -> dict[str, Any]:
        """Manifest-ready identity of this backend."""
        ...


class BaseBackend(ABC):
    """Convenience base implementing batching over :meth:`generate`."""

    name: str = "backend"

    @abstractmethod
    def generate(
        self, messages: Sequence[Message], config: GenerationConfig, **kwargs: Any
    ) -> GenerationOutput:
        """Produce one response."""

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        config: GenerationConfig,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate for several conversations. Override for real batching."""
        return [self.generate(messages, config, **kwargs) for messages in conversations]

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Manifest-ready identity."""


class HuggingFaceBackend(BaseBackend):
    """Local generation through transformers, with or without a LoRA adapter.

    This single class serves both the base and fine-tuned arms. Whether an adapter
    is attached is the *only* difference between them, which is precisely the
    control the experiment needs — anything else that differed would confound the
    comparison.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        adapter_path: Path | str | None = None,
        reasoning_mode: ReasoningMode | None = None,
        name: str | None = None,
    ) -> None:
        from kleos_models.models.loading import load_adapter_model, load_model

        self.model_config = model_config
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.name = name or ("finetuned" if adapter_path else "base")

        if self.adapter_path is not None:
            self.loaded = load_adapter_model(
                model_config, self.adapter_path, reasoning_mode=reasoning_mode
            )
        else:
            self.loaded = load_model(
                model_config, reasoning_mode=reasoning_mode, for_training=False
            )
        self.loaded.model.eval()

        from kleos_models.data.formatting import ConversationFormatter

        self.formatter = ConversationFormatter(
            self.loaded.tokenizer,
            max_seq_length=model_config.max_seq_length,
            template_kwargs=self.loaded.chat_template_kwargs(),
        )

    def generate(
        self, messages: Sequence[Message], config: GenerationConfig, **kwargs: Any
    ) -> GenerationOutput:
        import torch

        prompt = self.formatter.render_prompt(messages)
        inputs = self.loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.loaded.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[-1])

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.loaded.tokenizer.pad_token_id,
            "repetition_penalty": config.repetition_penalty,
        }
        # Passing temperature/top_p with do_sample=False triggers warnings and has
        # no effect; greedy decoding is the deterministic default for evaluation.
        if config.do_sample:
            generate_kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
            if config.top_k:
                generate_kwargs["top_k"] = config.top_k

        with torch.no_grad():
            output_ids = self.loaded.model.generate(**inputs, **generate_kwargs)

        completion_ids = output_ids[0][prompt_length:]
        text = self.loaded.tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Reasoning models emit a thinking span. Separate it from the answer:
        # graders score the decision and its justification, not hidden
        # chain-of-thought (spec section 39).
        reasoning: str | None = None
        answer = text
        if "</think>" in text:
            head, _, tail = text.partition("</think>")
            reasoning = head.replace("<think>", "").strip()
            answer = tail.strip()

        return GenerationOutput(
            text=answer,
            reasoning=reasoning,
            prompt_tokens=prompt_length,
            completion_tokens=int(completion_ids.shape[-1]),
            metadata={"backend": self.name, "reasoning_mode": self.loaded.reasoning_mode.value},
        )

    def describe(self) -> dict[str, Any]:
        description = self.loaded.describe()
        description["backend"] = self.name
        description["adapter_path"] = str(self.adapter_path) if self.adapter_path else None
        return description


class EchoBackend(BaseBackend):
    """Deterministic stub backend for testing the harness itself.

    Returns a canned response, so metrics, graders, consistency, OOD and report
    generation can all be exercised in CI with no model and no GPU. It is never a
    substitute for a real evaluation, and its ``describe()`` says so.
    """

    name = "echo"

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls = 0

    def generate(
        self, messages: Sequence[Message], config: GenerationConfig, **kwargs: Any
    ) -> GenerationOutput:
        self.calls += 1
        example_id = kwargs.get("example_id", "")
        text = self.responses.get(
            example_id,
            self.default or (messages[-1].content if messages else ""),
        )
        return GenerationOutput(
            text=text,
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(text.split()),
            metadata={"backend": self.name, "synthetic": True},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "synthetic": True,
            "warning": (
                "EchoBackend produces canned responses. Results from it exercise the "
                "harness and mean nothing about any model."
            ),
        }


class FrontierAPIBackend(BaseBackend):
    """Interface for a hosted frontier reference model (spec section 20).

    Intentionally not implemented. The arms exist in the design
    (``frontier_zero_shot``, ``frontier_orchestrated``) so that adding a provider
    later requires implementing :meth:`generate` here and nothing else — no change
    to tasks, graders, or the runner.
    """

    name = "frontier"

    def __init__(self, model: str = "unspecified", **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    def generate(
        self, messages: Sequence[Message], config: GenerationConfig, **kwargs: Any
    ) -> GenerationOutput:
        raise EvaluationError(
            "The frontier reference backend is not implemented.",
            details={"requested_model": self.model},
            suggestions=[
                "Frontier arms are part of the experimental design but no proprietary "
                "API client ships with this repository.",
                "Implement FrontierAPIBackend.generate() to add one; the runner, "
                "graders and task definitions need no changes.",
                "Meanwhile, restrict evaluation.arms to the open-weight arms.",
            ],
        )

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "implemented": False}


def build_backend(
    arm: str,
    model_config: ModelConfig,
    *,
    adapter_path: Path | str | None = None,
    reasoning_mode: ReasoningMode | None = None,
) -> BaseBackend:
    """Construct the backend for a research arm.

    Raises:
        EvaluationError: when a fine-tuned arm is requested without an adapter, or
            the arm is unknown.
    """
    if arm in ("arm0_base", "arm1_base_orchestrated"):
        return HuggingFaceBackend(model_config, reasoning_mode=reasoning_mode, name=arm)

    if arm in ("arm2_finetuned", "arm3_finetuned_orchestrated"):
        if adapter_path is None:
            raise EvaluationError(
                f"Arm {arm!r} evaluates a fine-tuned model but no adapter was supplied.",
                suggestions=[
                    "Pass --adapter outputs/<experiment-id>/adapter",
                    "Or set evaluation.adapter_path in the config.",
                    "Without an adapter this arm would silently duplicate the base "
                    "arm and the comparison would be meaningless.",
                ],
            )
        return HuggingFaceBackend(
            model_config, adapter_path=adapter_path, reasoning_mode=reasoning_mode, name=arm
        )

    if arm in ("frontier_zero_shot", "frontier_orchestrated"):
        return FrontierAPIBackend()

    raise EvaluationError(
        f"Unknown research arm {arm!r}.",
        details={"valid_arms": list(RESEARCH_ARMS)},
    )
