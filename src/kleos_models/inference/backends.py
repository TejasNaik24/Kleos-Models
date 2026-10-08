"""Generation backends (spec sections 19, 20)."""

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


def split_thinking(
    ids: Sequence[int], begin: int, end: int
) -> tuple[list[int] | None, list[int], bool]:
    """Split a completion's token ids into its thinking span and its answer."""
    tokens = list(ids)
    if tokens[:1] == [begin]:
        body = tokens[1:]
        if end in body:
            cut = body.index(end)
            return body[:cut], body[cut + 1 :], True
        return body, [], False
    if end in tokens:
        cut = tokens.index(end)
        return tokens[:cut], tokens[cut + 1 :], True
    return None, tokens, True


@dataclass
class PreparedPrompt:
    """A rendered, tokenized prompt, ready for the model."""

    inputs: dict[str, Any]
    prompt_length: int


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
    """Local generation through transformers, with or without a LoRA adapter."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        adapter_path: Path | str | None = None,
        reasoning_mode: ReasoningMode | None = None,
        name: str | None = None,
        fix_mistral_regex: bool | None = None,
        adapter_device: str | None = None,
    ) -> None:
        from kleos_models.models.loading import load_adapter_model, load_model

        self.model_config = model_config
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.name = name or ("finetuned" if adapter_path else "base")

        if self.adapter_path is not None:
            self.loaded = load_adapter_model(
                model_config,
                self.adapter_path,
                reasoning_mode=reasoning_mode,
                fix_mistral_regex=fix_mistral_regex,
                adapter_device=adapter_device,
            )
        else:
            self.loaded = load_model(
                model_config,
                reasoning_mode=reasoning_mode,
                for_training=False,
                fix_mistral_regex=fix_mistral_regex,
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
        prepared = self.prepare(messages)
        return self.finish(prepared, self.generate_ids(prepared, config))

    # Split in three so a GPU-billed host (ZeroGPU) runs only the middle step on the GPU;
    # generate() composes them unchanged for every other caller.

    def prepare(self, messages: Sequence[Message]) -> PreparedPrompt:
        """Render the chat template and tokenize. CPU only; tensors stay on the CPU."""
        prompt = self.formatter.render_prompt(messages)
        inputs = self.loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return PreparedPrompt(
            inputs=dict(inputs.items()),
            prompt_length=int(inputs["input_ids"].shape[-1]),
        )

    def generate_ids(self, prepared: PreparedPrompt, config: GenerationConfig) -> list[int]:
        """Run the model and return only the completion's token ids. Needs the GPU."""
        import torch

        device = next(self.loaded.model.parameters()).device
        inputs = {k: v.to(device) for k, v in prepared.inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.loaded.tokenizer.pad_token_id,
            "repetition_penalty": config.repetition_penalty,
        }
        # Sampling params only with do_sample: otherwise they warn and do nothing.
        if config.do_sample:
            generate_kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
            if config.top_k:
                generate_kwargs["top_k"] = config.top_k

        with torch.no_grad():
            output_ids = self.loaded.model.generate(**inputs, **generate_kwargs)

        return [int(token) for token in output_ids[0][prepared.prompt_length :].tolist()]

    def _thinking_markers(self) -> tuple[int, int] | None:
        """Ids of ``[THINK]`` and ``[/THINK]`` when the model thinks in them."""
        if self.loaded.reasoning_mode is not ReasoningMode.THINKING:
            return None
        tokenizer = self.loaded.tokenizer
        begin = tokenizer.convert_tokens_to_ids("[THINK]")
        end = tokenizer.convert_tokens_to_ids("[/THINK]")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if begin is None or end is None or unknown in (begin, end):
            return None
        return int(begin), int(end)

    def finish(self, prepared: PreparedPrompt, completion_ids: Sequence[int]) -> GenerationOutput:
        """Decode the completion and separate any reasoning span. CPU only."""
        tokenizer = self.loaded.tokenizer
        metadata = {"backend": self.name, "reasoning_mode": self.loaded.reasoning_mode.value}

        # Graders score the answer, not the thinking span (spec section 39).
        markers = self._thinking_markers()
        if markers is not None:
            # [THINK]/[/THINK] are special tokens; split on ids, as decoding erases the boundary.
            reasoning_ids, answer_ids, closed = split_thinking(completion_ids, *markers)
            return GenerationOutput(
                text=tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                reasoning=(
                    tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()
                    if reasoning_ids is not None
                    else None
                ),
                prompt_tokens=prepared.prompt_length,
                completion_tokens=len(completion_ids),
                # 'length': the budget ran out while thinking, so there is no answer.
                finish_reason="stop" if closed else "length",
                metadata=metadata,
            )

        text = tokenizer.decode(list(completion_ids), skip_special_tokens=True)
        reasoning: str | None = None
        answer = text
        if "</think>" in text:
            head, _, tail = text.partition("</think>")
            reasoning = head.replace("<think>", "").strip()
            answer = tail.strip()

        return GenerationOutput(
            text=answer,
            reasoning=reasoning,
            prompt_tokens=prepared.prompt_length,
            completion_tokens=len(completion_ids),
            metadata=metadata,
        )

    def describe(self) -> dict[str, Any]:
        description = self.loaded.describe()
        description["backend"] = self.name
        description["adapter_path"] = str(self.adapter_path) if self.adapter_path else None
        return description


class EchoBackend(BaseBackend):
    """Deterministic stub backend for testing the harness itself."""

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
    """Interface for a hosted frontier reference model (spec section 20)."""

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
    """Construct the backend for a research arm."""
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
