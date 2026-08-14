"""Shared fixtures and optional-dependency handling.

Tests that need torch/transformers/peft are marked and skipped cleanly when those
are absent, so the light suite runs everywhere while the heavy tests still exist
and run wherever the training extra is installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
CONFIGS_DIR = REPO_ROOT / "configs"


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = _available("torch")
HAS_TRANSFORMERS = _available("transformers")
HAS_PEFT = _available("peft")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip marked tests when their dependency is missing."""
    skip_torch = pytest.mark.skip(reason='needs torch + transformers: pip install -e ".[train]"')
    skip_peft = pytest.mark.skip(reason='needs peft: pip install -e ".[train]"')
    skip_cuda = pytest.mark.skip(reason="needs a CUDA GPU")

    cuda_available = False
    if HAS_TORCH:
        import torch

        cuda_available = torch.cuda.is_available()

    for item in items:
        if "requires_torch" in item.keywords and not (HAS_TORCH and HAS_TRANSFORMERS):
            item.add_marker(skip_torch)
        if "requires_peft" in item.keywords and not HAS_PEFT:
            item.add_marker(skip_peft)
        if "requires_cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def make_example(
    example_id: str = "test-000001",
    *,
    task: str = "notification_prioritization",
    domain: str = "career",
    user: str = "Rank these: alpha (due tomorrow), beta (due next month).",
    assistant: str = "1. alpha\n2. beta\n\nReasoning: alpha has the nearer deadline.",
    system: str | None = "You are the KLEOS reasoning layer.",
    scenario_family: str | None = None,
    quality_status: str = "reviewed",
    **axes: Any,
) -> dict[str, Any]:
    """Build a valid training-example dict, overridable per test."""
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})

    variation_axes: dict[str, Any] = {"domain": domain, "entities": "set_a"}
    variation_axes.update(axes)

    return {
        "id": example_id,
        "version": "1.0",
        "task": task,
        "messages": messages,
        "variation_axes": variation_axes,
        "metadata": {
            "source": "synthetic",
            "quality_status": quality_status,
            "scenario_family": scenario_family,
        },
    }


def make_eval_example(
    example_id: str = "eval-000001",
    *,
    task: str = "tool_routing",
    reference: dict[str, Any] | None = None,
    grader: str = "classification",
    split_tag: str = "in_distribution",
    ood_shift: str | None = None,
    scenario_family: str | None = None,
    **axes: Any,
) -> dict[str, Any]:
    """Build a valid evaluation-example dict."""
    variation_axes: dict[str, Any] = {"domain": "career"}
    variation_axes.update(axes)
    return {
        "id": example_id,
        "version": "1.0",
        "task": task,
        "messages": [{"role": "user", "content": "Which tool handles this?"}],
        "variation_axes": variation_axes,
        "metadata": {
            "source": "synthetic",
            "quality_status": "reviewed",
            "scenario_family": scenario_family,
        },
        "reference": reference or {"label": "memory_search", "options": ["memory_search", "none"]},
        "grader": grader,
        "split_tag": split_tag,
        "ood_shift": ood_shift,
    }


@pytest.fixture
def example_factory():
    """Factory for building valid training examples."""
    return make_example


@pytest.fixture
def eval_example_factory():
    """Factory for building valid evaluation examples."""
    return make_eval_example


@pytest.fixture
def training_examples():
    """A small set of parsed training examples spanning several axes."""
    from kleos_models.data.schemas import TrainingExample

    raw = [
        make_example(
            f"test-{i:06d}",
            domain=domain,
            entities=entities,
            urgency=urgency,
            scenario_family=f"family-{i % 4:02d}",
            user=f"Rank scenario {i}: item-{i}a (due soon), item-{i}b (later).",
            assistant=f"1. item-{i}a\n2. item-{i}b\n\nReasoning: nearer deadline first.",
        )
        for i, (domain, entities, urgency) in enumerate(
            [
                ("career", "set_a", "high"),
                ("research", "set_b", "medium"),
                ("coursework", "set_a", "low"),
                ("projects", "set_c", "high"),
                ("career", "set_b", "medium"),
                ("research", "set_c", "low"),
                ("coursework", "set_b", "high"),
                ("projects", "set_a", "medium"),
            ]
        )
    ]
    return [TrainingExample.model_validate(item) for item in raw]


@pytest.fixture
def fixture_examples():
    """The committed synthetic development fixtures."""
    from kleos_models.data.loaders import load_examples
    from kleos_models.data.schemas import TrainingExample

    examples, _ = load_examples(
        EXAMPLES_DIR / "synthetic_train.jsonl", model=TrainingExample, strict=True
    )
    return examples


@pytest.fixture
def fixture_eval_examples():
    """The committed evaluation fixtures."""
    from kleos_models.data.loaders import load_evaluation_examples

    return load_evaluation_examples(EXAMPLES_DIR / "synthetic_eval.jsonl", strict=True)


@pytest.fixture
def training_config_path() -> Path:
    return CONFIGS_DIR / "training" / "qlora_small.yaml"


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """A minimal chat tokenizer for testing formatting and masking.

    Deliberately mimics the structure of a real chat template — role markers and
    an end-of-turn token — so assistant-span masking is exercised for real
    without downloading a model. Encoding is word-level, which keeps expected
    token counts easy to reason about in assertions.
    """

    def __init__(self, *, supports_thinking: bool = False) -> None:
        self.supports_thinking = supports_thinking
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.chat_template = "fake"
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        if "enable_thinking" in kwargs and not self.supports_thinking:
            raise TypeError("this template does not accept enable_thinking")
        parts = [f"<|{m['role']}|> {m['content']} <|end|>" for m in conversation]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 100
            ids.append(self._vocab[token])
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        reverse = {v: k for k, v in self._vocab.items()}
        return " ".join(reverse.get(int(i), "<unk>") for i in ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def thinking_tokenizer() -> FakeTokenizer:
    return FakeTokenizer(supports_thinking=True)
