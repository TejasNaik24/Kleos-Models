"""KLEOS Logos v0.0.2: the text tower of Ministral 3 14B Reasoning, trained to think.

The Reasoning release has the Instruct release's architecture and text-tower
shape, so it loads through the same text-only view. What differs is behaviour:
it always thinks, writing ``[THINK]...[/THINK]`` before its answer, and its chat
template renders an assistant message's ``reasoning`` field as that span. These
tests pin the config, the adapter, the vendored template and the supervision of
exactly ``[THINK]trace[/THINK]answer</s>``.

The classes marked ``requires_torch`` need transformers >= 5 and run in the
project's Docker image (see tests/test_ministral3_text_view.py for the command).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import CONFIGS_DIR, REPO_ROOT, V007_RELEASE, make_example
from tests.test_logos_config import mistral3_container

from kleos_models.config import ModelConfig, ReasoningMode, load_model_config
from kleos_models.data.formatting import ConversationFormatter
from kleos_models.data.loaders import load_evaluation_examples
from kleos_models.data.schemas import TrainingExample
from kleos_models.errors import ModelCompatibilityError
from kleos_models.inference.generate import OrchestrationConfig, build_prompt
from kleos_models.models import adapters
from kleos_models.models.adapters import (
    Ministral3ReasoningTextAdapter,
    Ministral3TextAdapter,
    ReasoningCapability,
    get_adapter,
    resolve_load_plan,
)

MODELS = CONFIGS_DIR / "models"
REASONING_BASE = "mistralai/Ministral-3-14B-Reasoning-2512"
REVISION = "51f9210f3cd20f3452a80d5819d15dc61cc50630"
TEMPLATE_PATH = REPO_ROOT / "tests" / "fixtures" / "ministral3_reasoning_chat_template.jinja"
TEMPLATE_SHA256 = "6b5044075f09f4daa57beebe2d989d9fbe67dea351f220e1a84e19ddc893f2c2"
BENCHMARK_SHA256 = "a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266"


def reasoning_model() -> ModelConfig:
    return load_model_config(MODELS / "ministral3_14b_reasoning.yaml")


def instruct_model() -> ModelConfig:
    return load_model_config(MODELS / "ministral3_14b.yaml")


class TestReasoningModelConfig:
    def test_it_pins_the_reasoning_checkpoint(self):
        config = reasoning_model()
        assert config.base_model == REASONING_BASE
        assert config.revision == REVISION
        assert config.model_type == "ministral3_reasoning"
        assert config.architecture == "Mistral3ForConditionalGeneration"

    def test_it_is_trained_to_think(self):
        reasoning = reasoning_model().reasoning
        assert reasoning.supported is True
        assert reasoning.default_mode is ReasoningMode.THINKING
        assert reasoning.strip_thinking_from_targets is False

    def test_measured_settings(self):
        # Measured on kleos-policy-v0.0.7 with the pinned tokenizer: the longest
        # example is 736 tokens, and transformers flags the regex when unset.
        config = reasoning_model()
        assert config.max_seq_length == 1024
        assert config.fix_mistral_regex is True

    def test_everything_else_is_logos_v001s(self):
        reasoning, instruct = reasoning_model(), instruct_model()
        for field in (
            "family",
            "parameter_count",
            "is_moe",
            "is_multimodal",
            "context_limit",
            "dtype",
            "attn_implementation",
            "device_map",
            "quantization",
            "lora",
        ):
            assert getattr(reasoning, field) == getattr(instruct, field), field


class TestReasoningAdapter:
    def test_the_config_selects_the_reasoning_view(self):
        adapter = get_adapter(reasoning_model())
        assert isinstance(adapter, Ministral3ReasoningTextAdapter)
        assert isinstance(adapter, Ministral3TextAdapter)

    def test_it_always_thinks(self):
        adapter = get_adapter(reasoning_model())
        assert adapter.capabilities.reasoning is ReasoningCapability.ALWAYS_ON
        assert adapter.resolve_reasoning_mode(None) is ReasoningMode.THINKING

    def test_it_refuses_to_run_without_thinking(self):
        adapter = get_adapter(reasoning_model())
        with pytest.raises(ModelCompatibilityError, match="thinking-only") as raised:
            adapter.resolve_reasoning_mode(ReasoningMode.STANDARD)
        assert "qwen" not in str(raised.value.suggestions).lower()

    def test_the_tower_and_targets_are_the_instruct_views(self):
        reasoning = get_adapter(reasoning_model())
        instruct = get_adapter(instruct_model())
        assert reasoning.capabilities.model_type == "ministral3"
        assert reasoning.default_target_modules == instruct.default_target_modules

    def test_logos_v001_is_unchanged(self):
        adapter = get_adapter(instruct_model())
        assert type(adapter) is Ministral3TextAdapter
        assert adapter.capabilities.reasoning is ReasoningCapability.UNSUPPORTED

    def test_the_checkpoint_name_implies_the_reasoning_view(self):
        config = ModelConfig(name="m", family="mistral", base_model=REASONING_BASE)
        assert adapters._infer_model_type(config) == "ministral3_reasoning"

    def test_the_instruct_name_still_implies_the_plain_view(self):
        config = ModelConfig(
            name="m", family="mistral", base_model="mistralai/Ministral-3-14B-Instruct-2512-BF16"
        )
        assert adapters._infer_model_type(config) == "ministral3"

    def test_the_container_opens_as_the_reasoning_view(self):
        config = reasoning_model()
        plan = resolve_load_plan(config, mistral3_container())
        assert isinstance(plan.adapter, Ministral3ReasoningTextAdapter)
        assert plan.view is not None and plan.view.kind == "text_only"
        assert plan.config is config
        assert plan.flipped_from is None


class TestVendoredTemplate:
    def test_the_vendored_template_is_the_pinned_one(self):
        assert hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() == TEMPLATE_SHA256

    def test_it_renders_the_reasoning_field_as_the_thinking_span(self):
        text = TEMPLATE_PATH.read_text(encoding="utf-8")
        assert "msg.get('reasoning_content', msg.get('reasoning', none))" in text
        assert "'[THINK]' + block['thinking']" in text


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("bench")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_benchmark.py"),
            "--dataset",
            str(V007_RELEASE),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output / "benchmark.jsonl"


@pytest.mark.requires_v007
class TestBenchmarkPrompts:
    """The template adds a default 'how you should think' system prompt to any
    conversation that has none. Every KLEOS prompt brings its own, so it never
    appears; this pins that for the benchmark Logos v0.0.2 is evaluated on."""

    def test_the_benchmark_is_the_one_hermes_and_logos_v001_ran(self, benchmark):
        assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == BENCHMARK_SHA256

    def test_every_arm2_prompt_starts_with_a_system_message(self, benchmark):
        examples = load_evaluation_examples(benchmark)
        assert len(examples) == 349
        arm2 = OrchestrationConfig(enabled=False)
        assert all(build_prompt(e, arm2)[0].role == "system" for e in examples)


# ---------------------------------------------------------------------------
# On a real (tiny) checkpoint and tokenizer — Docker
# ---------------------------------------------------------------------------

TRACE_WORDS = ["score", "first", "wins", "so"]


def build_reasoning_tokenizer() -> Any:
    """A word-level tokenizer carrying the real Ministral 3 Reasoning chat template."""
    from tests.test_ministral3_text_view import MARKERS, WORDS
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for word in [*MARKERS, *WORDS, *TRACE_WORDS]:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=MARKERS,
    )
    tokenizer.chat_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return tokenizer


def thinking_example() -> TrainingExample:
    payload = make_example(
        "think-0001", system="KLEOS policy", user="Rank the tasks", assistant="1. alpha 2. beta"
    )
    payload["version"] = "1.1"
    payload["messages"][-1]["reasoning"] = "score alpha first so alpha wins"
    return TrainingExample.model_validate(payload)


@pytest.mark.requires_torch
@pytest.mark.slow
class TestReasoningTemplateSupervision:
    def test_the_trace_and_the_answer_are_supervised_exactly(self):
        tokenizer = build_reasoning_tokenizer()
        formatter = ConversationFormatter(
            tokenizer, max_seq_length=128, strip_reasoning=False, fail_on_target_truncation=True
        )
        assert not formatter._template_drops_trailing_system()
        formatted = formatter.format_example(thinking_example())
        assert formatted is not None
        supervised = [
            t
            for t, label in zip(formatted.input_ids, formatted.labels, strict=True)
            if label != -100
        ]
        assert tokenizer.decode(supervised, skip_special_tokens=False).split() == [
            "[THINK]",
            "score",
            "alpha",
            "first",
            "so",
            "alpha",
            "wins",
            "[/THINK]",
            "1.",
            "alpha",
            "2.",
            "beta",
            "</s>",
        ]

    def test_nothing_before_the_answer_is_supervised(self):
        tokenizer = build_reasoning_tokenizer()
        formatter = ConversationFormatter(tokenizer, max_seq_length=128, strip_reasoning=False)
        formatted = formatter.format_example(thinking_example())
        assert formatted is not None
        first = next(i for i, label in enumerate(formatted.labels) if label != -100)
        assert tokenizer.convert_ids_to_tokens(formatted.input_ids[first - 1]) == "[/INST]"

    def test_the_prompt_ends_where_the_model_starts_thinking(self):
        tokenizer = build_reasoning_tokenizer()
        formatter = ConversationFormatter(tokenizer, max_seq_length=128)
        prompt = formatter.render_prompt(thinking_example().messages[:-1])
        assert prompt.endswith("[/INST]")
        assert "[THINK]" not in prompt
        assert "HOW YOU SHOULD THINK" not in prompt

    def test_without_a_system_message_the_template_adds_its_own(self):
        tokenizer = build_reasoning_tokenizer()
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Rank the tasks"}], tokenize=False
        )
        assert "HOW YOU SHOULD THINK" in text


@pytest.mark.requires_torch
@pytest.mark.requires_peft
@pytest.mark.slow
class TestReasoningViewLoads:
    def test_the_official_layout_loads_as_a_thinking_text_view(self, tmp_path):
        from tests.test_ministral3_text_view import model_config, write_official_checkpoint

        from kleos_models.models.loading import load_model

        write_official_checkpoint(tmp_path)
        config = model_config(
            tmp_path,
            model_type="ministral3_reasoning",
            reasoning={
                "supported": True,
                "default_mode": "thinking",
                "strip_thinking_from_targets": False,
            },
        )
        loaded = load_model(config, for_training=False)
        assert isinstance(loaded.adapter, Ministral3ReasoningTextAdapter)
        assert type(loaded.model).__name__ == "Ministral3ForCausalLM"
        assert loaded.reasoning_mode is ReasoningMode.THINKING
        names = [n for n, _ in loaded.model.named_parameters()]
        assert not [n for n in names if "vision" in n or "projector" in n]
