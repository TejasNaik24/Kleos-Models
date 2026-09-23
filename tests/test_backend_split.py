"""The prepare / generate_ids / finish split generates exactly what generate() did.

ZeroGPU bills GPU time, so the Space runs only the middle step on a GPU and the
other two on the CPU, with the prepared prompt pickled across a process
boundary in between. This must be a pure refactor: the frozen 9/9 match was
measured through the monolithic ``generate`` this replaced. Here the split and
a verbatim copy of the pre-split method run side by side on a tiny random
Mistral with a real (offline) fast tokenizer, and must agree token for token.

Needs torch and transformers; skipped without them. It runs inside the Hermes
Docker image on CPU (see docs/deployment.md, "Verification record").
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from kleos_models.config import GenerationConfig, ReasoningMode
from kleos_models.data.schemas import Message

pytestmark = [pytest.mark.requires_torch]

WORDS = [
    "which",
    "task",
    "first",
    "the",
    "migration",
    "deadline",
    "decides",
    "it",
    "do",
    "now",
    "later",
    "none",
    "memory",
    "search",
]


def build_backend() -> Any:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    from kleos_models.inference.backends import HuggingFaceBackend

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"
    )

    torch.manual_seed(0)
    model = MistralForCausalLM(
        MistralConfig(
            vocab_size=len(vocab),  # every id the model can emit decodes
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=1,
        )
    ).eval()

    # Constructed without __init__, which would download a real checkpoint.
    backend = object.__new__(HuggingFaceBackend)
    backend.name = "split-test"
    backend.loaded = SimpleNamespace(
        model=model, tokenizer=tokenizer, reasoning_mode=ReasoningMode.STANDARD
    )
    backend.formatter = SimpleNamespace(
        render_prompt=lambda messages: " ".join(m.content for m in messages)
    )
    return backend


def monolithic_generate(backend: Any, messages: list[Message], config: GenerationConfig) -> Any:
    """HuggingFaceBackend.generate as it was at commit 728f214, verbatim in effect."""
    import torch

    from kleos_models.inference.backends import GenerationOutput

    prompt = backend.formatter.render_prompt(messages)
    inputs = backend.loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(backend.loaded.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[-1])
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": backend.loaded.tokenizer.pad_token_id,
        "repetition_penalty": config.repetition_penalty,
    }
    with torch.no_grad():
        output_ids = backend.loaded.model.generate(**inputs, **generate_kwargs)
    completion_ids = output_ids[0][prompt_length:]
    text = backend.loaded.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return GenerationOutput(
        text=text,
        prompt_tokens=prompt_length,
        completion_tokens=int(completion_ids.shape[-1]),
    )


PROMPTS = [
    [Message(role="user", content="which task first")],
    [
        Message(role="system", content="decides it"),
        Message(role="user", content="the migration deadline do now or later"),
    ],
]


@pytest.fixture(scope="module")
def backend() -> Any:
    return build_backend()


@pytest.mark.parametrize("messages", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", [1, 12])
def test_the_split_reproduces_the_monolithic_generate(backend, messages, max_new_tokens):
    config = GenerationConfig(do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)
    before = monolithic_generate(backend, messages, config)
    after = backend.generate(messages, config)
    assert after.text == before.text
    assert after.prompt_tokens == before.prompt_tokens
    assert after.completion_tokens == before.completion_tokens


@pytest.mark.parametrize("messages", PROMPTS)
def test_a_prepared_prompt_survives_the_trip_to_a_gpu_worker(backend, messages):
    # ZeroGPU pickles the arguments of the GPU function into a forked worker.
    # This test pickles its own freshly built object; nothing untrusted.
    config = GenerationConfig(do_sample=False, temperature=0.0, max_new_tokens=12)
    prepared = backend.prepare(messages)
    shipped = pickle.loads(pickle.dumps(prepared))
    ids = backend.generate_ids(shipped, config)
    assert all(type(token) is int for token in ids)
    assert backend.finish(prepared, ids).text == backend.generate(messages, config).text
