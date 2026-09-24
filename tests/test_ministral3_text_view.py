"""KLEOS Logos' text-only view, on a real (tiny) checkpoint in the official layout.

The Ministral 3 14B checkpoint is a vision-language model whose weights sit
under the Hub's pre-transformers-5 key layout (``language_model.model.*``, a
separate ``language_model.lm_head.weight``, ``vision_tower.*``,
``multi_modal_projector.*``). These tests write a tiny checkpoint in exactly that
layout and load it through the repository's own ``load_model``, proving:

* every language weight arrives exactly, the vision tower never loads, and
  ``lm_head`` stays untied;
* the text view computes what the full vision-language model computes on text;
* LoRA on a 40-layer view attaches 280 modules, with exactly the trainable count
  the feasibility estimator predicts, and gradients reach them;
* an adapter trained on the view saves, reloads and reproduces its outputs, and
  records the base revision in ``adapter_config.json`` (finding H-F1);
* the real Ministral 3 chat template renders system prompts in place, so the
  formatter needs no system merge and supervises exactly the answer;
* the vision-language path for other ``mistral3`` models still loads.

Requires torch, transformers >= 5 and peft; run in the project's Docker image:

    docker run --rm --platform linux/amd64 -u 0 -e HF_HUB_OFFLINE=1 \\
        -v "$PWD":/work:ro -w /work --entrypoint bash kleos-hermes:v0.0.6 \\
        -c 'pip install -q pytest==9.1.1 && pytest -p no:cacheprovider -q tests/test_ministral3_text_view.py'
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import REPO_ROOT, make_example

from kleos_models.config import ModelConfig, TrainingConfig
from kleos_models.data.formatting import ConversationFormatter
from kleos_models.data.schemas import TrainingExample
from kleos_models.models.adapters import Ministral3TextAdapter, get_adapter

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_peft, pytest.mark.slow]

TEMPLATE_PATH = REPO_ROOT / "tests" / "fixtures" / "ministral3_chat_template.jinja"
TEMPLATE_SHA256 = "2f545122222db8bb43ca0ea0c49e9185320a8670f7d35575b0da0eb48b1e8970"
REVISION = "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"
SEVEN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

#: Transformers-5 names of the full model, mapped to the Hub's layout.
TO_HUB_LAYOUT = (
    ("model.language_model.", "language_model.model."),
    ("lm_head.", "language_model.lm_head."),
    ("model.vision_tower.", "vision_tower."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
)

WORDS = [
    "Rank",
    "alpha",
    "beta",
    "the",
    "tasks",
    "1.",
    "2.",
    "deadline",
    "KLEOS",
    "policy",
    "decide",
]
MARKERS = [
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_CALLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[THINK]",
    "[/THINK]",
    "[IMG]",
    "[ARGS]",
    "[CALL_ID]",
]


def tiny_text_config(layers: int = 2, *, hidden: int = 32, vocab: int = 64) -> dict[str, Any]:
    return {
        "model_type": "ministral3",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": hidden * 2,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden // 2,
        "max_position_embeddings": 256,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def build_tokenizer() -> Any:
    """A word-level tokenizer carrying the real Ministral 3 chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for word in [*MARKERS, *WORDS]:
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


def write_official_checkpoint(directory: Path, *, layers: int = 2) -> dict[str, Any]:
    """A tiny Mistral3 vision-language checkpoint in the Hub's key layout."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, GenerationConfig, Mistral3Config

    config = Mistral3Config(
        text_config=tiny_text_config(layers),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_dim": 8,
            "image_size": 32,
            "patch_size": 16,
        },
        tie_word_embeddings=False,
    )
    config.architectures = ["Mistral3ForConditionalGeneration"]
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).eval()

    tensors: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        for new_prefix, hub_prefix in TO_HUB_LAYOUT:
            if name.startswith(new_prefix):
                name = hub_prefix + name[len(new_prefix) :]
                break
        tensors[name] = tensor.detach().clone().contiguous()
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(directory)
    GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0).save_pretrained(directory)
    build_tokenizer().save_pretrained(directory)
    return tensors


def model_config(directory: Path, *, model_type: str = "ministral3", **fields: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "name": "tiny-logos",
        "family": "mistral",
        "base_model": str(directory),
        "revision": REVISION,
        "model_type": model_type,
        "architecture": "Mistral3ForConditionalGeneration",
        "max_seq_length": 128,
        "quantization": {"mode": "none"},
        "device_map": None,
        "dtype": "float32",
    }
    values.update(fields)
    return ModelConfig.model_validate(values)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[Path, dict[str, Any]]:
    directory = tmp_path_factory.mktemp("tiny-ministral3")
    return directory, write_official_checkpoint(directory)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestTextView:
    def test_every_language_weight_arrives_exactly(self, checkpoint):
        import torch

        from kleos_models.models.loading import load_model

        directory, saved = checkpoint
        loaded = load_model(model_config(directory), for_training=False)
        model = loaded.model
        assert type(model).__name__ == "Ministral3ForCausalLM"
        assert isinstance(loaded.adapter, Ministral3TextAdapter)

        state = model.state_dict()
        for name, tensor in state.items():
            hub = (
                "language_model.lm_head." + name[len("lm_head.") :]
                if name.startswith("lm_head.")
                else "language_model." + name
            )
            assert torch.equal(tensor, saved[hub]), name
        text_keys = {k for k in saved if k.startswith("language_model.")}
        assert len(state) == len(text_keys)

    def test_the_vision_tower_stays_on_disk(self, checkpoint):
        from kleos_models.models.loading import load_model

        directory, saved = checkpoint
        loaded = load_model(model_config(directory), for_training=False)
        names = [n for n, _ in loaded.model.named_parameters()]
        assert not [n for n in names if "vision" in n or "projector" in n]
        info = loaded.load_metadata["view"]["loading_info"]
        assert info["missing_keys"] == info["mismatched_keys"] == 0
        assert info["unexpected_by_prefix"] == {
            "vision_tower.": sum(k.startswith("vision_tower.") for k in saved),
            "multi_modal_projector.": sum(k.startswith("multi_modal_projector.") for k in saved),
        }

    def test_lm_head_stays_untied(self, checkpoint):
        from kleos_models.models.loading import load_model

        directory, _ = checkpoint
        model = load_model(model_config(directory), for_training=False).model
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    def test_the_view_computes_what_the_full_model_computes(self, checkpoint):
        import torch
        from transformers import AutoModelForImageTextToText

        from kleos_models.models.loading import load_model

        directory, _ = checkpoint
        view = load_model(model_config(directory), for_training=False).model.eval()
        full = AutoModelForImageTextToText.from_pretrained(str(directory)).eval()
        input_ids = torch.tensor([[1, 20, 21, 22, 23, 24, 25, 2]])
        with torch.no_grad():
            expected = full(input_ids=input_ids).logits
            actual = view(input_ids=input_ids).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    def test_the_vision_language_path_still_loads(self, checkpoint):
        from kleos_models.models.loading import load_model

        directory, _ = checkpoint
        loaded = load_model(
            model_config(directory, model_type="mistral3", is_multimodal=True), for_training=False
        )
        assert type(loaded.model).__name__ == "Mistral3ForConditionalGeneration"
        assert "view" not in loaded.load_metadata


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class TestLoRAOnTheView:
    def test_a_40_layer_view_gets_280_modules_and_the_estimated_count(self):
        import torch
        from transformers import Ministral3Config, Ministral3ForCausalLM

        from kleos_models.models.feasibility import ModelShape, estimate_lora_parameters
        from kleos_models.models.peft_setup import attach_lora, verify_gradients_flow

        text = tiny_text_config(40, hidden=16)
        torch.manual_seed(0)
        model = Ministral3ForCausalLM(Ministral3Config(**text))
        # r=2: at r=16 a 16-wide toy model would be mostly adapter, and the
        # full-fine-tune guard would rightly refuse it. Logos itself is r=16.
        config = model_config(Path("/unused"), lora={"r": 2, "alpha": 4})
        result = attach_lora(
            model,
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        assert result.resolution.matched_module_count == 280
        shape = ModelShape(
            parameter_count=0,
            hidden_size=16,
            num_layers=40,
            intermediate_size=32,
            vocab_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        assert result.trainable_parameters == estimate_lora_parameters(shape, 2, SEVEN)

        batch = {
            "input_ids": torch.tensor([[1, 5, 6, 7, 2]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 6, 7, 2]]),
        }
        diagnostics = verify_gradients_flow(result.model, batch)
        assert diagnostics["lora_tensors"] == 560
        assert diagnostics["lora_tensors_with_nonzero_grad"] >= 280


# ---------------------------------------------------------------------------
# The real chat template
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_the_vendored_template_is_the_pinned_one(self):
        assert hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() == TEMPLATE_SHA256

    def test_the_system_prompt_is_rendered_in_place(self):
        tokenizer = build_tokenizer()
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "KLEOS policy"},
                {"role": "user", "content": "Rank the tasks"},
                {"role": "assistant", "content": "1. alpha 2. beta"},
            ],
            tokenize=False,
        )
        assert text.count("[SYSTEM_PROMPT]KLEOS policy[/SYSTEM_PROMPT]") == 1
        assert text.index("[SYSTEM_PROMPT]") < text.index("[INST]")

    def test_no_system_merge_and_exact_supervision(self):
        tokenizer = build_tokenizer()
        formatter = ConversationFormatter(tokenizer, max_seq_length=128)
        assert not formatter._template_drops_trailing_system()

        example = TrainingExample.model_validate(
            make_example(
                "tpl-0001",
                system="KLEOS policy",
                user="Rank the tasks",
                assistant="1. alpha 2. beta",
            )
        )
        formatted = formatter.format_example(example)
        assert formatted is not None
        supervised = [
            t
            for t, label in zip(formatted.input_ids, formatted.labels, strict=True)
            if label != -100
        ]
        assert tokenizer.decode(supervised, skip_special_tokens=False).split() == [
            "1.",
            "alpha",
            "2.",
            "beta",
            "</s>",
        ]


# ---------------------------------------------------------------------------
# Training on the view, end to end
# ---------------------------------------------------------------------------


class TestTrainingOnTheView:
    def test_training_saves_a_reloadable_adapter_pinned_to_its_base(self, checkpoint, tmp_path):
        import torch

        from kleos_models.config import load_config
        from kleos_models.data.loaders import DatasetBundle
        from kleos_models.experiments.manifest import build_manifest
        from kleos_models.models.loading import load_adapter_model, load_model
        from kleos_models.training.trainer import run_training

        directory, _ = checkpoint
        base = model_config(directory)
        config = load_config(
            REPO_ROOT / "configs" / "training" / "debug.yaml",
            overrides=[
                "training.max_steps=2",
                "training.save_steps=1",
                "training.eval_steps=1",
                "training.logging_steps=1",
                "training.optim=adamw_torch",
                "training.gradient_checkpointing=false",
                f"training.output_dir={tmp_path}",
            ],
        )
        config.model = base
        loaded = load_model(base, for_training=True)

        def example(index: int) -> TrainingExample:
            return TrainingExample.model_validate(
                make_example(
                    f"train-{index:04d}",
                    system="KLEOS policy",
                    user="Rank the tasks",
                    assistant="1. alpha 2. beta",
                )
            )

        bundle = DatasetBundle(train=[example(i) for i in range(4)], validation=[example(9)])
        manifest = build_manifest(config, kind="training", dataset_version="tiny")
        result = run_training(config, bundle, manifest, loaded=loaded, verify_gradients=True)

        adapter_config = json.loads((result.adapter_path / "adapter_config.json").read_text())
        assert adapter_config["revision"] == REVISION  # finding H-F1
        assert result.manifest.model["load"]["view"]["kind"] == "text_only"
        assert "memory_probe" not in result.manifest.metrics  # CPU: nothing to measure

        reloaded = load_adapter_model(base, result.adapter_path).model.eval()
        in_memory = loaded.model.eval()
        input_ids = torch.tensor([[1, 20, 21, 22, 2]])
        with torch.no_grad():
            torch.testing.assert_close(
                reloaded(input_ids=input_ids).logits,
                in_memory(input_ids=input_ids).logits,
                rtol=0,
                atol=1e-5,
            )
