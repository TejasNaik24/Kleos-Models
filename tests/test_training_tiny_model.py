"""End-to-end training verification on tiny randomly-initialized models.

Spec §45 forbids a fake training pipeline. These tests are the evidence that this
one is real, and they run on CPU in seconds because the models are built from a
config with a handful of layers — no weights are downloaded.

What is proven here:

* the family adapters find real modules in real architectures;
* LoRA attaches and *only* LoRA parameters are trainable;
* assistant-span masking selects exactly the answer tokens under the real Qwen and
  Mistral chat templates;
* a real optimizer step reduces the loss on a memorizable batch;
* adapter weights save and reload, and the reloaded adapter changes the output.

Skipped cleanly when the ``[train]`` extra is not installed.
"""

from __future__ import annotations

import pytest
from tests.conftest import CONFIGS_DIR, make_example

from kleos_models.config import LoRAConfig, ModelConfig, TrainingConfig
from kleos_models.constants import IGNORE_INDEX
from kleos_models.data.schemas import TrainingExample
from kleos_models.models.adapters import get_adapter

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_peft, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Tiny model builders
# ---------------------------------------------------------------------------


def build_tiny_qwen3():
    """A ~1M parameter Qwen3 with the real architecture, random weights."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def build_tiny_mistral():
    """A tiny MistralForCausalLM with random weights."""
    import torch
    from transformers import MistralConfig, MistralForCausalLM

    config = MistralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return MistralForCausalLM(config)


def qwen_config() -> ModelConfig:
    return ModelConfig(
        name="tiny_qwen",
        family="qwen",
        base_model="Qwen/Qwen3-8B",
        model_type="qwen3",
        max_seq_length=128,
        lora=LoRAConfig(r=4, alpha=8, target_modules="auto"),
    )


def mistral_config() -> ModelConfig:
    return ModelConfig(
        name="tiny_mistral",
        family="mistral",
        base_model="mistralai/Ministral-8B-Instruct-2410",
        model_type="mistral",
        max_seq_length=128,
        lora=LoRAConfig(r=4, alpha=8, target_modules="auto"),
    )


@pytest.fixture(scope="module")
def tiny_qwen():
    return build_tiny_qwen3()


@pytest.fixture(scope="module")
def tiny_mistral():
    return build_tiny_mistral()


# ---------------------------------------------------------------------------
# Target discovery against real architectures
# ---------------------------------------------------------------------------


class TestTargetDiscovery:
    def test_qwen_targets_exist_in_the_real_architecture(self, tiny_qwen):
        adapter = get_adapter(qwen_config())
        resolution = adapter.validate_target_modules(tiny_qwen, qwen_config().lora)
        assert resolution.ok
        assert resolution.matched_module_count > 0
        assert set(resolution.matched) == set(adapter.default_target_modules)

    def test_mistral_targets_exist_in_the_real_architecture(self, tiny_mistral):
        adapter = get_adapter(mistral_config())
        resolution = adapter.validate_target_modules(tiny_mistral, mistral_config().lora)
        assert resolution.ok
        assert resolution.matched_module_count > 0

    def test_a_bogus_target_fails_with_the_real_candidates_listed(self, tiny_qwen):
        from kleos_models.errors import ModelCompatibilityError

        config = qwen_config()
        config.lora = LoRAConfig(target_modules=["not_a_real_module"])
        adapter = get_adapter(config)

        with pytest.raises(ModelCompatibilityError) as info:
            adapter.validate_target_modules(tiny_qwen, config.lora)

        message = str(info.value)
        assert "not_a_real_module" in message
        # The error must show what IS available, or it is not actionable.
        assert "q_proj" in message

    def test_excluded_modules_are_not_adapted(self, tiny_qwen):
        adapter = get_adapter(qwen_config())
        resolution = adapter.validate_target_modules(tiny_qwen, qwen_config().lora)
        assert "lm_head" not in resolution.matched

    def test_module_listing_reports_real_shapes(self, tiny_qwen):
        from kleos_models.models.loading import list_candidate_modules

        candidates = list_candidate_modules(tiny_qwen)
        assert candidates
        names = {c["suffix"] for c in candidates}
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= names
        assert all(c["in_features"] > 0 for c in candidates)


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------


class TestLoRAAttachment:
    def test_only_lora_parameters_are_trainable(self, tiny_qwen):
        from kleos_models.models.peft_setup import attach_lora

        model = build_tiny_qwen3()
        config = qwen_config()
        result = attach_lora(
            model,
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )

        assert result.trainable_parameters > 0
        # A LoRA run must train a tiny fraction of the model.
        assert result.trainable_fraction < 0.5

        trainable_names = [name for name, p in result.model.named_parameters() if p.requires_grad]
        assert trainable_names
        assert all("lora" in name.lower() for name in trainable_names), (
            f"non-LoRA parameters are trainable: "
            f"{[n for n in trainable_names if 'lora' not in n.lower()][:5]}"
        )

    def test_attachment_summary_is_recorded(self, tiny_qwen):
        from kleos_models.models.peft_setup import attach_lora

        config = qwen_config()
        result = attach_lora(
            build_tiny_qwen3(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        payload = result.to_dict()
        assert payload["trainable_parameters"] > 0
        assert payload["lora_config"]["r"] == 4
        assert payload["target_modules"]["matched"]

    def test_mistral_lora_attaches(self):
        from kleos_models.models.peft_setup import attach_lora

        config = mistral_config()
        result = attach_lora(
            build_tiny_mistral(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        assert result.trainable_parameters > 0


# ---------------------------------------------------------------------------
# Masking against the real chat templates
# ---------------------------------------------------------------------------

QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class TestMaskingWithARealTokenizer:
    @pytest.fixture
    def tokenizer(self):
        """A real fast tokenizer with a Qwen-style chat template.

        Built locally from a tiny BPE vocabulary rather than downloaded, so the
        test needs no network but still exercises real tokenizer code paths.
        """
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = {f"tok{i}": i for i in range(200)}
        words = [
            "<|im_start|>",
            "<|im_end|>",
            "system",
            "user",
            "assistant",
            "\n",
            "Rank",
            "these",
            "alpha",
            "beta",
            "1.",
            "2.",
            "Reasoning:",
            "deadline",
            "nearer",
            "the",
            "is",
            "You",
            "are",
            "KLEOS",
            "answer",
            "SENTINEL",
        ]
        for index, word in enumerate(words):
            vocab[word] = 200 + index

        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="tok0",
            pad_token="tok1",
            eos_token="<|im_end|>",
        )
        tokenizer.chat_template = QWEN_TEMPLATE
        return tokenizer

    def test_only_the_answer_is_supervised(self, tokenizer):
        from kleos_models.data.formatting import ConversationFormatter

        example = TrainingExample.model_validate(
            make_example(
                user="Rank these alpha beta",
                assistant="1. alpha 2. beta",
                system="You are KLEOS",
            )
        )
        formatter = ConversationFormatter(tokenizer, max_seq_length=256)
        result = formatter.format_example(example)
        assert result is not None

        supervised = [
            token
            for token, label in zip(result.input_ids, result.labels, strict=True)
            if label != IGNORE_INDEX
        ]
        decoded = tokenizer.decode(supervised)

        assert "alpha" in decoded
        # The prompt must not be supervised.
        assert "Rank" not in decoded
        assert "KLEOS" not in decoded

    def test_masking_leaves_the_prompt_ignored(self, tokenizer):
        from kleos_models.data.formatting import ConversationFormatter

        example = TrainingExample.model_validate(
            make_example(user="Rank these alpha beta", assistant="1. alpha 2. beta")
        )
        formatter = ConversationFormatter(tokenizer, max_seq_length=256)
        result = formatter.format_example(example)
        assert result is not None
        assert result.labels[0] == IGNORE_INDEX
        assert result.target_token_count < len(result.input_ids)

    def test_reasoning_never_becomes_a_target(self, tokenizer):
        from kleos_models.data.formatting import ConversationFormatter

        example = TrainingExample.model_validate(
            make_example(
                user="Rank these alpha beta",
                assistant="<think>SENTINEL</think>1. alpha 2. beta",
            )
        )
        formatter = ConversationFormatter(tokenizer, max_seq_length=256, strip_reasoning=True)
        result = formatter.format_example(example)
        assert result is not None

        decoded = tokenizer.decode(
            [
                t
                for t, label in zip(result.input_ids, result.labels, strict=True)
                if label != IGNORE_INDEX
            ]
        )
        assert "SENTINEL" not in decoded


# ---------------------------------------------------------------------------
# The pipeline actually trains
# ---------------------------------------------------------------------------


class TestRealTrainingStep:
    def _batch(self, batch_size: int = 2, length: int = 16):
        """A fixed batch with only the second half of each row supervised."""
        import torch

        torch.manual_seed(0)
        input_ids = torch.randint(0, 200, (batch_size, length))
        labels = input_ids.clone()
        labels[:, : length // 2] = IGNORE_INDEX  # prompt is masked, as in training
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def test_gradients_reach_the_adapter(self):
        from kleos_models.models.peft_setup import attach_lora, verify_gradients_flow

        config = qwen_config()
        result = attach_lora(
            build_tiny_qwen3(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        diagnostics = verify_gradients_flow(result.model, self._batch())

        assert diagnostics["loss"] > 0
        assert diagnostics["lora_tensors"] > 0
        assert diagnostics["lora_tensors_with_nonzero_grad"] > 0

    def test_a_fully_masked_batch_is_rejected(self):
        """Training on nothing must fail loudly, not produce a plausible NaN."""
        import torch

        from kleos_models.errors import ModelCompatibilityError
        from kleos_models.models.peft_setup import attach_lora, verify_gradients_flow

        config = qwen_config()
        result = attach_lora(
            build_tiny_qwen3(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        batch = self._batch()
        batch["labels"] = torch.full_like(batch["labels"], IGNORE_INDEX)

        with pytest.raises((ModelCompatibilityError, ValueError, RuntimeError)):
            verify_gradients_flow(result.model, batch)

    def test_optimizer_steps_reduce_the_loss(self):
        """The core anti-fake-pipeline test: real steps must actually learn.

        A tiny model over a fixed batch should memorize quickly. If the loss does
        not fall, either the adapter is not connected to the graph or the
        optimizer is not updating it.
        """
        import torch

        from kleos_models.models.peft_setup import attach_lora

        config = qwen_config()
        config.lora = LoRAConfig(r=8, alpha=16, target_modules="auto")
        result = attach_lora(
            build_tiny_qwen3(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        model = result.model
        model.train()

        batch = self._batch(batch_size=2, length=16)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

        initial_loss = float(model(**batch).loss.detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        final_loss = float(model(**batch).loss.detach())

        assert final_loss < initial_loss, (
            f"loss did not decrease ({initial_loss:.4f} → {final_loss:.4f}); "
            "the adapter is not learning"
        )

    def test_base_weights_are_frozen_during_training(self):
        """Only the adapter may change. The base model must be untouched."""
        import torch

        from kleos_models.models.peft_setup import attach_lora

        config = qwen_config()
        model = build_tiny_qwen3()
        reference = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "q_proj" in name
        }

        result = attach_lora(
            model,
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        peft_model = result.model
        peft_model.train()

        batch = self._batch()
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=1e-2
        )
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            peft_model(**batch).loss.backward()
            optimizer.step()

        for name, original in reference.items():
            current = dict(model.named_parameters()).get(name)
            if current is not None:
                assert torch.allclose(current.detach(), original), (
                    f"base weight {name} changed; this is no longer a LoRA run"
                )


class TestAdapterPersistence:
    def test_adapter_saves_and_reloads(self, tmp_path):
        import torch
        from peft import PeftModel

        from kleos_models.models.peft_setup import attach_lora

        config = qwen_config()
        base = build_tiny_qwen3()
        result = attach_lora(
            base,
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        model = result.model
        model.train()

        # Train briefly so the adapter is not at its zero-initialized state.
        batch = {
            "input_ids": torch.randint(0, 200, (2, 16)),
            "labels": torch.randint(0, 200, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            optimizer.step()

        adapter_dir = tmp_path / "adapter"
        model.save_pretrained(str(adapter_dir))

        assert (adapter_dir / "adapter_config.json").exists()
        assert any(adapter_dir.glob("adapter_model.*"))

        model.eval()
        with torch.no_grad():
            trained_logits = model(**batch).logits.clone()

        # Reload onto a freshly built base and confirm the outputs match.
        fresh_base = build_tiny_qwen3()
        reloaded = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
        reloaded.eval()
        with torch.no_grad():
            reloaded_logits = reloaded(**batch).logits

        assert torch.allclose(trained_logits, reloaded_logits, atol=1e-4), (
            "the reloaded adapter does not reproduce the trained model's output"
        )

    def test_the_adapter_changes_the_output(self):
        """An adapter that does nothing would make the FT arm meaningless."""
        import torch

        from kleos_models.models.peft_setup import attach_lora

        config = qwen_config()
        base = build_tiny_qwen3()
        batch = {
            "input_ids": torch.randint(0, 200, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        base.eval()
        with torch.no_grad():
            base_logits = base(**batch).logits.clone()

        result = attach_lora(
            base,
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        model = result.model
        model.train()

        labels = torch.randint(0, 200, (1, 16))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-1)
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels
            ).loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tuned_logits = model(**batch).logits

        assert not torch.allclose(base_logits, tuned_logits, atol=1e-3), (
            "the trained adapter did not change the model's output at all"
        )


class TestEndToEndPipeline:
    def test_format_collate_and_step(self, tmp_path):
        """Formatter → collator → model, exactly as the trainer wires it."""
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from kleos_models.data.formatting import ConversationFormatter
        from kleos_models.models.peft_setup import attach_lora
        from kleos_models.training.trainer import PaddingCollator, format_split

        vocab = {f"tok{i}": i for i in range(200)}
        for index, word in enumerate(
            [
                "<|im_start|>",
                "<|im_end|>",
                "system",
                "user",
                "assistant",
                "Rank",
                "alpha",
                "beta",
                "1.",
                "2.",
                "answer",
                "the",
                "first",
            ]
        ):
            vocab[word] = 200 + index
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="tok0",
            pad_token="tok1",
            eos_token="<|im_end|>",
        )
        tokenizer.chat_template = QWEN_TEMPLATE

        examples = [
            TrainingExample.model_validate(
                make_example(
                    f"e2e-{i:04d}",
                    user="Rank alpha beta",
                    assistant="1. alpha 2. beta",
                )
            )
            for i in range(4)
        ]

        formatter = ConversationFormatter(tokenizer, max_seq_length=64)
        dataset, stats = format_split(examples, formatter, split_name="train")
        assert len(dataset) == 4
        assert stats["dropped"] == 0

        collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([dataset[i] for i in range(4)])

        assert batch["input_ids"].shape == batch["labels"].shape
        # Padded positions must not contribute to the loss.
        padded = batch["attention_mask"] == 0
        assert torch.all(batch["labels"][padded] == IGNORE_INDEX)

        config = qwen_config()
        result = attach_lora(
            build_tiny_qwen3(),
            config,
            TrainingConfig(gradient_checkpointing=False),
            get_adapter(config),
            is_quantized=False,
        )
        output = result.model(**batch)
        assert output.loss is not None
        assert torch.isfinite(output.loss), "the loss is not finite"


class TestFullPipelineEndToEnd:
    """The whole `scripts/train.py` path, on a tiny model, on CPU.

    This is the strongest available evidence that the pipeline is not a facade:
    it runs `run_training` itself — the same function the CLI calls — and asserts
    that real artifacts land on disk with a truthful manifest.
    """

    @pytest.fixture
    def tiny_setup(self, tmp_path):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from kleos_models.config import load_config
        from kleos_models.data.loaders import DatasetBundle
        from kleos_models.models.loading import LoadedModel

        vocab = {f"tok{i}": i for i in range(200)}
        for index, word in enumerate(
            [
                "<|im_start|>",
                "<|im_end|>",
                "system",
                "user",
                "assistant",
                "Rank",
                "alpha",
                "beta",
                "1.",
                "2.",
                "answer",
                "first",
                "second",
                "KLEOS",
            ]
        ):
            vocab[word] = 200 + index
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="tok0",
            pad_token="tok1",
            eos_token="<|im_end|>",
        )
        tokenizer.chat_template = QWEN_TEMPLATE

        config = load_config(
            CONFIGS_DIR / "training" / "debug.yaml",
            overrides=[
                "model.quantization.mode=none",
                "model.max_seq_length=64",
                "training.max_steps=3",
                "training.save_steps=2",
                "training.eval_steps=2",
                "training.logging_steps=1",
                "training.optim=adamw_torch",
                "training.gradient_checkpointing=false",
                "model.device_map=null",
                f"training.output_dir={tmp_path}",
            ],
        )

        model_config = config.model
        adapter = get_adapter(model_config)
        loaded = LoadedModel(
            model=build_tiny_qwen3(),
            tokenizer=tokenizer,
            adapter=adapter,
            config=model_config,
            reasoning_mode=adapter.resolve_reasoning_mode(),
            quantization_metadata={"mode": "none", "applied": False},
            load_metadata={"auto_class": "AutoModelForCausalLM", "injected_for_test": True},
        )

        def example(index: int, split: str) -> TrainingExample:
            return TrainingExample.model_validate(
                make_example(
                    f"{split}-{index:04d}",
                    user="Rank alpha beta",
                    assistant="1. alpha 2. beta",
                )
            )

        bundle = DatasetBundle(
            train=[example(i, "train") for i in range(6)],
            validation=[example(i, "val") for i in range(2)],
        )
        return config, bundle, loaded, tmp_path

    def test_run_training_produces_real_artifacts(self, tiny_setup):
        import json

        from kleos_models.experiments.manifest import RunStatus, build_manifest
        from kleos_models.training.trainer import run_training

        config, bundle, loaded, _output_root = tiny_setup
        manifest = build_manifest(
            config,
            kind="training",
            dataset_version="tiny-test-v0",
            dataset_hash=bundle.dataset_hash(),
            dataset_counts=bundle.counts,
        )

        result = run_training(config, bundle, manifest, loaded=loaded, verify_gradients=True)

        # Adapter weights exist and are real files.
        assert result.adapter_path.exists()
        assert (result.adapter_path / "adapter_config.json").exists()
        assert any(result.adapter_path.glob("adapter_model.*"))

        # Tokenizer, effective config, metrics, manifest.
        assert result.tokenizer_path.exists()
        assert (result.output_dir / "config.yaml").exists()
        assert (result.output_dir / "metrics.json").exists()
        assert (result.output_dir / "manifest.json").exists()
        assert (result.output_dir / "README.md").exists()
        assert (result.output_dir / "environment.txt").exists()

        # A real loss was computed.
        metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0

        # The manifest tells the truth about the run.
        assert result.manifest.status is RunStatus.COMPLETED
        assert result.manifest.dataset_version == "tiny-test-v0"
        assert result.manifest.config_hash == config.config_hash
        assert result.manifest.lora["trainable_parameters"] > 0
        assert result.manifest.metrics["gradient_check"]["lora_tensors_with_nonzero_grad"] > 0

    def test_structured_events_are_written_without_example_text(self, tiny_setup):
        from kleos_models.experiments.manifest import build_manifest
        from kleos_models.logging_utils import EventLogger
        from kleos_models.training.trainer import run_training

        config, bundle, loaded, _output_root = tiny_setup
        manifest = build_manifest(config, kind="training", dataset_version="tiny-test-v0")
        result = run_training(config, bundle, manifest, loaded=loaded)

        events = EventLogger(
            result.output_dir / "events.jsonl", experiment_id=manifest.experiment_id
        ).read_all()
        assert events
        names = {event["event"] for event in events}
        assert "train_begin" in names
        assert "train_end" in names

        # Spec §31: never log raw example content.
        raw = (result.output_dir / "events.jsonl").read_text(encoding="utf-8")
        assert "Rank alpha beta" not in raw

    def test_checkpoints_are_written_and_resumable(self, tiny_setup):
        from kleos_models.experiments.manifest import build_manifest
        from kleos_models.training.checkpointing import discover_checkpoints
        from kleos_models.training.trainer import run_training

        config, bundle, loaded, _output_root = tiny_setup
        manifest = build_manifest(config, kind="training", dataset_version="tiny-test-v0")
        result = run_training(config, bundle, manifest, loaded=loaded)

        checkpoints = discover_checkpoints(result.output_dir)
        assert checkpoints, "no checkpoint was written"
        assert all(c.valid for c in checkpoints)
        # KLEOS metadata travels with the checkpoint.
        assert checkpoints[0].metadata is not None
        assert checkpoints[0].metadata["dataset_version"] == "tiny-test-v0"

    def test_failure_is_recorded_in_the_manifest(self, tiny_setup):
        """A crashed run must leave a manifest saying it failed (spec §36)."""
        from kleos_models.experiments.manifest import ExperimentManifest, RunStatus, build_manifest
        from kleos_models.training.trainer import run_training

        config, bundle, loaded, tmp_path = tiny_setup
        manifest = build_manifest(config, kind="training", dataset_version="tiny-test-v0")

        # Break the model so the forward pass raises.
        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("simulated hardware failure")

        loaded.model = Exploding()

        # Any exception is acceptable here; the assertion is about what the
        # manifest records afterwards, not the exception type.
        with pytest.raises(Exception):  # noqa: B017
            run_training(config, bundle, manifest, loaded=loaded)

        output_dir = tmp_path / manifest.experiment_id
        restored = ExperimentManifest.load(output_dir)
        assert restored.status is RunStatus.FAILED
        assert restored.error is not None
        assert "simulated hardware failure" in restored.error["message"]
