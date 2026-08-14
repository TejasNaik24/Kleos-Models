# Troubleshooting

## CUDA out of memory

The error itself lists the detected GPU, the configuration that failed, and what
to change, in order. Work down that list.

The usual sequence:

1. Reduce `model.max_seq_length` — activation memory scales linearly with it.
2. Enable `training.gradient_checkpointing`.
3. Halve `per_device_train_batch_size` and double `gradient_accumulation_steps` —
   the effective batch size and learning dynamics stay the same.
4. Use `training.optim: paged_adamw_8bit`.
5. Check nothing else holds GPU memory (`nvidia-smi`; in a notebook, restart the
   runtime).
6. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

**If it happened at step 0**, the configuration never fit. Run
`python scripts/plan_run.py --config <config>` — it would have told you in a
second.

**If it happened mid-run**, it was probably an evaluation or checkpoint spike.
Lower `per_device_eval_batch_size` to 1.

**If the model is 24B or 30B on a 16GB GPU**, it does not fit and no amount of
tuning will make it. Use an 8B config or an A100.

## `AutoModelForCausalLM` fails on Mistral Small 3.2

Expected. `mistralai/Mistral-Small-3.2-24B-Instruct-2506` declares
`Mistral3ForConditionalGeneration` (`model_type: mistral3`), and transformers
registers it **only** in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`. It is a
vision-language model.

The KLEOS `Mistral3VLMAdapter` handles this: it loads via
`AutoModelForImageTextToText` and scopes LoRA to `language_model.*`. Use
`configs/models/mistral_small_3_2.yaml` rather than constructing a config by hand.

Confirm what a checkpoint really is:

```bash
python scripts/inspect_model.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

## `KeyError: 'qwen3_moe'`

transformers is older than 4.51. Install a tested version:

```bash
pip install -U "transformers>=4.56,<6"
```

## `TypeError: TrainingArguments got an unexpected keyword argument 'warmup_ratio'`

You are on transformers v5, which removed `warmup_ratio` in favour of
`warmup_steps` (a float below 1 is treated as a ratio).

`kleos_models.compat` handles this automatically. If you see this error, something
is constructing `TrainingArguments` directly instead of going through
`build_training_arguments`. If the installed transformers changed again, the shim
needs updating — the error message says so.

## `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`

Same cause: v5 renamed it to `processing_class`. Use
`compat.trainer_tokenizer_kwarg()`.

## Requesting non-thinking mode fails on Qwen3-30B-A3B-Thinking

Working as intended. That checkpoint is thinking-only — its chat template does not
accept `enable_thinking` and always emits a reasoning span.

Set `model.reasoning.default_mode: thinking`, or use `Qwen/Qwen3-8B` for a
switchable model. The pipeline refuses rather than silently producing a mangled
prompt, and it will not fake reasoning by injecting `<think>` tags.

## `401` / `403` / "gated repo" when downloading

The model requires accepting a licence.

1. Open the model page on Hugging Face **while signed in** and accept the terms.
2. Create a token at <https://huggingface.co/settings/tokens>.
3. `export HF_TOKEN=hf_...`, or add it as a Colab secret with notebook access.

The token alone is not enough — the licence must be accepted by that account.

## `bitsandbytes` is not installed / has no CUDA

bitsandbytes is CUDA-only in practice. There is no working macOS build, which is
why it is a separate `[quant]` extra rather than part of `[train]`.

- CUDA machine: `pip install -e ".[train,quant]"`
- Colab: `python scripts/colab_setup.py`
- macOS: you cannot run 4-bit QLoRA locally. Use Colab. The data, validation and
  evaluation-scoring pipelines all work fine locally.

## Colab: torch stops seeing the GPU after installing packages

A package replaced Colab's torch build with a generic wheel.

Runtime → Restart runtime, then use `python scripts/colab_setup.py`, which
installs torch-dependent packages with `--no-deps` specifically to prevent this.

## Colab: the runtime disconnected mid-training

Expected on the free tier. Re-run the setup cells, remount Drive, then:

```bash
python scripts/train.py --config <config> --resume-from-checkpoint auto
```

If you were not writing checkpoints to Drive, the run is gone. Set
`--output-dir /content/drive/MyDrive/kleos/outputs` next time.

## "No usable examples remain after formatting"

Every example produced zero supervised tokens — usually because
`max_seq_length` truncated the answer away.

```bash
python scripts/inspect_dataset.py --dataset <dir> --tokenizer <model> --show-masking
```

Raise `model.max_seq_length` or shorten the examples.

## Training runs but the loss never moves

The gradient check should catch a disconnected adapter before training starts. If
it passed and the loss is still flat, the plumbing is fine — look at the learning
rate, the data, or whether the answers are actually learnable.

To confirm the pipeline itself works:

```bash
python -m pytest tests/test_training_tiny_model.py -v
```

Those tests train a tiny model on CPU and assert the loss decreases.

## Dataset validation fails on `possible_sensitive_content`

The validator found something that looks like an email address, a token or a key.
**This repository is public** — treat it as real until proven otherwise.

If it is genuinely synthetic, rewrite it so it cannot be mistaken for real data
(use `name@example.com`, not a plausible address).

## Leakage detected between splits

A cross-split duplicate means an evaluation example is effectively present in
training, and any generalization number from it is measuring memorization.

1. Read `reports/leakage/leakage_summary.md`.
2. Deduplicate before splitting.
3. Use a held-out strategy (`entity_holdout`, `scenario_family_holdout`) so
   related examples cannot straddle the boundary.

Override with `--leakage-policy none` only if you understand why, and record that
you did.

## Split fails: "attribute has only 1 distinct value"

A held-out split needs at least two distinct values of the attribute it partitions
on. Your dataset has one.

Check coverage:

```bash
python scripts/inspect_dataset.py --dataset <dir> --coverage
```

Add examples covering another value, or use `strategy: group`.

## Consistency reports zero groups

No example carries `metadata.scenario_family`, so equivalent scenarios cannot be
grouped and consistency measures nothing. Add the field to perturbed examples.

## OOD reports "not measurable"

No benchmark example is tagged `split_tag: "ood"`. **No generalization claim can
be made** until some are. The report states this rather than omitting the section.

## Config error: unknown field

Configs reject unknown keys deliberately, so a typo fails loudly instead of
silently doing nothing. Check the spelling against `configs/*/README.md`.

```bash
python scripts/validate_configs.py
```

## Still stuck

```bash
python scripts/smoke_test.py -v
```

It checks each layer in order and reports exactly what failed and what was
skipped. Include its output in any bug report, along with `pip list | grep -E
"torch|transformers|peft|bitsandbytes"`.
