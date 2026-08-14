# Training

## Quick start

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml   # will it fit?
python scripts/train.py    --config configs/training/qlora_small.yaml --dry-run
python scripts/train.py    --config configs/training/qlora_small.yaml
```

On Colab, use `notebooks/02_train_qlora.ipynb` — see [colab.md](colab.md).

## What the pipeline does

Spec §15, step for step:

1. Load and validate config
2. Seed every RNG and record what was seeded
3. Load and validate the dataset; check leakage between splits
4. Print the pre-flight report (GPU, VRAM, CUDA, versions, model, LoRA, estimate)
5. Assess feasibility; apply and **record** any adjustment
6. Load the tokenizer and the quantized base model
7. Attach LoRA and validate the targets against the real model
8. Verify gradients reach the adapter
9. Format examples with assistant-only loss masking
10. Train, checkpointing as configured
11. Evaluate on the validation split
12. Save adapter, tokenizer, effective config, metrics, environment, manifest

## QLoRA

4-bit NF4 by default, with double quantization, a configurable compute dtype and a
frozen base model.

```yaml
model:
  quantization:
    mode: nf4
    compute_dtype: auto   # bf16 on capability >= 8.0, fp16 below
    double_quant: true
  lora:
    target_modules: auto  # family-appropriate, validated against the real model
    r: 16
    alpha: 32
    dropout: 0.05
```

If quantization is requested and bitsandbytes or CUDA is missing, the run **fails
with a diagnostic**. It never silently loads in full precision — that would turn a
"QLoRA run" into something the manifest misdescribes.

### `compute_dtype: auto` matters on Colab

A T4 is compute capability 7.5 and cannot do bfloat16. `auto` detects this and
selects float16, recording the reason in the manifest. Explicitly requesting bf16
on a T4 raises rather than silently degrading.

## Three silent failures this pipeline refuses to have

**1. Target modules that match nothing.** LoRA attaches, trains zero parameters,
and produces a normal-looking loss curve. Targets are validated against the loaded
model's actual `named_modules()`, and a miss is a hard error listing real
candidates.

**2. An adapter that receives no gradients.** A forward/backward pass runs before
the real loop and asserts LoRA parameters get non-zero gradients. Skip it with
`--skip-gradient-check` if you must.

**3. An accidental full fine-tune.** If more than 50% of parameters end up
trainable, the run stops unless `training.allow_full_finetune: true`.

## Assistant-only loss masking

Training on the prompt teaches the model to predict user messages, which is not
the behaviour we want. Everything except assistant content is set to `-100`.

Spans are located by **incremental chat-template application**: render the
conversation up to a turn with a generation prompt, render it again including that
turn, and take the token range between. This asks the template where the boundary
is instead of guessing from a delimiter string, so it works across families
without per-model parsing.

Reasoning spans are stripped before tokenization. We never train a model to emit
display chain-of-thought.

Inspect what would be supervised:

```bash
python scripts/inspect_dataset.py --dataset <dir> --tokenizer Qwen/Qwen3-8B --show-masking
```

It prints counts only, never example text, so it is safe to run against private
data.

## Memory

The knobs that matter, in the order worth trying:

| Setting | Effect |
| --- | --- |
| `model.max_seq_length` | Activation memory scales linearly. Biggest lever. |
| `training.gradient_checkpointing` | Large saving for ~20-30% slower steps. |
| `per_device_train_batch_size` | Halve it, double `gradient_accumulation_steps` — effective batch and learning dynamics unchanged. |
| `training.optim` | `paged_adamw_8bit` keeps optimizer state off the critical path. |
| `model.quantization.mode` | `nf4` for QLoRA. |

Estimate before running:

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml
```

### Feasibility tiers

| Tier | Meaning |
| --- | --- |
| `full_research` | Fits with headroom |
| `adapter_train` | Fits, but tight — evaluation spikes may still OOM |
| `smoke` | Only a reduced config fits |
| `inference_only` | Can be evaluated, not trained |
| `infeasible` | Cannot be loaded |

### Adjustments are recorded, never silent

If a configuration does not fit, the planner **proposes** changes, logs them, and
writes them to `manifest.adjustments[]` with before, after and reason.

```yaml
training:
  strict_config: true   # any required adjustment becomes a hard error
```

Turn this on for real research runs. It guarantees the run either used exactly the
configuration you specified or refused to start, so a comparison cannot be
invalidated by a memory-driven fallback nobody noticed.

## Checkpointing

Written for the reality of free Colab, where the runtime can vanish at any moment.

```bash
python scripts/train.py --config <config> --resume-from-checkpoint auto
```

- `auto` finds the newest **valid** checkpoint.
- A checkpoint half-written when a runtime was killed is detected as incomplete
  and skipped, rather than failing confusingly on resume.
- `save_total_limit` is honoured with a floor of 1 — retention can never leave you
  with nothing to resume from.
- KLEOS metadata (experiment id, dataset version and hash, config hash, seed)
  travels with each checkpoint, so a directory recovered from Drive months later
  still identifies its run.

On Colab, point `--output-dir` at Drive.

## Reproducibility

Every run records model + revision + dataset version + dataset hash + config hash
+ seed + git commit + environment.

```
outputs/<experiment-id>/
  adapter/  tokenizer/  config.yaml  manifest.json  metrics.json
  events.jsonl  environment.txt  training.log  README.md  checkpoint-*/
```

Two runs with the same `config_hash` used the same knobs. Git information degrades
gracefully — if the repository has no commits, the manifest records that fact
rather than omitting the field.

## Failed runs

A crashed run still writes its manifest, with `status: "failed"` and the failing
stage recorded. Nothing is deleted. `ExperimentRegistry` lists failures alongside
successes.

## Multi-model support

The same command trains any supported model:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --set model.name=ministral_8b
```

Everything family-specific — auto class, LoRA targets, exclusions, reasoning
handling, quantization exceptions — lives in the family adapter, not the pipeline.

Adding a family: write a config, subclass `ModelFamilyAdapter`, register it. The
training pipeline does not change.

## transformers version compatibility

`src/kleos_models/compat.py` handles the 4.56 ↔ 5.x differences:

| Concern | 4.x | 5.x |
| --- | --- | --- |
| LR warmup | `warmup_ratio` | `warmup_steps` (float < 1 = ratio) |
| Trainer tokenizer | `tokenizer=` | `processing_class=` |
| Load dtype | `torch_dtype=` | `dtype=` |
| Output dir | `overwrite_output_dir` | removed |
| Length grouping | `group_by_length` | removed |

KLEOS configs keep the spec's names; the shim emits whatever the installed version
accepts and records every translation in the manifest. It introspects the
installed classes rather than branching on a version number, so a backport or
release candidate does not break it.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md). OOM errors from this pipeline list
what to change, in order, along with the detected GPU and the configuration that
failed.
