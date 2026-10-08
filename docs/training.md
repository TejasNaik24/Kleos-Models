# Training

## Quick start

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml   # will it fit?
python scripts/train.py    --config configs/training/qlora_small.yaml --dry-run
python scripts/train.py    --config configs/training/qlora_small.yaml
```

`qlora_small.yaml` and `debug.yaml` include `mistral_nemo_12b.yaml` (Apache-2.0,
ungated, the Hermes base). On a free GPU, use `notebooks/02_train_qlora.ipynb`
([colab.md](colab.md)) or the Kaggle notebooks ([kaggle.md](kaggle.md)).

## What the pipeline does

`scripts/train.py`, in order:

1. Load and validate the config; seed every RNG and record what was seeded.
2. Load and validate the dataset, and check for leakage between splits
   (`--leakage-policy`, default `fatal`).
3. Tokenize the train split, tokenizer only, to measure the longest sequence.
4. Assess feasibility at that length. By default a run that does not fit is
   refused, or adjusted when `strict_config` is false, and every adjustment is
   recorded. `--feasibility record` records the assessment and leaves the decision
   to the memory probe. A `--dry-run` stops here and writes a manifest.
5. Load the tokenizer and the quantized base model.
6. Attach LoRA and validate the targets against the loaded model.
7. Format the examples with assistant-only loss masking.
8. Verify that gradients reach the adapter.
9. Measure memory on the longest batch (the memory probe).
10. Build the Trainer, recording every transformers compatibility translation,
    and check that a model spread over GPUs runs model parallel.
11. Train, checkpointing as configured.
12. Run a validation pass on the selected weights.
13. Save the adapter, tokenizer, effective config, metrics, environment report and
    manifest.

## The recipe used by every KLEOS run

All four runs (`kleos-v006-ministral8b-run1`, Hermes `kleos-v006-mistralnemo12b-run1`,
Logos v0.0.1 `kleos-v006-ministral314b-run1` and Logos v0.0.2
`kleos-v007-ministral314breasoning-run1`) used the same recipe. The values are
engineering defaults, not tuned hyperparameters.

| Setting | Value |
| --- | --- |
| Method | QLoRA: NF4, double quantization, base model frozen |
| Compute dtype | `auto`, which is float16 on a T4; every run trained in fp16 |
| LoRA | r 16, alpha 32, dropout 0.05, bias none |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; `lm_head` and `embed_tokens` excluded |
| LoRA modules, trainable parameters | Ministral-8B 252, 43,646,976; Hermes 280, 57,016,320; Logos v0.0.1 and v0.0.2 280, 60,948,480 |
| Loss | assistant tokens only; for Logos v0.0.2 the trace and the answer |
| Optimizer | `paged_adamw_8bit`, learning rate 2e-4, cosine schedule, warmup ratio 0.03, weight decay 0, max grad norm 1.0 |
| Batch | 1 per device, 8 accumulation steps: effective batch 8 |
| Length | `max_seq_length` 1024; nothing truncated |
| Epochs | 3 (309 steps over 820 training examples) |
| Memory | gradient checkpointing on |
| Checkpoints | every 50 steps (Ministral-8B, Hermes) or 25 (Logos); `save_total_limit` 3, plus the best |
| Selection | `load_best_model_at_end` on lowest `eval_loss` |
| `strict_config` | false (Ministral-8B, Hermes); true (Logos) |
| Seed | 42 |

Selected checkpoints, measured: step 200 (epoch 1.95) for Ministral-8B and Hermes,
`checkpoint-175` for both Logos runs (epoch 1.70 for v0.0.2). Decoding at
evaluation is greedy; [evaluation.md](evaluation.md) has the rest.

## QLoRA

```yaml
model:
  quantization:
    mode: nf4
    compute_dtype: auto   # bf16 on capability >= 8.0, fp16 below
    double_quant: true
  lora:
    target_modules: auto  # family defaults, validated against the real model
    r: 16
    alpha: 32
    dropout: 0.05
```

If quantization is requested and bitsandbytes or CUDA is missing, the run fails
with a diagnostic. It never falls back to full precision, which would make the
manifest misdescribe a "QLoRA run".

### `compute_dtype: auto` on a T4

A T4 is compute capability 7.5 and has no bfloat16. `auto` selects float16 there
and records the reason in the manifest. Requesting bf16 explicitly on a T4
raises rather than degrading.

## Three silent failures the pipeline refuses

**Target modules that match nothing.** LoRA attaches, trains zero parameters and
produces a normal-looking loss curve. Targets are validated against the loaded
model's `named_modules()`, and a miss is a hard error listing the real
candidates.

**An adapter that receives no gradients.** A forward and backward pass runs
before the real loop and asserts that LoRA parameters get non-zero gradients.
`--skip-gradient-check` turns it off.

**An accidental full fine-tune.** If more than 50% of parameters end up
trainable, the run stops unless `training.allow_full_finetune: true`.

## Assistant-only loss masking

Training on the prompt would teach the model to predict user messages. Every
token outside assistant content is labeled `-100`.

Spans are located by incremental chat-template application: render the
conversation up to a turn with a generation prompt, render it again through that
turn, and take the token range between. The template, not a delimiter string,
says where the boundary is, so the method works across families.

Some chat templates drop the system turn when the conversation ends with an
assistant message. Mistral-Nemo's does, so training text would have lost its
policy instructions while evaluation kept them. The formatter probes the live
template once and, when the system turn disappears, folds the system prompt into
the first user turn, reproducing what the template does at generation time
(deviation D5 in [experiments.md](experiments.md#deviations-log)). Ministral 3's
template renders the system prompt in place, so no merge happens there.

Inspect what would be supervised:

```bash
python scripts/inspect_dataset.py --dataset <path to release> \
    --tokenizer mistralai/Mistral-Nemo-Instruct-2407 --show-masking
```

It prints counts only, never example text, so it is safe to run against private
data.

## Reasoning-supervised runs

From schema 1.1, an assistant message may carry a `reasoning` field
([data-contract.md](data-contract.md#assistant-reasoning-schema-11)). What happens
to it depends on `model.reasoning.strip_thinking_from_targets`:

- **`true` (the default, every model except Logos v0.0.2).** The field is removed
  before tokenizing, and the count of affected examples is logged and recorded as
  `reasoning_dropped`.
- **`false` (`ministral3_14b_reasoning.yaml`).** The field reaches the chat
  template, which renders it as the thinking span, and the whole of
  `[THINK]trace[/THINK]answer</s>` is supervised.

A reasoning-supervised run trains on every trace whole. The formatter is built
with `fail_on_target_truncation`, so the run refuses to start if any example would
lose part of its target to `max_seq_length`, or all of it. `train.py` measures the
longest example before loading weights, so the refusal costs seconds. The
v0.0.7 traces lengthened the longest example from 448 to 736 tokens.

## `fix_mistral_regex`

transformers 5 flags the pre-tokenizer regex in some Mistral tokenizers as
differing from mistral-common's and can patch it at load time. The model config
field states the choice:

- `null` passes nothing (the library default, which every run before Logos used).
  It is hash-neutral while null, so earlier config hashes reproduce.
- `true` or `false` is passed explicitly and carried through training, evaluation
  and serving, so the three tokenize alike. Both Logos configs set `true`.

On kleos-policy-v0.0.6 the correction changes 0 of 1,350 examples (measured), so it
matters for user text at serving time. Requesting `true` on transformers older
than 5 raises.

## Memory

The settings that matter, in the order worth trying:

| Setting | Effect |
| --- | --- |
| `model.max_seq_length` | Activation memory scales with it. The largest lever. |
| `training.gradient_checkpointing` | A large saving for slower steps. |
| `per_device_train_batch_size` | Halve it and double `gradient_accumulation_steps`: the effective batch is unchanged. |
| `training.optim` | `paged_adamw_8bit` keeps optimizer state small and outside the allocator. |
| `model.quantization.mode` | `nf4` for QLoRA. |

Memory figures in this repository are in GB as the code reports them
(1 GB = 2³⁰ bytes).

### Estimate before running

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml
python scripts/plan_run.py --config configs/training/kleos_hermes_v006.yaml \
    --set-model configs/models/ministral3_14b.yaml --simulate-gpu t4-colab --seq-length 448
```

`--seq-length` matters: at batch size 1 the activation peak follows the longest
example present. Without it the estimate assumes `max_seq_length`, the worst
case. `--set-model` plans another model in the same recipe without writing a
config. `--simulate-gpu` takes the `t4-colab` preset (the capacity measured on
Colab's T4, 14.56 GB) or `NAME:TOTAL_GIB:MAJOR.MINOR[:COUNT]`.

The estimator is checked against measured training peaks
(`EMPIRICAL_ANCHORS` in `src/kleos_models/models/feasibility.py`). Before that
calibration it underestimated large-vocabulary models by about 5 GB (finding
H-F10 in the [Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)).

### Feasibility tiers

| Tier | Meaning |
| --- | --- |
| `full_research` | Fits with headroom |
| `adapter_train` | Fits, but tight |
| `smoke` | Only a reduced configuration fits |
| `inference_only` | Can be evaluated, not trained |
| `infeasible` | Cannot be loaded |

### Adjustments are recorded

If a configuration does not fit, the planner proposes changes, logs them and
writes them to `manifest.adjustments[]` with before, after and reason.

```yaml
training:
  strict_config: true   # any required adjustment becomes a hard error
```

Both Logos runs set it, so each run used exactly its registered configuration or
would have refused to start.

### The memory probe

Before step 1, `probe_training_peak` (`src/kleos_models/training/memory.py`) runs
two forward and backward passes on the longest micro-batch and records peak
allocated and reserved memory, and the memory still free once the paged optimizer
state exists. The result is in `manifest.metrics.memory_probe`.
`--skip-memory-probe` turns it off.

### The smoke gate

For the Logos runs, a ten-step smoke run at the real settings
(`configs/training/debug_logos.yaml`, `debug_logos_v002.yaml`) preceded training,
and `scripts/check_smoke_gate.py` read its manifest and returned GO or NO-GO:

- the run completed, and nothing was adjusted to make it fit;
- a text-only checkpoint view loaded every weight;
- LoRA attached where expected (`--expect-modules 280 --expect-trainable 60948480`);
- gradients reached the adapter;
- the probe left at least `--min-spare-gb` (0.15) free on the tightest GPU.

### Measured peaks

| Run | Hardware | Peak |
| --- | --- | --- |
| Ministral-8B | 1 × T4 | 9.67 GB |
| Hermes | 1 × T4 | 13.09 GB of 14.56 GB |
| Logos v0.0.1 | 1 × T4 | 13.60 GB allocated, 13.97 GB reserved, 0.35 GB spare after optimizer state (full run, probe before step 1, 448 tokens) |
| Logos v0.0.2 | 2 × T4 | 4.75 GB on GPU 0, 10.16 GB on GPU 1 (probe, 736 tokens) |

## Two GPUs

Logos v0.0.2 did not fit one T4 at 736 tokens (estimated 14.46 GB against a
14.41 GB budget) and trained on Kaggle's 2 × T4. The code handles a model spread
over several GPUs as follows:

- **Placement.** `model.device_map: auto` lets accelerate place the layers across
  every visible GPU. `cuda:0` keeps the model on one.
- **Estimate.** With `device_map: auto` the estimator splits the layers evenly and
  reports each GPU, gating on the fuller:

  ```bash
  python scripts/plan_run.py --config configs/training/kleos_logos_v002.yaml \
      --seq-length 736 --simulate-gpu T4:14.56:7.5:2
  ```

  It estimated 6.38 GB on GPU 0 and 8.53 GB on GPU 1. The measured split was
  uneven: GPU 0 held the embedding and layers 0 to 8, GPU 1 layers 9 to 39 and
  `lm_head`, and peaked at 10.16 GB (finding L-F6, in the
  [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings)).
  The total matched.
- **Probe.** The memory probe measures every GPU the model occupies and gates on
  the one with least to spare; the manifest records `gated_on_device` and a
  reading per GPU.
- **Model-parallel check.** `assert_model_parallel` (`src/kleos_models/training/trainer.py`)
  refuses to train a model spread over GPUs unless the Trainer runs it in place
  (`is_model_parallel` true and `n_gpu` 1). Otherwise the Trainer would wrap it in
  data parallelism. transformers 5.16.1 detects model parallelism from
  `hf_device_map` (verified in its source); the check stops a run if a later
  version does not.
- **Record.** The manifest's model block records where each module landed
  (`load.hf_device_map`).

## Checkpointing and resume

Written for free GPU sessions, which can end at any moment.

```bash
python scripts/train.py --config <config> --experiment-id <original-id> \
    --resume-from-checkpoint auto
```

- `--experiment-id` must be the id the run started with. It names the run's
  directory; a generated id embeds a timestamp, so `auto` would search a new,
  empty directory and the run would start again from step 0.
- `auto` takes the newest valid checkpoint. A checkpoint without
  `trainer_state.json` or a weights file (a save interrupted when a runtime was
  killed) is skipped. This is what caught a half-written checkpoint after a write
  reported as saved had not reached Drive (finding H-F4).
- `save_total_limit` has a floor of 1, and the best checkpoint so far is always
  kept on top of it, so `load_best_model_at_end` cannot fall back to the final
  weights (finding H-F9).
- The experiment id, dataset version and hash, config hash and seed travel with
  each checkpoint, so a recovered directory still identifies its run.
- `train_loss` in a resumed run's metrics is not a real mean: transformers divides
  the last session's loss sum by every step (finding H-F8).

Two gaps are known and open. Both are recorded in
[experiments/logos-findings.md](experiments/logos-findings.md):

- **L-F4: no guard against a fresh start over existing checkpoints.** Running
  without `--resume-from-checkpoint` into a folder that already holds checkpoints
  starts from step 0, and checkpoint rotation can then delete the old run's best
  checkpoint. Always resume with `auto`.
- **L-F5: `validate_checkpoint` does not require optimizer or scheduler state.** A
  checkpoint missing `optimizer.pt` or `scheduler.pt` passes, and the run resumes
  with a fresh optimizer and a restarted schedule, without a warning. The Kaggle
  training notebook sets such a checkpoint aside before resuming.

## Reproducibility

Every run records model, base revision, dataset version and hash, config hash,
seed, git commit and environment.

```
outputs/<experiment-id>/
  adapter/  tokenizer/  config.yaml  manifest.json  metrics.json
  events.jsonl  environment.txt  training.log  README.md  checkpoint-*/
```

Two runs with the same `config_hash` used the same settings. Git information
degrades gracefully: without commits, the manifest records that fact rather than
omitting the field.

## Failed runs

A crashed run still writes its manifest, with `status: "failed"` and the failing
stage. Nothing is deleted, and `ExperimentRegistry` lists failures beside
successes.

## Training another model

The model is whatever the training config includes (`includes.model`). To train
a different one, use or copy a training config that includes it, as
`kleos_hermes_v006.yaml`, `kleos_logos_v001.yaml` and `kleos_logos_v002.yaml` do.
`--set model.name=...` only renames the model entry; it does not load a different
checkpoint.

Everything family-specific (auto class, LoRA targets, exclusions, reasoning
capability, quantization exceptions, checkpoint view) lives in the family adapter
([architecture.md](architecture.md#the-model-family-adapter-layer)). Adding a
family means a model config and an adapter subclass, registered; the training
pipeline does not change.

## transformers compatibility

`src/kleos_models/compat.py` bridges transformers 4.56 to 5.x:

| Concern | 4.x | 5.x |
| --- | --- | --- |
| LR warmup | `warmup_ratio` | `warmup_steps` (a float below 1 is a ratio) |
| Trainer tokenizer | `tokenizer=` | `processing_class=` |
| Load dtype | `torch_dtype=` | `dtype=` |
| Removed arguments | `overwrite_output_dir` and others | dropped |
| Tokenizer regex patch | not available | `fix_mistral_regex` |

Configs keep one set of names; the shim emits what the installed version accepts
and records every translation in the manifest. It introspects the installed
classes rather than branching on a version number. Every KLEOS run used
transformers 5.16.1 and peft 0.20.0, and the Ministral 3 adapters need
transformers 5 or later (`ministral3` first appears in 5.0.0).

## Troubleshooting

See [troubleshooting.md](troubleshooting.md). Out-of-memory errors from this
pipeline list what to change, in order, with the detected GPU and the
configuration that failed. The run records behind the numbers on this page are
indexed in [experiments/README.md](experiments/README.md).
