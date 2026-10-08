# Troubleshooting

Known errors, their causes and their fixes. Findings referenced by id are indexed
in [experiments/README.md](experiments/README.md).

## Memory

### CUDA out of memory

The error lists the detected GPU, the configuration that failed, and what to
change, in order. Work down that list:

1. Reduce `model.max_seq_length`; activation memory scales with it.
2. Enable `training.gradient_checkpointing`.
3. Halve `per_device_train_batch_size` and double `gradient_accumulation_steps`;
   the effective batch size and learning dynamics stay the same.
4. Use `training.optim: paged_adamw_8bit`.
5. Check that nothing else holds GPU memory (`nvidia-smi`; in a notebook, restart
   the runtime).
6. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
   fragmentation.

**At step 0**, the configuration never fit. `python scripts/plan_run.py --config
<config>` answers that in a second, before any download.

**Mid-run**, it was probably an evaluation or checkpoint spike. Lower
`per_device_eval_batch_size` to 1.

**A 24B model on a 16 GB GPU** does not fit, and no tuning will make it. Use a
smaller config or an A100-class GPU.

### The estimate and a measured smoke run disagree

The estimator is arithmetic; the memory probe measures. When a smoke run at the
real settings passed its memory gate but the estimate refuses the full run,
`python scripts/train.py ... --feasibility record` records the assessment and
trains anyway, leaving the decision to the probe before step 1. Never use it to
skip checking.

## Models and transformers versions

### `AutoModelForCausalLM` fails on Mistral Small 3.2

Expected. `mistralai/Mistral-Small-3.2-24B-Instruct-2506` declares
`Mistral3ForConditionalGeneration` (`model_type: mistral3`), and transformers
registers that only in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`. It is a
vision-language model.

`Mistral3VLMAdapter` loads it through `AutoModelForImageTextToText` and scopes LoRA
to the language tower; use `configs/models/mistral_small_3_2.yaml` rather than a
hand-written config. On transformers 5 that scoping does not work (finding L-F1,
in [experiments/logos-findings.md](experiments/logos-findings.md)): no KLEOS run uses
the adapter, and `tests/test_vlm_targeting_finding.py` holds the gap as strict
expected failures.

Confirm what a checkpoint is:

```bash
python scripts/inspect_model.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

### A Ministral 3 checkpoint will not load: unknown `ministral3`

transformers is older than 5.0.0, where `ministral3` first appears. Reading the
container's `config.json` fails ("Could not read the config for ...") or the
architecture is not recognized. Install a 5.x release; every KLEOS run used
5.16.1:

```bash
pip install -U "transformers==5.16.1"
```

### `fix_mistral_regex=True was requested, but the installed transformers predates the flag`

Same cause: the Logos configs set `fix_mistral_regex: true`, which needs
transformers 5 or later. Upgrade rather than changing the flag, so training,
evaluation and serving keep tokenizing alike.

### A thinking-only model refuses `standard` mode

Working as intended. Ministral 3 Reasoning always thinks, so
`Ministral3ReasoningTextAdapter` refuses `model.reasoning.default_mode: standard`
rather than producing a prompt the model was not built for. Use
`configs/models/ministral3_14b_reasoning.yaml` as it is (`thinking`), or the
Instruct release (`configs/models/ministral3_14b.yaml`) for answers without
thinking.

The reverse also raises: requesting `thinking` from a model with no reasoning
mode. The pipeline never fakes reasoning by injecting `<think>` tags.

### The same tokenizer gives different token counts

Finding L-F2. With transformers 5.16.1, `AutoTokenizer` loaded from the Hub
repository (with `config.json`) builds `TokenizersBackend`; loaded from a folder of
tokenizer files alone it builds `LlamaTokenizer`, which splits the same text into
about 12% more tokens with different ids (442 against 492 for the longest Logos
v0.0.1 example). Load Ministral 3 tokenizers from the repository at the pinned
revision, never rebuilt from bare tokenizer files. A tokenizer saved by training
reloads as `TokenizersBackend` with identical ids.

### `TypeError: TrainingArguments got an unexpected keyword argument 'warmup_ratio'`

transformers 5 removed `warmup_ratio` in favor of `warmup_steps` (a float below 1
is a ratio). `kleos_models.compat` translates it. Seeing this error means
something constructs `TrainingArguments` directly instead of through
`build_training_arguments`.

### `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`

Same cause: transformers 5 renamed it to `processing_class`. Use
`compat.trainer_tokenizer_kwarg()`.

## Hugging Face access

### `401`, `403` or "gated repo" when downloading

The model requires accepting a license. Of the KLEOS bases only Ministral-8B is
gated; Mistral-Nemo and both Ministral 3 releases are not.

1. Open the model page on Hugging Face while signed in and accept the terms.
2. Create a token at <https://huggingface.co/settings/tokens>.
3. Set `HF_TOKEN` in the environment, or as a Colab secret with notebook access.

The token alone is not enough: the license must be accepted by that account.

## bitsandbytes

### Not installed, or no CUDA

bitsandbytes is CUDA-only in practice, with no working macOS build, which is why
it is a separate `[quant]` extra rather than part of `[train]`.

- CUDA machine: `pip install -e ".[train,quant]"`
- Colab or Kaggle: `python scripts/colab_setup.py`
- macOS: 4-bit QLoRA does not run locally. The data, validation and
  evaluation-scoring pipelines all work.

## Training

### Colab: torch stops seeing the GPU after installing packages

A package replaced Colab's torch build with a generic wheel. Runtime → Restart
runtime, then use `python scripts/colab_setup.py`, which installs torch-dependent
packages with `--no-deps` to prevent this.

### The runtime disconnected mid-training

Expected on free tiers. Re-run the setup, remount the storage that holds the run,
then:

```bash
python scripts/train.py --config <config> --experiment-id <original-id> \
    --resume-from-checkpoint auto
```

`--experiment-id` must be the id the run started with. A generated id is new on
every invocation, so `auto` would search an empty directory and the run would
start again from step 0. If checkpoints were not written to persistent storage,
the run is lost; point `--output-dir` at `<private storage>/outputs` next time
([colab.md](colab.md#7-send-checkpoints-to-drive)).

### A run was started fresh in a folder that already has checkpoints

Finding L-F4, not fixed. `train.py` does not refuse a fresh start where
checkpoints exist, and checkpoint rotation in the new run can delete the old run's
best checkpoint. Stop the run before its first save, move the old checkpoints
aside, and resume with `--resume-from-checkpoint auto` instead.

### A resumed run trains with a fresh optimizer

Finding L-F5, not fixed. `validate_checkpoint` accepts a checkpoint without
`optimizer.pt` or `scheduler.pt`, and transformers then resumes with a fresh
optimizer and a schedule restarted from warmup, without a warning. Before
resuming, check that the newest checkpoint holds both files; set it aside if not.
The Kaggle training notebook does this automatically.

### A model split over two GPUs is refused

`The model is spread over GPUs [0, 1], but the Trainer is not running it model
parallel` comes from `assert_model_parallel`. With `device_map: auto` on several
GPUs, training is safe only when the Trainer runs the split model in place;
otherwise it would treat the GPUs as data-parallel replicas. transformers 5.16.1
detects this from `hf_device_map`. Pin `transformers==5.16.1`, as the Kaggle
notebooks do, or set `model.device_map: cuda:0` to train on one GPU if the
feasibility check says it fits ([training.md](training.md#two-gpus)).

### A reasoning run refuses to start: target truncation

`N example(s) lost part of their target to max_seq_length=...` A model trained to
think (`strip_thinking_from_targets: false`) is trained on every trace whole, so
any truncation of a supervised token is refused before training. Raise
`model.max_seq_length` above the longest example; `train.py` prints the longest
train sequence before loading weights.

### "No usable examples remain after formatting"

Every example produced zero supervised tokens, usually because `max_seq_length`
truncated the answer away.

```bash
python scripts/inspect_dataset.py --dataset <path to release> --tokenizer <model> --show-masking
```

Raise `model.max_seq_length` or shorten the examples.

### Training runs but the loss never moves

The gradient check catches a disconnected adapter before training starts. If it
passed and the loss is flat, the plumbing is fine: look at the learning rate, the
data, or whether the answers are learnable. To confirm the pipeline itself:

```bash
python -m pytest tests/test_training_tiny_model.py -v
```

Those tests train a tiny model on CPU and assert that the loss decreases.

### `config_hash` differs from the pre-registered one

The hash covers the resolved config, including the dataset path and output
directory. A different `--dataset` or `--output-dir`, a `KLEOS_*` environment
variable, or a `--set` override all change it. The H8 and H9 hashes were computed
with the literal paths recorded in [experiments.md](experiments.md); the Kaggle
notebook refuses to train on a mismatch and to run with any `KLEOS_*` variable
set.

## Evaluation

### A resumed evaluation found 0 recorded generations on Google Drive

Finding L-F3, fixed. An earlier version held `<output>.partial.jsonl` open for the
whole arm, and Colab's Drive mount uploaded it only once closed, so a killed
session lost its generations. The writer has opened and closed the file per
record since commit `ed5a987`. On an older commit, update and evaluate again.

### Resume refused: the identity differs

The partial file records everything that could change a generation: arm, seeds,
example ids, benchmark sha256, decoding settings, orchestration prompt, model
config, adapter hashes, git commit, library versions and compute capability.
Resuming under a different one is refused, because the replayed and new
generations would not come from one run. Restore the original conditions (on
Kaggle, "Pin to original environment"), or start again with `--overwrite`.

### `compare.py` exits with code 2

The two results were graded against different benchmarks, checked by content
(finding H-F5). Rebuild the benchmark from the same sealed test split and
evaluate again.

### Consistency reports zero groups

No benchmark example carries the configured grouping key, so equivalent cases
cannot be grouped and consistency measures nothing. Add `metadata.group_id` (the
perturbations of one case) to the examples; `corrected.consistency` groups by it
([data-contract.md](data-contract.md#scenario_family-and-group_id)).

### OOD reports "not measurable"

No benchmark example is tagged `split_tag: "ood"`, or none is in distribution, so
there is no gap to measure. No generalization claim can be made until there is.
The report states this rather than omitting the section.

## Data

### Dataset validation fails on `possible_sensitive_content`

The validator found something that looks like an email address, a token or a key.
Treat it as real until proven otherwise ([privacy.md](privacy.md)). If it is
synthetic, rewrite it so it cannot be mistaken for real data (`name@example.com`,
not a plausible address).

### Leakage detected between splits

A cross-split duplicate means an evaluation example is effectively in training,
and any generalization number from it measures memorization.

1. Read `reports/leakage/leakage_summary.md`.
2. Deduplicate before splitting.
3. Use a held-out strategy (`format_holdout`, `entity_holdout`,
   `scenario_family_holdout`) so related examples cannot straddle the boundary.

Override with `--leakage-policy none` only with a recorded reason.

### Split fails: "attribute has only 1 distinct value"

A held-out split needs at least two distinct values of the attribute it
partitions on. Check coverage:

```bash
python scripts/inspect_dataset.py --dataset <path to release> --coverage
```

Add examples covering another value, or use `strategy: group`.

### Config error: unknown field

Configs reject unknown keys, so a typo fails loudly. Check the spelling against
`configs/*/README.md`:

```bash
python scripts/validate_configs.py
```

## Serving

### A ZeroGPU Space build fails resolving pydantic

Hugging Face installs `gradio[oauth,mcp]` beside a Space's requirements, and its
`mcp` extra requires `pydantic>=2.11.10,<=2.12.5`. The Space templates
(`deploy/zerogpu-space*/requirements.txt.template`) therefore pin
`pydantic==2.12.5`, while the Docker image uses 2.13.5. Keep that pin when
updating the templates; the full test suite passes on 2.12.5.

### The served answers differ from the recorded ones

Follow [deployment.md](deployment.md#diagnosing-a-reproduction-mismatch). Do not
serve until the difference is explained.

## Still stuck

```bash
python scripts/smoke_test.py -v
```

It checks each layer in order and reports what failed and what was skipped.
Include its output in any bug report, with
`pip list | grep -E "torch|transformers|peft|bitsandbytes"`.
