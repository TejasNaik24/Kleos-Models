# Training on Google Colab

Colab is the canonical environment for KLEOS training runs. This is the complete
procedure, written for someone who has not done it before.

## Before you start

You need a Google account. You do not need a paid Colab plan — the 8B models train
on the free tier. You do not need a Hugging Face account unless you want to use a
gated model (Mistral) or publish an adapter.

## 1. Open the notebook

Go to [colab.research.google.com](https://colab.research.google.com) →
**GitHub** tab → paste the repository URL → open
`notebooks/02_train_qlora.ipynb`.

Or open `notebooks/00_environment_check.ipynb` first if this is a fresh runtime
and you want to see what you were assigned.

## 2. Turn on the GPU

**Runtime → Change runtime type → T4 GPU → Save.**

This is the step people forget. Without it everything installs correctly and
training fails at the first CUDA call.

## 3. Understand what you were assigned

Free-tier Colab hands out different GPUs depending on load and recent usage. Run
the GPU cell and read the output.

| GPU | VRAM | Compute capability | bf16? | Trains 8B in 4-bit? |
| --- | ---: | ---: | --- | --- |
| Tesla T4 | 16GB | 7.5 | **no** | yes |
| L4 | 24GB | 8.9 | yes | yes, comfortably |
| A100 | 40GB | 8.0 | yes | yes, plus 24B/30B |

**The T4 cannot do bfloat16.** KLEOS configs use `compute_dtype: auto`, which
detects this and selects float16. You do not need to change anything, and the
choice is recorded in the run manifest.

## 4. Install dependencies

```python
!python scripts/colab_setup.py
```

### Why not just `pip install -e ".[train]"`

Colab ships a torch build compiled against its specific CUDA driver. Any install
that pulls torch as a dependency replaces that build with a generic wheel, and
CUDA then either stops working or starts crashing in confusing ways — often not
immediately.

`colab_setup.py` installs every torch-dependent package with `--no-deps` and
verifies CUDA still works afterwards. If you install packages manually later in
the session, use `--no-deps` for anything that depends on torch.

## 5. Authenticate (only if you need to)

Required for gated models such as Ministral-8B, and for publishing.

1. Sidebar → **key icon** (Secrets)
2. **Add new secret**, name it `HF_TOKEN`
3. Paste a token from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Enable **Notebook access**

The notebook reads it via `google.colab.userdata`. **Never paste a token into a
cell** — notebooks get shared, committed and screenshotted.

For a gated model you must also accept its licence on the model's Hugging Face
page while signed in. The token alone is not enough.

## 6. Check feasibility before downloading anything

```python
!python scripts/plan_run.py --all-models
```

This estimates peak memory for each model against the GPU you actually have, in
about a second, with no downloads. Expected on a free T4:

```
qwen3_8b                       full_research         9.6GB      +5.4GB
ministral_8b                   full_research         9.0GB      +6.0GB
mistral_small_3_2              infeasible           16.5GB      -1.5GB
qwen3_30b_a3b_thinking         infeasible           20.1GB      -5.1GB
```

The 24B and 30B models are not trainable on a free T4. That is a fact about the
hardware, not a configuration problem — they need an A100-class GPU.

## 7. Send checkpoints to Drive

**Do this before a long run.** Colab reclaims runtimes without warning, and
frequently overnight.

```python
from google.colab import drive

drive.mount("/content/drive")

OUTPUT_DIR = "/content/drive/MyDrive/kleos/outputs"
import os

os.environ["HF_HOME"] = "/content/drive/MyDrive/kleos/hf_cache"
```

Setting `HF_HOME` to Drive also means a reconnect does not re-download the base
model — which on a slow day is the difference between resuming in one minute and
in twenty.

## 8. Dry run

```python
!python scripts/train.py --config configs/training/qlora_small.yaml --dry-run
```

Validates config, dataset and feasibility, writes a manifest, and stops before
loading weights.

## 9. Train

```python
!python scripts/train.py \
    --config configs/training/qlora_small.yaml \
    --dataset data/examples \
    --output-dir {OUTPUT_DIR}
```

Before the loop starts you get GPU, VRAM, CUDA and library versions, model and
quantization settings, LoRA configuration and a memory estimate. Then a gradient
check confirms the adapter actually receives gradients — a setup that trains
nothing would otherwise still produce a plausible-looking loss curve.

## 10. When the runtime dies

It will. Re-run the setup cells, remount Drive, then:

```python
!python scripts/train.py \
    --config configs/training/qlora_small.yaml \
    --output-dir {OUTPUT_DIR} \
    --resume-from-checkpoint auto
```

`auto` finds the newest **valid** checkpoint. A checkpoint half-written when the
runtime was killed is detected as incomplete and skipped, rather than causing a
confusing failure on resume.

## Staying connected longer

- Keep the browser tab open and interact with it occasionally.
- Free runtimes are capped at roughly 12 hours, and idle ones are reclaimed much
  sooner.
- Set `save_steps` low enough that losing the interval since the last checkpoint
  is acceptable.
- Do not rely on browser auto-clicker hacks; they violate Colab's terms and get
  accounts limited.

## Common problems

**`CUDA out of memory`**
The error itself lists what to change, in order. Start with `max_seq_length`. If
it happened at step 0, the configuration never fit — run `plan_run.py`. If it
happened mid-run, it was probably an evaluation spike: lower
`per_device_eval_batch_size`.

**`torch` stops seeing the GPU after installing something**
A package replaced Colab's torch. Runtime → Restart, then re-run
`colab_setup.py`, which uses `--no-deps` for exactly this reason.

**`401` or `403` downloading a model**
Gated repository. Accept the licence on its model page while signed in, and check
`HF_TOKEN` is set as a Colab secret with notebook access enabled.

**`KeyError: 'qwen3_moe'` or an unrecognized architecture**
transformers is too old. `colab_setup.py` pins `>=4.56,<6`.

**Training seems to work but the loss never moves**
Run `python scripts/smoke_test.py`. The gradient check should catch a
disconnected adapter before training starts; if it passed and the loss is still
flat, the learning rate or the data is the problem, not the plumbing.

More detail: [troubleshooting.md](troubleshooting.md).

## What a completed run leaves behind

```
outputs/<experiment-id>/
  adapter/          ← the model artifact
  tokenizer/
  config.yaml       ← fully resolved effective config
  manifest.json     ← complete provenance
  metrics.json
  events.jsonl      ← structured event stream
  environment.txt   ← the pre-flight report
  README.md
  checkpoint-*/
```

Keep the whole directory. The adapter alone is not reproducible — the manifest is
what ties it to a dataset version, a config hash, a seed and a commit.

## Next

`notebooks/03_evaluate.ipynb` compares the adapter against the base model under
identical conditions.

A training loss curve is not a result. Evaluation decides.
