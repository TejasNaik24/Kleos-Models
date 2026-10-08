# Training on Google Colab

Free Colab gives one GPU per session, usually a T4. Hermes v0.0.6 and Logos
v0.0.1 were both trained and evaluated there. This page is the general procedure;
the Logos v0.0.1 run itself is in
[runbooks/logos-v001-colab.md](runbooks/logos-v001-colab.md). For runs that need
two GPUs, see [kaggle.md](kaggle.md).

## Before starting

- A Google account. A paid Colab plan is not needed: Hermes (12B) and the Logos
  v0.0.1 text tower (14B) trained in 4-bit on the free T4 (measured).
- A Hugging Face account only for a gated model (Ministral-8B) or for publishing.
  Mistral-Nemo and both Ministral 3 releases are ungated.
- Space on Google Drive for checkpoints, and optionally for the model cache
  (below).

## 1. Open the notebook

At [colab.research.google.com](https://colab.research.google.com), open the
**GitHub** tab, paste the repository URL and open `notebooks/02_train_qlora.ipynb`.
On a fresh runtime, `notebooks/00_environment_check.ipynb` shows what was
assigned before anything is installed.

## 2. Turn on the GPU

**Runtime → Change runtime type → T4 GPU → Save.**

Without this, everything installs and training fails at the first CUDA call.

## 3. Check what was assigned

Free Colab assigns GPUs by load and recent usage. Run the GPU cell and read the
output.

| GPU | VRAM | Compute capability | bfloat16 |
| --- | ---: | ---: | --- |
| Tesla T4 | 16 GB (14.56 GB reported by torch on Colab) | 7.5 | no |
| L4 | 24 GB | 8.9 | yes |
| A100 | 40 GB | 8.0 | yes |

A T4 cannot do bfloat16. KLEOS configs use `compute_dtype: auto`, which detects
this and selects float16, and the run manifest records the choice. Nothing needs
to change.

## 4. Install dependencies

```python
!python scripts/colab_setup.py
```

Colab ships a torch build compiled against its CUDA driver. Any install that pulls
torch as a dependency replaces that build with a generic wheel, and CUDA then
stops working or crashes in confusing ways, often not immediately.
`colab_setup.py` installs every torch-dependent package with `--no-deps` and
verifies CUDA afterwards. Use `--no-deps` for anything torch-dependent installed
later in the session.

## 5. Authenticate, if needed

Required for gated models such as Ministral-8B, and for publishing.

1. Sidebar → key icon (Secrets).
2. **Add new secret**, named `HF_TOKEN`.
3. Paste a token from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Enable **Notebook access**.

The notebook reads it through `google.colab.userdata`. Never paste a token into
a cell: notebooks get shared, committed and screenshotted. For a gated model, the
account must also accept the model's license on its Hugging Face page; the token
alone is not enough.

## 6. Check feasibility before downloading anything

```python
!python scripts/plan_run.py --all-models
```

This estimates peak memory for every model config against the GPU actually
present, in about a second, with no downloads. Its output for a free T4 (the
`t4-colab` preset, which reproduces Colab's T4) on 2026-10-07:

```
model                          tier              est. peak   headroom
------------------------------ ---------------- ---------- ----------
ministral3_14b                 smoke                15.6GB      -1.2GB
ministral3_14b_reasoning       smoke                15.6GB      -1.2GB
ministral_8b                   adapter_train        13.5GB      +0.9GB
mistral_nemo_12b               smoke                15.0GB      -0.5GB
mistral_small_3_2              infeasible           22.8GB      -8.4GB
```

These figures assume each config's `max_seq_length`, the worst case. The KLEOS
data is much shorter, and `train.py` sizes its own check by the longest real
example. `--seq-length` gives the planner that length:

| Config | Base | Gated | On a free T4 |
| --- | --- | --- | --- |
| `mistral_nemo_12b` | Mistral-Nemo-Instruct-2407 (Hermes) | no | trains; 13.09 GB peak measured |
| `ministral_8b` | Ministral-8B-Instruct-2410 | yes | trains; 9.67 GB peak measured |
| `ministral3_14b` | Ministral 3 14B Instruct, text tower (Logos v0.0.1) | no | trains; 13.60 GB at the longest batch measured, 0.35 GB spare |
| `ministral3_14b_reasoning` | Ministral 3 14B Reasoning, text tower (Logos v0.0.2) | no | does not fit at 736 tokens (estimated); trained on Kaggle's 2 × T4 |
| `mistral_small_3_2` | Mistral Small 3.2 24B, vision-language | not verified | does not fit; needs an A100-class GPU; never trained |

No configuration change makes the 24B model fit a T4.

## 7. Send checkpoints to Drive

Do this before a long run. Colab reclaims runtimes without warning, often
overnight.

Section 7 of `02_train_qlora.ipynb` mounts Google Drive and points the run at it:

```python
from google.colab import drive
import os

drive.mount("<drive mount point>")
OUTPUT_DIR = "<private storage>/outputs"
os.environ["HF_HOME"] = "<private storage>/hf_cache"
```

`<private storage>` is a folder on the mounted Drive. Keeping `HF_HOME` on Drive
means a reconnect does not re-download the base model, at the cost of Drive space
equal to the checkpoint (about 24.5 GB for Mistral-Nemo, 28 GB for the Ministral 3
container). With little Drive space, keep `HF_HOME` on the runtime disk and accept
the re-download, as the Logos v0.0.1 runbook does. One run filled its Drive quota
partway through (deviation D9 in [experiments.md](experiments.md#deviations-log)).

Drive is a mounted file system with its own upload behavior. A checkpoint reported
as saved once turned out to lack its weights 18 hours later (finding H-F4), and an
evaluation resume file that stayed open never reached Drive (finding L-F3, fixed).
The record behind both is indexed in [experiments/README.md](experiments/README.md).

## 8. Dry run

```python
!python scripts/train.py --config configs/training/qlora_small.yaml --dry-run
```

Validates config, dataset and feasibility, writes a manifest, and stops before
loading weights.

## 9. Train

Pin the experiment id first: step 10 needs the same one to resume.

```python
EXPERIMENT_ID = "my-first-run"
```

```python
!python scripts/train.py \
    --config configs/training/qlora_small.yaml \
    --dataset data/examples \
    --output-dir {OUTPUT_DIR} \
    --experiment-id {EXPERIMENT_ID}
```

Before the loop starts, the run prints GPU, VRAM, CUDA and library versions, the
model and quantization settings, the LoRA configuration and a memory estimate.
A gradient check then confirms that the adapter receives gradients, and the memory
probe measures the longest batch.

## 10. When the runtime dies

Re-run the setup cells, remount Drive, then:

```python
!python scripts/train.py \
    --config configs/training/qlora_small.yaml \
    --output-dir {OUTPUT_DIR} \
    --experiment-id {EXPERIMENT_ID} \
    --resume-from-checkpoint auto
```

`--experiment-id` must be the id the run started with. It names the run's
directory; a generated id is new on every invocation, so `auto` would search an
empty directory and the run would start again from step 0.

`auto` takes the newest valid checkpoint and skips one half-written when the
runtime was killed. Always resume this way: a fresh start into a folder that
already holds checkpoints is not guarded against and can delete the best one
(finding L-F4, open; [training.md](training.md#checkpointing-and-resume)).

An evaluation resumes the same way: re-run the same `evaluate.py` command with
`--resume` ([evaluation.md](evaluation.md#resuming-an-evaluation)).

## Staying connected

- Keep the browser tab open and interact with it occasionally.
- Free runtimes are capped at roughly 12 hours, and idle ones are reclaimed much
  sooner. The free GPU usage limit can also end a session (deviation D9).
- Set `save_steps` low enough that losing the interval since the last checkpoint
  is acceptable.
- Do not use browser auto-clickers; they violate Colab's terms and get accounts
  limited.

## Common problems

**`CUDA out of memory`.** The error lists what to change, in order; start with
`max_seq_length`. At step 0 the configuration never fit: run `plan_run.py`.
Mid-run, it was probably an evaluation spike: lower `per_device_eval_batch_size`.

**`torch` stops seeing the GPU after installing something.** A package replaced
Colab's torch. Runtime → Restart, then re-run `colab_setup.py`.

**`401` or `403` downloading a model.** A gated repository. Accept the license on
its model page while signed in, and check that `HF_TOKEN` is a Colab secret with
notebook access.

**Training runs but the loss never moves.** Run `python scripts/smoke_test.py`.
The gradient check catches a disconnected adapter before training starts; if it
passed, look at the learning rate or the data.

More: [troubleshooting.md](troubleshooting.md).

## What a completed run leaves

```
outputs/<experiment-id>/
  adapter/          the model artifact
  tokenizer/
  config.yaml       fully resolved effective config
  manifest.json     complete provenance
  metrics.json
  events.jsonl      structured event stream
  environment.txt   the pre-flight report
  training.log
  README.md
  checkpoint-*/
```

Keep the whole directory. The adapter alone is not reproducible; the manifest ties
it to a dataset version, a config hash, a seed and a commit.

## Next

`notebooks/03_evaluate.ipynb` evaluates the adapter against the base model under
identical conditions ([evaluation.md](evaluation.md)). Whether fine-tuning helped
is decided there, not by the training loss.
