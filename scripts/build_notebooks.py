#!/usr/bin/env python3
"""Generate the Colab notebooks from source-of-truth cell definitions.

Notebooks are JSON and awful to hand-edit or review in a diff. They are generated
here instead, so notebook content is reviewable Python and stays consistent with
the CLI it wraps.

Regenerate after changing any cell:

    python scripts/build_notebooks.py

``tests/test_notebooks.py`` validates the output and checks for hard-coded
secrets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

REPO_URL = "https://github.com/kleos/kleos-models.git"


def markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().split("\n"),
    }


def notebook(cells: list[dict[str, Any]], *, title: str) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kleos": {"title": title},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------

CLONE_CELL = code(
    f"""
# Clone the repository (skip if already present) and enter it.
import os
from pathlib import Path

REPO_DIR = Path("/content/kleos-models")

if not REPO_DIR.exists():
    !git clone {REPO_URL} {{REPO_DIR}}
else:
    print(f"{{REPO_DIR}} already exists; pulling latest")
    !cd {{REPO_DIR}} && git pull --ff-only

os.chdir(REPO_DIR)
print("working directory:", Path.cwd())
"""
)

SETUP_CELL = code(
    """
# Install dependencies WITHOUT touching Colab's torch build.
#
# Reinstalling torch on Colab replaces the build compiled against this runtime's
# CUDA driver, and CUDA then silently stops working. scripts/colab_setup.py uses
# --no-deps for every package that would otherwise pull torch along.
!python scripts/colab_setup.py
"""
)

AUTH_CELL = code(
    """
# Hugging Face authentication.
#
# Needed only for gated base models (Mistral) or to publish an adapter.
# Use Colab Secrets (the key icon in the left sidebar), never a literal token in
# a cell — notebooks get shared and committed.
import os

try:
    from google.colab import userdata

    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN loaded from Colab secrets.")
    else:
        print("No HF_TOKEN secret set.")
except Exception as exc:
    print(f"Colab secrets unavailable ({type(exc).__name__}).")
    print("Set os.environ['HF_TOKEN'] manually if you need gated models.")

if not os.environ.get("HF_TOKEN"):
    print()
    print("Without a token you can still use ungated models such as Qwen/Qwen3-8B.")
    print("To add one: sidebar key icon -> Add new secret -> name HF_TOKEN ->")
    print("enable 'Notebook access'.")
"""
)

GPU_CELL = code(
    """
# What GPU did Colab actually assign? Free-tier allocation varies.
!nvidia-smi

from kleos_models.models.feasibility import probe_gpu

gpu = probe_gpu()
print()
print(gpu.render())
print()
if not gpu.available:
    print("NO GPU ASSIGNED.")
    print("Runtime -> Change runtime type -> T4 GPU, then re-run this cell.")
elif not gpu.bf16_supported:
    print(f"Note: {gpu.name} (compute capability {gpu.capability_string}) has no")
    print("bfloat16 support. KLEOS configs use compute_dtype: auto, which selects")
    print("float16 here automatically. Nothing to change.")
"""
)


# ---------------------------------------------------------------------------
# 00 — environment check
# ---------------------------------------------------------------------------


def build_environment_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                """
# KLEOS 00 — Environment check

Run this first on any new Colab runtime. It answers two questions before you
spend a session on anything expensive:

1. Is the environment set up correctly?
2. **Which models can this particular runtime actually train?**

Free-tier Colab assigns different GPUs at different times. A T4 (16GB) trains an
8B model in 4-bit comfortably and cannot train a 24B model at all. Finding that
out here takes seconds; finding it out after a 28GB download does not.
"""
            ),
            markdown("## 1. Clone the repository"),
            CLONE_CELL,
            markdown("## 2. Install dependencies"),
            SETUP_CELL,
            markdown(
                """
## 3. Check the assigned GPU

The key numbers are total VRAM and compute capability. Compute capability below
8.0 (a T4 is 7.5) means no bfloat16, so training uses float16 instead.
"""
            ),
            GPU_CELL,
            markdown("## 4. Authenticate (optional)"),
            AUTH_CELL,
            markdown(
                """
## 5. Feasibility across every model config

This is the cell worth reading carefully. It estimates peak memory for each
supported model **on the GPU you were actually assigned**, with no downloads.
"""
            ),
            code(
                """
!python scripts/plan_run.py --all-models
"""
            ),
            markdown(
                """
### Reading the table

| Tier | Meaning |
| --- | --- |
| `full_research` | Fits with headroom. Use it. |
| `adapter_train` | Fits, but tight. Evaluation spikes may still OOM. |
| `smoke` | Only a reduced config fits — the research config does not. |
| `inference_only` | Can be evaluated here, not trained. |
| `infeasible` | Cannot even be loaded. |

On a free T4 expect `qwen3_8b` and `ministral_8b` to be trainable, and
`mistral_small_3_2` and `qwen3_30b_a3b_thinking` to be infeasible. That is not a
bug — those need an A100-class GPU. The estimates are deliberately slightly
pessimistic.
"""
            ),
            markdown("## 6. Smoke test"),
            code(
                """
!python scripts/smoke_test.py
"""
            ),
            markdown(
                """
Everything above should pass or be explicitly skipped. A skip means an optional
dependency is missing; a failure means something is wrong and training will not
work.

## Next

- `01_dataset_validation.ipynb` — check your dataset before training on it
- `02_train_qlora.ipynb` — the canonical training notebook
"""
            ),
        ],
        title="Environment check",
    )


# ---------------------------------------------------------------------------
# 01 — dataset validation
# ---------------------------------------------------------------------------


def build_dataset_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                """
# KLEOS 01 — Dataset validation

Validate a dataset before training on it. Everything here runs on CPU in seconds
and needs no GPU, so you can do dataset work on any runtime.

What gets checked:

- **schema** — every example matches the versioned data contract
- **duplicates and leakage** — exact, normalized and near-duplicates, id
  collisions, scenario repeats, entity leakage
- **coverage** — how many examples cover each *situation type*, not just how many
  examples there are
- **privacy** — patterns that suggest private data reached a public repository
- **policy vs facts** — examples that teach a private fact instead of a decision
  policy
"""
            ),
            markdown("## 1. Setup"),
            CLONE_CELL,
            SETUP_CELL,
            markdown(
                """
## 2. Choose the dataset

By default this validates the bundled **development fixtures**, which are
synthetic and exist only to exercise the pipeline. They are not the KLEOS
research dataset.

To validate the real dataset produced by the private `kleos-training-data`
repository, mount Drive and point `DATASET` at it.
"""
            ),
            code(
                """
DATASET = "data/examples"   # or "/content/drive/MyDrive/kleos/dataset"

# Uncomment to mount Drive for a private dataset:
# from google.colab import drive
# drive.mount("/content/drive")

print("validating:", DATASET)
"""
            ),
            markdown("## 3. Validate schema, content and leakage"),
            code(
                """
!python scripts/validate_dataset.py --dataset {DATASET} --leakage-report reports/leakage
"""
            ),
            markdown(
                """
A cross-split duplicate is the finding that matters most: it means an evaluation
example is effectively present in training, and any "generalization" number
computed from it is measuring memorization.
"""
            ),
            markdown("## 4. Quality and coverage report"),
            code(
                """
!python scripts/inspect_dataset.py --dataset {DATASET} --coverage
"""
            ),
            markdown(
                """
### What to look for

- **Constant axes.** An axis with one value cannot support a claim about that
  axis.
- **Empty cells.** Situation types with no examples at all.
- **Thin cells.** Present but too sparse to support a per-cell claim.
- **Missing metadata.** Especially `scenario_family`, without which consistency
  testing cannot group anything.

A dataset is not diverse because it is large.
"""
            ),
            markdown(
                """
## 5. Exact token statistics (optional)

Approximate counts are fine for triage. For the real numbers, load the tokenizer
of the model you plan to train.
"""
            ),
            code(
                """
!python scripts/inspect_dataset.py --dataset {DATASET} --tokenizer Qwen/Qwen3-8B
"""
            ),
            markdown(
                """
## 6. Create a versioned split

Random splitting is development-only. Generalization claims need a held-out
strategy — otherwise paraphrases of the same scenario land on both sides of the
boundary.
"""
            ),
            code(
                """
# Entity-held-out: does the learned policy transfer to unseen entities?
!python scripts/split_dataset.py \\
    --input {DATASET}/synthetic_train.jsonl \\
    --output /content/kleos-dataset-v1 \\
    --strategy entity_holdout \\
    --seed 42 \\
    --version kleos-policy-v0.1.0
"""
            ),
            markdown(
                """
The split is verified for overlap before it is written, and the manifest records
the strategy, seed, held-out values and per-file hashes.

## Next

`02_train_qlora.ipynb` — train an adapter on this dataset.
"""
            ),
        ],
        title="Dataset validation",
    )


# ---------------------------------------------------------------------------
# 02 — QLoRA training (the canonical entry point)
# ---------------------------------------------------------------------------


def build_training_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                """
# KLEOS 02 — QLoRA training

**This notebook is the canonical way to train a KLEOS adapter.**

It runs the real pipeline: 4-bit quantized base model, LoRA adapter, assistant-
only loss masking, a genuine `Trainer` loop, checkpointing, and a full experiment
manifest.

## Before you start

1. **Set the runtime to GPU.** Runtime → Change runtime type → T4 GPU.
2. **Expect to be disconnected.** Free Colab reclaims runtimes without warning.
   Section 7 writes checkpoints to Drive so a disconnect costs minutes, not the
   session.
3. **Check feasibility first.** Section 5 tells you whether your chosen model fits
   the GPU you were assigned.
"""
            ),
            markdown("## 1. Clone and install"),
            CLONE_CELL,
            SETUP_CELL,
            markdown("## 2. Authenticate"),
            AUTH_CELL,
            markdown("## 3. Check the GPU"),
            GPU_CELL,
            markdown(
                """
## 4. Choose a configuration

`configs/training/debug.yaml` runs ten steps and proves the pipeline works.
`configs/training/qlora_small.yaml` is the real run.

Model configs available:

| Config | Fits a free T4? |
| --- | --- |
| `qwen3_8b` | yes (4-bit) |
| `ministral_8b` | yes (4-bit) — gated, needs HF_TOKEN |
| `mistral_small_3_2` | no — needs A100 |
| `qwen3_30b_a3b_thinking` | no — needs A100 |
"""
            ),
            code(
                """
CONFIG = "configs/training/qlora_small.yaml"
MODEL = "configs/models/qwen3_8b.yaml"
DATASET = "data/examples"   # replace with your dataset directory

# Start with the debug config to prove the pipeline runs end to end:
# CONFIG = "configs/training/debug.yaml"

print(f"config : {CONFIG}")
print(f"model  : {MODEL}")
print(f"dataset: {DATASET}")
"""
            ),
            markdown(
                """
## 5. Will it fit?

Do this **before** downloading weights. If the answer is no, change the model
here rather than discovering it during training.
"""
            ),
            code(
                """
!python scripts/plan_run.py --config {CONFIG} --set model.name=check
"""
            ),
            markdown(
                """
If this reports `infeasible` or `inference_only`, switch to an 8B config. The
pipeline will refuse to start rather than silently shrink your configuration into
something that is no longer comparable with other runs.
"""
            ),
            markdown("## 6. Validate the dataset"),
            code(
                """
!python scripts/validate_dataset.py --dataset {DATASET}
"""
            ),
            markdown(
                """
## 7. Persist checkpoints to Drive (strongly recommended)

Colab runtimes disappear. Writing checkpoints to Drive means a disconnect costs
you the time since the last checkpoint, not the whole session.
"""
            ),
            code(
                """
USE_DRIVE = True   # set False to keep checkpoints on the ephemeral runtime disk

if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = "/content/drive/MyDrive/kleos/outputs"
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Cache weights on Drive too, so a reconnect does not re-download them.
    os.environ["HF_HOME"] = "/content/drive/MyDrive/kleos/hf_cache"
else:
    OUTPUT_DIR = "outputs"

print("checkpoints ->", OUTPUT_DIR)
"""
            ),
            markdown(
                """
## 8. Dry run

Validates the config, dataset and feasibility, writes a manifest, and stops
before loading any weights. Cheap insurance.
"""
            ),
            code(
                """
!python scripts/train.py --config {CONFIG} --dataset {DATASET} --output-dir {OUTPUT_DIR} --dry-run
"""
            ),
            markdown(
                """
## 9. Train

The pre-flight report prints GPU, VRAM, CUDA and library versions, the model and
quantization settings, the LoRA configuration and a memory estimate. Training then
verifies that gradients actually reach the adapter before starting the real loop —
a setup that trains nothing would otherwise still produce a plausible loss curve.
"""
            ),
            code(
                """
!python scripts/train.py \\
    --config {CONFIG} \\
    --dataset {DATASET} \\
    --output-dir {OUTPUT_DIR}
"""
            ),
            markdown(
                """
## 10. Resume after a disconnect

If the runtime died, re-run sections 1-3 and 7, then this cell. `auto` finds the
newest **valid** checkpoint; a half-written checkpoint from an interrupted save is
detected and skipped rather than causing a confusing failure.
"""
            ),
            code(
                """
!python scripts/train.py \\
    --config {CONFIG} \\
    --dataset {DATASET} \\
    --output-dir {OUTPUT_DIR} \\
    --resume-from-checkpoint auto
"""
            ),
            markdown("## 11. Inspect the run"),
            code(
                """
from kleos_models.experiments.registry import ExperimentRegistry

registry = ExperimentRegistry(OUTPUT_DIR)
print(registry.render())

latest = registry.scan()[0] if registry.scan() else None
if latest:
    print()
    print(latest.manifest.summary())
    ADAPTER = str(latest.directory / "adapter")
    print()
    print("adapter:", ADAPTER)
"""
            ),
            markdown(
                """
The manifest ties this artifact to model + revision + dataset version + dataset
hash + config hash + seed + git commit + environment. Check `adjustments` — if the
run was automatically reshaped to fit memory, it is recorded there, and it affects
comparability.

## 12. Optional: publish the adapter

Nothing is published automatically. This uploads adapter weights and a generated
model card, and refuses to upload training data, `.env`, or anything matching the
private-data scanner.
"""
            ),
            code(
                """
# Review what would be uploaded first.
# !python scripts/publish_adapter.py --adapter {ADAPTER} --repo-id YOUR_USERNAME/kleos-qwen3-8b --dry-run

# Then publish:
# !python scripts/publish_adapter.py --adapter {ADAPTER} --repo-id YOUR_USERNAME/kleos-qwen3-8b
"""
            ),
            markdown(
                """
## Next

`03_evaluate.ipynb` — compare this adapter against the base model.

**A training loss curve is not a result.** Whether this adapter is better is
decided by evaluation on a held-out split, not by the loss reaching a low number.
"""
            ),
        ],
        title="QLoRA training",
    )


# ---------------------------------------------------------------------------
# 03 — evaluation
# ---------------------------------------------------------------------------


def build_evaluation_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                """
# KLEOS 03 — Evaluate and compare

Run the base and fine-tuned arms under **identical conditions** and compare them
honestly.

The comparison reports per-task deltas with bootstrap confidence intervals, plus
OOD and consistency separately. An improvement that is not statistically
significant is reported as *not significant*, not as a win.
"""
            ),
            markdown("## 1. Setup"),
            CLONE_CELL,
            SETUP_CELL,
            AUTH_CELL,
            GPU_CELL,
            markdown("## 2. Point at the run to evaluate"),
            code(
                """
CONFIG = "configs/training/qlora_small.yaml"
BENCHMARK = "data/examples/synthetic_eval.jsonl"

USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = "/content/drive/MyDrive/kleos/outputs"
else:
    OUTPUT_DIR = "outputs"

from kleos_models.experiments.registry import ExperimentRegistry

registry = ExperimentRegistry(OUTPUT_DIR)
runs = registry.filter(kind="training")
if not runs:
    raise SystemExit(f"No training runs found under {OUTPUT_DIR}. Run notebook 02 first.")

latest = runs[0]
ADAPTER = str(latest.directory / "adapter")
print(latest.manifest.summary())
print()
print("adapter:", ADAPTER)
"""
            ),
            markdown(
                """
## 3. Evaluate the base model (arm 0)

No adapter. This is the baseline the fine-tuned arm must beat.
"""
            ),
            code(
                """
!python scripts/evaluate.py \\
    --config {CONFIG} \\
    --arm arm0_base \\
    --benchmark {BENCHMARK} \\
    --output {OUTPUT_DIR}/base_results.json
"""
            ),
            markdown(
                """
## 4. Evaluate the fine-tuned model (arm 2)

Identical base weights, identical decoding, identical benchmark. **The adapter is
the only difference** — that is what makes the comparison meaningful.
"""
            ),
            code(
                """
!python scripts/evaluate.py \\
    --config {CONFIG} \\
    --arm arm2_finetuned \\
    --adapter {ADAPTER} \\
    --benchmark {BENCHMARK} \\
    --output {OUTPUT_DIR}/finetuned_results.json
"""
            ),
            markdown("## 5. Compare"),
            code(
                """
!python scripts/compare.py \\
    --base {OUTPUT_DIR}/base_results.json \\
    --finetuned {OUTPUT_DIR}/finetuned_results.json \\
    --report reports/latest
"""
            ),
            markdown(
                """
### How to read the comparison

- **Per task, not blended.** A single aggregate would hide a model that improves
  one task while harming another — which may be the most interesting result
  available.
- **Significance.** A positive delta marked *(not significant)* is not a win. It
  means the evaluation set is too small or too noisy to tell.
- **OOD delta.** An in-distribution gain with an OOD loss is consistent with
  fitting surface features rather than learning a transferable policy.
- **Consistency delta.** Whether the model reaches the same decision under
  logically irrelevant perturbations. This is the clearest signal of a learned
  policy versus a surface pattern.
- **Where it failed.** Always read this section.

If OOD is reported as *not measured*, no generalization claim can be made from
this run — tag benchmark examples with `split_tag: "ood"` to enable it.
"""
            ),
            markdown("## 6. Read the report"),
            code(
                """
from pathlib import Path
from IPython.display import Markdown, display

summary = Path("reports/latest/summary.md")
if summary.exists():
    display(Markdown(summary.read_text()))
else:
    print("No report found; check that section 5 completed.")
"""
            ),
            markdown(
                """
## A note on interpreting results

If fine-tuning wins, quantify the win and check it survives OOD.
If it loses, that is a valid, publishable result — analyse data quality, policy
learnability, capacity and evaluation design.
If it improves one task and harms another, that is likely the most informative
outcome the experiment can produce.

Record the run either way. `docs/experiments.md` holds the pre-registered
hypotheses this evaluation is meant to test.
"""
            ),
        ],
        title="Evaluate and compare",
    )


def main() -> int:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_environment_check.ipynb": build_environment_notebook(),
        "01_dataset_validation.ipynb": build_dataset_notebook(),
        "02_train_qlora.ipynb": build_training_notebook(),
        "03_evaluate.ipynb": build_evaluation_notebook(),
    }

    for filename, content in notebooks.items():
        path = NOTEBOOKS_DIR / filename
        path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        cells = len(content["cells"])
        print(f"  wrote {filename:34} ({cells} cells)")

    print(f"\n✓ Generated {len(notebooks)} notebook(s) in {NOTEBOOKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
