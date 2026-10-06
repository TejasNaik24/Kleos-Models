#!/usr/bin/env python3
"""Generate the Kaggle notebooks for KLEOS Logos v0.0.2 (H9).

Logos v0.0.2 trains on Kaggle's free 2 x T4: its reasoning traces lengthen the
longest example to 736 tokens, past what one T4 holds at Logos' settings. Kaggle
runs a notebook unattended ("Save Version" -> "Save & Run All"), so every cell
here runs without a human and stops the run on the first failure. A failing
``!command`` would not stop it, so commands go through ``run()``, which raises.

    python scripts/build_kaggle_notebooks.py

writes notebooks/kaggle/logos_v002_train.ipynb and logos_v002_evaluate.ipynb.
``tests/test_kaggle_notebooks.py`` checks them. docs/logos.md has the runbook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_notebooks import code, markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO_ROOT / "notebooks" / "kaggle"

REPO_URL = "https://github.com/TejasNaik24/Kleos-Models.git"
EXPERIMENT_ID = "kleos-v007-ministral314breasoning-run1"
#: The two paths H9's config hash is computed with (tests/test_logos_v002_config.py).
#: The dataset sits in /tmp, which Kaggle never saves: the private release cannot
#: reach a version's output, however the session ends.
DATASET = "/tmp/kleos-data/kleos-policy-v0.0.7"
OUTPUTS = "/kaggle/working/outputs"
#: Kaggle stops a GPU session at 12 hours. A long command is stopped here instead,
#: so the version ends normally and its output (checkpoints, partial results) is
#: saved for the next version to resume from.
SESSION_HOURS = 11.25
#: Pre-registered under H9 (docs/experiments.md).
CONFIG_HASH = "d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9"
TEST_SHA256 = "a4decaaf029b227366846d44018b0b1b669d21f068800a9f3175e15ec4783980"
BENCHMARK_SHA256 = "a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266"
PINS = (
    "transformers==5.16.1 peft==0.20.0 accelerate==1.14.0 bitsandbytes==0.50.2 tokenizers==0.23.2"
)


def kaggle_notebook(cells: list[dict[str, Any]], *, title: str) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kleos": {"title": title},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------


def settings_cell() -> dict[str, Any]:
    return code(f'''
import time

SESSION_START = time.time()

# Set COMMIT to the full sha of the pushed Kleos-Models commit that pre-registered
# H9. A branch name is refused: the run must be reproducible from its manifest.
COMMIT = "SET-ME"

REPO_URL = "{REPO_URL}"
REPO = "/tmp/kleos-models"  # code stays out of the saved output
EXPERIMENT_ID = "{EXPERIMENT_ID}"
DATASET = "{DATASET}"
OUTPUTS = "{OUTPUTS}"
CONFIG = "configs/training/kleos_logos_v002.yaml"
CONFIG_HASH = "{CONFIG_HASH}"
TEST_SHA256 = "{TEST_SHA256}"
BENCHMARK_SHA256 = "{BENCHMARK_SHA256}"

if len(COMMIT) != 40 or any(c not in "0123456789abcdef" for c in COMMIT):
    raise SystemExit("Set COMMIT to the 40-character sha of the pushed commit (not 'main').")
''')


def helpers_cell() -> dict[str, Any]:
    return code(f'''
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

PY = sys.executable
# Kaggle ends a GPU session at 12 hours; long commands stop here instead, so this
# version still finishes and saves its output for the next one to resume from.
SESSION_DEADLINE = SESSION_START + {SESSION_HOURS} * 3600
STOPPED = False


def run(*args, cwd=None, deadline=False):
    """Run a command, streaming its output, and stop the notebook if it fails.

    With deadline=True the command is stopped at SESSION_DEADLINE; every later
    run() is then skipped, and the version ends normally.
    """
    global STOPPED
    command = " ".join(str(a) for a in args)
    if STOPPED:
        print("skipped (session deadline reached):", command, flush=True)
        return ""
    print("$", command, flush=True)
    process = subprocess.Popen(
        [str(a) for a in args],
        cwd=cwd or REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []

    def pump():
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while process.poll() is None:
        if deadline and time.time() > SESSION_DEADLINE:
            print("SESSION DEADLINE: stopping so this version's output is saved. "
                  "Attach it as input and Save & Run All again to resume.", flush=True)
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=10)
            STOPPED = True
            return "".join(lines)
        time.sleep(2)
    reader.join()
    if process.returncode != 0:
        raise SystemExit(f"FAILED (exit {{process.returncode}}): {{command}}")
    return "".join(lines)


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def copy_writable(source, target):
    """Copy a tree out of /kaggle/input (read-only) so it can be written and removed."""
    shutil.copytree(source, target, copy_function=shutil.copyfile)
    for root, _, files in os.walk(target):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)
''')


def hardware_cell() -> dict[str, Any]:
    return code("""
# Two T4s (Settings -> Accelerator -> GPU T4 x2) and room for the 28 GB checkpoint.
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
).strip().splitlines()
print("\\n".join(gpus))
if len(gpus) != 2 or not all("T4" in g for g in gpus):
    raise SystemExit("Need exactly 2 x Tesla T4: Settings -> Accelerator -> GPU T4 x2.")
for path in ("/kaggle/working", "/tmp"):
    print(f"{path}: {shutil.disk_usage(path).free / 1024**3:.1f} GB free")
if shutil.disk_usage("/tmp").free < 30 * 1024**3:
    raise SystemExit("/tmp needs 30 GB free for the model weights (HF_HOME).")
""")


def clone_cell() -> dict[str, Any]:
    return code("""
shutil.rmtree(REPO, ignore_errors=True)
run("git", "clone", "--quiet", REPO_URL, REPO, cwd="/tmp")
run("git", "checkout", "--quiet", COMMIT)
print(run("git", "log", "-1", "--format=%H %s").strip())
""")


def install_cell() -> dict[str, Any]:
    return code(f'''
# Keeps Kaggle's torch (reinstalling it breaks CUDA), then pins the libraries
# Hermes and Logos v0.0.1 ran with. --no-deps stops pip replacing torch.
run(PY, "scripts/colab_setup.py")
run(PY, "-m", "pip", "install", "-q", "--no-deps", *"{PINS}".split())
''')


def environment_cell() -> dict[str, Any]:
    return code("""
# Weights on /tmp, never /kaggle/working (saved output, 20 GB). No KLEOS_*
# variables: they change config_hash.
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Line-by-line logs from child processes, kept even if one is stopped.
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
stray = [k for k in os.environ if k.startswith("KLEOS_")]
if stray:
    raise SystemExit(f"Unset {stray}: KLEOS_* variables change config_hash.")
run(PY, "-c", "import torch, transformers, peft, bitsandbytes; "
    "print(torch.__version__, torch.cuda.device_count(), transformers.__version__, "
    "peft.__version__, bitsandbytes.__version__)")
""")


def dataset_cell() -> dict[str, Any]:
    return code("""
# The private Kaggle dataset holding releases/kleos-policy-v0.0.7 (Add Input),
# copied to the path the config hash was computed with, in /tmp, which Kaggle
# never saves. Identical copies count once; two different releases stop the run.
RELEASE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "RELEASE.lock")
releases = {}
for path in sorted(glob.glob("/kaggle/input/**/RELEASE.lock", recursive=True)):
    directory = os.path.dirname(path)
    if all(os.path.exists(os.path.join(directory, n)) for n in RELEASE_FILES):
        digest = tuple(sha256(os.path.join(directory, n)) for n in RELEASE_FILES)
        releases.setdefault(digest, directory)
if len(releases) != 1:
    raise SystemExit(f"Attach exactly one kleos-policy-v0.0.7 release; found {sorted(releases.values())}.")
source = next(iter(releases.values()))
shutil.rmtree(DATASET, ignore_errors=True)
copy_writable(source, DATASET)
if sha256(os.path.join(DATASET, "test.jsonl")) != TEST_SHA256:
    raise SystemExit("test.jsonl is not the sealed one (sha256 a4decaaf...).")
print("dataset:", source, "->", DATASET)
""")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_notebook() -> dict[str, Any]:
    cells = [
        markdown(f"""
# KLEOS Logos v0.0.2 — training (H9)

Fine-tunes the text tower of **Ministral 3 14B Reasoning** on
**kleos-policy-v0.0.7** with QLoRA, on Kaggle's free **2 x T4**, exactly as
pre-registered under H9 in `docs/experiments.md`.

**Before you run** (docs/logos.md has the full runbook):

1. Settings → Accelerator → **GPU T4 x2**; Settings → **Internet on**.
2. Add Input → your private dataset holding `kleos-policy-v0.0.7`.
3. Set `COMMIT` in the first code cell to the pushed commit's full sha.
4. **Save Version → Save & Run All (Commit).** The run is unattended and stops
   at the first failure. ESTIMATED 5–6 hours.

**To resume** a run that stopped: add this notebook's previous version **output**
as an input, then Save & Run All again. Its checkpoints are copied back and
training continues from the latest one.

Nothing here publishes anything. The adapter lands in this version's output:
`outputs/{EXPERIMENT_ID}/adapter`.
"""),
        settings_cell(),
        helpers_cell(),
        markdown("## Hardware, code, libraries"),
        hardware_cell(),
        clone_cell(),
        install_cell(),
        environment_cell(),
        markdown("## Data"),
        dataset_cell(),
        markdown("""
## Resume, config hash, plan

A previous version's checkpoints are restored if attached. The config hash must
be the pre-registered one before anything trains.
"""),
        code("""
run_dir = os.path.join(OUTPUTS, EXPERIMENT_ID)
previous = [
    p for p in glob.glob(f"/kaggle/input/**/outputs/{EXPERIMENT_ID}", recursive=True)
    if glob.glob(os.path.join(p, "checkpoint-*"))
]
if len(previous) > 1:
    raise SystemExit(f"Attach at most one previous version; found {previous}.")
if previous and not os.path.exists(run_dir):
    copy_writable(previous[0], run_dir)
# Finding L-F5: a checkpoint missing optimizer or scheduler state would resume with
# a fresh optimizer and a restarted schedule, silently. Set such a checkpoint
# aside, visibly; auto-resume then uses the newest complete one.
REQUIRED = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")
for checkpoint in sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                         key=lambda p: int(p.rsplit("-", 1)[1]), reverse=True):
    missing = [n for n in REQUIRED if not os.path.exists(os.path.join(checkpoint, n))]
    if not missing:
        break
    aside = os.path.join(run_dir, "incomplete-" + os.path.basename(checkpoint))
    os.rename(checkpoint, aside)
    print(f"L-F5: {os.path.basename(checkpoint)} lacks {missing}; set aside as {aside}.")
RESUMING = bool(glob.glob(os.path.join(run_dir, "checkpoint-*")))
print("resuming from:", previous[0] if previous else "-", "| checkpoints:",
      sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))))
"""),
        code("""
printed = run(PY, "-c", "import sys; sys.path.insert(0, 'src'); "
    "from kleos_models.config import load_config; "
    f"print(load_config({CONFIG!r}, dataset_path={DATASET!r}, output_dir={OUTPUTS!r}).config_hash)")
if printed.strip().splitlines()[-1] != CONFIG_HASH:
    raise SystemExit("config_hash differs from the one pre-registered under H9; stop.")
run(PY, "scripts/plan_run.py", "--config", CONFIG, "--seq-length", "736")
"""),
        markdown("""
## Smoke run and gate

Ten steps at the real settings, on scratch space, then the go/no-go gate on the
tighter GPU's measured memory. Skipped when resuming a run that already passed it.
"""),
        code("""
if not RESUMING:
    shutil.rmtree("/tmp/logos-v002-smoke", ignore_errors=True)
    run(PY, "scripts/train.py", "--config", "configs/training/debug_logos_v002.yaml",
        "--dataset", DATASET, "--output-dir", "/tmp/logos-v002-smoke",
        "--experiment-id", "kleos-logos-v002-smoke-001", deadline=True)
    run(PY, "scripts/check_smoke_gate.py",
        "--manifest", "/tmp/logos-v002-smoke/kleos-logos-v002-smoke-001/manifest.json",
        "--expect-modules", "280", "--expect-trainable", "60948480", "--min-spare-gb", "0.15")
"""),
        markdown("""
## Train Logos v0.0.2

The pre-registered run. `--resume-from-checkpoint auto` continues from the latest
checkpoint when one was restored above, and starts fresh otherwise.
"""),
        code("""
run(PY, "scripts/train.py", "--config", CONFIG,
    "--dataset", DATASET, "--output-dir", OUTPUTS,
    "--experiment-id", EXPERIMENT_ID, "--resume-from-checkpoint", "auto", deadline=True)
"""),
        code("""
# What this version's output holds. Training loss is not a result; H9 is decided
# by the evaluation notebook and compare.py.
with open(os.path.join(run_dir, "manifest.json")) as handle:
    manifest = json.load(handle)
print("config_hash:", manifest.get("config_hash"))
metrics = manifest.get("metrics", {})
print("recorded metrics:", sorted(metrics))
for key in ("memory_probe", "peak_memory_gb", "eval_loss"):
    if key in metrics:
        print(f"{key}: {metrics[key]}")
print(sorted(os.listdir(run_dir)))
print(subprocess.check_output(["du", "-sh", run_dir], text=True).strip())
if STOPPED:
    print("NOT FINISHED: stopped at the session deadline. Add this version's output "
          "as an input and Save & Run All again; training resumes from its checkpoints.")
"""),
    ]
    return kaggle_notebook(cells, title="KLEOS Logos v0.0.2 — training (H9)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def build_evaluation_notebook() -> dict[str, Any]:
    cells = [
        markdown(f"""
# KLEOS Logos v0.0.2 — evaluation (H9)

Evaluates the trained adapter (`arm2_finetuned`) on the sealed benchmark (sha256
`a11ffad7…`): greedy decoding, seed 42, `max_new_tokens` 1024. Each answer's
thinking is split off at `[/THINK]` and stored beside it; only the answer is
graded.

**Before you run:**

1. Settings → Accelerator → **GPU T4 x2**; **Internet on**.
2. Add Input → the `kleos-policy-v0.0.7` dataset, and the **training notebook's
   output** (the version that finished).
3. Set `COMMIT` to the same sha the training run used.
4. **Save Version → Save & Run All (Commit).** ESTIMATED 7–8 hours.

**To resume**, also add this notebook's previous version output: its partial file
is copied back and finished generations are replayed, not regenerated.

Then download `outputs/{EXPERIMENT_ID}/arm2_finetuned.json` and run the H9
comparison on your own machine (docs/logos.md, "H9 comparison").
"""),
        settings_cell(),
        helpers_cell(),
        markdown("## Hardware, code, libraries, data"),
        hardware_cell(),
        clone_cell(),
        install_cell(),
        environment_cell(),
        dataset_cell(),
        markdown("## The adapter, the benchmark, any partial run"),
        code("""
adapters = sorted(
    os.path.dirname(p)
    for p in glob.glob(f"/kaggle/input/**/outputs/{EXPERIMENT_ID}/adapter/adapter_config.json",
                       recursive=True)
)
if len(adapters) != 1:
    raise SystemExit(f"Attach exactly one finished training output; found {adapters}.")
ADAPTER = adapters[0]
RESULT = os.path.join(OUTPUTS, EXPERIMENT_ID, "arm2_finetuned.json")
os.makedirs(os.path.dirname(RESULT), exist_ok=True)
partials = glob.glob(f"/kaggle/input/**/outputs/{EXPERIMENT_ID}/arm2_finetuned.json.partial.jsonl",
                     recursive=True)
if len(partials) > 1:
    raise SystemExit(f"Attach at most one previous evaluation version; found {partials}.")
if partials and not os.path.exists(RESULT + ".partial.jsonl"):
    shutil.copyfile(partials[0], RESULT + ".partial.jsonl")
print("adapter:", ADAPTER, "| partial restored:", bool(partials))

run(PY, "scripts/build_benchmark.py", "--dataset", DATASET, "--output", "/tmp/benchmark")
if sha256("/tmp/benchmark/benchmark.jsonl") != BENCHMARK_SHA256:
    raise SystemExit("The benchmark is not the sealed one (sha256 a11ffad7...).")
"""),
        markdown("## Evaluate arm2 (re-running this cell resumes)"),
        code("""
run(PY, "scripts/evaluate.py", "--config", CONFIG, "--arm", "arm2_finetuned",
    "--adapter", ADAPTER, "--benchmark", "/tmp/benchmark/benchmark.jsonl",
    "--output", RESULT, "--resume", deadline=True)
"""),
        code("""
# A first look. H9 is decided only by compare.py against Hermes (docs/logos.md).
if not os.path.exists(RESULT):
    print("NOT FINISHED in this session. Add this version's output as an input and "
          "Save & Run All again: finished generations are replayed, not regenerated.")
else:
    with open(RESULT) as handle:
        result = json.load(handle)
    print("overall:", result.get("overall"))
    print("answerable:", ((result.get("corrected") or {}).get("subsets") or {}).get("answerable"))
    stats = result.get("generation_stats") or {}
    for key in ("responses", "thinking_truncated", "hit_max_new_tokens", "parse_failures",
                "completion_tokens", "reasoning_responses", "reasoning_chars", "latency_seconds"):
        print(f"{key}: {stats.get(key)}")
"""),
    ]
    return kaggle_notebook(cells, title="KLEOS Logos v0.0.2 — evaluation (H9)")


def build_output_probe() -> dict[str, Any]:
    """Two minutes, no GPU: does Kaggle keep a FAILED version's output?

    The resume runbook relies on attaching a stopped or failed version's output.
    Run this once (Save & Run All); it fails on purpose. Then, in any notebook,
    Add Input -> this notebook's output, and look for probe/marker.txt.
    """
    cells = [
        markdown("""
# Kaggle output probe (run once, before Logos v0.0.2)

Writes a marker file, then fails on purpose. Afterwards, in any notebook, add this
notebook's output as an input: if `probe/marker.txt` is there, Kaggle keeps a
failed version's output, and the resume runbook can rely on it (docs/logos.md §11).
"""),
        code("""
import os

os.makedirs("/kaggle/working/probe", exist_ok=True)
with open("/kaggle/working/probe/marker.txt", "w") as handle:
    handle.write("written before the failure")
print("marker written")
"""),
        code("""
raise SystemExit("Deliberate failure: now check that this version's output can be attached.")
"""),
    ]
    return kaggle_notebook(cells, title="Kaggle output probe")


NOTEBOOKS = {
    "logos_v002_train.ipynb": build_training_notebook,
    "logos_v002_evaluate.ipynb": build_evaluation_notebook,
    "output_probe.ipynb": build_output_probe,
}


def render(builder: Any) -> str:
    return json.dumps(builder(), indent=1) + "\n"


def main() -> int:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in NOTEBOOKS.items():
        (KAGGLE_DIR / filename).write_text(render(builder), encoding="utf-8")
        print(f"  wrote notebooks/kaggle/{filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
