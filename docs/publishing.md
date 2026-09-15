# Publishing adapters

## Publishing is never automatic

Nothing is uploaded as a side effect of training, evaluation, or running a
notebook. Publishing requires this explicit command:

```bash
python scripts/publish_adapter.py \
    --adapter outputs/<experiment-id>/adapter \
    --repo-id YOUR_USERNAME/kleos-qwen3-8b
```

## Always dry-run first

```bash
python scripts/publish_adapter.py --adapter <path> --repo-id <you>/<name> --dry-run
```

Prints exactly which files would be uploaded, which were refused and why, and
writes the generated model card for review. Nothing leaves the machine.

## What gets uploaded

An **allowlist**, not a blocklist — anything not named here is refused, so a new
artifact type cannot leak by default.

| Uploaded | Refused |
| --- | --- |
| `adapter_config.json` | any `*.jsonl` (datasets) |
| `adapter_model.safetensors` | `.env` and `.env.*` |
| `config.yaml` (effective config) | `checkpoint-*/` |
| `manifest.json` | `*.log`, `events.jsonl` |
| `metrics.json` | key material |
| generated `README.md` model card | **all tokenizer files** (see below) |
| | anything failing the content scan |

Every eligible text file is additionally scanned for secrets before upload. A file
that matches is refused with the reason printed.

## Why no tokenizer is published

**The adapter repository holds adapter artifacts. The tokenizer comes from the
pinned base model.** This is deliberate, and it is all-or-nothing.

A PEFT adapter is not self-contained — loading it requires the base model, so the
base repository is *always* a dependency and its tokenizer is always at hand.
KLEOS never adds tokens and never resizes embeddings (LoRA excludes
`embed_tokens` and `lm_head`, visible in any run's `adapter_config.json`), so the
tokenizer is byte-identical to the base model's. Shipping ~17MB of identical
vocabulary alongside every adapter buys nothing.

The all-or-nothing rule matters more than the size saving. `tokenizer.json` for
Ministral-8B is ~17MB — above the 5MB content-scan cap — while
`tokenizer_config.json` is a few hundred bytes. Admitting files individually
produced a repository carrying tokenizer *config* with no vocabulary source:
`AutoTokenizer.from_pretrained` on it fails. **A partial tokenizer bundle is
worse than none**, because it looks complete.

Two layers enforce this: the run's `tokenizer/` directory is never enumerated,
and every tokenizer filename is refused by name even if one appears in the
adapter directory. `tests/test_publishing.py` pins both.

The scan cap was **not** raised to accommodate this. `_MAX_SCAN_BYTES` remains
5MB and still applies to `manifest.json`, `config.yaml` and every other scanned
text file — widening a general control to admit one known file is the wrong
trade.

## Authentication

```bash
export HF_TOKEN=hf_...     # write scope
```

On Colab, use Colab Secrets. A token must never appear in a config file, a
notebook cell, or a commit.

## The model card

Generated from the run manifest, covering: model name, base model and revision,
fine-tuning method, LoRA configuration, hyperparameters, training hardware,
dataset description and **privacy statement**, tasks, evaluation methodology,
results, limitations, intended use, prohibited uses, licence, and the full
reproducibility chain.

Preview it without uploading:

```bash
python scripts/publish_adapter.py --adapter <path> --repo-id <you>/<name> --card-only
```

### It will not claim the model is better

With no evaluation attached, the card says so plainly:

> **No evaluation results were attached to this release.** Nothing is therefore
> claimed about this adapter's performance.

Attach results to include them:

```bash
python scripts/publish_adapter.py \
    --adapter outputs/<id>/adapter \
    --repo-id <you>/<name> \
    --results outputs/finetuned_results.json \
    --comparison reports/exp-001/metrics.json
```

With a comparison attached, the card reports per-task deltas including
regressions, and states plainly when general capability dropped. If OOD was not
measured, it says no generalization claim is made.

This is deliberate: a published adapter is not evidence that fine-tuning helped.

## Exporting without publishing

```bash
python scripts/export_adapter.py --run outputs/<experiment-id> --output exports/my-adapter
```

Same safety checks, writes to a local directory. Useful for review before
publishing, or for sharing through another channel.

## Base-model revision

An adapter is deltas against **specific** base weights. Serving or reloading it
against a different revision pairs it with weights it was never trained on, and
nothing raises an error — the judgment just degrades.

KLEOS pins the base revision in two places, and
`tests/test_revision_pinning.py` asserts they stay equal:

| File | Purpose |
| --- | --- |
| `configs/models/ministral_8b.yaml` | training and evaluation |
| `configs/deployment/kleos_v006_ministral8b.yaml` | serving |

Current pin: `2f494a194c5b980dfb9772cb92d26cbb671fce5a` (verified 2026-09-15).

The generated card emits the revision in its load snippet:

```python
BASE = "mistralai/Ministral-8B-Instruct-2410"
REVISION = "2f494a194c5b980dfb9772cb92d26cbb671fce5a"
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)
model = PeftModel.from_pretrained(base, "<repo-id>")
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
```

If a run used a moving pointer, the card says so outright rather than implying
reproducibility it cannot offer. The v0.0.6 adapter is in that category — it was
trained with `revision: main` and its base commit is not recoverable. See
`docs/experiments.md`.

To resolve a new pin:

```python
from huggingface_hub import model_info
model_info("mistralai/Ministral-8B-Instruct-2410", token=...).sha
```

A sha identifies one checkpoint. Never copy one config's pin into another.

## Licences

The adapter derives from its base model and is generally subject to that model's
licence terms. Check them before redistributing. The KLEOS code is MIT.

The generated card records the base model and sets `license: other`, so a reader
knows to check.

**Ministral-8B is under the Mistral Research Licence, which is non-commercial.**
Commercial use — including offering KLEOS as a paid product or service — requires
a separate agreement with Mistral. Licence terms attach to a revision, so pinning
also fixes the terms that were accepted. The repository is gated: access is
granted per Hugging Face account and needs `HF_TOKEN` at load time.

## Never publish

- Raw or sanitized training data
- `.env` or any credential
- A private dataset, or a manifest flagged `contains_private_data: true`
- An adapter you cannot trace to a manifest — provenance is the point

## After publishing

Check the rendered card on the model page, confirm no private data is present, and
confirm the results section matches what your evaluation actually showed.
