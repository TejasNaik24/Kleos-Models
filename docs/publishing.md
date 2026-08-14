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
| tokenizer files | `checkpoint-*/` |
| `config.yaml` (effective config) | `*.log`, `events.jsonl` |
| `manifest.json` | key material |
| `metrics.json` | anything failing the content scan |
| generated `README.md` model card | |

Every eligible text file is additionally scanned for secrets before upload. A file
that matches is refused with the reason printed.

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

## Licences

The adapter derives from its base model and is generally subject to that model's
licence terms. Check them before redistributing. The KLEOS code is MIT.

The generated card records the base model and sets `license: other`, so a reader
knows to check.

## Never publish

- Raw or sanitized training data
- `.env` or any credential
- A private dataset, or a manifest flagged `contains_private_data: true`
- An adapter you cannot trace to a manifest — provenance is the point

## After publishing

Check the rendered card on the model page, confirm no private data is present, and
confirm the results section matches what your evaluation actually showed.
