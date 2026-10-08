# Publishing adapters

Two scripts move an adapter off the machine that trained it, for two different
audiences:

| Script | Destination | Contents | Used for |
| --- | --- | --- | --- |
| `scripts/publish_adapter.py` | a Hugging Face model repository, public unless `--private` | adapter weights and config, effective config, manifest, metrics, a generated model card; no tokenizer | releasing an adapter for others to load |
| `scripts/upload_deployment_package.py` | a private Hugging Face repository; refuses a public one | a verified deployment package: adapter, frozen tokenizer files, packaged model config, manifest | the ZeroGPU Space that serves the model |

The KLEOS adapters (Hermes v0.0.6, Logos v0.0.2) are currently distributed only
the second way: each Space downloads its package from a private repository at a
pinned commit. None has been published publicly. [deployment.md](deployment.md)
covers packages; the rest of this page covers `publish_adapter.py`.

## Publishing is never automatic

Nothing is uploaded as a side effect of training, evaluation or running a
notebook. Publishing requires this explicit command:

```bash
python scripts/publish_adapter.py \
    --adapter outputs/<experiment-id>/adapter \
    --repo-id <owner>/kleos-<model>-adapter
```

## Dry-run first

```bash
python scripts/publish_adapter.py --adapter <path> --repo-id <owner>/kleos-<model>-adapter --dry-run
```

It prints which files would be uploaded, which were refused and why, and writes
the generated model card for review. Nothing leaves the machine.

## What gets uploaded

An allowlist (`ALLOWED_UPLOAD_NAMES` in `src/kleos_models/publishing.py`), not a
blocklist: anything not named is refused, so a new artifact type cannot leak by
default.

| Uploaded | Refused |
| --- | --- |
| `adapter_config.json` | any `*.jsonl` (datasets) |
| `adapter_model.safetensors` or `adapter_model.bin` | `.env` and `.env.*` |
| `config.yaml` (effective config) | `checkpoint-*` |
| `manifest.json` | `*.log`, `events.jsonl` |
| `metrics.json` | key material (`*.key`, `*.pem`) |
| generated `README.md` model card | every tokenizer file (below) |
| | anything else not on the allowlist |
| | any text file failing the content scan |

Every eligible text file is scanned for secrets and personal data before upload,
and a match is refused with the reason printed. Text files larger than 5 MB are
refused, not skipped.

## Why no tokenizer is published

The adapter repository holds adapter artifacts; the tokenizer comes from the
pinned base model. This is all-or-nothing.

A PEFT adapter is not self-contained: loading it requires the base model, so the
base repository is always a dependency and its tokenizer is always at hand. KLEOS
never adds tokens or resizes embeddings (LoRA excludes `embed_tokens` and
`lm_head`, visible in any run's `adapter_config.json`), so the tokenizer is
byte-identical to the base model's.

The all-or-nothing rule matters more than the size. `tokenizer.json` for
Ministral-8B is about 17 MB, above the 5 MB scan limit, while
`tokenizer_config.json` is a few hundred bytes. With the size limit as the only
filter, an upload would have admitted the config and refused the vocabulary,
producing a repository that looks like it carries a tokenizer and fails in
`AutoTokenizer.from_pretrained` (audit finding F2, in the
[Ministral-8B artifact audit](experiments/kleos-v006-ministral8b-run1-artifact-audit.md#findings)).
A partial tokenizer bundle is worse than none.

Two layers enforce the rule: the run's `tokenizer/` directory is never enumerated,
and every tokenizer filename (`TOKENIZER_ARTIFACTS`: `tokenizer.json`,
`tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json`,
`chat_template.jinja` and others) is refused by name even if it appears in the
adapter directory. `tests/test_publishing.py` pins both. The scan limit was not
raised to admit the tokenizer; it still applies to every other text file.

A deployment package is different: it carries the frozen tokenizer files with
their hashes, because the serving runtime must tokenize exactly as training did.
The audit record is indexed in [experiments/README.md](experiments/README.md).

## Authentication

```bash
export HF_TOKEN=<token with write scope>
```

On Colab or Kaggle, use the platform's secrets store. A token never appears in a
config file, a notebook cell or a commit.

## The model card

Generated from the run manifest: model name, base model and revision, method,
LoRA configuration, hyperparameters, training hardware, dataset description and
privacy statement, tasks, evaluation method, results, limitations, intended use,
prohibited uses, license, and the reproducibility chain.

Preview it without uploading:

```bash
python scripts/publish_adapter.py --adapter <path> --repo-id <owner>/kleos-<model>-adapter --card-only
```

### It does not claim the model is better

With no evaluation attached, the card says so:

> **No evaluation results were attached to this release.** Nothing is therefore
> claimed about this adapter's performance.

Attach results to include them:

```bash
python scripts/publish_adapter.py \
    --adapter outputs/<experiment-id>/adapter \
    --repo-id <owner>/kleos-<model>-adapter \
    --results <private storage>/outputs/<experiment-id>/arm2_finetuned.json \
    --comparison reports/exp-001/metrics.json
```

With a comparison attached, the card reports per-task differences including
regressions, and states when general capability dropped. If OOD was not measured,
it says no generalization claim is made. A published adapter is not evidence that
fine-tuning helped.

### The load snippet

The card's usage snippet loads the base with `AutoModelForCausalLM` at the pinned
revision and attaches the adapter with `PeftModel`. That fits a plain causal LM
base (Mistral-Nemo, Ministral-8B). A Ministral 3 base is a vision-language
container that KLEOS loads through a text-only view (`Ministral3TextAdapter`), with
`fix_mistral_regex: true`; the generic snippet does not do either, and has not
been tested on a Logos adapter.

## Exporting without publishing

```bash
python scripts/export_adapter.py --run outputs/<experiment-id> --output exports/<name>
```

Same safety checks, written to a local directory, for review before publishing or
for sharing through another channel.

## Base-model revision

An adapter is a set of deltas against specific base weights. Serving or reloading
it against a different revision pairs it with weights it was never trained on,
and nothing raises an error; the judgment degrades. Each base KLEOS has trained
on is now pinned to a commit. `tests/test_revision_pinning.py` asserts the pins and
that no config copied another's; it and `tests/test_logos_deployment_record.py`
assert that each training config and its serving record agree:

| Base | Revision | Verified | Pinned in |
| --- | --- | --- | --- |
| `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` | 2026-09-15 | `configs/models/ministral_8b.yaml`, `configs/deployment/kleos_v006_ministral8b.yaml` |
| `mistralai/Mistral-Nemo-Instruct-2407` (Hermes) | `04d8a90549d23fc6bd7f642064003592df51e9b3` | 2026-09-15 | `configs/models/mistral_nemo_12b.yaml`, `configs/deployment/kleos_hermes_v006.yaml` |
| `mistralai/Ministral-3-14B-Instruct-2512-BF16` (Logos v0.0.1) | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | 2026-09-24 | `configs/models/ministral3_14b.yaml` |
| `mistralai/Ministral-3-14B-Reasoning-2512` (Logos v0.0.2) | `51f9210f3cd20f3452a80d5819d15dc61cc50630` | 2026-10-06 | `configs/models/ministral3_14b_reasoning.yaml`, `configs/deployment/kleos_logos_v002.yaml` |

`mistral_small_3_2.yaml` stays on `revision: main`; it has never been trained.
For the Ministral 3 bases the test also records the sha256 of the tokenizer files
and the chat template at the pinned revision.

The generated card emits the revision in its load snippet:

```python
BASE = "mistralai/Mistral-Nemo-Instruct-2407"
REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)
model = PeftModel.from_pretrained(base, "<owner>/kleos-<model>-adapter")
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
```

If a run used a moving pointer, the card says so rather than implying a
reproducibility it cannot offer. The Ministral-8B run is in that category: it was
trained with `revision: main`, and the commit it resolved to is not recoverable
(deviation D7, [experiments.md](experiments.md#base-model-revision)).
`--serving-revision` states the revision consumers should load in that case,
reported beside the training revision.

To resolve a new pin:

```python
from huggingface_hub import model_info

model_info("mistralai/Mistral-Nemo-Instruct-2407").sha
```

A sha identifies one checkpoint. Never copy one config's pin into another.

## Licenses

The KLEOS code is MIT. An adapter derives from its base model and is generally
subject to that model's license; the generated card sets `license: other` and
names the base, so a reader knows to check.

| Base | License | Notes |
| --- | --- | --- |
| Mistral-Nemo-Instruct-2407 | Apache-2.0 | ungated; attribution and a statement of modification are required when redistributing a derivative |
| Ministral 3 14B Instruct and Reasoning | Apache-2.0 | ungated; same attribution requirement |
| Ministral-8B-Instruct-2410 | Mistral AI Research License | gated; non-commercial. Commercial use, including offering KLEOS as a paid product or service, requires a separate agreement with Mistral AI |
| Mistral Small 3.2 24B | Apache-2.0 | config only; never trained |

License terms attach to a revision, so pinning also fixes the terms that were
reviewed.

## Never publish

- Raw or sanitized training data.
- `.env` or any credential.
- A private dataset, or a manifest flagged `contains_private_data: true`.
- An adapter that cannot be traced to a manifest.

## After publishing

Check the rendered card on the model page, confirm that no private data is
present, and confirm that the results section matches what the evaluation showed.
