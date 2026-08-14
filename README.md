# KLEOS Models

Research engineering for KLEOS behavioural fine-tuning: dataset contracts, QLoRA
training, and a controlled evaluation harness for Qwen and Mistral open-weight
models.

> **Research status:** infrastructure phase. No KLEOS fine-tuning result exists
> yet. The examples shipped here are development fixtures, not the research
> dataset, and this repository makes **no claim** that fine-tuning improves
> anything. Determining that is the experiment.

---

## 1. What this is

This repository is the public model/research engineering repository for KLEOS, an
AI operating system for computer science students. It contains everything needed
to prepare data, train LoRA/QLoRA adapters, and evaluate them rigorously — and
nothing that touches the KLEOS application or its production data.

## 2. The research question

> Per subtask, can a behaviourally fine-tuned open-weight model outperform a
> prompt-engineered orchestration baseline on judgment and correctness metrics?

This is an empirical question with a real possibility of a negative answer. The
codebase is built so that "fine-tuning did not help" is a result it can report
cleanly, not a failure it hides.

The model is meant to learn **decision policies** — prioritization, evidence-aware
recommendation, relevance judgment, tool routing, briefing behaviour. It is
explicitly _not_ meant to memorize any user's facts. Private facts belong in
retrieval and memory, never in weights.

## 3. Public/private data boundary

**This repository is public. It must never contain real user data.**

| Lives here                     | Lives in the private `kleos-training-data` repo |
| ------------------------------ | ----------------------------------------------- |
| Code, schemas, configs         | Raw conversations and memory records            |
| Synthetic development fixtures | Sanitized real examples                         |
| Dataset manifests and loaders  | Supabase exports                                |
| Aggregate metrics, model cards | Private evaluation traces                       |

The private repository produces a versioned artifact:

```
dataset/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

This repository consumes it by path and never needs to know where it came from:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset /path/to/private/dataset
```

Safeguards: `.gitignore` blocks data and output directories, and
`scripts/check_no_private_data.py` scans for emails, tokens, keys and Supabase
URLs. It runs in CI and can be installed as a pre-commit hook. See
[SECURITY.md](SECURITY.md) and [docs/privacy.md](docs/privacy.md).

## 4. Repository structure

```
configs/          layered YAML: base, models, datasets, training, evaluation
data/
  schema/         JSON Schema for the data contract
  examples/       synthetic development fixtures (NOT research data)
docs/             architecture, data contract, training, evaluation, colab, privacy…
notebooks/        Colab workflow (the canonical training entry point)
scripts/          CLI entry points
src/kleos_models/
  config.py       typed configuration
  compat.py       transformers 4.56 ↔ 5.x compatibility
  data/           schemas, loading, validation, formatting, splitting, leakage
  models/         family adapters, loading, quantization, PEFT, feasibility
  training/       QLoRA pipeline, callbacks, checkpointing, memory
  evaluation/     metrics, graders, consistency, OOD, faithfulness, reports
  inference/      generation backends, orchestration scaffolding
  experiments/    manifests, registry, environment capture
tests/            real tests, no GPU required
```

### The one architectural rule worth knowing

`config`, `data`, `experiments` and the scoring half of `evaluation` **never
import torch**. Dataset work, validation, leakage checking, re-scoring and CI all
run on any laptop in seconds. Only `models/`, `training/` and `inference/` need
the `[train]` extra. `tests/test_import_isolation.py` enforces this.

## 5. Installation

Python 3.11+.

```bash
# Data, validation, evaluation-scoring work. No torch, installs in seconds.
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Add model loading and training (CPU or CUDA).
pip install -e ".[train,dev]"

# Add 4-bit quantization. CUDA only — bitsandbytes has no macOS build.
pip install -e ".[train,quant,dev]"
```

On Colab, use `python scripts/colab_setup.py`, which installs the right packages
**without reinstalling torch** (reinstalling torch on Colab breaks CUDA).

## 6. Environment setup

Every secret is optional; nothing in the data or evaluation pipeline needs one.

```bash
cp .env.example .env   # then fill in what you need
```

`HF_TOKEN` is required only to download gated base models or to publish an
adapter. On Colab, use Colab Secrets rather than a `.env` file — see
[docs/colab.md](docs/colab.md).

## 7. Dataset format

JSONL, one example per line, validated against a versioned schema:

```json
{
  "id": "example-000001",
  "version": "1.0",
  "task": "notification_prioritization",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "variation_axes": {
    "domain": "career",
    "urgency": "high",
    "entities": "unseen"
  },
  "metadata": {
    "source": "synthetic",
    "quality_status": "reviewed",
    "scenario_family": "deadline-conflict-01"
  }
}
```

`variation_axes` is not decoration. It is what lets the pipeline answer _how many
examples cover each situation type_ rather than mistaking a large dataset for a
diverse one. Full contract: [docs/data-contract.md](docs/data-contract.md).

## 8. Running validation

```bash
python scripts/validate_dataset.py --dataset data/examples
python scripts/validate_dataset.py --dataset data/examples --leakage-report reports/leakage
python scripts/inspect_dataset.py  --dataset data/examples --coverage
```

Validation covers schema, duplicate ids, coverage gaps, placeholder text,
sensitive-content patterns, and examples that teach a private fact rather than a
policy. Leakage detection covers exact, normalized and near-duplicates
(MinHash/LSH), id collisions, scenario repeats and entity leakage.

## 9. Running a smoke test

```bash
python scripts/smoke_test.py
```

Checks Python version, imports, configs, schemas, the data pipeline and the
evaluation harness. Add `--model configs/models/qwen3_8b.yaml` to also check
tokenizer loading and PEFT setup when the `[train]` extra is installed. Run it
before spending a GPU session.

## 10. Running training

Colab is the canonical training environment — open
[`notebooks/02_train_qlora.ipynb`](notebooks/02_train_qlora.ipynb).

Locally, on a CUDA machine:

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml   # will it fit?
python scripts/train.py    --config configs/training/qlora_small.yaml
```

Before training starts you get the GPU, VRAM, compute capability, CUDA and library
versions, the model and quantization settings, the LoRA configuration, and a
memory estimate. If the run cannot fit, it says so with numbers instead of
OOM-ing an hour later.

## 11. Resuming training

Colab runtimes die. Resume is a first-class path:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --resume-from-checkpoint auto
```

`auto` finds the newest **valid** checkpoint — partial directories from an
interrupted save are detected and skipped rather than causing a confusing failure.
Checkpoint retention never deletes the last remaining checkpoint.

## 12. Evaluation

```bash
python scripts/evaluate.py --config configs/evaluation/default.yaml \
                           --arm arm0_base --output outputs/base_results.json

python scripts/evaluate.py --config configs/evaluation/default.yaml \
                           --arm arm2_finetuned \
                           --adapter outputs/<experiment-id>/adapter \
                           --output outputs/finetuned_results.json
```

Measures judgment, not tone: classification, ranking (nDCG, Kendall τ), set F1,
rubric scores, faithfulness, consistency under logically irrelevant perturbations,
and OOD performance reported separately from in-distribution.

## 13. Comparing base vs fine-tuned

```bash
python scripts/compare.py --base outputs/base_results.json \
                          --finetuned outputs/finetuned_results.json \
                          --report reports/experiment-001
```

Per-task base, fine-tuned, absolute delta, relative delta, OOD delta and
consistency delta — no single blended score unless you pass `--show-aggregate`.
Deltas are paired per example with bootstrap confidence intervals, and an
improvement that is not significant is reported as _not significant_ rather than
as a win.

## 14. Hugging Face publishing

Nothing is ever published automatically.

```bash
python scripts/publish_adapter.py --adapter outputs/<experiment-id>/adapter \
                                  --repo-id YOUR_USERNAME/kleos-qwen3-8b
```

Publishes adapter weights and a generated model card. It refuses to upload raw
training data, `.env`, or any file matching the private-data scanner, and the
model card will not claim the model is better unless the evaluation shows it.

## 15. Reproducibility

Every run writes a manifest tying the artifact to model + revision + dataset
version + dataset hash + config hash + seed + git commit + environment. Failed
runs are recorded too (`status: "failed"`), because hiding them is how a research
record stops being trustworthy.

```
outputs/<experiment-id>/
  adapter/  tokenizer/  config.yaml  manifest.json  metrics.json
  events.jsonl  environment.txt  README.md  checkpoint-*/
```

Automatic configuration adjustments — a reduced sequence length, an optimizer
fallback — are recorded in `manifest.adjustments[]`, never applied silently. Set
`training.strict_config: true` to make any adjustment a hard error instead.

## 16. Privacy

See [docs/privacy.md](docs/privacy.md) and [SECURITY.md](SECURITY.md). The short
version: no real user data, no credentials, no production database access, and
logging that records digests and counts rather than example text.

## 17. Current limitations

Stated plainly, because the alternative is misleading:

- **No GPU training has been executed in this repository yet.** The QLoRA path is
  implemented for CUDA and is launched from Colab. Verification so far covers the
  data, config, evaluation and training-setup layers, plus a real LoRA training
  step on a tiny randomly-initialized model on CPU.
- **The bundled dataset is synthetic development fixtures.** It exists to exercise
  the pipeline. Any number computed from it describes the plumbing, not KLEOS.
- **Hyperparameters are engineering defaults, not tuned values.**
- **The heuristic rubric grader is coarse.** It checks structure — required points,
  citations, unsupported-claim phrasings — and is not a substitute for human or
  LLM judging on open-ended quality.
- **Frontier reference arms are interface-only** (spec §20).
- **Model sizes constrain where they can run.** On a free 16GB T4, Qwen3-8B and
  Ministral-8B are trainable in 4-bit; Mistral-Small-24B and Qwen3-30B-A3B are
  not. `scripts/plan_run.py` tells you before you waste a session.

## 18. Supported models

| Config                   | Checkpoint                                      | Notes                          |
| ------------------------ | ----------------------------------------------- | ------------------------------ |
| `qwen3_8b`               | `Qwen/Qwen3-8B`                                 | Dense, switchable thinking     |
| `qwen3_30b_a3b_thinking` | `Qwen/Qwen3-30B-A3B-Thinking-2507`              | MoE, ~3B active, thinking-only |
| `mistral_small_3_2`      | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | **Vision-language model**      |
| `ministral_8b`           | `mistralai/Ministral-8B-Instruct-2410`          | Scale-matched to Qwen3-8B      |

Two facts worth knowing before you write a config:

**Mistral Small 3.2 cannot be loaded with `AutoModelForCausalLM.`** It declares
`Mistral3ForConditionalGeneration` (`model_type: mistral3`) and transformers
registers it only for image-text-to-text. The family adapter loads it with
`AutoModelForImageTextToText` and scopes LoRA to `language_model.*`, keeping the
vision tower frozen and unquantized.

**Qwen3-30B-A3B is a mixture of experts** — 48 layers × 128 experts. LoRA targets
attention only; adapting the experts would create ~18,000 adapter modules. It is
also thinking-only, so requesting non-thinking mode raises an error rather than
silently producing a broken prompt.

Inspect any model's real architecture:

```bash
python scripts/inspect_model.py --model Qwen/Qwen3-8B
```

## 19. Documentation

| Document                                           | Contents                                 |
| -------------------------------------------------- | ---------------------------------------- |
| [docs/architecture.md](docs/architecture.md)       | How the layers fit together              |
| [docs/data-contract.md](docs/data-contract.md)     | Schema, variation axes, policy-not-facts |
| [docs/training.md](docs/training.md)               | QLoRA pipeline, memory, checkpointing    |
| [docs/evaluation.md](docs/evaluation.md)           | Metrics, graders, consistency, OOD       |
| [docs/experiments.md](docs/experiments.md)         | **Pre-registered hypotheses**            |
| [docs/colab.md](docs/colab.md)                     | Step-by-step Colab procedure             |
| [docs/privacy.md](docs/privacy.md)                 | Data boundary and safeguards             |
| [docs/publishing.md](docs/publishing.md)           | Hugging Face publishing                  |
| [docs/troubleshooting.md](docs/troubleshooting.md) | OOM, gated repos, version issues         |

## 20. Research integrity

Committed to in code, not just in prose:

- Evaluation examples are never cherry-picked; per-example records are always kept.
- Failed runs stay in the registry.
- OOD, consistency and capability results are reported separately, never blended.
- No superiority claim without a significance test behind it.
- Synthetic development data is labelled as such everywhere it appears.

If fine-tuning wins, quantify the win. If it loses, analyse why. If it improves one
task and harms another, that is likely the most interesting result available.

## 21. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `make check` before opening a PR.

## 22. Licence

MIT for the code — see [LICENSE](LICENSE). Base models carry their own licences,
which you must accept on their Hugging Face pages; adapters are generally subject
to the base model's terms.
