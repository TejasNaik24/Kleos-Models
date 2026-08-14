# Architecture

## What this repository is for

KLEOS Models exists to answer one question:

> Per subtask, can a behaviourally fine-tuned open-weight model outperform a
> prompt-engineered orchestration baseline on judgment and correctness metrics?

Every design decision below follows from that being an *empirical* question with a
real chance of a negative answer. The system is built to produce a trustworthy
answer, not a favourable one.

## Layers

```
                    ┌─────────────────────────────────────────┐
   configs/*.yaml ──▶  config.py — typed, composable, hashed   │
                    └────────────────────┬────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌───────────────┐              ┌──────────────────┐            ┌──────────────────┐
│ data/         │              │ models/          │            │ evaluation/      │
│ schemas       │              │ family adapters  │            │ metrics, graders │
│ loaders       │              │ loading          │            │ consistency, OOD │
│ validation    │              │ quantization     │            │ faithfulness     │
│ splitting     │              │ PEFT             │            │ reports          │
│ leakage       │              │ feasibility      │            │                  │
│ formatting    │              └────────┬─────────┘            └────────┬─────────┘
└───────┬───────┘                       │                               │
        │                               ▼                               │
        │                      ┌──────────────────┐                     │
        └─────────────────────▶│ training/        │                     │
                               │ QLoRA pipeline   │                     │
                               │ callbacks        │                     │
                               │ checkpointing    │                     │
                               └────────┬─────────┘                     │
                                        │                               │
                                        ▼                               ▼
                               ┌─────────────────────────────────────────────┐
                               │ experiments/ — manifests, registry          │
                               └─────────────────────────────────────────────┘
```

## The dependency-isolation rule

**`config`, `data`, `experiments` and the scoring half of `evaluation` never import
torch or transformers at module scope.**

This is the single most load-bearing structural decision in the repository.

Why it matters:

- Dataset preparation, validation, leakage checking and re-scoring run on any
  laptop in seconds, with a dependency install measured in seconds rather than
  gigabytes.
- CI runs the full logic suite without a GPU image.
- A Colab notebook can print a feasibility table for every model **before**
  installing or downloading anything heavy.
- Re-analysing saved results months later does not require reconstructing a
  training environment.

`models/`, `training/` and `inference/` do need torch — but they import it *inside
functions*, so even those modules are importable for introspection.

Enforced by `tests/test_import_isolation.py`, which installs a `sys.meta_path`
blocker and imports every module with torch made unavailable. Breaking the rule
is a test failure, not a code-review opinion.

## Configuration

Layered YAML with `extends` (defaults) and `includes` (section fragments), so one
training config consumes an unmodified model config:

```yaml
extends: ../base.yaml
includes:
  model: ../models/qwen3_8b.yaml
  dataset: ../datasets/example.yaml
```

Resolution order: `extends` → `includes` → own keys → `--set` overrides. Unknown
keys are rejected, so a typo fails loudly instead of silently doing nothing.

`ExperimentConfig.config_hash` is a sha256 over the fully resolved config. Two
runs with the same hash used the same knobs — which makes "same config, different
seed" a checkable claim rather than an assertion.

## The model-family adapter layer

Everything that differs between Qwen and Mistral lives behind
`ModelFamilyAdapter`. The training and evaluation pipelines never branch on model
family, so a Qwen-vs-Mistral comparison measures the model rather than the harness.

Adding a family requires a config plus an adapter subclass. **The training
pipeline does not change.**

| Concern | Why it must be per-family |
| --- | --- |
| Auto class | `AutoModelForCausalLM` vs `AutoModelForImageTextToText` |
| LoRA targets + scope prefix | `language_model.*` for `mistral3` |
| Exclusions | vision tower, MoE experts, MoE router |
| Reasoning capability | unsupported / switchable / always-on |
| Chat template kwargs | `enable_thinking` exists only on some Qwen models |
| Quantization exceptions | never quantize a router or an unused vision tower |

### Two facts this layer exists to encode

**Mistral Small 3.2 is a vision-language model.** Its config declares
`Mistral3ForConditionalGeneration` / `model_type: mistral3`, and transformers
registers `mistral3` **only** in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.
`AutoModelForCausalLM.from_pretrained` on that checkpoint fails. LoRA is scoped to
`language_model.*`, and the vision tower stays frozen and unquantized because
KLEOS data is text-only.

**Qwen3-30B-A3B is MoE and thinking-only.** 48 layers × 128 experts: targeting the
experts would create roughly 18,400 adapter modules, and each expert would see a
fraction of the tokens. Attention-only targeting is the correct default, and the
router is excluded from both LoRA and quantization. The chat template does not
accept `enable_thinking`, so requesting non-thinking mode raises.

## Reasoning as a capability

Reasoning is modelled as a declared property of a checkpoint
(`ReasoningCapability`), validated against what a run requests. Asking a
thinking-only model for non-thinking mode is an error with an explanation, not a
silently mangled prompt.

We never inject `<think>` strings to simulate reasoning — that would measure
formatting compliance, not reasoning. Reasoning spans are stripped from training
targets and from multi-turn history, so the model is never trained to emit display
chain-of-thought.

## Feasibility planning

`models/feasibility.py` estimates peak VRAM from model shape, quantization, LoRA
rank, optimizer and activation size — pure arithmetic, no GPU required. It emits a
tier:

`infeasible` → `inference_only` → `smoke` → `adapter_train` → `full_research`

Adjustments are **proposed and recorded**, never silently applied.
`training.strict_config: true` turns any required adjustment into a hard error, so
a research comparison cannot be invalidated by a memory-driven fallback nobody
noticed.

## Training

Standard Hugging Face `Trainer` plus PEFT — the canonical, readable path. Not trl:
its 1.x API churn would add a moving dependency to the most important code in the
repository for no gain here.

Three silent-failure modes are actively guarded:

1. **Targets that match nothing** → validated against the loaded model's real
   `named_modules()`, with candidates listed on failure.
2. **An adapter that trains nothing** → a forward/backward check runs before the
   real loop and asserts LoRA parameters receive non-zero gradients.
3. **A quiet full fine-tune** → refused unless `allow_full_finetune` is set.

Assistant-only loss masking is computed by incremental chat-template application:
render up to a turn with a generation prompt, render including it, and take the
token range between. Family-agnostic, because it asks the template where the
boundary is instead of guessing from a delimiter.

## Evaluation

Arms are the experimental conditions (spec §20). The runner takes the arm as a
parameter and changes nothing else — same benchmark, decoding, graders and seeds —
so `arm0_base` vs `arm2_finetuned` differs only by the adapter.

Per-example records are always retained, so any aggregate can be recomputed and
checked. In-distribution, OOD, consistency and capability results are reported
**separately**; there is no blended score unless one is explicitly requested.

## Experiments

Every run writes a manifest tying the artifact to model + revision + dataset
version + dataset hash + config hash + seed + git commit + environment. Failed runs
write one too, with `status: "failed"`.

The registry lists failures alongside successes. A registry that hid them would
make reporting a favourable subset effortless and invisible.

## Boundaries

- **No Supabase.** Training consumes exported, versioned dataset artifacts. The
  research pipeline never touches production infrastructure.
- **No dependency on the private repo.** The private repository produces a
  directory; this one consumes a path.
- **No automatic publishing.** Uploading requires an explicit command and passes
  an allowlist plus a content scan.

## Further reading

| Document | Contents |
| --- | --- |
| [data-contract.md](data-contract.md) | Schema, variation axes, policy-not-facts |
| [training.md](training.md) | QLoRA pipeline, memory, checkpointing |
| [evaluation.md](evaluation.md) | Metrics, graders, consistency, OOD |
| [experiments.md](experiments.md) | Pre-registered hypotheses |
| [colab.md](colab.md) | The canonical training workflow |
| [privacy.md](privacy.md) | Public/private boundary |
| [troubleshooting.md](troubleshooting.md) | OOM, gated repos, version issues |
