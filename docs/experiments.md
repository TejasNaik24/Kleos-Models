# Experiments and pre-registration

This document holds the **pre-registered hypotheses** for the KLEOS fine-tuning
research (spec §37).

## Why pre-register

Deciding what counts as success *after* seeing results is how well-intentioned
research produces unreliable conclusions. Writing the hypotheses, metrics and
decision rules down first means a negative result stays a result rather than
becoming a prompt to go looking for a different metric.

**Rules of engagement:**

1. Hypotheses, primary metrics and decision thresholds are fixed **before** the
   run that tests them.
2. Changing a metric after seeing results is permitted only as an explicitly
   labelled exploratory analysis, never as the headline.
3. Failed and negative runs stay in the registry.
4. Evaluation code must not be tuned against observed results.

---

## Status

**One run completed: `kleos-v006-ministral8b-run1` (2026-09-15).** Ministral-8B
QLoRA on `kleos-policy-v0.0.6`, evaluated against the prompt-engineered
orchestration baseline on the 349-example held-out split.

H1 is supported, with deviations recorded below. H3 is supported directionally
but the absolute consistency number stays poor. H2 is **not measurable** on this
benchmark. H4–H7 remain untested.

Read the Deviations log before quoting any number from here: the run departed
from the pre-registered protocol in three ways, and 22% of the test label space
turned out to be unlearnable from the training split.

---

## H1 — Primary hypothesis

> Behavioural fine-tuning on KLEOS policy data improves task-specific judgment
> and correctness relative to the corresponding prompt-engineered baseline.

| | |
| --- | --- |
| **Arms** | `arm0_base` vs `arm2_finetuned` |
| **Primary metric** | Per-task grader score on the held-out test split |
| **Split** | `entity_holdout` — generalization to unseen entities |
| **Decision rule** | Improvement counts only if the paired bootstrap 95% CI excludes zero |
| **Reported per task** | Yes. No blended aggregate. |
| **Status** | **SUPPORTED** — `kleos-v006-ministral8b-run1`, 2026-09-15 (see deviations D1, D2) |

**Prediction:** unknown. A negative result is a genuinely likely outcome and is
publishable.

**What would falsify it:** no task improves with a CI excluding zero.

**Result.** Ministral-8B QLoRA vs `arm1_base_orchestrated`, n=349, paired
bootstrap. **All seven tasks improved at p<0.05; none regressed.**

Figures below are the **corrected** scores, re-graded 2026-09-15 after audit
finding F1 (unbounded nDCG). See "Correction" below for the superseded numbers.

| Task | Baseline | Fine-tuned | Δ | Verdict |
| --- | --- | --- | --- | --- |
| mission_control_briefing | 0.5323 | 0.9708 | +0.4385 | improved (p<0.05) |
| workspace_reasoning | 0.5228 | 0.9336 | +0.4108 | improved (p<0.05) |
| context_prioritization | 0.5054 | 0.8924 | +0.3870 | improved (p<0.05), n=8 |
| memory_conflict_resolution | 0.4781 | 0.8192 | +0.3412 | improved (p<0.05) |
| tool_routing | 0.3919 | 0.7181 | +0.3262 | improved (p<0.05) |
| notification_prioritization | 0.6253 | 0.8871 | +0.2617 | improved (p<0.05) |
| recommendation_generation | 0.3285 | 0.4569 | +0.1284 | improved (p<0.05) |

Overall **0.4744 → 0.8015**, a gap of **+0.3271**. Secondary metrics are unchanged
by the correction: faithfulness 0.7297 → 0.8331, citation precision 0.4470 →
0.6676, fabricated citations 194 → 116 responses, parse failures 13 → 0.

`context_prioritization` has n=8 and its interval should not be leaned on.
`recommendation_generation` remains the weakest task in absolute terms (0.4569)
for the reasons in D3 and H3, even though its improvement is now significant.

### Correction — F1 re-grade, 2026-09-15

The first report was produced with an unbounded nDCG that credited repeated
predicted items, letting a degenerate answer score above a perfect ranking. Both
arms were re-graded **offline from their stored responses** — no inference re-run,
no model loaded, source artifacts left byte-identical — using `scripts/rescore.py`.

| | Reported | Corrected | Change |
| --- | ---: | ---: | ---: |
| `arm1_base_orchestrated` | 0.5231 | **0.4744** | −0.0487 |
| `arm2_finetuned` | 0.8015 | **0.8015** | **0.0000** |
| Gap | +0.2784 | **+0.3271** | +0.0487 |
| Tasks significant | 6 / 7 | **7 / 7** | +1 |

**210 of 349 baseline responses changed; zero fine-tuned responses changed.** The
inflation existed only in the baseline, so the originally reported effect was
*understated*. `recommendation_generation` crossed from "improved (not
significant)" to p<0.05. No task changed direction and none regressed, so the H1
conclusion holds and is strengthened rather than revised.

That 210-vs-0 split is a finding in its own right: 60% of baseline answers
contained a repeated item in their extracted ranking, and **not one** fine-tuned
answer did. `tool_routing` was the only task whose baseline was untouched by the
correction.

Superseded artifacts are retained: `report_v006/` (original) alongside
`report_v006_rescored/`, and `eval_arm{1,2}_full.json` alongside
`…rescored.json`.

---

## H2 — Does any gain generalize out of distribution?

> Improvements from fine-tuning persist on out-of-distribution examples (unseen
> domains, unseen formats, conflicting evidence).

| | |
| --- | --- |
| **Primary metric** | `ood_score`, reported separately from `in_distribution_score` |
| **Secondary** | `generalization_gap` (in-distribution − OOD) |
| **Decision rule** | Gains generalize if the OOD delta CI excludes zero |
| **Status** | **NOT MEASURABLE** on `kleos-policy-v0.0.6` — see below |

**The interesting failure case:** in-distribution improves while OOD degrades.
That pattern is consistent with fitting surface features of the training
distribution rather than learning a transferable policy, and the comparison
report calls it out explicitly.

**Requires:** benchmark examples tagged `split_tag: "ood"` with an `ood_shift`.
Without them OOD is not measurable and **no generalization claim may be made**.

**2026-09-15.** The v0.0.6 benchmark is tagged, but *every* example is
`split_tag: "ood"` / `ood_shift: "unseen_formats"` — the release holds out
`format=json` entirely, so there is no in-distribution population to compare
against and `generalization_gap` is undefined. The OOD score itself is valid and
is what H1 reports; the *gap* is not computable. Measuring H2 needs a benchmark
carrying both populations, which v0.0.6 does not.

Related and worth stating plainly: `format_valid` was **0.0000 for both arms** on
all 349 examples. Neither the baseline nor the fine-tuned model emitted JSON,
even though every test prompt contains JSON input. The decision policy crossed
the format boundary; the output format did not. The H1 numbers are therefore
measuring judgment on prose answers in both arms, which is why the ranking
grader had to be made format-agnostic first (D4).

---

## H3 — Does behaviour survive format changes?

> The fine-tuned model reaches the same decision under logically irrelevant
> perturbations: rewording, evidence reordering, format changes, irrelevant
> added context.

| | |
| --- | --- |
| **Primary metric** | `correct_agreement_rate` (agrees **and** is right) |
| **Secondary** | `agreement_rate`, flips attributed per perturbation kind |
| **Decision rule** | Consistency improves if the delta CI excludes zero |
| **Status** | **IMPROVED, STILL POOR** — `kleos-v006-ministral8b-run1`, 2026-09-15 |

Reported as two numbers on purpose. A model that is *consistently wrong* scores
1.0 on agreement and 0.0 on correct agreement — collapsing them would hide that.

This is the most direct available test of "policy versus surface pattern".

**Requires:** `metadata.scenario_family` on perturbed examples.

**Result (15 groups, n=349).**

| Metric | Baseline | Fine-tuned |
| --- | --- | --- |
| `agreement_rate` | 0.067 | 0.333 |
| `correct_agreement_rate` | **0.000** | 0.333 |
| Groups flipping under an irrelevant perturbation | 14 / 15 | 10 / 15 |

Fine-tuning helps, and the two-number split earns its keep: the baseline's
`correct_agreement_rate` of exactly 0.000 means it never both agreed with itself
*and* was right. But 10 of 15 groups still flip under paraphrase or evidence
reordering, so the fine-tuned model is **not** demonstrating a stable policy in
absolute terms.

The abstention analysis says the same thing more sharply. Splitting the 78
"should decline" cases by scenario family:

| Family | Training signal | Test cases | Model correct |
| --- | --- | --- | --- |
| `mem.stale_explicit_conflict` | 32/32 abstain (unconditional) | 20 | **20/20** |
| `route.ask_when_underspecified` | 45/45 abstain (unconditional) | 30 | **30/30** |
| `rec.verify_before_recommending` | 25/55 abstain (**conditional**) | 25 | **0/25** |
| `rec.abstain_without_evidence` | 3/9 abstain (**conditional**) | 3 | **0/3** |

The model abstains perfectly where the entire family always abstains, and never
where abstention depends on the scenario. It learned *"this kind of question →
decline"*, not *"the evidence is insufficient → decline"*. On the 271 cases where
committing is correct it scored 1.0000, i.e. it never wrongly abstains — the
error is one-sided overconfidence.

That is the clearest evidence in this run that what transferred is a family-level
shortcut rather than the intended policy, and it is the finding most worth acting
on in the next dataset revision.

---

## H4 — Does fine-tuning damage general capability?

> Narrow behavioural fine-tuning does not measurably degrade general reasoning.

| | |
| --- | --- |
| **Primary metric** | `relative_delta` on the fixed capability suite |
| **Decision rule** | Regression if `relative_delta < -0.01` with a CI excluding zero |
| **Status** | Not yet run |

The suite must stay **constant across experiments** — changing it invalidates
comparison with every prior run. Its version is recorded in every result
(`CAPABILITY_SUITE_VERSION`).

A task gain paired with a capability loss is a trade-off, not an improvement, and
must be reported as such.

---

## H5 — Does model scale change the effect?

> The effect of fine-tuning differs between an 8B and a ~24-30B model.

| | |
| --- | --- |
| **Arms** | `ministral_8b` vs `mistral_small_3_2` |
| **Status** | Not yet run — `mistral_small_3_2` needs an L4 (≥22.5GB) or A100 |

**Re-scoped 2026-09-15.** This hypothesis previously named `qwen3_8b` vs
`qwen3_30b_a3b_thinking`. Qwen is permanently excluded from KLEOS, so the scale
comparison is now between the two Mistral candidates.

**Confound to state, not to hide:** `mistral_small_3_2` is a 24B *vision-language*
model (`Mistral3ForConditionalGeneration`). Comparing it against text-only
Ministral-8B varies scale **and** modality together. There is no scale-matched
text-only Mistral in the registry, so this confound cannot be designed away — it
must be reported alongside any result.

Feasibility (measured, `max_seq_length` 1024): Ministral-8B needs ~8.1GB and runs
on a free T4; `mistral_small_3_2` needs ~16.5GB and does **not** fit a 16GB T4.

---

## H6 — Does KLEOS policy learning transfer across model families?

> If fine-tuning helps one family, it helps the other.

| | |
| --- | --- |
| **Arms** | ~~`qwen3_8b` vs `ministral_8b`~~ |
| **Status** | **CLOSED — not testable under the current model policy (2026-09-15)** |

**Why closed.** This hypothesis required a cross-family comparison, and the only
scale-matched counterpart in the registry was `qwen3_8b`. Qwen is permanently
excluded from KLEOS, so there is no second family to compare against: both
remaining candidates (`ministral_8b`, `mistral_small_3_2`) are Mistral.

The Qwen configs and `QwenDenseAdapter` / `QwenMoEAdapter` remain in the
repository — the family abstraction is what keeps the pipeline architecture-
agnostic, and deleting them would not make the pipeline simpler. They are simply
not used by KLEOS.

Reopening this would require adding a non-Mistral, non-Qwen family (e.g. Llama or
Gemma) with its own adapter. That is a deliberate scope decision, not an
oversight, and no cross-family claim may be made until it is taken.

---

## H7 — Do fine-tuning and orchestration interact?

> Fine-tuning and the KLEOS orchestration layer are complementary rather than
> redundant.

| | |
| --- | --- |
| **Arms** | all four: `arm0`, `arm1`, `arm2`, `arm3` |
| **Analysis** | (arm3 − arm2) vs (arm1 − arm0) |
| **Status** | **Partially measured** — `arm1` and `arm2` complete on n=349 (2026-09-15); `arm0` and `arm3` outstanding |

`arm1_base_orchestrated` = 0.4744 and `arm2_finetuned` = 0.8015 are already on
record from `kleos-v006-ministral8b-run1`. Completing this needs `arm0_base` and
`arm3_finetuned_orchestrated` on the **same 349-example benchmark** — the
comparison pairs by `example_id`, so the existing 20-example `arm0` probe
(0.5659) is a biased head-of-file slice and is **not** usable here.

`arm3` reuses the adapter already trained, so it costs one evaluation pass
(~90 min on a T4) and no further training.

If orchestration helps the base model but not the fine-tuned one, the adapter has
likely internalized what the scaffolding was supplying — an interesting and
actionable finding for the product.

---

## Fixed experimental protocol

Applies to every hypothesis above.

| Element | Commitment |
| --- | --- |
| Split | Held-out strategy, never `random` |
| Decoding | Greedy (`do_sample: false`), identical across arms |
| Benchmark | Identical across arms, fixed before the run |
| Graders | Fixed before the run |
| Significance | Paired bootstrap, 2000 iterations, 95% CI |
| Reporting | Per task; no blended aggregate |
| Seeds | Recorded; multiple seeds only meaningful with sampling |
| Leakage | Checked and reported before training |

## Recording a run

```bash
python scripts/run_experiment.py --config configs/training/qlora_small.yaml \
                                 --dataset /path/to/private/dataset
```

Every run writes a manifest. To list them, failures included:

```python
from kleos_models.experiments.registry import ExperimentRegistry

print(ExperimentRegistry("outputs").render())
```

## Results log

Append one row per completed experiment. **Include failed and negative runs.**

| Date | Hypothesis | Model | Dataset | Config hash | Outcome | Report |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-15 | H1 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Supported.** 7/7 tasks improved at p<0.05, none regressed. Overall 0.4744 → 0.8015 (corrected; originally reported 0.5231 → 0.8015 with 6/7 significant, before the F1 nDCG re-grade). | `outputs/report_v006_rescored/summary.md` (original: `report_v006/`) |
| 2026-09-15 | H2 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Not measurable.** Benchmark is 100% OOD; no in-distribution population, so no gap. | same run |
| 2026-09-15 | H3 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Improved, still poor.** correct_agreement 0.000 → 0.333; 10/15 groups still flip. Abstention is a family-level shortcut. | same run |

Run provenance: experiment id `kleos-v006-ministral8b-run1`, seed 42, 3 epochs
(309 steps, effective batch 8, `max_seq_length` 1024), best checkpoint selected
on validation loss at **step 200 / epoch 1.95** (`eval_loss` 0.03987; the third
epoch degraded it to 0.04469). Tesla T4, fp16, NF4 double-quant, `paged_adamw_8bit`.
transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, torch 2.11.0+cu128.
Benchmark `benchmark.jsonl` sha256 `a11ffad75f5147f9…`, derived from the sealed
test split by `scripts/build_benchmark.py` (reproducible; release unmodified).

Artifact hashes, the adapter configuration, and an independent verification of the
H1 numbers against the stored evaluation JSONs are recorded in
[experiments/kleos-v006-ministral8b-run1-artifact-audit.md](experiments/kleos-v006-ministral8b-run1-artifact-audit.md).
Finding **F1 from that audit is now resolved**: nDCG is bounded, both arms were
re-graded offline, and the corrected result is the one reported above. **F3 is
resolved for future runs** — see "Base-model revision" below; the historical run
remains unpinned and is not backfilled. **F2** (tokenizer packaging) remains
open.

## Deviations log

Record any departure from the protocol above, with the reason, at the time it
happens.

| Date | Deviation | Reason |
| --- | --- | --- |
| 2026-09-15 | **D1.** H1 pre-registers `arm0_base` vs `arm2_finetuned`; the run compared **`arm1_base_orchestrated`** vs `arm2_finetuned`. | The stated research question names the *prompt-engineered orchestration* baseline, which is arm1. arm0 is the weaker comparison and would have flattered the result. Only a 20-example arm0 probe exists (0.5659), which is a biased head-of-file slice and is **not** comparable to the full run. |
| 2026-09-15 | **D2.** H1 pre-registers an `entity_holdout` split; the sealed release uses **`format_holdout`** on `format=json`. | The dataset was sealed upstream with that strategy. Consequence: the test split is a format-transfer probe, not an entity-generalization probe, so H1's result speaks to a different kind of generalization than pre-registered. |
| 2026-09-15 | **D3.** 78 of 349 test cases (22%) expect a `deciding_factor` label that appears **zero times** in the training targets: `request_ambiguous` (30), `missing_input` (25), `stale_explicit_conflict` (20), `insufficient_separation` (3). | Discovered during analysis, not by design. Those are exactly the 78 abstention cases, so they are unwinnable by construction. Overall `deciding_factor` reads 0.6332; on the 271 answerable cases it is **0.816**. Affects both arms identically, so the H1 comparison stays fair, but absolute `deciding_factor` figures must carry this caveat. Fix belongs in the next dataset release, not here. |
| 2026-09-15 | **D4.** `extract_ranking` was corrected before the run: it returned whole prose lines instead of resolving them to candidate names. | Measurement instrument, changed *before* any result was produced, so no reported number is affected. Without it every prose answer scored 0.0 and the run would have reported fine-tuning as catastrophic — a false negative caused by the grader measuring formatting instead of ordering. |
| 2026-09-15 | **D5.** `ConversationFormatter` now folds the system prompt into the first user turn when the chat template drops it. | Mistral's template injects the system message into the *last* message, so during training (which ends on the assistant turn) the system prompt was silently discarded, while evaluation kept it. Every example would have trained without its policy instructions. Fixed before the run; detected by probing the live template rather than branching on model family. |
| 2026-09-15 | **D6.** A composite `kleos_policy` grader was added; it was not in the original protocol. | The benchmark needs ranking, deciding factor and abstention scored together, with `format_valid` reported **separately and excluded from the score**. Without that separation a format failure is indistinguishable from a judgment failure — which, given D2, is the difference between a real result and a wrong one. |
| 2026-09-15 | **D7.** The v0.0.6 run used `revision: main` for the base model. The commit it resolved to is **not recoverable** from the preserved artifacts. | Nothing in the pipeline resolved or recorded a Hub commit sha — every code path echoes back the requested pointer. The Colab HF cache that held it was wiped. See "Base-model revision" below. Future runs are pinned; the historical record is **not** backfilled. |

---

## Base-model revision

**v0.0.6 was trained with `revision: main`, an unmoored pointer.**

The manifest records `"revision": "main"` because that is what was *requested*.
Every recording path — `_describe_model_from_config`, `ModelFamilyAdapter.describe()`,
`models/loading.py` — echoes the configured value; nothing ever asked the Hub
what `main` resolved to. The only place the resolved sha existed was the Colab
HF cache, and that runtime is long gone (evidenced by the full 16GB re-download
on every later session).

**The exact commit used by the v0.0.6 run is therefore not provable, and we do
not claim one.** In particular:

> `2f494a194c5b980dfb9772cb92d26cbb671fce5a` is the revision verified on
> **2026-09-15**. It is **not** asserted to be the commit v0.0.6 trained
> against. It may be the same commit; nothing in the preserved artifacts can
> establish that either way, so no claim is made.

What the artifacts *do* establish is architectural compatibility — `MistralForCausalLM`,
`model_type: ministral`, 252 adapted modules (36 layers × 7 projections), and an
adapter file size that arithmetically confirms 43,646,976 fp32 parameters. None
of that identifies a commit: an upstream re-upload that changed weights without
changing shapes would be invisible to all of it.

**Future training and deployment are pinned** to
`2f494a194c5b980dfb9772cb92d26cbb671fce5a`, in
[`configs/models/ministral_8b.yaml`](../configs/models/ministral_8b.yaml) and
[`configs/deployment/kleos_v006_ministral8b.yaml`](../configs/deployment/kleos_v006_ministral8b.yaml).
`tests/test_revision_pinning.py` asserts both, and asserts they cannot drift
apart.

**The historical config was deliberately not edited.** `model.revision`
participates in `config_hash`, so inserting the sha would change it from the
recorded `3fbb3f90ed9662ee…` and the manifest would describe a configuration
that never ran. The run's manifest records `git.commit: 2ad7c9ae` with
`git.dirty: false`, so `git checkout 2ad7c9ae` reconstructs the exact historical
config — git preserves the record, and pinning at HEAD costs nothing.

The honest statement about v0.0.6 reproducibility: **reproducible modulo upstream
not having moved.** That caveat cannot be retroactively removed.

---

## The standard this is held to

The most important outcome is not "KLEOS fine-tuning worked". It is:

> We designed a controlled experiment capable of determining whether KLEOS-specific
> fine-tuning actually improves task performance.

If fine-tuning wins, quantify the win. If it loses, analyse why. If it improves
one task and harms another, that may be the most interesting result available.
