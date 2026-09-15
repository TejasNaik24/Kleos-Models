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
bootstrap. Six of seven tasks improved at p<0.05; none regressed.

| Task | Baseline | Fine-tuned | Δ | Verdict |
| --- | --- | --- | --- | --- |
| mission_control_briefing | 0.5636 | 0.9708 | +0.4072 | improved (p<0.05) |
| workspace_reasoning | 0.5865 | 0.9336 | +0.3471 | improved (p<0.05) |
| context_prioritization | 0.5544 | 0.8924 | +0.3380 | improved (p<0.05), n=8 |
| tool_routing | 0.3919 | 0.7181 | +0.3262 | improved (p<0.05) |
| memory_conflict_resolution | 0.5291 | 0.8192 | +0.2901 | improved (p<0.05) |
| notification_prioritization | 0.7064 | 0.8871 | +0.1807 | improved (p<0.05) |
| recommendation_generation | 0.4158 | 0.4569 | +0.0411 | **not significant** |

Overall 0.5231 (95% CI 0.4979–0.5483) → 0.8015 (0.7768–0.8262); the intervals do
not overlap. Secondary: faithfulness 0.7297 → 0.8331, citation precision 0.4470 →
0.6676, fabricated citations 194 → 116 responses, parse failures 13 → 0.

`recommendation_generation` is the one task that did not move, and the reason is
known rather than mysterious — see D3. `context_prioritization` has n=8 and its
interval should not be leaned on.

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
| **Arms** | `qwen3_8b` vs `qwen3_30b_a3b_thinking` |
| **Status** | Not yet run — requires an A100-class GPU |

**Confound to control:** Qwen3-30B-A3B is thinking-only and Qwen3-8B is
configurable. Reasoning mode and scale therefore vary together unless Qwen3-8B is
deliberately run in thinking mode. State which was done.

---

## H6 — Does KLEOS policy learning transfer across model families?

> If fine-tuning helps one family, it helps the other.

| | |
| --- | --- |
| **Arms** | `qwen3_8b` vs `ministral_8b` |
| **Status** | Not yet run |

**Why `ministral_8b` and not `mistral_small_3_2`:** Ministral-8B is a text-only
dense model at the same scale as Qwen3-8B. Comparing Qwen3-8B against a 24B
multimodal Mistral would confound family with scale *and* modality, and the result
would be uninterpretable.

Interpretation:
- both improve → the policy is learnable across architectures
- one improves → investigate architecture, tokenizer, template, or capacity
- neither improves → examine data quality, policy learnability and evaluation
  design before concluding anything about the models

---

## H7 — Do fine-tuning and orchestration interact?

> Fine-tuning and the KLEOS orchestration layer are complementary rather than
> redundant.

| | |
| --- | --- |
| **Arms** | all four: `arm0`, `arm1`, `arm2`, `arm3` |
| **Analysis** | (arm3 − arm2) vs (arm1 − arm0) |
| **Status** | Not yet run |

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
| 2026-09-15 | H1 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Supported.** 6/7 tasks improved at p<0.05, none regressed. Overall 0.5231 → 0.8015. | `outputs/report_v006/summary.md` |
| 2026-09-15 | H2 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Not measurable.** Benchmark is 100% OOD; no in-distribution population, so no gap. | same run |
| 2026-09-15 | H3 | Ministral-8B-Instruct-2410 (QLoRA r=16) | kleos-policy-v0.0.6 (`3cc9a744…`) | `3fbb3f90ed9662ee` | **Improved, still poor.** correct_agreement 0.000 → 0.333; 10/15 groups still flip. Abstention is a family-level shortcut. | same run |

Run provenance: experiment id `kleos-v006-ministral8b-run1`, seed 42, 3 epochs
(309 steps, effective batch 8, `max_seq_length` 1024), best checkpoint selected
on validation loss at **step 200 / epoch 1.95** (`eval_loss` 0.03987; the third
epoch degraded it to 0.04469). Tesla T4, fp16, NF4 double-quant, `paged_adamw_8bit`.
transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, torch 2.11.0+cu128.
Benchmark `benchmark.jsonl` sha256 `a11ffad75f5147f9…`, derived from the sealed
test split by `scripts/build_benchmark.py` (reproducible; release unmodified).

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

---

## The standard this is held to

The most important outcome is not "KLEOS fine-tuning worked". It is:

> We designed a controlled experiment capable of determining whether KLEOS-specific
> fine-tuning actually improves task performance.

If fine-tuning wins, quantify the win. If it loses, analyse why. If it improves
one task and harms another, that may be the most interesting result available.
