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

**No hypothesis below has been tested yet.** The infrastructure exists; the
research dataset does not. Every result cell reads "not yet run".

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
| **Status** | Not yet run |

**Prediction:** unknown. A negative result is a genuinely likely outcome and is
publishable.

**What would falsify it:** no task improves with a CI excluding zero.

---

## H2 — Does any gain generalize out of distribution?

> Improvements from fine-tuning persist on out-of-distribution examples (unseen
> domains, unseen formats, conflicting evidence).

| | |
| --- | --- |
| **Primary metric** | `ood_score`, reported separately from `in_distribution_score` |
| **Secondary** | `generalization_gap` (in-distribution − OOD) |
| **Decision rule** | Gains generalize if the OOD delta CI excludes zero |
| **Status** | Not yet run |

**The interesting failure case:** in-distribution improves while OOD degrades.
That pattern is consistent with fitting surface features of the training
distribution rather than learning a transferable policy, and the comparison
report calls it out explicitly.

**Requires:** benchmark examples tagged `split_tag: "ood"` with an `ood_shift`.
Without them OOD is not measurable and **no generalization claim may be made**.

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
| **Status** | Not yet run |

Reported as two numbers on purpose. A model that is *consistently wrong* scores
1.0 on agreement and 0.0 on correct agreement — collapsing them would hide that.

This is the most direct available test of "policy versus surface pattern".

**Requires:** `metadata.scenario_family` on perturbed examples.

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
| _(none yet)_ | | | | | | |

## Deviations log

Record any departure from the protocol above, with the reason, at the time it
happens.

| Date | Deviation | Reason |
| --- | --- | --- |
| _(none yet)_ | | |

---

## The standard this is held to

The most important outcome is not "KLEOS fine-tuning worked". It is:

> We designed a controlled experiment capable of determining whether KLEOS-specific
> fine-tuning actually improves task performance.

If fine-tuning wins, quantify the win. If it loses, analyse why. If it improves
one task and harms another, that may be the most interesting result available.
