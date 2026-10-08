# Experiments and pre-registration

This document is the pre-registration and results record of the KLEOS
fine-tuning research: hypotheses H1 to H9, each written down before the run that
tests it, the result recorded against each, the fixed protocol, the results log
and the deviations log. The introduction and the status section are current as
of 2026-10-07. Everything after them is the record as written at the time.
Editorial changes for public release are limited to editor's notes, retargeted
links, and these two sections.

Identifiers used throughout:

- **H1–H9**: the pre-registered hypotheses, in this document.
- **D1–D13**: departures from the protocol, in the [deviations log](#deviations-log).
- **F1–F3**: findings of the [Ministral-8B artifact audit](experiments/kleos-v006-ministral8b-run1-artifact-audit.md#findings).
- **H-F1–H-F14**: findings recorded with the [Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings).
  H-F11 to H-F14 concern how the evaluation measures and were found while Logos
  v0.0.1 was designed.
- **L-F1–L-F8**: findings from the Logos work, in
  [Logos findings](experiments/logos-findings.md).

The index of every record, the evidence labels (VERIFIED, ESTIMATED, SOURCE,
OBSERVED, INFERRED, NOT VERIFIED) and the placeholders used in the records are
in [docs/experiments/README.md](experiments/README.md).

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

As of 2026-10-07, four runs are complete. All are QLoRA fine-tunes, and all were
evaluated on the same 349-item held-out benchmark (sha256 `a11ffad7…`):

| Run | Model | Base | Data | Completed |
| --- | --- | --- | --- | --- |
| `kleos-v006-ministral8b-run1` | Research only (the base is under the Mistral AI Research License) | Ministral-8B-Instruct-2410 | `kleos-policy-v0.0.6` | 2026-09-15 |
| `kleos-v006-mistralnemo12b-run1` | **KLEOS Hermes v0.0.6** | Mistral-Nemo-Instruct-2407 (Apache-2.0) | `kleos-policy-v0.0.6` | 2026-09-22 |
| `kleos-v006-ministral314b-run1` | **KLEOS Logos v0.0.1** | Ministral-3-14B-Instruct-2512-BF16, text tower (Apache-2.0) | `kleos-policy-v0.0.6` | 2026-10-01 |
| `kleos-v007-ministral314breasoning-run1` | **KLEOS Logos v0.0.2** | Ministral-3-14B-Reasoning-2512, text tower (Apache-2.0) | `kleos-policy-v0.0.7` | 2026-10-06 |

| Hypothesis | Status | Date |
| --- | --- | --- |
| [H1](#h1--primary-hypothesis): fine-tuning beats the prompt-engineered baseline | **Supported** on Ministral-8B (with deviations D1 and D2); **replicated** by Hermes | 2026-09-15; 2026-09-22 |
| [H2](#h2--does-any-gain-generalize-out-of-distribution): gains generalize out of distribution | **Not measurable** on the v0.0.6 benchmark, which is entirely out of distribution | 2026-09-15 |
| [H3](#h3--does-behaviour-survive-format-changes): decisions survive irrelevant perturbations | **Improved, still poor**; reproduced exactly by Hermes | 2026-09-15; 2026-09-22 |
| [H4](#h4--does-fine-tuning-damage-general-capability): no loss of general capability | Not run | |
| [H5](#h5--does-model-scale-change-the-effect): the effect depends on scale | Not run; re-scoped to two Mistral models | 2026-09-15 |
| [H6](#h6--does-kleos-policy-learning-transfer-across-model-families): transfer across model families | **Closed**: not testable with one model family | 2026-09-15 |
| [H7](#h7--do-fine-tuning-and-orchestration-interact): fine-tuning and orchestration interact | Partially measured: `arm1` and `arm2` only, in three runs | 2026-10-01 |
| [H8](#h8--does-a-stronger-base-make-a-better-kleos-model): a stronger base makes a better model (Logos v0.0.1) | **H8a supported** (fine-tuning helps Logos); **H8b, the primary comparison with Hermes, inconclusive** | H8b 2026-09-29; H8a 2026-10-01 |
| [H9](#h9--does-a-logos-trained-to-think-beat-hermes): a Logos trained to think beats Hermes (Logos v0.0.2) | **Supported: better.** Answerable subset +0.0409, cluster 95% CI +0.0040 to +0.0780 | 2026-10-06 |

How to read the table:

- **Deviations come first.** Read the [deviations log](#deviations-log) before
  quoting a number. D1 (the baseline is `arm1`, not `arm0`), D2 (a format
  holdout, not an entity holdout) and D3 (78 of the 349 test items, 22%, expect
  labels that never occur in v0.0.6 training) apply to every run on v0.0.6.
- **Intervals.** From H8 on, intervals resample groups (`group_id`) rather than
  examples ([protocol amendment](#protocol-amendment--clustered-intervals-2026-09-24)).
- **H8 and H9 together.** Fine-tuning helps the stronger base, but Logos v0.0.1
  was not measurably better than Hermes on v0.0.6. Logos v0.0.2 is measurably
  better. Its base release and its training data changed together, and each
  model is a single training run, so H9 does not say which change produced the
  gain or how large it is.

The run card below was written when Hermes completed, and is kept as recorded.

### KLEOS Hermes — `kleos-v006-mistralnemo12b-run1`, completed 2026-09-22

| | |
| --- | --- |
| KLEOS model | **Hermes** (the smaller/faster model; Logos is the larger one) |
| Base | `mistralai/Mistral-Nemo-Instruct-2407` |
| Revision | `04d8a90549d23fc6bd7f642064003592df51e9b3` (pinned, audited 2026-09-15) |
| Licence | **Apache-2.0, ungated** |
| Dataset | `kleos-policy-v0.0.6`, unchanged and read-only |
| Config | [`configs/training/kleos_hermes_v006.yaml`](../configs/training/kleos_hermes_v006.yaml) |
| Result | `arm1_base_orchestrated` **0.4755 → `arm2_finetuned` 0.8051**, +0.3295 (95% CI 0.3068–0.3521); 7/7 tasks significant at p < 0.001, none regressed |
| Full report | [experiments/kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md) |

Hermes exists because Ministral-8B is under the Mistral Research Licence, which
is non-commercial. That does not invalidate `kleos-v006-ministral8b-run1` — it
remains sound research — but it cannot back a product-facing model. Mistral Nemo
is Apache-2.0 and resolves to the same `MistralDenseAdapter` with no code
changes.

It is a **separate experiment**, not a re-run: different base, different scale
(12B vs 8B). The valid comparison is within the Hermes run —
`arm1_base_orchestrated` vs `arm2_finetuned` on the same benchmark — and that is
the result above. Set beside the Ministral run it is descriptive only: scale,
architecture and pretraining are confounded, so no difference between the two may
be attributed to scale. *(Correction: this section previously said
`assert_comparable` would block pooling the two runs. It does not — it compares
dataset, task and model family, and both runs are family `mistral`, so it passes
them without a warning. Recorded as finding H-F6 in the report.)*

Read beside Ministral, the answer to the open question is yes: the behavioural
signal transferred across Mistral base checkpoints, landing within 0.004 of
Ministral on both arms. So did every limitation — consistency, the abstention
shortcut and the output-format failure reproduce exactly (see H3).

**The research artifact is frozen.** Hermes was subsequently prepared for
serving, which produced a *separate* deployment artifact: a package with the base
revision pinned, the tokenizer files frozen and hashed, and a manifest that a
loader verifies before answering any request. The research artifact was not
edited to match it — its `adapter_config.json` still records `revision: null`,
and the deployment package records both the pin and that original absence. No
number above is affected. On 2026-09-23 the package, loaded through the serving
path, reproduced the frozen evaluation's responses 9/9 byte for byte — a
reproducibility check on the artifact, not a new score. It did again the same
day on a free Hugging Face ZeroGPU Space (Blackwell GPU), also 9/9. See
[deployment.md](deployment.md).

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
| **Status** | **SUPPORTED** — `kleos-v006-ministral8b-run1`, 2026-09-15 (see deviations D1, D2); **replicated** by `kleos-v006-mistralnemo12b-run1` (KLEOS Hermes), 2026-09-22 |

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

### Replication — KLEOS Hermes, 2026-09-22

`kleos-v006-mistralnemo12b-run1` repeated the test on Mistral-Nemo-12B with the
same data, benchmark, graders, decoding and hyperparameters, graded with the
corrected nDCG from the start.

| Task | arm1 | arm2 | Δ | |
| --- | ---: | ---: | ---: | --- |
| workspace_reasoning | 0.5190 | 0.9512 | +0.4323 | improved (p<0.001) |
| context_prioritization | 0.5935 | 1.0000 | +0.4065 | improved (p<0.001), n=8 |
| mission_control_briefing | 0.5681 | 0.9716 | +0.4035 | improved (p<0.001) |
| memory_conflict_resolution | 0.4756 | 0.8553 | +0.3798 | improved (p<0.001) |
| tool_routing | 0.3929 | 0.6679 | +0.2750 | improved (p<0.001) |
| recommendation_generation | 0.2974 | 0.4946 | +0.1971 | improved (p<0.001) |
| notification_prioritization | 0.6201 | 0.8105 | +0.1904 | improved (p<0.001) |

Overall **0.4755 → 0.8051**, a gap of **+0.3295** (95% CI 0.3068–0.3521). **H1 is
supported again: 7/7 improved, 7/7 significant, none regressed.** `compare.py`
prints these p-values as `p=0.0`; with 2,000 two-sided resamples that means no
resample crossed zero, i.e. p < 0.001. Faithfulness 0.7513 → 0.8531, citation
precision 0.5938 → 0.7192, fabricated citations 147 → 98 responses, parse
failures 0 → 0.

Beside Ministral this is descriptive, not a tested comparison: both arms land
within 0.004 of the Ministral figures on a different base model. Details, hashes
and findings: [experiments/kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md).

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
| **Status** | **IMPROVED, STILL POOR** — `kleos-v006-ministral8b-run1`, 2026-09-15; reproduced exactly by `kleos-v006-mistralnemo12b-run1`, 2026-09-22 |

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

**Correction note — the grouping unit, 2026-09-24 (finding H-F11).** The groups
above are *scenario families*, and 6 of the 15 test families mix cases whose
correct answers differ. A model that answered every case correctly would score
`agreement_rate` **0.600 (9/15)** and log **57 "flips"** under this grouping, so
"10/15 groups flip" overstates instability: part of it is the metric's ceiling,
not the model. The unit of logical equivalence is `group_id`: 78 test groups, each
label-identical by construction, where an oracle scores 1.000. The numbers above
are unchanged and stay as reported. From H8 on, consistency is reported by
`group_id` beside the family figure, each with its oracle ceiling
([evaluation.md](evaluation.md#which-unit)); `scripts/rescore.py
--mode annotate` re-reports both earlier runs that way without touching their files.

**Replication — KLEOS Hermes, 2026-09-22.** On Mistral-Nemo-12B every number above
reproduced exactly:

| Metric | Hermes arm1 | Hermes arm2 |
| --- | --- | --- |
| `agreement_rate` | 0.133 | 0.333 |
| `correct_agreement_rate` | **0.000** | 0.333 |
| Groups flipping under an irrelevant perturbation | 13 / 15 | 10 / 15 |

The abstention table is identical case for case: 20/20 and 30/30 on the two
unconditional families, 0/25 and 0/3 on the two conditional ones, 271/271 where
committing is correct. A base with 50% more parameters did not move consistency or
abstention at all. That makes it very likely the shortcut lives in the v0.0.6 data
and recipe rather than in the base model — and that the fix is a dataset revision,
not a larger model. `format_valid` was again 0.0000 in both arms.

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
comparison is now between the two Mistral candidates. *[editor's note: the two
configs named here were later removed from the repository; see the editor's note
under H6.]*

**Confound to state, not to hide:** `mistral_small_3_2` is a 24B *vision-language*
model (`Mistral3ForConditionalGeneration`). Comparing it against text-only
Ministral-8B varies scale **and** modality together. There is no scale-matched
text-only Mistral in the registry, so this confound cannot be designed away — it
must be reported alongside any result.

Feasibility (measured, `max_seq_length` 1024): Ministral-8B needs ~8.1GB and runs
on a free T4; `mistral_small_3_2` needs ~16.5GB and does **not** fit a 16GB T4.

**Exploratory, not a test of H5 (2026-09-22).** The Hermes run puts a 12B Mistral
beside the 8B one on identical data and protocol: fine-tuning gaps of +0.3295 and
+0.3271, with identical consistency and abstention. These are not H5's
pre-registered arms, and the two bases differ in architecture and pretraining as
well as size, so this observation cannot support or refute H5. It is recorded so
it is not later mistaken for one.

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

> *[editor's note, 2026-10-07: the paragraph above records the decision as
> closed on 2026-09-15. In the public-release change of 2026-10-07 the Qwen configs,
> adapters and their tests were removed from the repository; the family
> abstraction (`ModelFamilyAdapter`) stays, with the Mistral adapters as its
> only implementations.]*

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
| **Status** | **Partially measured** — `arm1` and `arm2` complete on n=349 for Ministral (2026-09-15), Hermes (2026-09-22) and Logos (2026-10-01); `arm0` and `arm3` outstanding in all three |

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

## H8 — Does a stronger base make a better KLEOS model?

> Under the same data and the same recipe, a more capable base (Ministral 3 14B,
> MMLU 79.4) yields better KLEOS decisions than Hermes' base (Mistral-Nemo 12B,
> MMLU 68.0).

**Pre-registered 2026-09-24, before any Logos training.** Model selection and
feasibility: [logos.md](logos.md).

| | |
| --- | --- |
| **KLEOS model** | **Logos v0.0.1**: `mistralai/Ministral-3-14B-Instruct-2512-BF16` @ `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`, text tower, QLoRA NF4 r=16 on all seven projections |
| **Config** | [`configs/training/kleos_logos_v001.yaml`](../configs/training/kleos_logos_v001.yaml), `config_hash` **`18008c6716a58afc284c64eb7e1e96c9bfce34b8da2b8c489b6d59c0c45b6f71`** with `--dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output-dir /content/drive/MyDrive/kleos-private/outputs` and no `KLEOS_*` variables set. *[editor's note: the literal `--dataset` and `--output-dir` paths are inputs to the recorded `config_hash` and are kept for that reason.]* |
| **Experiment id** | `kleos-v006-ministral314b-run1` |
| **Dataset** | `kleos-policy-v0.0.6`, sealed, unchanged |
| **Benchmark** | `benchmark.jsonl`, sha256 `a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266`, built from `test.jsonl` (`a4decaaf029b2273…`); greedy decoding, `max_new_tokens` 512, seed 42, grader `kleos_policy`; identical to Hermes' |
| **Code** | The commit that adds this entry. The run's manifest records the commit it ran; a later code change must be declared as a deviation |
| **Status** | Trained 2026-09-27; best checkpoint `checkpoint-175` (`eval_loss` 0.03824). **H8b: inconclusive** (2026-09-29). **H8a: supported** (2026-10-01). Both below |
| **Full report** | [experiments/kleos-v006-ministral314b-run1-report.md](experiments/kleos-v006-ministral314b-run1-report.md) |

### H8a — Does fine-tuning help Logos?

| | |
| --- | --- |
| **Arms** | Logos `arm1_base_orchestrated` vs Logos `arm2_finetuned` (H1's comparison, D1) |
| **Primary metric** | Per-task `kleos_policy` score |
| **Statistics** | Paired **cluster** bootstrap by `group_id`, 2,000 iterations, 95% CI. The example-level bootstrap is reported beside it for continuity with H1 |
| **Decision rule** | A task improves only if its cluster CI excludes zero. **Supported** if at least one task improves and none regresses with a CI excluding zero |

### H8b — Is Logos better than Hermes? (primary)

| | |
| --- | --- |
| **Arms** | Hermes `arm2_finetuned` (stored, re-reported with `rescore.py --mode annotate`) vs Logos `arm2_finetuned` |
| **Primary population** | The **271 answerable** benchmark items (`reference.confident: true`, 61 groups) |
| **Primary metric** | Mean `kleos_policy` score, paired by `example_id` |
| **Statistics** | Paired cluster bootstrap by `group_id`, 2,000 iterations, 95% CI of Logos − Hermes |
| **Decision rule** | **better**: CI entirely above 0. **worse**: entirely below 0. **equivalent**: entirely within ±0.02. Otherwise **inconclusive** |
| **Command** | `compare.py --cross-model --primary-subset answerable --equivalence-margin 0.02` |

**Why the answerable subset.** The other 78 items carry four decline labels that
never occur in training (D3), so on v0.0.6 neither model can learn them and they
cannot separate two bases. They are reported as a secondary row, not dropped.

**Prediction:** unknown. Hermes and Ministral-8B scored within 0.004 of each other
overall, which is a reason to expect a small difference. Base capability rose far
more between Nemo and Ministral 3 than between Ministral-8B and Nemo, which is a
reason to expect a larger one. "Equivalent" is a publishable answer: it would say
the v0.0.6 data, not the base, is the limit.

**Secondary, reported, not decisive:** per-task deltas with cluster intervals; the
should-decline subset; consistency by `group_id` with its oracle ceiling (all
groups and answerable groups); the family-level consistency for continuity;
`format_valid`; generation statistics (length, latency, answers cut off at
`max_new_tokens`); faithfulness with the citation heuristic's gold floor.

**Declared differences from Hermes' run** (none in the training arithmetic):
the base model, tokenizer and chat template; `fix_mistral_regex: true` (0 of 1,350
examples tokenized differently); evaluation and checkpoint cadence 25 steps
instead of 50; `strict_config: true`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
the best checkpoint protected from pruning (H-F9); a gradient check at the
configured batch size and a memory probe, both before the Trainer re-seeds.

**Confounds, stated:**

- **One seed each.** H8b compares two single training runs. A difference within
  what a seed change could cause cannot be ruled out. The cluster interval covers
  evaluation noise, not training noise.
- **The bases differ in more than size:** pretraining data, distillation from
  Mistral Small 3.1, tokenizer and template, and release date.
- **Library versions.** transformers, peft, accelerate, bitsandbytes and
  tokenizers are pinned to Hermes' versions; Colab's torch may differ from Hermes'
  2.11.0.
- **The checkpoint grid.** 25-step selection can pick a checkpoint Hermes' 50-step
  grid could not have.

**What would falsify H8:** H8b's interval entirely below zero, or entirely within
±0.02.

### H8b result — 2026-09-29

**Inconclusive.** On the 271 answerable items, Logos − Hermes = **−0.0168**:

| Arm (answerable subset) | Mean | Cluster 95% CI |
| --- | --: | --- |
| Hermes `arm2_finetuned` (re-reported) | 0.8976 | 0.8634–0.9301 |
| Logos `arm2_finetuned` | 0.8808 | 0.8381–0.9191 |
| **Logos − Hermes, paired** | **−0.0168** | **−0.0546 to +0.0177** (61 groups) |

The interval spans zero and reaches beyond −0.02, so by the pre-registered rule
H8b is neither better, worse nor equivalent. H8's central claim, that a stronger
base makes a better KLEOS model, is neither supported nor falsified.

- **What the interval does say:** its upper end, +0.0177, is below +0.02. At the
  95% level these data do not support a Logos advantage as large as the
  equivalence margin. A Hermes lead of up to about 0.055 is not ruled out either.
- **The confounds above still apply:** one training seed per model, so training
  noise isn't in the interval.

**Secondary rows. They are reported, not decisive, and the seven tasks are not
corrected for multiple comparisons.**

- **Should-decline subset** (78 items, 17 groups): −0.0111, cluster CI −0.0429 to
  +0.0195.
- **`recommendation_generation`** (n = 41): 0.4946 → 0.3906 (−0.1040). Regressed in
  both bootstraps: examples p < 0.0005, groups p = 0.006.
- **`workspace_reasoning`** (n = 46): −0.0142. Regressed by groups (p = 0.014),
  not by examples.
- **`notification_prioritization`** (n = 38): +0.0837. Improved by examples
  (p = 0.007), not by groups.
- **Every other task:** within ±0.021 and not significant in either bootstrap.
- **Consistency by `group_id`,** oracle 1.000: Hermes 0.769 → Logos 0.833 over 78
  groups, and 0.770 → 0.853 over the 61 answerable groups. No interval is
  computed for it.
- **Overall means** (descriptive): Hermes 0.8051, Logos 0.7896.

**Provenance:**

- **Hermes' side:** read from its frozen results (`arm2_finetuned.json`, sha256
  `428400f5…`, as registered in its report). `rescore.py --mode annotate` wrote a
  separate `.annotated.json`; stored scores reproduced with drift 0, and the
  source is unmodified.
- **Logos' side:** one uninterrupted evaluation pass, with 0 generations replayed
  (D10).
- **Both:** the same benchmark, sha256 `a11ffad7…`.
- **Report:** `outputs/report_h8b_hermes_vs_logos/summary.md`.

### H8a result — 2026-10-01

**Supported.** By the pre-registered rule, a task improves only if its cluster
interval excludes zero. 5 of 7 tasks improved and none regressed:

| Task | n | Logos base | Logos fine-tuned | Δ | By groups | By examples |
| --- | --: | --: | --: | --: | --- | --- |
| `memory_conflict_resolution` | 111 | 0.3763 | 0.8350 | +0.4586 | improved, p < 0.0005 | improved, p < 0.0005 |
| `mission_control_briefing` | 43 | 0.5552 | 0.9646 | +0.4094 | improved, p < 0.0005 | improved, p < 0.0005 |
| `workspace_reasoning` | 46 | 0.5366 | 0.9370 | +0.4005 | improved, p < 0.0005 | improved, p < 0.0005 |
| `notification_prioritization` | 38 | 0.5888 | 0.8942 | +0.3054 | improved, p < 0.0005 | improved, p < 0.0005 |
| `tool_routing` | 62 | 0.4021 | 0.6501 | +0.2479 | improved, p < 0.0005 | improved, p < 0.0005 |
| `context_prioritization` | 8 | 0.6204 | 0.9995 | +0.3792 | not estimable (8 items) | improved, p < 0.0005 |
| `recommendation_generation` | 41 | 0.3263 | 0.3906 | +0.0643 | not significant | not significant |
| **Overall** (descriptive) | 349 | **0.4469** | **0.7896** | +0.3427 | | |

- **Answerable subset** (271 items, 61 groups): 0.4857 → 0.8808, **+0.3952**
  (cluster CI +0.3443 to +0.4438).
- **Should-decline subset** (78 items, 17 groups): 0.3123 → 0.4724, +0.1601
  (cluster CI +0.0712 to +0.2409). The four decline labels never occur in
  training (D3), so this gain comes from the other parts of the score.
- **Consistency by `group_id`,** oracle 1.000: 0.154 → 0.833, and 0.164 → 0.853 on
  the answerable groups.
- **Counting:** `compare.py`'s closing line counts "6 significant" by the example
  bootstrap. The pre-registered count uses group intervals, so it is 5.

**Against Hermes' H1 (descriptive; different bases):**

- **Hermes improved all seven tasks**, including `recommendation_generation`
  (+0.1971: 0.2974 → 0.4946).
- **Logos' base starts higher on that task (0.3263) but ends lower (0.3906).**
  This is the one task where fine-tuning did markedly less for Logos, and it is
  the task behind Logos' secondary regression in H8b.
- **Base Logos scores 0.3123 on the should-decline items, against base Hermes'
  0.1923.**

**Provenance:**

- **`arm1_base_orchestrated`:** ran at `ed5a987` (clean), the declared D11
  commit, across two sessions. The second session replayed 172 recorded
  generations and generated 177.
- **`arm2_finetuned`:** ran at `ec4f9e3` in one pass (D10).
- **Both:** the same benchmark, sha256 `a11ffad7…`.
- **Report:** `outputs/kleos-v006-ministral314b-run1/report_logos_v001/summary.md`.

---

## H9 — Does a Logos trained to think beat Hermes?

> Logos v0.0.2, Ministral 3 14B Reasoning fine-tuned to write the KLEOS
> policies' own reasoning before it answers, makes better KLEOS decisions than
> KLEOS Hermes.

**Pre-registered 2026-10-06, before any Logos v0.0.2 training.** Design and
runbook: [logos.md](logos.md#what-changed-in-v002), [runbooks/logos-v002-kaggle.md](runbooks/logos-v002-kaggle.md).

| | |
| --- | --- |
| **KLEOS model** | **Logos v0.0.2**: `mistralai/Ministral-3-14B-Reasoning-2512` @ `51f9210f3cd20f3452a80d5819d15dc61cc50630`, text tower, QLoRA NF4 r=16 on all seven projections (280 modules, 60,948,480 trainable parameters) |
| **Config** | [`configs/training/kleos_logos_v002.yaml`](../configs/training/kleos_logos_v002.yaml), `config_hash` **`d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9`** with `--dataset /tmp/kleos-data/kleos-policy-v0.0.7 --output-dir /kaggle/working/outputs` and no `KLEOS_*` variables set. *[editor's note: the literal `--dataset` and `--output-dir` paths are inputs to the recorded `config_hash` and are kept for that reason.]* |
| **Experiment id** | `kleos-v007-ministral314breasoning-run1` |
| **Dataset** | `kleos-policy-v0.0.7`, sealed (release content hash `b53afa4216bf6973…`). Train and validation answers carry a policy-derived reasoning trace (schema 1.1) and a "What decided it" line; `test.jsonl` is byte-identical to v0.0.6's (`a4decaaf029b2273…`) |
| **Benchmark** | The H8 file, unchanged: sha256 `a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266`; greedy decoding, seed 42, grader `kleos_policy`, **`max_new_tokens` 1024**. Each completion is split at `[/THINK]` (token 35) and only the answer after it is graded |
| **Hardware** | Kaggle, 2 × Tesla T4, the model's layers spread over both by `device_map: auto` (model parallel), through [the Kaggle notebooks](../notebooks/kaggle/). The dataset is copied to `/tmp`, which Kaggle never saves |
| **Code** | The commit that adds this entry. The run's manifest records the commit it ran; a later code change must be declared as a deviation |
| **Status** | Trained 2026-10-06 on Kaggle (4.4 h, one session); best checkpoint `checkpoint-175` (`eval_loss` 0.0293). Evaluated 2026-10-06. **H9: better** (below) |
| **Full report** | [experiments/kleos-v007-ministral314breasoning-run1-report.md](experiments/kleos-v007-ministral314breasoning-run1-report.md) |

### H9 — Is Logos v0.0.2 better than Hermes? (primary)

| | |
| --- | --- |
| **Arms** | Hermes `arm2_finetuned` (stored, re-reported with `rescore.py --mode annotate`, as for H8b) vs Logos v0.0.2 `arm2_finetuned` |
| **Primary population** | The **271 answerable** benchmark items (`reference.confident: true`, 61 groups), as in H8b |
| **Primary metric** | Mean `kleos_policy` score, paired by `example_id` |
| **Statistics** | Paired cluster bootstrap by `group_id`, 2,000 iterations, 95% CI of Logos v0.0.2 − Hermes |
| **Decision rule** | **better**: CI entirely above 0. **worse**: entirely below 0. **equivalent**: entirely within ±0.02. Otherwise **inconclusive** |
| **Command** | `compare.py --cross-model --primary-subset answerable --equivalence-margin 0.02` |

**Why the answerable subset stays primary.** It is H8b's population, so H9 and
H8b read on the same scale. v0.0.7 trains the four decline labels for the first
time, so the 78 should-decline items are the row the new data was built to move.
Making them primary now, knowing that, would pick the population most likely to
flatter the result. They are reported as the first secondary row.

**No arm1.** H9 does not ask whether fine-tuning helps; H8a answered that for
this architecture. An arm costs about 7 GPU hours (ESTIMATED) of Kaggle's 30-hour
weekly quota. Fine-tuning's own effect on v0.0.2 is therefore not measured.

**Prediction:**

- **Should-decline subset:** likely to improve, because its labels are now trained.
- **Answerable subset (primary): unknown.** H8b found no effect of a stronger base
  on v0.0.6. The traces add an explicit derivation of every decision, which may
  help, or may only restate what the answers already taught.
- **"Inconclusive" or "equivalent" is a publishable answer.** It would say the
  reasoning traces did not move the decisions the benchmark measures.

**Secondary, reported, not decisive:**

- the should-decline subset and all 349 items;
- Logos v0.0.2 vs Logos v0.0.1 `arm2_finetuned`, from v0.0.1's stored results;
- per-task deltas with cluster intervals;
- `generation_stats.thinking_truncated` (answers whose thinking never closed), the
  trace length distribution, completion tokens and latency (thinking costs time);
- `format_valid`, consistency by `group_id` with its oracle ceiling, and
  faithfulness with the citation heuristic's gold floor.

**Declared differences from Logos v0.0.1's run.** The training section is
v0.0.1's, value for value (`tests/test_logos_v002_config.py`). What differs:

- the base: the Reasoning release, with its own tokenizer files and chat template;
  `fix_mistral_regex: true` changes 0 of 1,350 v0.0.7 examples (measured);
- the data: kleos-policy-v0.0.7, with the traces supervised as the model's thinking.
  The longest example is 736 tokens against 448, and a run that would cut any
  supervised token refuses to start;
- the evaluation's `max_new_tokens`, 1024 instead of 512;
- the hardware: Kaggle's 2 × T4 instead of Colab's single T4, with the model spread
  over both GPUs, a per-GPU memory probe and a refusal to train a split model under
  `DataParallel`. One T4 does not hold the longest example at these settings
  (estimated 14.46 GB peak against a 14.41 GB budget; 2 × T4: 6.38 and 8.53 GB);
- **what selects the adapter.** The best checkpoint is still the one with the lowest
  validation loss (`eval_loss`), but that loss now covers trace and answer together,
  and the trace is 58% of the validation targets by characters (575 of 988 per
  answer, on average). Selection therefore rewards predicting the trace more than
  the answer, where v0.0.1's rewarded the answer alone. No answer-only loss is
  recorded;
- results schema 3, which stores each answer's trace and `finish_reason`.

**Confounds, stated:**

- **Two changes at once.** Against Hermes, both the base and the data differ.
  Hermes is not retrained on v0.0.7 (decided 2026-10-06), so H9 can say whether
  Logos v0.0.2 as built beats Hermes as shipped, but not which change did it. The
  v0.0.1 comparison changes the base release and the data together too.
- **The thinking budget.** Logos v0.0.2 gets 1,024 new tokens where Hermes had 512,
  because its thinking spends them. A completion whose thinking never closes has
  no answer and scores as given; how many did is reported.
- **Greedy decoding on a reasoning model.** Mistral's model card recommends
  sampling at temperature 1 for this release. Greedy decoding is kept so every KLEOS evaluation is deterministic and
  comparable. A loop it causes shows as `thinking_truncated` or
  `hit_max_new_tokens`.
- **One seed each.** The cluster interval covers evaluation noise, not training
  noise.
- **Hardware and libraries.** transformers, peft, accelerate, bitsandbytes and
  tokenizers are pinned as before; Kaggle's torch may differ from Colab's. The
  split across two GPUs changes the order of floating-point work, not the
  arithmetic.
- **What the traces are.** They are written from the same policies that define the
  benchmark's reference answers, as every KLEOS training answer is. H9 measures how
  well a model learns those policies, not general reasoning.

**What would falsify H9:** the interval entirely below zero, or entirely within
±0.02.

### H9 result — 2026-10-06

**Better.** On the 271 answerable items, Logos v0.0.2 − Hermes = **+0.0409**:

| Arm (answerable subset) | Mean | Cluster 95% CI |
| --- | --: | --- |
| Hermes `arm2_finetuned` (annotated) | 0.8976 | 0.8634–0.9301 |
| Logos v0.0.2 `arm2_finetuned` | 0.9385 | 0.9083–0.9663 |
| **Logos v0.0.2 − Hermes, paired** | **+0.0409** | **+0.0040 to +0.0780** (61 groups, p = 0.031) |

The interval lies entirely above zero, so by the pre-registered rule H9 is
**supported**: Logos v0.0.2 makes better KLEOS decisions than Hermes as shipped.

**How to read it:**

- **The gain is real but its size is uncertain.** The lower end, +0.004, is close
  to zero, and each model is a single training run, so training noise is not in
  the interval.
- **It does not say which change did it.** The base model and the data changed
  together, and Hermes was not retrained on v0.0.7.

**Secondary rows. They are reported, not decisive, and the seven tasks are not
corrected for multiple comparisons.**

- **Should-decline subset** (78 items, 17 groups): 0.4835 → 0.5855 (+0.1020).
  Cluster CI −0.0018 to +0.2062: not significant by groups.
- **Per task, by groups:**
  - **improved:** `notification_prioritization` +0.1324 (p < 0.0005),
    `recommendation_generation` +0.0893 (p = 0.002) and
    `memory_conflict_resolution` +0.0737 (p = 0.048);
  - **not significant:** `tool_routing` +0.0330, `workspace_reasoning` +0.0172 and
    `mission_control_briefing` −0.0153;
  - **not estimable by groups:** `context_prioritization` (8 items).
- **Logos v0.0.1 → v0.0.2** (same benchmark; also changes the base release and
  the data together):
  - answerable +0.0577 (cluster CI +0.0234 to +0.0952);
  - should-decline +0.1131 (+0.0028 to +0.2296, p = 0.046);
  - no task regressed.
- **Thinking:**
  - 349 of 349 answers carried a trace, averaging 549 characters;
  - none ran out of budget while thinking (`thinking_truncated` 0);
  - 6 answers looped after the trace until the 1,024-token budget ran out. They
    were scored as given (L-F7 in the report).
- **Consistency by `group_id`,** oracle 1.000: 0.769 → 0.833 over all groups, and
  0.770 → 0.869 over the answerable groups.
- **`format_valid`:** 0.0 for all three models, unchanged (D2, D6).
- **Overall means** (descriptive): Hermes 0.8051, Logos v0.0.1 0.7896, Logos
  v0.0.2 0.8596.

**Provenance:**

- **Logos v0.0.2's side:**
  - one uninterrupted pass at `a17ace7`, the pre-registration commit, with 0
    generations replayed;
  - `config_hash` matched before training;
  - the results file's `benchmark_fingerprint` was verified after download.
- **Hermes' side:** its annotated results, read-only (sha256 `99fdcdc3…`).
- **Both:** the same benchmark, sha256 `a11ffad7…`.
- **Reports:** `outputs/report_h9_hermes_vs_logos_v002/summary.md` and
  `outputs/report_h9_logos_v001_vs_v002/summary.md`.
- **Deviations:** D12 and D13.

---

## Fixed experimental protocol

Applies to every hypothesis above.

| Element | Commitment |
| --- | --- |
| Split | Held-out strategy, never `random` |
| Decoding | Greedy (`do_sample: false`), identical across arms |
| Benchmark | Identical across arms, fixed before the run |
| Graders | Fixed before the run |
| Significance | Paired bootstrap, 2000 iterations, 95% CI. From H8 on, resampling groups (see amendment) |
| Reporting | Per task; no blended aggregate |
| Seeds | Recorded; multiple seeds only meaningful with sampling |
| Leakage | Checked and reported before training |

### Protocol amendment — clustered intervals, 2026-09-24

**From H8 on, every interval resamples groups (`group_id`), not examples.** The
349 benchmark items come in 78 groups of perturbations of one case, and those are
not independent. An example-level bootstrap treats them as 349 independent draws
and gives intervals that are too narrow (finding H-F14). The example-level
intervals are still printed beside the cluster ones, so every earlier number stays
reproducible. **No earlier result is re-decided**: H1 and H3 keep their
pre-registered decision rules and their reported values. Corrected figures for
the earlier runs, if produced, are labelled as re-reports.

Consistency is reported by `group_id` beside the configured family grouping, each
with the score an oracle would get (H-F11). The answerable and should-decline
subsets are reported separately.

## Recording a run

```bash
python scripts/run_experiment.py --config configs/training/qlora_small.yaml \
                                 --dataset <path to release>
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
| 2026-09-22 | H1 | Mistral-Nemo-Instruct-2407 (QLoRA r=16) — **KLEOS Hermes** | kleos-policy-v0.0.6 (`3cc9a744…`) | `b2328857c6026dd7` | **Supported (replicated).** 7/7 tasks improved at p<0.001, none regressed. Overall 0.4755 → 0.8051 (+0.3295, 95% CI 0.3068–0.3521). | `outputs/kleos-v006-mistralnemo12b-run1/report_hermes_v006/summary.md`; [report](experiments/kleos-v006-mistralnemo12b-run1-report.md) |
| 2026-09-22 | H2 | Mistral-Nemo-Instruct-2407 (QLoRA r=16) — **KLEOS Hermes** | kleos-policy-v0.0.6 (`3cc9a744…`) | `b2328857c6026dd7` | **Not measurable.** Same benchmark, 100% OOD. | same run |
| 2026-09-22 | H3 | Mistral-Nemo-Instruct-2407 (QLoRA r=16) — **KLEOS Hermes** | kleos-policy-v0.0.6 (`3cc9a744…`) | `b2328857c6026dd7` | **Improved, still poor — identical to Ministral.** correct_agreement 0.000 → 0.333; 10/15 groups still flip; abstention table identical case for case. | same run |
| 2026-09-24 | H8 | Ministral-3-14B-Instruct-2512-BF16, text tower (QLoRA r=16) — **KLEOS Logos v0.0.1** | kleos-policy-v0.0.6 (`3cc9a744…`) | `18008c6716a58afc` | **Pre-registered; not run.** | [logos.md](logos.md) |
| 2026-09-29 | H8b | **KLEOS Logos v0.0.1** vs **KLEOS Hermes**, both `arm2_finetuned` | kleos-policy-v0.0.6 (`3cc9a744…`) | `18008c6716a58afc` (Logos) | **Inconclusive.** Answerable subset: Logos − Hermes −0.0168, cluster 95% CI −0.0546 to +0.0177 (271 items, 61 groups). Neither better, worse nor equivalent at ±0.02. | `outputs/report_h8b_hermes_vs_logos/summary.md`; [H8b result](#h8b-result--2026-09-29) |
| 2026-10-01 | H8a | Ministral-3-14B-Instruct-2512-BF16, text tower (QLoRA r=16) — **KLEOS Logos v0.0.1**, `arm1_base_orchestrated` vs `arm2_finetuned` | kleos-policy-v0.0.6 (`3cc9a744…`) | `18008c6716a58afc` | **Supported.** 5/7 tasks improved by group intervals; none regressed. `recommendation_generation` +0.0643 n.s.; `context_prioritization` not estimable by groups (8 items). Overall 0.4469 → 0.7896; answerable +0.3952 (cluster CI +0.3443 to +0.4438). | `outputs/kleos-v006-ministral314b-run1/report_logos_v001/summary.md`; [H8a result](#h8a-result--2026-10-01) |
| 2026-10-06 | H9 | Ministral-3-14B-Reasoning-2512, text tower (QLoRA r=16) — **KLEOS Logos v0.0.2** | kleos-policy-v0.0.7 (`b53afa42…`) | `d1961583546b5761` | **Pre-registered; not run.** | [H9](#h9--does-a-logos-trained-to-think-beat-hermes) |
| 2026-10-06 | H9 | **KLEOS Logos v0.0.2** vs **KLEOS Hermes**, both `arm2_finetuned` | kleos-policy-v0.0.7 (`b53afa42…`) | `d1961583546b5761` (Logos) | **Supported: better.** Answerable subset: Logos − Hermes +0.0409, cluster 95% CI +0.0040 to +0.0780 (271 items, 61 groups). Overall 0.8051 → 0.8596. | `outputs/report_h9_hermes_vs_logos_v002/summary.md`; [H9 result](#h9-result--2026-10-06); [report](experiments/kleos-v007-ministral314breasoning-run1-report.md) |

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
remains unpinned and is not backfilled. **F2** (tokenizer packaging) is **also
resolved** — commit `f759aac` made the exclusion explicit by name rather than an
accident of the scan cap; the marker in the audit was added late, on 2026-09-22.

Hermes run provenance: experiment id `kleos-v006-mistralnemo12b-run1`, seed 42,
3 epochs (309 steps, effective batch 8, `max_seq_length` 1024), base pinned to
`04d8a90549d23fc6bd7f642064003592df51e9b3`, git `4cd76c42` (clean). Best
checkpoint selected on validation loss at **step 200 / epoch 1.95** (`eval_loss`
0.04343; 0.04515 by the end of epoch 3) — the same step as Ministral. The exported
adapter is byte-identical to `checkpoint-200` (SHA-256 `dc121fa3…5b857b32`).
Training took 11,396 s across two sessions with an 18-hour interruption (D8);
peak VRAM 13.09 GB of 14.56 GB, no OOM, 4 fp16-skipped optimizer steps. Tesla T4,
fp16, NF4 double-quant, `paged_adamw_8bit`; same library versions as above. Same
benchmark (`a11ffad75f5147f9…`), rebuilt and verified in each evaluation session.
Hashes, the full training record and ten findings (H-F1–H-F10) are in
[experiments/kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md).

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
| 2026-09-18 | **D8.** Hermes training was interrupted after step 250 and resumed from `checkpoint-250` eighteen hours later, on a different T4 instance. Steps 251–309 ran in the second session. | The Colab runtime died mid-save; `validate_checkpoint` rejected the half-written `checkpoint-300`. Resume state was verified faithful: gradient-check loss bit-identical across sessions, scheduler lag carried over, and the two independent evaluations of step 300 agree to 1.4 × 10⁻⁵. **No effect on the result:** the selected adapter (`checkpoint-200`) was written before the interruption. D1–D6 apply to the Hermes run unchanged; D7 does not (its base is pinned). |
| 2026-09-27 | **D9.** Logos training ran across four Colab sessions between 2026-09-24 and 2026-09-27, and resumed twice. Session 1 was disconnected after the step-125 evaluation, leaving `checkpoint-125` without weights; `validate_checkpoint` rejected it, so session 2 resumed from `checkpoint-100`. Session 2 hit the Google Drive storage quota partway through, and the free GPU usage limit ended it between steps 250 and 275. Session 3 made no progress. Session 4 resumed from `checkpoint-250` and finished. | Free-tier Colab limits. Both resume points were complete: `checkpoint-100` by its size (368 MB, a full checkpoint), and `checkpoint-250` file by file (weights, optimizer, scheduler, RNG and scaler state). **No effect on the selected adapter:** `checkpoint-175` was written in session 2. The final validation pass on the loaded best weights reproduced its `eval_loss` (0.0382), and the exported adapter is byte-identical to it (sha256 `f3e8dcdc…70a7`). `train_loss` in `metrics.json` (0.0007) is a resume artifact: transformers divides the last session's loss sum by all 309 steps. Do not quote it. `training.log` may lack killed sessions' lines (L-F3); `events.jsonl` does not. |
| 2026-09-29 | **D10.** Logos `arm2_finetuned` was evaluated in three sessions. The generations of the first two never reached Drive; the third generated all 349 in one pass (`resume`: 0 replayed, 349 new). | Finding L-F3 ([experiments/logos-findings.md](experiments/logos-findings.md)): the partial file was held open, and Colab's Drive mount uploads only closed files. **No effect on the result**, which is a single uninterrupted run at the pre-registered commit. |
| 2026-09-29 | **D11. Declared before it runs:** Logos `arm1_base_orchestrated` will run at a commit later than H8's, which contains the L-F3 fix. The fix opens and closes the partial file for every record instead of holding one handle for the run. | An arm takes 2–3 hours, longer than a free session reliably lasts, and without the fix a killed session loses all its work. The change touches file handling only: generation, prompts, grading, configs and the benchmark are unchanged. The arm's resume identity records the commit it ran. **Done as declared, 2026-10-01:** the arm ran at `ed5a987` (clean), and the second of its two sessions resumed 172 recorded generations (L-F3 confirmed fixed). |
| 2026-10-06 | **D12.** Both Kaggle notebooks ran on Kaggle's "Latest Container Image"; the runbook's "Pin to original environment" was not set. | The setting matters only when a run is resumed in a later version, which neither was: training finished in one session and the evaluation in one pass (0 replayed). The training libraries are installed at pinned versions over the image either way; Kaggle's torch is recorded in the run's `environment.txt`. **No effect on the result.** |
| 2026-10-06 | **D13.** The training run is Version 2 of its Kaggle notebook. Version 1 stopped in its first cell, because `COMMIT` still held its placeholder. The evaluation result is Version 2 of its notebook. | Version 1 ran no code beyond the guard that refused it: no data was copied and no model was loaded. Both recorded runs used commit `a17ace7`, the pre-registration commit, and the training run's `config_hash` matched before training. **No effect on the result.** |

---

## Base-model revision

*[editor's note: this section and the next, to the end of the document, are kept
as written from the pre-registration, including their first-person wording.]*

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
