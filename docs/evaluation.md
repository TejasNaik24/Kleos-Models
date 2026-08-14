# Evaluation

## Quick start

```bash
python scripts/evaluate.py --config <config> --arm arm0_base       --output outputs/base.json
python scripts/evaluate.py --config <config> --arm arm2_finetuned  --adapter <path> --output outputs/ft.json
python scripts/compare.py  --base outputs/base.json --finetuned outputs/ft.json --report reports/exp-001
```

## The principle

**Conditions are held identical across arms.** Same benchmark, same decoding, same
graders, same seeds. The runner takes the arm as a parameter and changes nothing
else, so `arm0_base` vs `arm2_finetuned` differs only by the adapter.

`compare.py` checks this and emits a comparability warning when it is violated —
different benchmarks, different decoding settings, a missing adapter path, or a
different base model.

## Research arms

| Arm | Model | Orchestration |
| --- | --- | --- |
| `arm0_base` | base | none |
| `arm1_base_orchestrated` | base | KLEOS scaffolding |
| `arm2_finetuned` | base + LoRA | controlled task context |
| `arm3_finetuned_orchestrated` | base + LoRA | KLEOS scaffolding |
| `frontier_zero_shot` | frontier | **interface only** |
| `frontier_orchestrated` | frontier | **interface only** |

arm0 vs arm2 isolates the fine-tuning effect. arm0 vs arm1 isolates orchestration.
Running all four separates them and shows whether they interact.

Frontier arms exist in the design so a provider can be added later by implementing
one `Backend` method — no changes to tasks, graders or the runner.

## Metrics

Judgment and correctness, not tone.

| Metric | Use |
| --- | --- |
| `exact_match` | short exact answers |
| `token_f1` | partial textual overlap |
| accuracy, macro/micro P/R/F1 | classification |
| `ndcg` | ranking quality |
| `kendall_tau` | pairwise ordering agreement |
| `spearman_footrule` | how far items moved |
| `top_1_accuracy` | did it get the most important item right |
| set precision/recall/F1 | selection tasks |

Macro-F1 is reported alongside accuracy because KLEOS label distributions are
imbalanced — most notifications are not urgent — and accuracy alone would let a
majority-class predictor look competent.

## Graders

| Grader | For | Reference |
| --- | --- | --- |
| `exact_match` | short answers | `text` |
| `classification` | single-label decisions | `label`, `options` |
| `set_match` | item selection | `items` |
| `ranking` | prioritization | `ranking` |
| `heuristic_rubric` | open-ended, offline | `required_points`, `evidence_ids`, … |
| `llm_judge` | open-ended | **opt-in**, requires an explicit judge callable |

Graders parse responses tolerantly — fenced JSON, bare JSON, numbered lists,
bulleted lists, or mention order. Models wrap answers in prose far more often than
they emit clean JSON, and treating that as a wrong answer would measure format
compliance rather than judgment.

### `heuristic_rubric`

Deliberately coarse and completely offline. It checks structural properties:
required points covered, evidence cited, unsupported-claim phrasings absent, an
actionable recommendation present.

It is **not** a substitute for human or LLM judging on open-ended quality. It
exists so rubric-graded tasks have a reproducible, zero-cost baseline that runs in
CI and cannot drift.

### `llm_judge`

Interface only — supply a `judge_fn`. The judge model id is recorded in every
result, because scores from different judges are not comparable and must never be
pooled.

## Consistency (spec §22)

The clearest available test of *policy versus surface pattern*.

Group examples that are logically equivalent — same situation, differing only in
wording, evidence order, formatting, or irrelevant additions — and check whether
the model reaches the same decision.

Two numbers, reported separately on purpose:

| Metric | Meaning |
| --- | --- |
| `agreement_rate` | the group reached one decision |
| `correct_agreement_rate` | it agreed **and** was right |

A model that is consistently wrong scores 1.0 on the first and 0.0 on the second.
Collapsing them would hide that.

Flips are attributed to perturbation kind, showing which axis the model is fragile
along. Requires `metadata.scenario_family` on the perturbed examples.

## Out-of-distribution (spec §23)

Three numbers, **never blended**:

```
in_distribution_score
ood_score
generalization_gap
```

Plus a per-shift breakdown: unseen entities, unseen domains, unseen formats,
unseen source types, context-length shift, reordered evidence, conflicting
evidence.

There is deliberately no "overall OOD" field. A single number would hide the case
that matters most — improving in-distribution while degrading OOD, which is
consistent with fitting surface features rather than learning a policy.

**Without OOD-tagged examples, OOD is not measurable and no generalization claim
may be made.** The report says so explicitly rather than omitting the section.

## Faithfulness

Is the answer grounded in the supplied context?

| Metric | Checks |
| --- | --- |
| `evidence_coverage` | decisive evidence actually used |
| `citation_precision` | cited evidence ids exist |
| `unsupported_claim_rate` | specific claims absent from the context |

These are text-level heuristics, not entailment checks. They detect claims and
citations that do not appear in the supplied context; they do not verify semantic
support. Stated as such wherever they are reported.

## Capability preservation (spec §24)

Fine-tuning on a narrow distribution can damage general ability. A small **fixed**
suite runs identically before and after:

```
base_score  fine_tuned_score  delta  relative_delta
```

The suite must stay constant across experiments — consistency matters more than
coverage here. Its version is recorded in every result.

A task gain paired with a capability loss is a trade-off, not an improvement.

## Comparison

```bash
python scripts/compare.py --base outputs/base.json --finetuned outputs/ft.json \
                          --report reports/exp-001
```

Output: per-task base, fine-tuned, absolute delta, relative delta, verdict — plus
OOD, consistency, faithfulness and capability deltas.

**No blended aggregate** unless you pass `--show-aggregate`. A single number across
tasks hides exactly the per-task trade-offs that matter.

Deltas are paired per `example_id` and accompanied by a bootstrap confidence
interval, so an improvement that is not statistically significant is reported as
*not significant* rather than as a win.

### Reading the conclusion

| Pattern | Reported as |
| --- | --- |
| some tasks up, some down | **MIXED** — possibly the most interesting outcome |
| all down | regression — a valid, publishable result |
| up but CI includes zero | suggestive, **not demonstrated** |
| up with CI excluding zero | improvement — check OOD and capability before generalizing |
| no change | fine-tuning neither helped nor harmed |

## Testing the harness without a GPU

```bash
python scripts/evaluate.py --config <config> --echo
```

`EchoBackend` returns canned responses, exercising metrics, graders, consistency,
OOD and reporting end to end with no model. Its `describe()` states that the
results mean nothing about any model — it validates the plumbing, not a model.
