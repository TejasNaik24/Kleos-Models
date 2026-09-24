# Evaluation

## Quick start

```bash
python scripts/evaluate.py --config <config> --arm arm0_base       --output outputs/base.json
python scripts/evaluate.py --config <config> --arm arm2_finetuned  --adapter <path> --output outputs/ft.json
python scripts/compare.py  --base outputs/base.json --finetuned outputs/ft.json --report reports/exp-001
```

A dropped runtime costs one example, not the arm: re-run the same `evaluate.py`
command with `--resume` (see "Resuming an evaluation" below).

## The principle

**Conditions are held identical across arms.** Same benchmark, same decoding, same
graders, same seeds. The runner takes the arm as a parameter and changes nothing
else, so `arm0_base` vs `arm2_finetuned` differs only by the adapter.

`compare.py` checks this and emits a comparability warning when it is violated —
different decoding settings, a missing adapter path, or a different base model.
Two results graded against different benchmarks are refused outright (exit code
2). Identity is established by content, never by path (finding H-F5): the
benchmark file's sha256 when both results recorded it, otherwise the targets of
the shared examples (id, task, reference decision). Colab rebuilds the benchmark
at the same path every session, so a path proves nothing either way.

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
along. The grouping key is `consistency.group_key` in the evaluation config
(`scenario_family` in the v0.0.6 configs).

### Consistency: which unit

A group is only a consistency test if every member has the **same correct
answer**. Otherwise a model that answers everything correctly still "flips", and
the metric measures the grouping, not the model.

On the v0.0.6 benchmark, scenario families fail that test: 6 of the 15 test
families mix different correct labels. A perfect model scores `agreement_rate`
0.600 (9/15) and logs 57 flips when grouped by family. `metadata.group_id` is the
real unit, perturbations of one case: 78 groups, label-identical by construction,
where the perfect model scores 1.000 (finding H-F11).

Every result therefore carries, in its `corrected.consistency` block, the
consistency by `group_id` (all groups, and answerable groups only) and by
`scenario_family`, **each beside the score an oracle would get**. An agreement rate
is readable only next to its oracle ceiling. The original `consistency` block keeps
the configured key, so earlier numbers stay reproducible.

## Subsets: answerable and should-decline

A benchmark item whose reference says `confident: false` should be declined. On
v0.0.6 those 78 items carry four decline labels that never occur in training, so
no model trained on v0.0.6 can learn them. The other 271 items are answerable.
The two subsets measure different things: base-model headroom lives in the
answerable subset. Each is reported separately (`corrected.subsets`, and a
per-subset row in every comparison), never instead of the overall score.

## Confidence intervals: clusters, not examples

The 349 benchmark items are 78 groups of perturbations of one case. Resampling
examples as if independent makes intervals too narrow (finding H-F14). So beside
the original example-level bootstrap, every mean and every paired delta also gets
a **cluster bootstrap** that resamples whole groups (`metrics.cluster_bootstrap_mean`,
`metrics.paired_cluster_bootstrap_difference`; 2,000 iterations, seed 42). With
fewer than 5 groups the interval is reported as not estimable rather than
computed. From H8 on, verdicts use the cluster interval (see the protocol
amendment in [experiments.md](experiments.md)).

A bootstrap p-value of 0 means no resample crossed zero. It is printed as
`p<0.0005` for 2,000 iterations, never as `p=0`.

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

Two of them are unreliable on the KLEOS benchmark, and are labelled so in every
report:

- **Citation precision and "fabricated citations" are UNCALIBRATED.** The
  heuristic matches any word after "evidence", "source", "doc", "item" or "ref",
  and it flags 47.9% of the gold test answers (finding H-F12). Read a model's count
  only against that gold floor: `scripts/rescore.py --mode annotate --gold-targets
  <split files>` runs the same heuristic over the reference answers and records the
  counts.
- **Evidence coverage is vacuous** when no benchmark item names decisive
  `evidence_ids`: every response then scores 1.0 by definition (H-F13). Reports say
  "vacuous" instead of presenting the 1.0.

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
*not significant* rather than as a win. Each row shows two verdicts: resampling
examples (the original method) and resampling groups (the "cluster verdict").

### Comparing two models

```bash
python scripts/compare.py --cross-model \
    --base hermes_arm2.annotated.json --finetuned logos_arm2.json \
    --primary-subset answerable --equivalence-margin 0.02
```

`--cross-model` compares two models on the same arm; different base models are
then the point, not a warning. `--primary-subset` names the pre-registered primary
comparison, decided on that subset's cluster interval:

| Interval (right − left) | Verdict |
| --- | --- |
| entirely above 0 | **better** |
| entirely below 0 | **worse** |
| entirely within ±margin | **equivalent** |
| anything else | **inconclusive** |

Exit codes: 0 compared; 2 not comparable; 1 only with `--fail-on-regression`, when
the comparison regressed.

### Results written before these fields existed

`scripts/rescore.py --mode annotate` adds the corrected measures to a stored result
without changing any original field: per-record `group_id`, `subset` and grader
`details` (joined from the benchmark), the benchmark's sha256, `generation_stats`
and the `corrected` block. It first re-grades every stored response and stops if
any stored score does not reproduce. The input file is only read. `--summary`
writes an original-vs-corrected table.

### Reading the conclusion

| Pattern | Reported as |
| --- | --- |
| some tasks up, some down | **MIXED** — possibly the most interesting outcome |
| all down | regression — a valid, publishable result |
| up but CI includes zero | suggestive, **not demonstrated** |
| up with CI excluding zero | improvement — check OOD and capability before generalizing |
| no change | fine-tuning neither helped nor harmed |

## What every result records

Results are schema version 2. The version-1 fields keep their meaning and values;
version 2 adds:

| Field | Contents |
| --- | --- |
| `benchmark_sha256`, `benchmark_fingerprint` | the benchmark file's hash, and a hash of the graded targets |
| per-record `group_id`, `subset`, `details` | the grouping facts, and what the grader extracted: predicted ranking, label and confidence (finding H-F7) |
| `generation_stats` | completion and prompt tokens, latency, and how many answers used the whole `max_new_tokens` budget (probably cut off) |
| `corrected` | the corrected measures above |
| `resource_usage` | model load time, wall time, peak VRAM, GPU |
| `resume` | the run's identity hash, and how many generations were replayed |

Results are written atomically: a runtime that dies mid-write leaves the previous
file or the new one, never half of one.

## Resuming an evaluation

An arm of the KLEOS benchmark takes 2–3 hours on a free T4. `evaluate.py` appends
every finished generation to `<output>.partial.jsonl` and flushes it to disk. After
a disconnect, run the same command with `--resume`:

- Recorded generations are replayed; only the rest is generated. Decoding is
  greedy, so the result is the result of an uninterrupted run. Grading always
  reruns over everything.
- The partial file starts with an **identity**: arm, seeds, example ids, the
  benchmark's sha256, decoding settings, the orchestration prompt, the model config,
  the adapter's weight hashes, the git commit, library versions and the GPU's
  compute capability. A resume under any other identity is refused.
- A torn last line (the runtime died mid-write) is dropped and its example
  generated again.
- A finished result is never overwritten: `--resume` on a finished result with the
  same identity does nothing; anything else needs `--overwrite`.

## Testing the harness without a GPU

```bash
python scripts/evaluate.py --config <config> --echo
```

`EchoBackend` returns canned responses, exercising metrics, graders, consistency,
OOD and reporting end to end with no model. Its `describe()` states that the
results mean nothing about any model — it validates the plumbing, not a model.
