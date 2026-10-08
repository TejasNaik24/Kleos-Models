# Evaluation

## Quick start

```bash
python scripts/evaluate.py --config <config> --arm arm1_base_orchestrated --output outputs/arm1.json
python scripts/evaluate.py --config <config> --arm arm2_finetuned --adapter <path> --output outputs/arm2.json
python scripts/compare.py  --base outputs/arm1.json --finetuned outputs/arm2.json --report reports/exp-001
```

A dropped runtime costs at most the answer being generated: run the same
`evaluate.py` command again with `--resume` ([Resuming an evaluation](#resuming-an-evaluation)).

## The principle

Conditions are held identical across arms: same benchmark, same decoding, same
graders, same seeds. The runner takes the arm as a parameter and changes nothing
else.

`compare.py` checks this and warns when it is violated (different decoding
settings, a missing adapter path, a different base model). Two results graded
against different benchmarks are refused outright (exit code 2). Identity is
established by content, never by path (finding H-F5 in the
[Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)):
the benchmark file's sha256 when both results recorded it, otherwise the targets
of the shared examples (id, task, reference decision). Colab rebuilds the
benchmark at the same path every session, so a path proves nothing either way.
The findings and decisions referenced on this page are indexed in
[experiments/README.md](experiments/README.md).

## Research arms

| Arm | Model | Orchestration |
| --- | --- | --- |
| `arm0_base` | base | none |
| `arm1_base_orchestrated` | base | KLEOS scaffolding |
| `arm2_finetuned` | base + LoRA | controlled task context |
| `arm3_finetuned_orchestrated` | base + LoRA | KLEOS scaffolding |
| `frontier_zero_shot` | frontier | interface only |
| `frontier_orchestrated` | frontier | interface only |

arm0 against arm2 isolates the fine-tuning effect; arm0 against arm1 isolates
orchestration; all four together show whether they interact.

The Ministral-8B, Hermes and Logos v0.0.1 runs compared `arm1_base_orchestrated`
with `arm2_finetuned`, because the research question names the prompt-engineered
orchestration baseline (deviation D1 in
[experiments.md](experiments.md#deviations-log)). Logos v0.0.2 was evaluated on
`arm2_finetuned` only, for its pre-registered comparison with Hermes (H9).
`arm0_base` and `arm3_finetuned_orchestrated` have not been run, so H7 is open.

The frontier arms exist in the design so a provider can be added by implementing
`FrontierAPIBackend.generate()`; tasks, graders and the runner need no changes.
Until then the backend raises.

## The KLEOS benchmark

`scripts/build_benchmark.py` derives the benchmark from the sealed test split of a
dataset release, reproducibly. For kleos-policy-v0.0.6 and v0.0.7 (whose test
splits are byte-identical) it has 349 items over the 7 tasks, sha256
`a11ffad7…`, in 78 groups:

- **271 answerable items** (`reference.confident: true`).
- **78 should-decline items** (`reference.confident: false`). Their four decline
  labels never occur in v0.0.6's training targets, so no model trained on v0.0.6
  can learn them (deviation D3). kleos-policy-v0.0.7 trains them for the first
  time.

The release was split with `format_holdout` on `format=json`, so the whole test
split is a format-transfer probe (deviation D2).

## Metrics

The metrics measure judgment and correctness; tone is not scored.

| Metric | Use |
| --- | --- |
| `exact_match` | short exact answers |
| `token_f1` | partial textual overlap |
| accuracy, macro and micro P/R/F1 | classification |
| `ndcg` | ranking quality |
| `kendall_tau` | pairwise ordering agreement |
| `spearman_footrule` | how far items moved |
| `top_1_accuracy` | whether the most important item came first |
| set precision, recall, F1 | selection tasks |

Macro-F1 is reported beside accuracy because KLEOS label distributions are
imbalanced (most notifications are not urgent), and accuracy alone would let a
majority-class predictor look competent.

## Graders

| Grader | For | Reference keys |
| --- | --- | --- |
| `exact_match` | short answers | `text` |
| `classification` | single-label decisions | `label`, `options` |
| `set_match` | item selection | `items` |
| `ranking` | prioritization | `ranking` |
| `kleos_policy` | the KLEOS benchmark | `ranking`, `label`, `options`, `confident` |
| `heuristic_rubric` | open-ended, offline | `required_points`, `forbidden_points`, `evidence_ids`, `expected_decision` |
| `llm_judge` | open-ended | opt-in; requires an explicit judge callable |

Graders parse responses tolerantly: fenced JSON, bare JSON, numbered lists,
bulleted lists, or mention order. Models wrap answers in prose far more often
than they emit clean JSON, and scoring that as wrong would measure format
compliance rather than judgment.

### `kleos_policy`

The composite grader every KLEOS result is scored with
(`src/kleos_models/evaluation/graders.py`). It was added before the first run
(deviation D6) because the benchmark needs three judgments scored together:

| Component | Measures | How it is read |
| --- | --- | --- |
| `ranking` | the ordering | 0.6 × nDCG + 0.4 × top-1 accuracy against `reference.ranking` |
| `deciding_factor` | the stated reason | the `deciding_factor` key of a JSON answer, a "What decided it:" line, or the first allowed label mentioned; compared with `reference.label` |
| `confidence` | commit or decline | an explicit boolean in a JSON answer, otherwise a list of abstention phrasings taken from the KLEOS corpus; compared with `reference.confident` |

The score (`judgment`) is the mean of the components the reference supports.
`format_valid` (the response is a JSON object) is reported as a sub-score and
excluded from the score, so a format failure is not mistaken for a judgment
failure. The predicted and expected ranking, label and confidence are kept in each
record's `details` (finding H-F7).

### `heuristic_rubric`

Coarse and fully offline. It checks structural properties: required points
covered, evidence cited, unsupported-claim phrasings absent, an actionable
recommendation present. It is not a substitute for human or model judging of
open-ended quality; it gives rubric-graded tasks a reproducible, zero-cost
baseline that runs in CI.

### `llm_judge`

Interface only: supply a `judge_fn`. The judge model id is recorded in every
result, because scores from different judges are not comparable and must never
be pooled.

## Thinking models

A thinking model's completion is `[THINK]trace[/THINK]answer`. `[THINK]` and
`[/THINK]` are special tokens (`[/THINK]` is token 35 in the Ministral 3
Reasoning tokenizer), and decoding with special tokens skipped erases the
boundary. So `split_thinking` (`src/kleos_models/inference/backends.py`) cuts the
completion on token ids before decoding:

- The graders read only the answer. The trace is stored beside it, in the
  record's `reasoning` field.
- A completion whose thinking never closes has an empty answer, is graded as
  such, and gets `finish_reason: "length"`. `generation_stats.thinking_truncated`
  counts them; Logos v0.0.2 had 0 of 349.
- `finish_reason` reads `"stop"` whenever the trace closed, even if the answer was
  then cut by `max_new_tokens`. The only record of that cut is
  `generation_stats.hit_max_new_tokens`: 6 answers in Logos v0.0.2's evaluation
  (finding L-F7, in the
  [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings)).
  The evaluation code is unchanged; a served reply reports `"length"` in both
  cases.

A model without these tokens falls back to splitting decoded text on
`</think>`.

## Consistency

The clearest test of policy versus surface pattern: group examples that are
logically equivalent (same situation, differing only in wording, evidence order,
formatting or irrelevant additions) and check whether the model reaches the same
decision.

Two numbers, reported separately:

| Metric | Meaning |
| --- | --- |
| `agreement_rate` | the group reached one decision |
| `correct_agreement_rate` | it agreed and was right |

A model that is consistently wrong scores 1.0 on the first and 0.0 on the second.
Flips are attributed to the perturbation kind, showing which axis the model is
fragile along.

### Which unit

A group is a consistency test only if every member has the same correct answer.
Otherwise a model that answers everything correctly still "flips", and the metric
measures the grouping.

On the v0.0.6 benchmark, scenario families fail that test: 6 of the 15 test
families mix different correct labels. Grouped by family, a perfect model scores
`agreement_rate` 0.600 (9/15) and logs 57 flips. `metadata.group_id` is the real
unit, the perturbations of one case: 78 groups, label-identical by construction,
where the perfect model scores 1.000 (finding H-F11).

Every result therefore carries, in `corrected.consistency`, the consistency by
`group_id` (all groups, and answerable groups only) and by `scenario_family`, each
beside the score an oracle would get. An agreement rate is readable only next to
its oracle ceiling. The original `consistency` block keeps the configured key
(`consistency.group_key`, `scenario_family` in the v0.0.6 configs), so earlier
numbers stay reproducible.

## Subsets: answerable and should-decline

The two subsets measure different things. Base-model headroom lives in the
answerable subset; the should-decline subset measures the decline labels, which
v0.0.6 never taught. The pre-registered head-to-head comparisons (H8b, H9) are
decided on the answerable subset.
Each is reported separately (`corrected.subsets`, and a per-subset row in every
comparison), beside the overall score.

## Confidence intervals: clusters, not examples

The 349 items are 78 groups of perturbations of one case. Resampling examples as
if independent makes intervals too narrow (finding H-F14). So beside the original
example-level bootstrap, every mean and every paired difference also gets a
cluster bootstrap that resamples whole groups (`metrics.cluster_bootstrap_mean`,
`metrics.paired_cluster_bootstrap_difference`; 2,000 iterations, seed 42). With
fewer than 5 groups the interval is reported as not estimable. From H8 on,
verdicts use the cluster interval
([protocol amendment](experiments.md#protocol-amendment--clustered-intervals-2026-09-24)).

A bootstrap p-value of 0 means no resample crossed zero. It is printed as
`p<0.0005` for 2,000 iterations, never as `p=0`.

## Out-of-distribution

Three numbers, never blended:

```
in_distribution_score
ood_score
generalization_gap
```

Plus a per-shift breakdown: unseen entities, domains, formats and source types,
context-length shift, reordered evidence, conflicting evidence.

There is no "overall OOD" field. A single number would hide the case that matters
most: improving in distribution while degrading out of it, which is consistent
with fitting surface features rather than learning a policy.

Without both in-distribution and OOD-tagged examples, the gap is not measurable
and no generalization claim is made; the report says so rather than omitting the
section. The KLEOS benchmark is entirely OOD by construction (format holdout), so
H2 is recorded as not measurable.

## Faithfulness

Is the answer grounded in the supplied context?

| Metric | Checks |
| --- | --- |
| `evidence_coverage` | decisive evidence used |
| `citation_precision` | cited evidence ids exist |
| `unsupported_claim_rate` | specific claims absent from the context |

These are text-level heuristics, not entailment checks: they detect claims and
citations that do not appear in the context, and do not verify semantic support.
Two are unreliable on the KLEOS benchmark and are labeled so in every report:

- **Citation precision and "fabricated citations" are uncalibrated.** The
  heuristic matches any word after "evidence", "source", "doc", "item" or "ref",
  and flags 47.9% of the gold test answers (finding H-F12). Read a model's count
  only against that gold floor: `scripts/rescore.py --mode annotate
  --gold-targets <path to release>/test.jsonl` runs the same heuristic over the
  reference answers and records the counts.
- **Evidence coverage is vacuous** when no benchmark item names decisive
  `evidence_ids`: every response then scores 1.0 by definition (finding H-F13).
  Reports say "vacuous" instead of presenting the 1.0.

## Capability preservation

Fine-tuning on a narrow distribution can damage general ability. A small fixed
suite runs identically before and after:

```
base_score  fine_tuned_score  delta  relative_delta
```

The suite stays constant across experiments, and its version is recorded in every
result. A task gain paired with a capability loss is reported as a trade-off.
No capability suite has been run on a KLEOS model yet
(`capability.enabled: false` in the KLEOS evaluation configs), so H4 is open.

## Comparison

```bash
python scripts/compare.py --base outputs/arm1.json --finetuned outputs/arm2.json \
    --report reports/exp-001
```

Output: per task, the base and fine-tuned scores, absolute and relative
difference, and a verdict, plus OOD, consistency, faithfulness and capability
differences. There is no blended aggregate unless `--show-aggregate` is passed: a
single number across tasks hides the per-task trade-offs.

Differences are paired per `example_id` with a bootstrap confidence interval, so
an improvement that is not statistically significant is reported as not
significant. Each row shows two verdicts: resampling examples (the original
method) and resampling groups (the cluster verdict).

### Comparing two models

```bash
python scripts/compare.py --cross-model \
    --base hermes_arm2.annotated.json --finetuned logos_arm2.json \
    --primary-subset answerable --equivalence-margin 0.02
```

`--cross-model` compares two models on the same arm; a different base model is
then the point, not a warning. `--primary-subset` names the pre-registered primary
comparison, decided on that subset's cluster interval:

| Cluster interval (right minus left) | Verdict |
| --- | --- |
| entirely above 0 | better |
| entirely below 0 | worse |
| entirely within ±margin | equivalent |
| anything else, or not estimable | inconclusive |

A "better" or "worse" interval that also lies within the margin is reported with
that note. H8b was decided this way (inconclusive) and so was H9 (better).

Exit codes: 0 compared; 2 not comparable; 1 only with `--fail-on-regression`, when
the comparison regressed.

### Results written before these fields existed

`scripts/rescore.py` works on stored results, offline, without a GPU:

- `--mode regrade` (the default) re-grades every stored response with the current
  grader and writes a copy with updated scores and aggregates, for correcting a
  grader defect.
- `--mode annotate` keeps every original field and value and adds beside them
  per-record `group_id`, `subset` and grader `details` (joined from the benchmark),
  the benchmark's sha256, `generation_stats` and the `corrected` block. It first
  re-grades every response and stops if any stored score does not reproduce.
  `--summary` writes an original-versus-corrected table.

The input file is only read; the output goes to a new path.

### Reading a comparison

| Pattern | Reported as |
| --- | --- |
| some tasks up, some down | mixed |
| all down | a regression, reported like any other result |
| up, but the interval includes zero | suggestive, not demonstrated |
| up, interval excluding zero | an improvement; check OOD and capability before generalizing |
| no change | fine-tuning neither helped nor harmed |

## What every result records

Results are schema version 3 (`RESULTS_SCHEMA_VERSION` in
`src/kleos_models/evaluation/runner.py`). Versions 2 and 3 only add fields; the
version-1 fields keep their meaning and values.

| Field | Since | Contents |
| --- | --- | --- |
| `benchmark_sha256`, `benchmark_fingerprint` | 2 | the benchmark file's hash, and a hash of the graded targets |
| per-record `group_id`, `subset`, `details` | 2 | the grouping facts, and what the grader extracted: predicted ranking, label and confidence |
| `generation_stats` | 2 | completion and prompt tokens, latency, parse failures, empty responses, and how many answers used the whole `max_new_tokens` budget (`hit_max_new_tokens`) |
| `corrected` | 2 | the corrected measures: consistency by group with oracle ceilings, subsets, cluster intervals, faithfulness caveats |
| `resource_usage` | 2 | model load time, wall time, peak VRAM, GPU |
| `resume` | 2 | the run's identity hash, and how many generations were replayed |
| per-record `finish_reason` | 3 | `"length"` when a thinking model ran out of budget before closing its trace |
| per-record `reasoning` | 3 | a thinking model's trace, split off before grading |
| `generation_stats.thinking_truncated`, `reasoning_responses`, `reasoning_chars` | 3 | how many answers ran out while thinking, how many carried a trace, and the trace lengths |

`--no-responses` omits the model's text from the file; the trace is withheld with
the answer. Results are written atomically: a runtime that dies mid-write leaves
the previous file or the new one, never half of one.

## Resuming an evaluation

An arm of the KLEOS benchmark takes hours on a free GPU. Measured times for the
349 items:

| Model and arm | Hardware | Measured |
| --- | --- | --- |
| Hermes, `arm2_finetuned` | Colab, 1 × T4 | 5,634 s (about 1.6 h) |
| Hermes, `arm1_base_orchestrated` | Colab, 1 × T4 | 8,620 s (about 2.4 h; longer answers) |
| Logos v0.0.1, `arm2_finetuned` | Colab, 1 × T4 | 5,996 s of generation |
| Logos v0.0.1, `arm1_base_orchestrated` | Colab, 1 × T4 | 12,804 s of generation, over two sessions |
| Logos v0.0.2, `arm2_finetuned` | Kaggle, 2 × T4 | 18,791 s (5.2 h), 53.8 s per answer, plus 229 s to load the model |

Hermes' figures are the recorded arm durations; the Logos figures are as the run
reports state them.

`evaluate.py` appends every finished generation to `<output>.partial.jsonl`,
opening and closing the file for each record. After a disconnect, run the same
command with `--resume`:

- Recorded generations are replayed; only the rest is generated. Decoding is
  greedy, so the result equals that of an uninterrupted run. Grading always
  reruns over everything.
- The partial file starts with an identity: arm, seeds, example ids, the
  benchmark's sha256, decoding settings, the orchestration prompt, the model
  config, the adapter's weight hashes, the git commit, library versions and the
  GPU's compute capability. A resume under any other identity is refused.
- A torn last line (the runtime died mid-write) is dropped and its example
  generated again.
- A finished result is never overwritten: `--resume` on a finished result with
  the same identity does nothing; anything else needs `--overwrite`.

An earlier version held the partial file open for the whole arm, and on Colab's
Drive mount a killed session lost every generation (finding L-F3, recorded in
[experiments/logos-findings.md](experiments/logos-findings.md)). The writer has
opened and closed the file per record since commit `ed5a987`, confirmed on Colab
on 2026-10-01. `evaluate.py` prints no per-answer progress while it runs
(finding L-F8).

## Testing the harness without a GPU

```bash
python scripts/evaluate.py --config <config> --echo
```

`EchoBackend` returns canned responses, exercising metrics, graders, consistency,
OOD and reporting end to end with no model. Its `describe()` states that the
results mean nothing about any model: it validates the plumbing.
