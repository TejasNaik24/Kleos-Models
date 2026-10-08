# Evaluation configurations

| Config | Benchmark | Arms listed | Graders | `max_new_tokens` | Used by |
| --- | --- | --- | --- | --- | --- |
| `default.yaml` | `data/examples/synthetic_eval.jsonl` (synthetic fixtures) | `arm0_base`, `arm2_finetuned` | `classification`, `ranking`, `heuristic_rubric` | 512 | `debug.yaml`, `qlora_small.yaml` |
| `kleos_policy_v006.yaml` | The benchmark built from the sealed v0.0.6 test split (sha256 `a11ffad7…`) | all four | `kleos_policy` | 512 | Ministral-8B, KLEOS Hermes |
| `kleos_logos_v001.yaml` | Same | Same | Same | 512 | KLEOS Logos v0.0.1 |
| `kleos_logos_v002.yaml` | Same | Same | Same | 1024 | KLEOS Logos v0.0.2 |

`kleos_logos_v001.yaml` extends `kleos_policy_v006.yaml` and overrides nothing,
so Hermes and Logos v0.0.1 are evaluated identically; `tests/test_logos_config.py`
enforces that. `kleos_logos_v002.yaml` extends v0.0.1's with one change,
`max_new_tokens` 512 to 1024 (enforced by `tests/test_logos_v002_config.py`),
because Logos v0.0.2 always thinks before it
answers and the thinking counts against the budget. The longest training trace
is 298 tokens and the longest benchmark answer 199 (measured), so 1024 leaves
room for twice the longest trace plus the longest answer. The change was declared
under H9 in [docs/experiments.md](../../docs/experiments.md) before the run.

The benchmark is derived from a sealed test split by `scripts/build_benchmark.py`;
the release itself is never modified. `KLEOS_BENCHMARK_PATH` overrides its path:

```bash
python scripts/build_benchmark.py --dataset <path to release>
```

## Conditions are held identical across arms

Same benchmark, same decoding settings, same graders, same seeds; only the arm
changes. If any of these differed between the baseline and the fine-tuned model,
the difference in score would be uninterpretable. `compare.py` checks for this and
emits a comparability warning when it finds a difference.

## Research arms

| Arm | Model | Orchestration |
| --- | --- | --- |
| `arm0_base` | base | none |
| `arm1_base_orchestrated` | base | KLEOS scaffolding |
| `arm2_finetuned` | base + LoRA | controlled task context |
| `arm3_finetuned_orchestrated` | base + LoRA | KLEOS scaffolding |
| `frontier_zero_shot` | frontier | none (interface only, not implemented) |
| `frontier_orchestrated` | frontier | scaffolding (interface only, not implemented) |

`arm0` against `arm2` isolates the fine-tuning effect, `arm0` against `arm1` the
orchestration effect, and all four together show whether the two interact. The
recorded runs evaluated `arm1_base_orchestrated` and `arm2_finetuned` for
Ministral-8B, Hermes and Logos v0.0.1, and `arm2_finetuned` only for Logos v0.0.2;
`arm0` and `arm3` have not been run on the full benchmark. The runs and their
reports are indexed in
[docs/experiments/README.md](../../docs/experiments/README.md).

## Graders

| Grader | For | Reference keys |
| --- | --- | --- |
| `exact_match` | Short exact answers | `text` |
| `classification` | Single-label decisions | `label` (+ `options`) |
| `set_match` | "Which items are relevant" | `items` |
| `ranking` | Prioritization | `ranking` |
| `heuristic_rubric` | Open-ended answers, offline | `required_points`, `evidence_ids`, … |
| `llm_judge` | Open-ended answers | Opt-in; needs an explicit judge callable |
| `kleos_policy` | The KLEOS benchmark: ordering, stated deciding factor and commitment, scored together | `ranking`, `label` (+ `options`), `confident` |

`kleos_policy` is the composite grader every KLEOS benchmark item names. Its score
is the mean of the components the reference provides: the ranking (0.6 × nDCG +
0.4 × top-1 accuracy), whether the stated deciding factor matches `label`, and
whether the answer commits or declines as `confident` expects. `format_valid`
(whether the answer is a JSON object) is recorded beside the score and excluded
from it by design (deviation D6,
[deviations log](../../docs/experiments.md#deviations-log)).

`heuristic_rubric` is deliberately coarse. It checks structure (required points
covered, evidence cited, unsupported-claim phrasings absent, an actionable
recommendation present) and is no substitute for human judging. It gives
rubric-graded tasks a reproducible, zero-cost baseline that runs in CI.

`llm_judge` records the judge model id in every result. Scores from different
judges are not comparable and are never pooled.

## Decoding

`do_sample: false` throughout, so evaluation is deterministic and a re-run
reproduces the number. Several `seeds` only mean something with
`do_sample: true`; the runner warns when seeds are configured without sampling.

For a model that thinks, the completion is split at the end-of-thinking token
(`[/THINK]`, token 35) before grading, and only the answer is graded. A completion
whose thinking never closes has an empty answer, is scored as given, and is
counted in `generation_stats.thinking_truncated`.

## Out-of-distribution and consistency measures

Tag benchmark examples with `split_tag: "ood"` and an `ood_shift` value to make
out-of-distribution performance measurable. Without them, `in_distribution_score`,
`ood_score` and `generalization_gap` cannot be computed and no generalization
claim can be made; the report states this instead of omitting the section. Every
item of the v0.0.6 benchmark is out of distribution, so the gap is not measurable
there (hypothesis H2,
[docs/experiments.md](../../docs/experiments.md#h2--does-any-gain-generalize-out-of-distribution)).

`consistency.group_key` (default `scenario_family`) groups logically equivalent
perturbations for the consistency measure. The runner also reports corrected
measures beside the original ones, with no setting needed:
consistency by `metadata.group_id`, separate answerable and should-decline
subsets, and confidence intervals that resample groups instead of examples
(findings H-F11 to H-F14,
[Hermes run report, section 11](../../docs/experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)).
The metrics are documented in [docs/evaluation.md](../../docs/evaluation.md).
