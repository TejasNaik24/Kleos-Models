# Evaluation configuration

## Conditions are held identical across arms

Same benchmark, same decoding settings, same graders, same seeds. Only the arm
changes. If any of these differed between base and fine-tuned, the difference in
score would be uninterpretable — `compare.py` checks for exactly this and emits a
comparability warning when it finds it.

## Research arms (spec §20)

| Arm | Model | Orchestration |
| --- | --- | --- |
| `arm0_base` | base | none |
| `arm1_base_orchestrated` | base | KLEOS scaffolding |
| `arm2_finetuned` | base + LoRA | controlled task context |
| `arm3_finetuned_orchestrated` | base + LoRA | KLEOS scaffolding |
| `frontier_zero_shot` | frontier | none — **interface only** |
| `frontier_orchestrated` | frontier | scaffolding — **interface only** |

arm0 vs arm2 isolates the fine-tuning effect. arm0 vs arm1 isolates the
orchestration effect. Running all four separates the two and reveals whether they
interact.

## Graders

| Grader | For | Reference key |
| --- | --- | --- |
| `exact_match` | short exact answers | `text` |
| `classification` | single-label decisions | `label` (+ `options`) |
| `set_match` | "which items are relevant" | `items` |
| `ranking` | prioritization | `ranking` |
| `heuristic_rubric` | open-ended answers, offline | `required_points`, `evidence_ids`, … |
| `llm_judge` | open-ended answers | **opt-in**, needs an explicit judge callable |

`heuristic_rubric` is deliberately coarse. It checks structure — required points
covered, evidence cited, unsupported-claim phrasings absent, an actionable
recommendation present — and is not a substitute for human judging. It exists so
rubric-graded tasks have a reproducible, zero-cost baseline that runs in CI.

`llm_judge` records the judge model id in every result. Scores from different
judges are not comparable and must never be pooled.

## Decoding

`do_sample: false` by default, so evaluation is deterministic and a re-run
reproduces the number. Multiple `seeds` only mean something with
`do_sample: true` — the runner warns if you configure seeds without sampling.

## OOD and consistency

Tag benchmark examples with `split_tag: "ood"` and an `ood_shift` value to make
OOD measurable. Without them, `in_distribution_score`, `ood_score` and
`generalization_gap` cannot be computed and **no generalization claim can be
made** — the report says so explicitly rather than omitting the section.

Consistency needs `metadata.scenario_family` on perturbed examples so logically
equivalent scenarios can be grouped.
