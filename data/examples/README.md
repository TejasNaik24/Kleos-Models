# Development fixtures

> **These are synthetic development fixtures, not the KLEOS research dataset.**
> They exist to exercise the pipeline. Any metric computed from them describes the
> plumbing, not KLEOS.

## Files

| File | Contents |
| --- | --- |
| `synthetic_train.jsonl` | 13 training examples across 4 tasks |
| `synthetic_eval.jsonl` | 9 evaluation examples with references |
| `manifest.json` | Dataset manifest with content hashes |

## What they are designed to exercise

- **Loading and validation** — every example satisfies the data contract
- **Splitting** — varied `entities`, `domain` and `format` so held-out strategies
  have something to partition on
- **Leakage detection** — distinct enough that a clean dataset reports clean
- **Consistency testing** — `eval-consistency-*` share one `scenario_family` and
  differ only in wording, evidence order and format, so a correct model must
  reach the same decision on all three
- **OOD reporting** — two examples tagged `split_tag: "ood"`, one for an unseen
  domain and one for conflicting evidence
- **Rubric grading** — a briefing example with `required_points` and
  `evidence_ids`

`eval-ood-0002` deliberately inverts the recency policy: source reliability
outranks recency there. A model that learned "always prefer the newer record" as a
surface rule gets it wrong, which is the point.

## They teach policy, not facts

Every example uses generic entities (`internship-application`, `alpha-task`) and
justifies decisions from supplied evidence. None teaches a fact about a person.
See [../../docs/data-contract.md](../../docs/data-contract.md).

## Regenerating

```bash
python scripts/prepare_dataset.py --generate-fixtures
```

Deterministic — regenerating produces byte-identical files, and CI checks that the
committed files match their generator.
