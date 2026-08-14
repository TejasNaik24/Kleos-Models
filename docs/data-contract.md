# Data contract

The versioned schema every KLEOS example must satisfy. Source of truth:
`src/kleos_models/data/schemas.py`. The JSON Schema in `data/schema/` is generated
from it, and a test fails if they drift.

## Example format

JSONL, one complete JSON object per line.

```json
{
  "id": "example-000001",
  "version": "1.0",
  "task": "notification_prioritization",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "variation_axes": {
    "domain": "career",
    "entities": "set_a",
    "urgency": "high",
    "evidence_quality": "mixed",
    "conflicting_evidence": "none",
    "format": "bullets"
  },
  "metadata": {
    "source": "synthetic",
    "quality_status": "reviewed",
    "scenario_family": "deadline-conflict-01",
    "perturbation_kind": "paraphrase"
  }
}
```

## Required fields

| Field | Rule |
| --- | --- |
| `id` | Unique within the dataset version. 3-128 chars of `[A-Za-z0-9._:-]`. |
| `version` | Schema version. Currently `"1.0"`. |
| `task` | One of the registered tasks (below). |
| `messages` | At least 2. Ends with `assistant`. |
| `variation_axes.domain` | Required. Every other axis is optional but reported. |
| `metadata.source` | Provenance. |
| `metadata.quality_status` | Review state. |

## Conversation rules

Enforced by the schema, because each violation produces broken supervision:

- The final message must be from the assistant — otherwise there is nothing to
  learn.
- A system message, if present, must be first.
- The first assistant message must be preceded by a user message.
- No two consecutive assistant messages — that makes the supervised span
  ambiguous.
- No empty or whitespace-only content.
- `role: "tool"` requires a `name`.

## Tasks

| Task | What the model must learn |
| --- | --- |
| `notification_prioritization` | Rank competing items by urgency, evidence and impact |
| `tool_routing` | Choose the right tool, or none |
| `mission_control_briefing` | Synthesize a structured briefing from context |
| `context_prioritization` | Select and order the most relevant context |
| `recommendation_generation` | Recommend actions grounded in supplied evidence |
| `memory_conflict_resolution` | Resolve contradictions using recency and reliability |
| `workspace_reasoning` | Reason within workspace scoping rules |

Each task is trainable in isolation. The research plan explicitly rejects the
assumption that one monolithic "KLEOS personality" fine-tune is the right approach.

## Policy, not private facts

**This is the most important rule in the contract.**

The model should learn generalizable decision procedures. Private facts belong in
retrieval and memory, never in weights.

Good — teaches a policy:

```
Given these competing opportunities, prioritize the one with the closest deadline
and the strongest evidence of impact.
```

Bad — teaches a fact about a person:

```
Tejas has a Motorola project and therefore should prioritize Motorola.
```

The second memorizes something that belongs in a retrieval system, will go stale,
and cannot generalize to anyone else.

The validator flags phrasings that look like fact-teaching
(`possible_fact_memorization`). It is a heuristic and it flags for human review —
it cannot catch everything, so this rule is ultimately enforced by whoever writes
the data.

## Variation axes

Not decoration. Axes are what let the pipeline answer *how many examples cover
each situation type* rather than mistaking a large dataset for a diverse one.

| Axis | Example values |
| --- | --- |
| `domain` | career, research, coursework, projects |
| `entities` | set_a, set_b, unseen |
| `urgency` | high, medium, low |
| `deadlines` | imminent, distant, none, mixed |
| `evidence_quality` | strong, weak, mixed |
| `conflicting_evidence` | none, present |
| `context_length` | short, medium, long |
| `presentation_order` | as_given, reversed, shuffled |
| `format` | prose, bullets, json, mixed |
| `source_type` | email, document, note, tool_output |
| `workspace` | personal, team, course |
| `difficulty` | easy, medium, hard |
| `ambiguity` | low, medium, high |

Extra axes are allowed and reported as unregistered, so a new axis can be piloted
before being promoted into `constants.py`.

Coverage is reported at three levels:

- **Marginal** — counts per value of each axis
- **Joint** — counts per *combination* (the situation type)
- **Gaps** — empty and thin cells

```bash
python scripts/inspect_dataset.py --dataset <dir> --coverage
```

Watch for: a constant axis (no contrast available), empty cells (situation types
with no examples), and thin cells (too sparse for a per-cell claim).

## Metadata

| Field | Purpose |
| --- | --- |
| `source` | `synthetic`, `synthetic_seeded`, `real_sanitized`, `expert_authored`, `development_fixture` |
| `quality_status` | `draft`, `auto_generated`, `reviewed`, `rejected` |
| `scenario_family` | Groups logically related examples |
| `group_id` | Generic grouping key for splitting |
| `perturbation_of` / `perturbation_kind` | Links a perturbation to its original |
| `author` | Pipeline or role — **never a person's name** |

### `scenario_family` carries a lot of weight

It does two jobs:

1. **Splitting.** Related examples never straddle the train/test boundary.
2. **Consistency testing.** Perturbations of one scenario are grouped so the
   harness can check whether the model reaches the same decision under logically
   irrelevant variation.

Without it, consistency testing has nothing to group and silently measures
nothing. The runner warns when this happens.

## Evaluation examples

Same shape plus grading fields. The assistant turn is optional; the `reference` is
what matters.

```json
{
  "id": "eval-000001",
  "task": "tool_routing",
  "messages": [{"role": "user", "content": "..."}],
  "variation_axes": {"domain": "career"},
  "reference": {"label": "memory_search",
                "options": ["memory_search", "web_search", "none"]},
  "grader": "classification",
  "split_tag": "in_distribution",
  "ood_shift": null
}
```

| Grader | Reference keys |
| --- | --- |
| `exact_match` | `text` |
| `classification` | `label`, optional `options` |
| `set_match` | `items` |
| `ranking` | `ranking` |
| `heuristic_rubric` | `required_points`, `forbidden_points`, `evidence_ids`, `expected_decision` |

`split_tag` must be `"in_distribution"`, `"ood"` or `"capability"`. **Without
OOD-tagged examples, OOD is not measurable and no generalization claim can be
made** — the report says so explicitly rather than omitting the section.

`ood_shift` values: `unseen_entities`, `unseen_domains`, `unseen_formats`,
`unseen_source_types`, `context_length_shift`, `reordered_evidence`,
`conflicting_evidence`.

## Dataset manifests

Every dataset version carries a `manifest.json`:

```json
{
  "version": "kleos-policy-v0.1.0",
  "schema_version": "1.0",
  "created_at": "2026-08-14T00:00:00+00:00",
  "example_count": 1200,
  "splits": {"train": 840, "validation": 180, "test": 180},
  "split_strategy": "entity_holdout",
  "split_seed": 42,
  "file_hashes": {"train.jsonl": "sha256:..."},
  "content_hash": "sha256:...",
  "contains_private_data": false
}
```

**Dataset versions are immutable.** `split_dataset.py` refuses to overwrite an
existing version directory, because silently rewriting a version breaks every
comparison that referenced it.

`contains_private_data: true` marks the private artifact. Loaders warn loudly on
it; such a dataset must never be committed here or published with an adapter.

## Validating

```bash
python scripts/validate_dataset.py --dataset <dir>
python scripts/validate_dataset.py --dataset <dir> --leakage-report reports/leakage
python scripts/inspect_dataset.py  --dataset <dir> --coverage
```

Checks: schema, duplicate ids, coverage gaps, quality status, placeholder text,
sensitive-content patterns, policy-vs-fact phrasing, length against
`max_seq_length`, and leakage across splits.

## Consuming the private dataset

The private `kleos-training-data` repository produces:

```
dataset/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

This repository consumes it by path:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset /path/to/private/dataset
```

Neither repository imports the other. Never copy the private artifact into this
public repository.
