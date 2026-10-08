# Data contract

The versioned schema every KLEOS example must satisfy. The source of truth is
`src/kleos_models/data/schemas.py`. The JSON Schema in `data/schema/` is generated
from it, and a test fails if the two drift.

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
    "group_id": "deadline-conflict-01-case-03",
    "perturbation_kind": "paraphrase"
  }
}
```

## Required fields

| Field | Rule |
| --- | --- |
| `id` | Unique within the dataset version. 3 to 128 characters of `[A-Za-z0-9._:-]`. |
| `version` | Schema version: `"1.0"`, or `"1.1"` for an example whose assistant message carries `reasoning` (below). This package writes `"1.0"` and reads both. |
| `task` | One of the registered tasks (below). |
| `messages` | At least 2, ending with `assistant`. |
| `variation_axes.domain` | Required. Every other axis is optional but reported. |
| `metadata.source` | Provenance. |
| `metadata.quality_status` | Review state. |

## Conversation rules

Enforced by the schema, because each violation produces broken supervision:

- The final message is from the assistant; otherwise there is nothing to learn.
- A system message, if present, is first.
- The first assistant message is preceded by a user message.
- No two consecutive assistant messages, which would make the supervised span
  ambiguous.
- No empty or whitespace-only content.
- `role: "tool"` requires a `name`.

## Assistant reasoning (schema 1.1)

From kleos-policy-v0.0.7, an assistant message may carry an optional `reasoning`
field: the policy-derived trace behind its answer, written step by step. It is
allowed only on assistant messages, never blank, and serialized only when present,
so an example without it keeps its exact bytes and content hash.

- A model trained to think (`model.reasoning.strip_thinking_from_targets: false`,
  Logos v0.0.2) receives it through the chat template as its thinking span,
  `[THINK]reasoning[/THINK]answer`, and is trained on all of it. Such a run refuses
  to start if truncation would cut any supervised token
  ([training.md](training.md#reasoning-supervised-runs)).
- Every other model has the field removed before tokenizing. The number of
  examples affected is logged and recorded as `reasoning_dropped`.
- The field enters an example's content hash (and so the dataset hash) only when
  present. Leakage detection compares conversations without it: only train and
  validation carry traces, so a trace would otherwise hide a train copy of a test
  conversation.

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

Each task is trainable in isolation; the research plan does not assume that one
monolithic "KLEOS personality" fine-tune is the right approach.

## Policy, not private facts

The most important rule in the contract: the model learns generalizable decision
procedures, and private facts belong in retrieval and memory, never in weights.

Teaches a policy:

```
Given these competing opportunities, prioritize the one with the closest deadline
and the strongest evidence of impact.
```

Teaches a fact about a person:

```
The user is interviewing at Example Corp, so Example Corp items always come first.
```

The second memorizes something that belongs in a retrieval system, goes stale,
and cannot generalize to anyone else.

The validator flags phrasings that look like fact-teaching
(`possible_fact_memorization`) for human review. It is a heuristic and cannot
catch everything, so the rule is ultimately enforced by whoever writes the data.
[privacy.md](privacy.md) states the wider public/private rule.

## Variation axes

Axes let the pipeline answer how many examples cover each situation type, instead
of mistaking a large dataset for a diverse one.

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
before it is added to `constants.py`.

Coverage is reported at three levels:

- **Marginal:** counts per value of each axis.
- **Joint:** counts per combination (the situation type).
- **Gaps:** empty and thin cells.

```bash
python scripts/inspect_dataset.py --dataset <path to release> --coverage
```

Look for a constant axis (no contrast available), empty cells (situation types
with no examples) and thin cells (too sparse for a per-cell claim).

## Metadata

| Field | Purpose |
| --- | --- |
| `source` | `synthetic`, `synthetic_seeded`, `real_sanitized`, `expert_authored`, `development_fixture` |
| `quality_status` | `draft`, `auto_generated`, `reviewed`, `rejected` |
| `scenario_family` | Groups related scenarios; the split key for `scenario_family_holdout` |
| `group_id` | The perturbations of one case; the unit for consistency and cluster intervals |
| `perturbation_of`, `perturbation_kind` | Link a perturbation to its original, and name the axis it varies |
| `author` | Pipeline or role, never a person's name |

### `scenario_family` and `group_id`

The two keys do different jobs:

- **`scenario_family`** groups related scenarios. The `scenario_family_holdout`
  strategy splits on it, so related examples never straddle the train/test
  boundary. A family can contain cases with different correct answers.
- **`group_id`** groups the perturbations of one case, which share one correct
  answer by construction. That makes it the consistency unit: whether the model
  reaches the same decision under logically irrelevant variation is a meaningful
  question only within such a group. It is also the cluster for confidence
  intervals ([evaluation.md](evaluation.md#which-unit)).

Grouping by family instead measures the grouping: on the v0.0.6 benchmark a
perfect model scores 0.600 agreement by family and 1.000 by `group_id` (finding
H-F11 in the
[Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings);
the record is indexed in [experiments/README.md](experiments/README.md)).

Group-aware splitting uses a configured `split.group_key` when set, otherwise
`group_id`, then `scenario_family`, then the example id. Without either key,
consistency testing has nothing to group and measures nothing; the runner warns
when that happens.

## Evaluation examples

Same shape plus grading fields. The assistant turn is optional; the `reference`
is what matters.

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
| `kleos_policy` | `ranking`, optional `label`, `options`, `confident` |
| `heuristic_rubric` | `required_points`, `forbidden_points`, `evidence_ids`, `expected_decision` |

`reference.confident` also assigns the subset: `true` is answerable, `false` is
should-decline ([evaluation.md](evaluation.md#subsets-answerable-and-should-decline)).

`split_tag` is `"in_distribution"`, `"ood"` or `"capability"`. Without OOD-tagged
examples, OOD is not measurable and no generalization claim is made; the report
says so rather than omitting the section.

`ood_shift` values: `unseen_entities`, `unseen_domains`, `unseen_formats`,
`unseen_source_types`, `context_length_shift`, `reordered_evidence`,
`conflicting_evidence`.

## Dataset manifests

Every dataset version carries a `manifest.json`. Its shape, with the counts and
split of kleos-policy-v0.0.6 and the timestamp and hashes elided:

```json
{
  "version": "kleos-policy-v0.0.6",
  "schema_version": "1.0",
  "created_at": "...",
  "example_count": 1350,
  "splits": {"train": 820, "validation": 181, "test": 349},
  "split_strategy": "format_holdout",
  "split_seed": 42,
  "holdout_values": ["json"],
  "file_hashes": {"train.jsonl": "sha256:..."},
  "content_hash": "sha256:...",
  "contains_private_data": false
}
```

`format_holdout` sends every `format=json` example to the test split and draws
validation from the seen formats, so early stopping does not leak the held-out
signal. The test split is then a format-transfer probe (deviation D2 in
[experiments.md](experiments.md#deviations-log)).

Dataset versions are immutable. `split_dataset.py` refuses to overwrite an
existing version directory, because rewriting a version would break every
comparison that referenced it.

`contains_private_data: true` marks a private artifact. Loaders warn loudly on
it; such a dataset is never committed here or published with an adapter.

## Validating

```bash
python scripts/validate_dataset.py --dataset <path to release>
python scripts/validate_dataset.py --dataset <path to release> --leakage-report reports/leakage
python scripts/inspect_dataset.py  --dataset <path to release> --coverage
```

Checks: schema, duplicate ids, coverage gaps, quality status, placeholder text,
sensitive-content patterns, policy-versus-fact phrasing, length against
`max_seq_length`, and leakage across splits.

## Consuming the private dataset

The private `kleos-training-data` repository produces a release directory:

```
<path to release>/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

Released versions also carry `provenance.json` and `RELEASE.lock`. This repository
consumes the directory by path:

```bash
python scripts/train.py --config configs/training/kleos_hermes_v006.yaml \
    --dataset <path to release>
```

Neither repository imports the other, and the private artifact is never copied
into this one.
