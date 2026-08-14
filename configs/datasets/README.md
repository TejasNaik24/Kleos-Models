# Dataset configurations

## `example.yaml`

Points at `data/examples/`, the synthetic development fixtures.

**These are not the KLEOS research dataset.** They exist to exercise loading,
validation, splitting, leakage detection, formatting, training and evaluation.
Numbers computed from them describe the pipeline, not KLEOS.

## `private_artifact.yaml`

Template for consuming the artifact produced by the private
`kleos-training-data` repository. The private repo exports:

```
dataset/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

Point at it by path — from anywhere on disk, outside this repository:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset /path/to/private/dataset
```

This public repository never depends on the private one, never embeds private
data, and never reaches into Supabase (spec §27, §51).

## Split strategies

`strategy: random` is for development only. A random split of a dataset
containing paraphrases and reordered variants of the same scenario puts
near-copies on both sides of the boundary, and the resulting "generalization"
number measures memorization.

| Strategy | Tests |
| --- | --- |
| `random` | nothing — development convenience only |
| `group` | related examples stay together |
| `entity_holdout` | does behaviour transfer to unseen entities? |
| `domain_holdout` | does it transfer to unseen domains? |
| `format_holdout` | does it survive a different presentation format? |
| `scenario_family_holdout` | related scenarios never straddle the split |

Validation is drawn from *seen* values even under a held-out strategy, so early
stopping cannot leak the OOD signal the test split is meant to measure.

## Quality gate

`filters.quality_statuses` defaults to `[reviewed]`. Widening it is a deliberate
act that changes what the experiment measures — do it consciously and record it.
