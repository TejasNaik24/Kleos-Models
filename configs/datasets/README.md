# Dataset configurations

| Config | Dataset | Used by |
| --- | --- | --- |
| `example.yaml` | The synthetic development fixtures in `data/examples/` | `debug.yaml`, `qlora_small.yaml`, tests and CI |
| `kleos_policy_v006.yaml` | The sealed `kleos-policy-v0.0.6` release, consumed pre-split | Ministral-8B, KLEOS Hermes, KLEOS Logos v0.0.1 |
| `kleos_policy_v007.yaml` | The sealed `kleos-policy-v0.0.7` release, consumed pre-split | KLEOS Logos v0.0.2 |
| `private_artifact.yaml` | Template for any release exported by the private `kleos-training-data` repository | A starting point for new releases |

## `example.yaml`

Points at `data/examples/`, the synthetic development fixtures. They are not the
KLEOS research dataset. They exist to exercise loading, validation, splitting,
leakage detection, formatting, training and evaluation, and numbers computed from
them describe the pipeline, not KLEOS.

## `private_artifact.yaml`

Template for consuming a release produced by the private `kleos-training-data`
repository, which exports:

```
dataset/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

Point at it by path, anywhere on disk outside this repository:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset <path to release>
```

`--dataset` overrides the config's `path`, which otherwise reads
`KLEOS_DATASET_PATH`. This repository never depends on the private one, never
embeds private data and never connects to the KLEOS application's database. The
boundary is set out in [docs/privacy.md](../../docs/privacy.md).

## Split strategies

`strategy: random` is for development only. A random split of a dataset that
contains paraphrases and reordered variants of the same scenario puts
near-copies on both sides of the boundary, and the resulting "generalization"
number measures memorization.

| Strategy | What the test split measures |
| --- | --- |
| `random` | Nothing: development convenience only |
| `group` | Related examples stay on one side of the split |
| `entity_holdout` | Whether behavior transfers to unseen entities |
| `domain_holdout` | Whether it transfers to unseen domains |
| `format_holdout` | Whether it survives a different presentation format |
| `scenario_family_holdout` | Related scenarios never straddle the split |

Validation is drawn from seen values even under a held-out strategy, so early
stopping cannot leak the out-of-distribution signal the test split is meant to
measure.

## Quality gate

`filters.quality_statuses` defaults to `[reviewed]`. Widening it changes what an
experiment measures; record the change in the run's report. In both sealed
releases every example is `reviewed`, so there the gate acts as a tripwire: a
release containing drafts fails it instead of being trained on.

## `kleos_policy_v006.yaml` and `kleos_policy_v007.yaml`

The sealed KLEOS releases, consumed pre-split from outside this repository: the
loader never re-splits them, and the `split` block only records how the release
was partitioned upstream (`format_holdout` on `format=json`). Both have 820
training, 181 validation and 349 test examples; the test split is JSON only, so
it is a pure format-transfer probe that is never trained on or used for model
selection.

v0.0.7 is v0.0.6 plus two additions on every training and validation answer: a
policy-derived `reasoning` trace (schema 1.1), and a final "What decided it"
line, so the four decline labels v0.0.6 never trained now occur in training. Its
`test.jsonl` is byte-identical to v0.0.6's, so both build the same benchmark
(sha256 `a11ffad7…`). The data contract is in
[docs/data-contract.md](../../docs/data-contract.md).
