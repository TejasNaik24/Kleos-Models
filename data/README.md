# Data

## What is committed, and what is not

`data/` is deny-by-default in `.gitignore`. Everything here is ignored unless
explicitly allowlisted, so a file with an unexpected name cannot slip into a
public commit.

| Path | Contents | Committed? |
| --- | --- | --- |
| `schema/` | JSON Schema: describes shape, holds no data | yes |
| `examples/` | Synthetic development fixtures: generated, no real data | yes |
| `README.md` | This file | yes |
| `raw/` | Private or raw data | no, only `.gitkeep` |
| `processed/` | Generated datasets | no, only `.gitkeep` |
| anything else | | no |

A real export dropped in as `data/my_export.jsonl`, `data/memories.json` or
`data/kleos-policy-v0.1.0/train.jsonl` is ignored in full.

`tests/test_gitignore.py` verifies this against real git behavior, so the
guarantee cannot quietly regress.

Real training data is built by [Kleos-Training-Data](https://github.com/TejasNaik24/Kleos-Training-Data), kept in private storage, and consumed by
path, never copied here. See [../docs/privacy.md](../docs/privacy.md).

## `examples/` is not the research dataset

> The included examples are development fixtures only and must not be interpreted
> as the KLEOS research dataset.

They exist to exercise loading, validation, splitting, leakage detection,
formatting, training and evaluation. Any number computed from them describes the
pipeline, not KLEOS.

They are generated deterministically:

```bash
python scripts/prepare_dataset.py --generate-fixtures
```

## `schema/`

Generated from `src/kleos_models/data/schemas.py`, which is the source of truth.
`tests/test_data_schema.py` and a CI step fail if the committed files drift from
it.

```bash
python scripts/prepare_dataset.py --emit-schemas
```

## Using the real dataset

The [Kleos-Training-Data](https://github.com/TejasNaik24/Kleos-Training-Data) pipeline produces a versioned release. Point
at it by path, anywhere on disk outside this repository:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset <path to release>
```

Never copy it here.
