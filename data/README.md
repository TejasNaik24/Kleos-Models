# Data

## What is committed, and what is not

`data/` is **deny-by-default** in `.gitignore`. Everything here is ignored unless
explicitly allowlisted, so a file with an unexpected name cannot slip into a
public commit.

| Path | Contents | Committed? |
| --- | --- | --- |
| `schema/` | JSON Schema — describes shape, holds no data | **yes** |
| `examples/` | Synthetic development fixtures — generated, no real data | **yes** |
| `README.md` | This file | **yes** |
| `raw/` | Private/raw data | **no** — only `.gitkeep` |
| `processed/` | Generated datasets | **no** — only `.gitkeep` |
| anything else | | **no** |

Drop a real export in as `data/my_export.jsonl`, `data/memories.json` or
`data/kleos-policy-v0.1.0/train.jsonl` and git will ignore all of it.

`tests/test_gitignore.py` verifies this against real git behaviour, so the
guarantee cannot quietly regress.

Your actual training data belongs in the **separate private repository** and is
consumed by path — never copied here. See
[../docs/privacy.md](../docs/privacy.md).

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
A test fails if they drift.

```bash
python scripts/prepare_dataset.py --emit-schemas
```

## Using the real dataset

The private `kleos-training-data` repository produces a versioned artifact. Point
at it by path — from anywhere on disk, outside this repository:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset /path/to/private/dataset
```

Never copy it here.
