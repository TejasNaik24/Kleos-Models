# Contributing

Contributions are welcome: bug reports, fixes, tests, documentation, and new
model families or graders. For a security issue or an accidental data exposure,
follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # data, configuration and evaluation work; no torch
pip install -e ".[train,dev]"        # adds model loading and training
pip install -e ".[train,quant,dev]"  # adds 4-bit quantization (CUDA only)
pip install -e ".[serve,dev]"        # adds the FastAPI inference service
```

`make install`, `make install-train`, `make install-cuda` and `make install-serve`
do the same inside `.venv`. `make help` lists every target.

## Checks

```bash
make check    # ruff check, ruff format --check, mypy, pytest
```

Or individually:

```bash
ruff check .
ruff format --check .
mypy src/kleos_models
pytest
python scripts/check_no_private_data.py .
python scripts/validate_configs.py
python scripts/smoke_test.py
```

Run bare `pytest`, as CI does. `python -m pytest` also puts the working directory
on `sys.path`, which hides import errors that CI then hits. The suite needs no
GPU and no model download; tests that need torch, PEFT or CUDA are marked and
skip when the dependency is absent.

CI (`.github/workflows/ci.yml`) runs, on every push and pull request:

| Job | What it checks |
| --- | --- |
| Private-data scan | Secrets, keys and forbidden files, before any dependency is installed |
| Lint and format | `ruff check`, `ruff format --check` |
| Type check | `mypy src/kleos_models` |
| Tests (Python 3.11, 3.12, 3.13) | The full suite without torch |
| Tests with torch (CPU) | The full suite, including tiny-model training |
| Configs, schemas and fixtures | Every config resolves; generated schemas, fixtures and Colab notebooks match their generators |
| Smoke test and pipeline scripts | Smoke test, dataset inspection, leakage report, split, echo evaluation, comparison, feasibility planning |

## Design constraints

### The light layer stays torch-free

`config`, `data`, `experiments`, the scoring half of `evaluation`, and the
verification and client side of `serving` must import without torch or
transformers at module scope. Put heavy imports inside functions.
`tests/test_import_isolation.py` enforces the rule. It keeps dataset work, CI,
re-scoring and package verification fast, and lets a web backend call a model
without installing torch.

### Unimplemented paths raise

Code that appears to do something must do it. A path that is not implemented
raises a clear error instead of returning something that looks like a result.
`FrontierAPIBackend` shows the pattern. Errors carry actionable suggestions
(see `src/kleos_models/errors.py`).

### Tests check behavior

No `assert True`. A test fails when the behavior it describes breaks. Prefer a
property over an implementation detail: "the split never overlaps" survives a
refactor; "the function calls `_assign_by_fraction`" does not.

### Research integrity

A change to evaluation, reporting or experiments preserves these properties:

- Per-example records are always retained.
- Out-of-distribution, consistency and capability results are reported
  separately, never blended into one number.
- No superiority claim is made without a significance test; from H8 on, the
  interval is a cluster bootstrap by `group_id`.
- Failed runs stay in the registry.
- Automatic configuration changes are recorded in `manifest.adjustments[]`;
  `training.strict_config: true` makes any of them an error.

### Recorded hashes keep reproducing

`config_hash` covers the fully resolved configuration, defaults included. A new
config field therefore defaults to `None` and is listed in the model's
`HASH_NEUTRAL_FIELDS`, so it is omitted while unset and every recorded hash still
reproduces. Do not remove, rename or change the default of an existing field.
`tests/test_logos_config.py` and `tests/test_logos_v002_config.py` pin the
hashes of the Hermes, Logos v0.0.1 and Logos v0.0.2 runs.

### The record is frozen

The run reports under `docs/experiments/` and the result sections of
`docs/experiments.md` are not edited after the fact. A correction is added
beside the original, dated, and the superseded number stays visible. A change
that can invalidate a comparison between runs is marked **[research-affecting]**
in [CHANGELOG.md](CHANGELOG.md).

## Adding a model family

The training pipeline should not need to change.

1. Add `configs/models/<name>.yaml`. Take architecture facts from the
   checkpoint's real `config.json`, and pin `revision` to a commit sha.
2. If the architecture needs different loading, targeting or reasoning handling,
   subclass `ModelFamilyAdapter` in `src/kleos_models/models/adapters.py` and
   register it with `@register_adapter`.
3. Check the real architecture:
   ```bash
   python scripts/inspect_model.py --model <checkpoint-id>
   ```
4. Check feasibility on the target GPU:
   ```bash
   python scripts/plan_run.py --all-models --simulate-gpu t4-colab
   ```
5. Add tests to `tests/test_adapters.py`.

A branch like `if family == "..."` in the training pipeline means the
abstraction is in the wrong place.

## Adding a grader or metric

1. Implement it in `src/kleos_models/evaluation/graders.py` (subclass `Grader`)
   or `metrics.py`.
2. Register it (`register_grader`, or an entry in `GRADER_REGISTRY`).
3. Test it against a case whose right answer is known by hand.
4. Return a float where higher is better; invert if necessary.

## Generated artifacts

Some files are generated. Regenerate them instead of editing them by hand:

```bash
python scripts/prepare_dataset.py --emit-schemas       # data/schema/
python scripts/prepare_dataset.py --generate-fixtures  # data/examples/
python scripts/build_notebooks.py                      # notebooks/*.ipynb (Colab)
python scripts/build_kaggle_notebooks.py               # the Kaggle notebooks in notebooks/kaggle
```

CI fails if the schemas, the fixtures or the Colab notebooks drift from their
generators, and `tests/test_kaggle_notebooks.py` fails if a Kaggle notebook does.

## Style

- Type hints on public functions.
- Docstrings and comments explain why, not what the signature already says.
- Small functions with explicit interfaces.
- Named constants in `src/kleos_models/constants.py`, not magic numbers.
- No hard-coded absolute paths, no private paths, no account identifiers.
- US English in code and documentation.

## Commit messages

Say what changed and why. If the change affects experimental behavior (a metric,
a split, a default hyperparameter, a grader), say so explicitly, because it may
invalidate comparisons with existing runs.

## Pull request checklist

- [ ] `make check` passes.
- [ ] `python scripts/check_no_private_data.py .` passes; no data, outputs,
      credentials or private paths are included.
- [ ] Generated files are regenerated, not hand-edited.
- [ ] New behavior has a test that fails without the change.
- [ ] `tests/test_logos_config.py` and `tests/test_logos_v002_config.py` pass
      unchanged.
- [ ] A research-affecting change is marked in [CHANGELOG.md](CHANGELOG.md)
      under `[Unreleased]`.
- [ ] Documentation that describes the changed behavior is updated.
