# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # light: data, config, evaluation work
pip install -e ".[train,dev]"  # add model loading and training
```

## Before opening a PR

```bash
make check    # ruff, format, mypy, pytest
```

Or individually:

```bash
ruff check .
ruff format .
mypy src/kleos_models
pytest
python scripts/check_no_private_data.py .
```

## The rules that are not negotiable

### 1. No private data. Ever.

This repository is public. Run the scanner before committing. See
[docs/privacy.md](docs/privacy.md).

### 2. The light layer stays torch-free

`config`, `data`, `experiments` and the scoring half of `evaluation` must import
without torch or transformers at module scope. Put heavy imports **inside
functions**.

`tests/test_import_isolation.py` enforces this. It is not a style preference — it
is what keeps dataset work, CI and re-analysis fast.

### 3. Real tests only

No `assert True`. A test should fail if the behaviour it describes breaks.

Prefer testing a property over an implementation detail: "the split never
overlaps" survives a refactor; "the function calls `_assign_by_fraction`" does not.

### 4. No fake pipelines

Code that appears to do something must do it. If a path is not implemented, raise
a clear error — do not stub it out with something that looks like it worked. See
`FrontierAPIBackend` for the pattern.

### 5. Research integrity is a code property

If you touch evaluation or reporting, preserve these:

- per-example records are always retained
- OOD, consistency and capability results are reported separately
- no superiority claim without a significance test
- failed runs stay in the registry
- automatic configuration changes are recorded, never silent

## Adding a model family

You should not need to touch the training pipeline.

1. Add `configs/models/<name>.yaml`. Get architecture facts from the checkpoint's
   real `config.json`, not from a blog post.
2. If the architecture needs different loading, targeting or reasoning handling,
   subclass `ModelFamilyAdapter` and `@register_adapter` it.
3. Verify against the real architecture:
   ```bash
   python scripts/inspect_model.py --model <checkpoint-id>
   ```
4. Add tests to `tests/test_adapters.py`.

If you find yourself adding `if family == "..."` to the training pipeline, the
abstraction is in the wrong place.

## Adding a grader or metric

1. Implement it in `evaluation/graders.py` or `metrics.py`.
2. Register it.
3. Test it against a case where you know the right answer by hand.
4. Metrics return a float where **higher is better**. Invert if necessary.

## Regenerating artifacts

Some files are generated. Regenerate rather than hand-editing:

```bash
python scripts/prepare_dataset.py --emit-schemas       # data/schema/
python scripts/prepare_dataset.py --generate-fixtures  # data/examples/
python scripts/build_notebooks.py                      # notebooks/
```

CI fails if these drift from their generators.

## Style

- Type hints on public functions
- Docstrings explaining *why*, not restating the signature
- Small functions, explicit interfaces
- No magic numbers — named constants in `constants.py`
- No hard-coded absolute paths
- Errors carry actionable suggestions (see `errors.py`)

## Commit messages

Explain what changed and why. If it changes experimental behaviour — a metric, a
split, a default hyperparameter — say so explicitly, because it may invalidate
comparisons with existing runs.
