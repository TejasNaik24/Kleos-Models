# Notebooks

**Google Colab is the canonical environment for KLEOS training runs.** These
notebooks are the primary entry point.

| Notebook | Purpose | Needs a GPU? |
| --- | --- | --- |
| `00_environment_check.ipynb` | What runtime did I get, and what can it train? | no |
| `01_dataset_validation.ipynb` | Validate and split a dataset | no |
| `02_train_qlora.ipynb` | **Train an adapter** | yes |
| `03_evaluate.ipynb` | Compare base vs fine-tuned | yes |

## Opening in Colab

[colab.research.google.com](https://colab.research.google.com) → **GitHub** tab →
paste the repository URL → pick a notebook.

Then **Runtime → Change runtime type → T4 GPU**.

Full walkthrough: [../docs/colab.md](../docs/colab.md).

## They are generated, not hand-edited

Notebook JSON is unreadable in a diff, so these are generated from reviewable
Python:

```bash
python scripts/build_notebooks.py
```

Edit `scripts/build_notebooks.py`, regenerate, and commit both. CI fails if they
drift.

`tests/test_notebooks.py` validates the output: valid nbformat, no stored outputs,
no hard-coded secrets, and every referenced script, config and CLI flag actually
exists.

## Two things they will not do

**They will not reinstall torch.** Colab's torch is compiled against its CUDA
driver; replacing it breaks CUDA in confusing ways. `scripts/colab_setup.py`
installs everything else with `--no-deps`.

**They will not publish anything automatically.** The publish command in
`02_train_qlora.ipynb` is commented out deliberately.
