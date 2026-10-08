# Notebooks

Two generated sets. The Colab notebooks run the general pipeline on a free T4.
The Kaggle notebooks run the KLEOS Logos v0.0.2 experiment, pre-registered as H9
in [docs/experiments.md](../docs/experiments.md), unattended on Kaggle's free
2 × T4. The run they produced is indexed in
[docs/experiments/README.md](../docs/experiments/README.md).

## Colab

| Notebook | Purpose | GPU |
| --- | --- | --- |
| `00_environment_check.ipynb` | Report the assigned runtime and which model configs it can train | not required |
| `01_dataset_validation.ipynb` | Validate, inspect and split a dataset | not required |
| `02_train_qlora.ipynb` | Train a QLoRA adapter, with checkpoints on Google Drive and resume after a disconnect | T4 |
| `03_evaluate.ipynb` | Evaluate the base and fine-tuned arms under identical conditions and compare them | T4 |

To open one: [colab.research.google.com](https://colab.research.google.com) →
GitHub tab → paste the repository URL → pick the notebook. Then select Runtime →
Change runtime type → T4 GPU. The full walkthrough is
[docs/colab.md](../docs/colab.md).

## Kaggle

| Notebook | Purpose | Accelerator |
| --- | --- | --- |
| `kaggle/logos_v002_train.ipynb` | Smoke run and memory gate, then the full Logos v0.0.2 training run at the pre-registered `config_hash`; resumes from a previous version's output | GPU T4 x2 |
| `kaggle/logos_v002_evaluate.ipynb` | Evaluate the trained adapter (`arm2_finetuned`) on the sealed benchmark, with each answer's thinking split off at `[/THINK]`; resumes from a partial result | GPU T4 x2 |
| `kaggle/output_probe.ipynb` | Writes a marker file and then fails on purpose, to check that a failed version's output can be attached as an input, which resume relies on. About two minutes | none |

Each runs unattended (Save Version → Save & Run All), and every command
stops the notebook on its first failure. On 2026-10-06 the training notebook ran
17,746 s in total (install, download, smoke run and 4.43 h of training) and the
evaluation 18,791 s (5.2 h), as measured in the
[Logos v0.0.2 run report](../docs/experiments/kleos-v007-ministral314breasoning-run1-report.md).
The how-to is [docs/kaggle.md](../docs/kaggle.md) and the full runbook
[docs/runbooks/logos-v002-kaggle.md](../docs/runbooks/logos-v002-kaggle.md).

## Generated, not hand-edited

Notebook JSON is unreadable in a diff, so both sets are generated from reviewable
Python:

```bash
python scripts/build_notebooks.py          # the Colab notebooks
python scripts/build_kaggle_notebooks.py   # the Kaggle notebooks
```

Edit the builder, regenerate, and commit both. CI catches drift in both sets: its
"Check notebooks are current" step regenerates the Colab notebooks and fails on
any difference under `notebooks/`, and `tests/test_kaggle_notebooks.py` fails
unless every Kaggle notebook equals what its builder writes.

The tests also validate the content. `tests/test_notebooks.py` checks valid
nbformat, no stored outputs, no hard-coded secrets, tokens read from Colab
secrets, and that every referenced script, config and CLI flag exists.
`tests/test_kaggle_notebooks.py` checks the pre-registered paths and config hash,
that the private dataset and the model weights never reach a version's saved
output, that long runs end before the session limit, and that nothing is
published.

## What they never do

They never reinstall torch. Colab's torch is compiled against its CUDA driver, and
replacing it breaks CUDA in confusing ways, so `scripts/colab_setup.py` installs
everything else with `--no-deps`.

They never publish anything automatically. The publish command in
`02_train_qlora.ipynb` is commented out, and the Kaggle notebooks contain none.
