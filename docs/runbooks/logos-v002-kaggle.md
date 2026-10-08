# Logos v0.0.2 on Kaggle

This runbook trains and evaluates KLEOS Logos v0.0.2
(`kleos-v007-ministral314breasoning-run1`) on Kaggle's free 2 × T4 with the two
generated notebooks in [`notebooks/kaggle`](../../notebooks/kaggle), then runs
the H9 comparison on a local machine, CPU only. The recorded run followed it on
2026-10-06. Model background and the results are in [Logos](../logos.md); ids
and placeholders are explained in
[docs/experiments/README.md](../experiments/README.md).

Both notebooks run unattended (Save Version → Save & Run All) and halt at the
first failure. They keep the dataset, the model weights and the code in `/tmp`,
which Kaggle never saves, and publish nothing. Only `outputs/` reaches a
version's saved output.

## Before starting

1. **Push the commit to run** to GitHub and note its full 40-character sha; both
   notebooks clone it. The recorded run used
   `a17ace7fd42a5d91796f6bbca3aa60bcc659d90d`, the commit that pre-registered H9.
2. **Prepare a Kaggle account** with phone verification, which GPUs and internet
   access require. The free GPU quota is about 30 hours a week; the recorded run
   used about 10.5 GPU hours.
3. **Upload the release as a private Kaggle dataset** (New Dataset → upload the
   files → Private): the six files of `<path to release>/kleos-policy-v0.0.7`,
   that is `train.jsonl`, `validation.jsonl`, `test.jsonl`, `manifest.json`,
   `provenance.json` and `RELEASE.lock`.
4. **Run the output probe once** (2 minutes, no GPU):
   `output_probe.ipynb` from `notebooks/kaggle`, with Save & Run All. It writes a marker
   and then fails on purpose. In any notebook, Add Input → the probe's output: if
   `probe/marker.txt` is there, a failed or stopped version's output can be
   attached, and the resume steps below work. If it is not, a run that stops must
   start over, and resume cannot be relied on.

**The config hash depends on the paths.** H9's pre-registered `config_hash`
(`d1961583546b5761…`) was computed with the literal `--dataset` and
`--output-dir` paths recorded under
[H9 in docs/experiments.md](../experiments.md#h9--does-a-logos-trained-to-think-beat-hermes),
and with no `KLEOS_*` variable set. The notebooks use exactly those paths and
refuse to train if the hash differs. Do not change them.

## Training

ESTIMATED 5 to 6 hours, smoke run included. The recorded run trained for 4.4
hours, about 4.9 with setup, in one session.

1. Import `logos_v002_train.ipynb` from `notebooks/kaggle` (File → Import
   Notebook).
2. Settings → Accelerator → **GPU T4 x2**; Settings → **Internet on**; Settings →
   Environment → **Pin to original environment**. A resumed evaluation must run
   the same library versions, or it refuses to resume. The recorded run did not
   set the pin (deviation D12, [deviations log](../experiments.md#deviations-log));
   that had no effect, because neither notebook resumed.
3. Add Input → the dataset.
4. In the first code cell, set `COMMIT` to the full sha. A placeholder or a branch
   name stops the notebook in that cell, which is what ended Version 1 of the
   recorded training notebook (deviation D13).
5. **Save Version → Save & Run All (Commit).** The notebook runs, in order:
   1. checks the hardware and disk: exactly two T4s, and at least 30 GB free in
      `/tmp`;
   2. clones at `COMMIT` and pins the libraries;
   3. copies the dataset to the pre-registered path in `/tmp` and checks
      `test.jsonl` (sha256 `a4decaaf…`);
   4. refuses to continue unless `config_hash` is `d1961583…`;
   5. plans against the live GPUs;
   6. runs the 10-step smoke test and the per-GPU memory gate (skipped when
      resuming a run that already passed it);
   7. trains, with `--resume-from-checkpoint auto`.

   A long command is stopped at 11¼ hours, before Kaggle's 12-hour limit, so the
   version still ends normally and saves its output; the last cell then says NOT
   FINISHED. The adapter lands in the finished version's output, at
   `outputs/kleos-v007-ministral314breasoning-run1/adapter`.
6. **If it stopped** (the 12-hour limit, an error, the quota):
   1. Add Input → this notebook's previous version **output**.
   2. Save & Run All again.

   The checkpoints are copied back and training resumes from the latest complete
   one. A checkpoint missing optimizer or scheduler state is set aside with an
   `L-F5` line (finding L-F5, [Logos findings](../experiments/logos-findings.md)).
   The notebook always resumes with `auto` and never starts fresh over existing
   checkpoints (finding L-F4).

## Evaluation

ESTIMATED 7 to 8 hours. The recorded evaluation took 5.2 hours, 53.8 s per
answer, in one pass.

1. Import `logos_v002_evaluate.ipynb` from `notebooks/kaggle`, with the same settings
   (including the pinned environment) and the same dataset.
2. Add Input → the training notebook's finished output.
3. Set `COMMIT` to the same sha.
4. **Save & Run All.** The notebook:
   1. rebuilds the benchmark and checks its sha256 (`a11ffad7…`);
   2. evaluates `arm2_finetuned` with `--resume`.
5. **If it stopped,** also attach this notebook's previous version output. Its
   partial file is restored, and finished generations are replayed, not
   regenerated.

## The H9 comparison

On a local machine, CPU only, in a checkout of this repository with the package
installed. Replace each `<private storage>` with the private folder that holds
the run outputs before running a command.

1. Download `outputs/kleos-v007-ministral314breasoning-run1/arm2_finetuned.json`
   from the evaluation notebook's output into
   `<private storage>/outputs/kleos-v007-ministral314breasoning-run1/`. The
   recorded file's sha256 is `0dd74661e1c5a9b1…`.
2. Locate Hermes' annotated results,
   `<private storage>/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json`
   (sha256 `99fdcdc3…`). Cell 17 of the
   [Logos v0.0.1 Colab runbook](logos-v001-colab.md#compare) makes it.
3. Run the primary comparison:

    ```bash
    python scripts/compare.py --cross-model \
        --base <private storage>/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json \
        --finetuned <private storage>/outputs/kleos-v007-ministral314breasoning-run1/arm2_finetuned.json \
        --primary-subset answerable --equivalence-margin 0.02 \
        --report outputs/report_h9_hermes_vs_logos_v002
    ```

4. Run the secondary comparison, Logos v0.0.1 against v0.0.2, the same way with
   v0.0.1's stored results (sha256 `09243630…`) as the base:

    ```bash
    python scripts/compare.py --cross-model \
        --base <private storage>/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json \
        --finetuned <private storage>/outputs/kleos-v007-ministral314breasoning-run1/arm2_finetuned.json \
        --primary-subset answerable --equivalence-margin 0.02 \
        --report outputs/report_h9_logos_v001_vs_v002
    ```

`compare.py` notes that the decoding settings differ (`max_new_tokens` 1024
against 512). That difference is declared under H9. The recorded verdict is in
[H9 result](../experiments.md#h9-result--2026-10-06).

## Risks recorded before the run

Written on 2026-10-06, before training:

- **First run of this code on two GPUs.** The per-GPU probe loop and the
  model-parallel check can only execute on real GPUs. The smoke run is their
  first execution, and it halts the notebook before training if either fails.
- **Kaggle is not Colab.** Its torch version and disk had not been measured. The
  notebook prints both and refuses to start with less than 30 GB free in `/tmp`.
- **Greedy decoding on a reasoning model** can loop until `max_new_tokens`. Such
  answers are counted, not hidden (H9, confounds).

How they turned out on 2026-10-06: the smoke run, the memory gate and the
model-parallel check passed. The two GPUs were loaded unevenly, 4.75 GB and 10.16
GB at the longest batch, with 4.0 GB spare on the fuller one (finding L-F6). 6 of
349 answers looped after a closed trace until the 1,024-token budget ran out
(finding L-F7), and were scored as given.
