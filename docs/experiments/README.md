# Experiment records

This folder holds the research record of the KLEOS fine-tuning work: a report for
each completed run, the findings and serving records that go with them, and the
conventions they share. The pre-registered hypotheses, the fixed protocol, the
results log and the deviations log are in [experiments.md](../experiments.md).
The model pages, [Hermes](../hermes.md) and [Logos](../logos.md), summarize the
same results for each model.

## Records

| Record | What it covers | Written |
| --- | --- | --- |
| [Ministral-8B artifact audit](kleos-v006-ministral8b-run1-artifact-audit.md) | `kleos-v006-ministral8b-run1`, the first KLEOS run: artifact hashes, the H1 result checked against the stored evaluation files, findings F1 to F3 | 2026-09-15 |
| [Hermes run report](kleos-v006-mistralnemo12b-run1-report.md) | `kleos-v006-mistralnemo12b-run1`, KLEOS Hermes v0.0.6: training, evaluation, the H1 replication, findings H-F1 to H-F14 | 2026-09-22; H-F11 to H-F14 added 2026-09-24 |
| [Logos v0.0.1 run report](kleos-v006-ministral314b-run1-report.md) | `kleos-v006-ministral314b-run1`: training over four Colab sessions, H8a and H8b | 2026-10-01 |
| [Logos v0.0.2 run report](kleos-v007-ministral314breasoning-run1-report.md) | `kleos-v007-ministral314breasoning-run1`: training and evaluation on Kaggle's 2 × T4, H9, findings L-F6 to L-F8 | 2026-10-06 |
| [Logos findings](logos-findings.md) | Findings L-F1 to L-F5 in full, and an index of L-F6 to L-F8 | 2026-09-24 to 2026-10-06 |
| [Serving verification records](serving-verification-records.md) | Hermes v0.0.6 reproduced 9 of 9 on a Colab T4 and on ZeroGPU; Logos v0.0.2 reproduced 8 of 9 on ZeroGPU, and the decision to serve it as a Beta | 2026-09-23; 2026-10-07 |
| [v0.0.7 repair specification](../datasets/kleos-policy-v0.0.7-repair-spec.md) | What the data release after `kleos-policy-v0.0.6` had to fix, with a status preface | 2026-09-24 |

The run artifacts themselves (adapters, checkpoints, manifests, result files) are
private and are not in this repository. Each report's hash register is the record
of them.

## Identifiers

Each id is assigned once and never reused. A finding keeps its id after it is
fixed, and its status is recorded beside it.

| Id | Meaning | Defined in |
| --- | --- | --- |
| H1 to H9 | Pre-registered hypotheses | [experiments.md](../experiments.md#status) |
| D1 to D13 | Deviations from the pre-registered protocol, recorded when they happened | [experiments.md, deviations log](../experiments.md#deviations-log) |
| F1 to F3 | Findings of the Ministral-8B artifact audit | [audit, findings](kleos-v006-ministral8b-run1-artifact-audit.md#findings) |
| H-F1 to H-F14 | Findings recorded with the Hermes run. H-F11 to H-F14 concern how the evaluation measures | [Hermes run report, section 11](kleos-v006-mistralnemo12b-run1-report.md#11-findings) |
| L-F1 to L-F8 | Findings from the Logos work: L-F1 to L-F5 from v0.0.1, L-F6 to L-F8 from v0.0.2 | [Logos findings](logos-findings.md) |
| `arm0` to `arm3` | Evaluation arms: `arm0_base`, `arm1_base_orchestrated`, `arm2_finetuned`, `arm3_finetuned_orchestrated` | [evaluation.md](../evaluation.md) |

## Evidence labels

The records label how each claim is known, with these six labels:

| Label | Meaning |
| --- | --- |
| **VERIFIED** | Checked in this repository against the pinned artifacts: by a test, a hash or the library source |
| **ESTIMATED** | Computed, not measured, for example by the memory estimator or from observed quota use |
| **SOURCE** | Quoted from a published source, which is linked |
| **OBSERVED** | What was seen on a platform (finding L-F3) |
| **INFERRED** | The cause deduced from what was observed (finding L-F3) |
| **NOT VERIFIED** | Stated in advance and not yet checked |

Unlabeled run figures (losses, peaks, scores, timings) are measurements recorded
by the run. Units are kept as each record states them: GB and GiB are not
converted. The code and the manifests report memory in "GB" meaning 2^30 bytes,
the same quantity as GiB; figures are quoted in the unit each record used.
Scores have four decimals and confidence intervals are 95%. With 2,000 bootstrap
resamples, a p-value bound means that no resample crossed zero. H1 prints this
as p < 0.001; from H8 on it is p < 0.0005 in either bootstrap.

## Placeholders

The public records replace private locations and account identifiers with
placeholders:

| Placeholder | Stands for |
| --- | --- |
| `<private storage>` | The private storage outside this repository that holds the sealed releases and the run outputs |
| `<kaggle notebook>` | A private Kaggle notebook's id |
| `<owner>/<space>`, `<owner>/<package-repo>` | A private Hugging Face Space and the private package repository it loads |
| `<path to release>` | A local copy of a sealed data release |

Paths such as `outputs/<experiment-id>/adapter/` in the reports are relative to
the private outputs folder. Two exceptions are kept on purpose: the literal
`--dataset` and `--output-dir` paths under H8 and H9 in
[experiments.md](../experiments.md) are inputs to the recorded `config_hash`, so
they stay as recorded, each marked with an editor's note.

## The record notice

A frozen record is not edited after the fact: no number, table, finding or
decision in it changes. A later correction is added beside the original as a
dated note, as with the F1 re-grade in the audit and the H-F11 correction note in
experiments.md. Every frozen record in this folder, and the repair
specification, opens with this notice:

> This is a frozen research record, written while the work was done and
> preserved as written. Editorial changes for public release are limited to
> replacing private storage paths and account identifiers with placeholders
> (marked as editor's notes) and retargeting links to documents that moved. No
> number, table, finding or decision was changed. Conventions:
> docs/experiments/README.md.

An editorial change made for public release is marked inline as
*[editor's note: …]*. Frozen records keep their original spelling.
