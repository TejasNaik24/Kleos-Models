<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kleos-mark-dark.svg">
    <img src="assets/kleos-mark-light.svg" width="40" alt="KLEOS">
  </picture>
</p>

# KLEOS Models documentation

Behavioral fine-tuning and pre-registered evaluation of open-weight models for
KLEOS, an AI operating system for computer science students. The repository
overview, the results tables and the quickstart are in the
[README](../README.md). This page lists every documentation page and the order in
which to read them for a given purpose.

## Reader paths

### Results

1. The results tables in the [README](../README.md).
2. [experiments.md](experiments.md): the status of hypotheses H1 to H9, each
   result beside its decision rule, and the deviations log.
3. The model pages: [KLEOS Hermes](hermes.md) and [KLEOS Logos](logos.md).
4. The run reports in [experiments/](experiments/README.md) for the measurements
   behind each number.

### Run the pipeline

1. [architecture.md](architecture.md): the layers, the configuration system and
   what each model-family adapter encodes.
2. [data-contract.md](data-contract.md): the schema every example satisfies.
3. [training.md](training.md), then [evaluation.md](evaluation.md).
4. [colab.md](colab.md) for one free T4, or [kaggle.md](kaggle.md) for an
   unattended run on two.
5. [troubleshooting.md](troubleshooting.md) when something fails.

The configuration files are described beside them, in
[configs/models](../configs/models/README.md),
[configs/training](../configs/training/README.md),
[configs/datasets](../configs/datasets/README.md) and
[configs/evaluation](../configs/evaluation/README.md).

### Serve a model

1. [deployment.md](deployment.md): deployment packages, pinning, serving profiles
   and the ZeroGPU host.
2. [serving-api.md](serving-api.md): the API that the KLEOS backend integrates
   against.
3. The runbooks: [deploy Hermes](runbooks/deploy-hermes-zerogpu.md) and
   [deploy Logos](runbooks/deploy-logos-zerogpu.md).
4. [publishing.md](publishing.md): base-model pins and licenses, and the two ways
   an adapter can leave a training run.

### The record

1. [experiments.md](experiments.md): the pre-registration and every result.
2. [experiments/README.md](experiments/README.md): the index of the frozen run
   records and their conventions.
3. The run reports, the artifact audit, the Logos findings and the serving
   verification records listed below.
4. The [v0.0.7 repair specification](datasets/kleos-policy-v0.0.7-repair-spec.md),
   written from the audit of v0.0.6; the v0.0.7 release implements part of it.

### Policies

[privacy.md](privacy.md) (the public/private boundary),
[SECURITY.md](../SECURITY.md), [CONTRIBUTING.md](../CONTRIBUTING.md),
[CHANGELOG.md](../CHANGELOG.md) and the [license](../LICENSE).

## All pages

| Page | Kind | Purpose |
| --- | --- | --- |
| [README](../README.md) | Overview | What the repository is, results, models, method, quickstart, limitations and citation |
| [hermes.md](hermes.md) | Model | KLEOS Hermes v0.0.6: base and pins, recipe, the H1 replication, consistency and abstention, serving status and limitations |
| [logos.md](logos.md) | Model | The KLEOS Logos line: v0.0.1 (research only) and v0.0.2 (served as Beta), the choice of Ministral 3, measured fits, H8 and H9 in brief, serving and limitations |
| [experiments.md](experiments.md) | Record | Pre-registered hypotheses H1 to H9 with status and results, the results log, the deviations log and the fixed protocol |
| [experiments/README.md](experiments/README.md) | Record index | The frozen run records, the identifier conventions and the evidence labels |
| [Ministral-8B artifact audit](experiments/kleos-v006-ministral8b-run1-artifact-audit.md) | Record | Audit of the first run, `kleos-v006-ministral8b-run1` (2026-09-15), with findings F1 to F3 |
| [Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md) | Record | Training and evaluation of KLEOS Hermes, `kleos-v006-mistralnemo12b-run1`, with findings H-F1 to H-F14 |
| [Logos v0.0.1 run report](experiments/kleos-v006-ministral314b-run1-report.md) | Record | Training and evaluation of KLEOS Logos v0.0.1, `kleos-v006-ministral314b-run1`, tested under H8 |
| [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md) | Record | Training and evaluation of KLEOS Logos v0.0.2, `kleos-v007-ministral314breasoning-run1`, tested under H9, with findings L-F6 to L-F8 |
| [experiments/logos-findings.md](experiments/logos-findings.md) | Record | Findings L-F1 to L-F5 from the Logos v0.0.1 work, and an index of L-F6 to L-F8 |
| [experiments/serving-verification-records.md](experiments/serving-verification-records.md) | Record | Verification of the served models: Hermes on Colab and on ZeroGPU (9 of 9), and Logos v0.0.2's smoke test over two quota windows and acceptance decision |
| [architecture.md](architecture.md) | Reference | Layers, the torch-free rule, configuration and `config_hash`, the model-family adapters, feasibility, training guards, evaluation principles and serving |
| [training.md](training.md) | Reference | The training pipeline: recipe, QLoRA settings, guards, loss masking, reasoning-supervised runs, memory, two GPUs, checkpointing, resume and reproducibility |
| [evaluation.md](evaluation.md) | Reference | Arms, graders including `kleos_policy`, the thinking split, results schema 3, clustered intervals, cross-model verdicts, rescoring and resume |
| [data-contract.md](data-contract.md) | Reference | The versioned example schema, the consistency and split keys, and the dataset manifest |
| [colab.md](colab.md) | How-to | Training and evaluating on a free Colab T4 |
| [kaggle.md](kaggle.md) | How-to | Unattended runs on Kaggle's free 2 × T4: private dataset, output probe, session guard and resume |
| [troubleshooting.md](troubleshooting.md) | How-to | Known errors, what causes them and what to change |
| [deployment.md](deployment.md) | Reference | Deployment packages, revision pinning, tokenizer and runtime contracts, serving profiles, the inference service, the container, the ZeroGPU host and the verification status |
| [serving-api.md](serving-api.md) | Reference | The integration API: configuration variables, request, reply and status for contracts v1 and v2, fallback rules, measured capacity and latency |
| [runbooks/deploy-hermes-zerogpu.md](runbooks/deploy-hermes-zerogpu.md) | Runbook | Deploying KLEOS Hermes to a private ZeroGPU Space |
| [runbooks/deploy-logos-zerogpu.md](runbooks/deploy-logos-zerogpu.md) | Runbook | Deploying KLEOS Logos v0.0.2 to a private ZeroGPU Space, with the smoke test over two quota windows |
| [runbooks/logos-v001-colab.md](runbooks/logos-v001-colab.md) | Runbook | Training and evaluating Logos v0.0.1 on Colab, as run for H8 |
| [runbooks/logos-v002-kaggle.md](runbooks/logos-v002-kaggle.md) | Runbook | Training and evaluating Logos v0.0.2 on Kaggle, and the H9 comparison commands |
| [publishing.md](publishing.md) | Reference | Base-model pins and licenses; publishing a public adapter compared with uploading a private deployment package |
| [privacy.md](privacy.md) | Policy | The public/private boundary: what may be committed and what stays in private storage |
| [datasets/kleos-policy-v0.0.7-repair-spec.md](datasets/kleos-policy-v0.0.7-repair-spec.md) | Record | Requirements for the v0.0.7 data release, written from the audit of v0.0.6, with a status note on what shipped |
| [SECURITY.md](../SECURITY.md) | Policy | Reporting a vulnerability, accidental data exposure, secrets handling and automated checks |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Project | Setup, checks, design constraints, adding a model family or a grader, generated artifacts and the pull-request checklist |
| [CHANGELOG.md](../CHANGELOG.md) | Project | Release history |
| [configs/models](../configs/models/README.md) | Directory guide | The five model configs with measured training memory, vision-language checkpoints and revision pinning |
| [configs/training](../configs/training/README.md) | Directory guide | Every training config with its status, the frozen configs and their pinned hashes |
| [configs/datasets](../configs/datasets/README.md) | Directory guide | The fixture, sealed-release and template dataset configs, and the split strategies |
| [configs/evaluation](../configs/evaluation/README.md) | Directory guide | The four evaluation configs, the arms and the graders |
| [data](../data/README.md) | Directory guide | What under `data/` is committed, and why the fixtures are not the research dataset |
| [data/examples](../data/examples/README.md) | Directory guide | The synthetic development fixtures and what each exercises |
| [notebooks](../notebooks/README.md) | Directory guide | The Colab and Kaggle notebooks and how they are generated |
| [tests/fixtures](../tests/fixtures/README.md) | Directory guide | The vendored chat templates and their pinned hashes |
| [Hermes Space card](../deploy/zerogpu-space/README.md) | Space card | The Hugging Face card of the Hermes ZeroGPU Space |
| [Logos Space card](../deploy/zerogpu-space-logos/README.md) | Space card | The Hugging Face card of the Logos ZeroGPU Space |

## Conventions

Identifiers. H1 to H9 are the pre-registered hypotheses in
[experiments.md](experiments.md), and D1 to D13 the dated deviations from the
pre-registered protocol in its deviations log. F1 to F3 are the findings of the
[Ministral-8B artifact audit](experiments/kleos-v006-ministral8b-run1-artifact-audit.md),
H-F1 to H-F14 the findings in section 11 of the
[Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md), and
L-F1 to L-F8 the Logos findings (L-F1 to L-F5 in
[logos-findings.md](experiments/logos-findings.md), L-F6 to L-F8 in section 10 of
the [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md)).
A page expands and links an identifier at its first mention and uses the bare
identifier after that.

Placeholders. `<owner>/<space>`, `<owner>/<package-repo>`,
`<private storage>/outputs/<experiment-id>`, `<path to release>` and
`<commit sha>` stand for values that are private or specific to one deployment.
Secrets appear as environment variables such as `$HF_TOKEN`, never as values.

Evidence labels. A number is marked measured when it was read from a run (a
manifest, the memory probe, a log), estimated when it comes from the feasibility
estimator or from arithmetic, and not tested when neither exists. The frozen
records use their own labels, defined in
[experiments/README.md](experiments/README.md#evidence-labels): VERIFIED,
ESTIMATED, SOURCE, OBSERVED, INFERRED and NOT VERIFIED. Numbers keep the
precision and units they were recorded in; GiB and GB are not converted. The
code and the manifests report memory in "GB" meaning 2^30 bytes, the same
quantity as GiB; figures are quoted in the unit each record used. Dates are
ISO 8601.

Frozen records. The run reports, the audit, the Logos findings and the serving
verification records under `experiments/`, and the record sections of
`experiments.md`, were written while the work was done and are preserved as
written, including their British spelling. Each record file under
`experiments/` carries a notice at the top.
Editorial changes for public release are limited to replacing private storage
paths and account identifiers with placeholders, marked as editor's notes, and
retargeting links to documents that moved. The repair specification is preserved
the same way, with a status note on what shipped in v0.0.7.
