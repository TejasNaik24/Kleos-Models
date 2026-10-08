# Changelog

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes that can invalidate comparisons between existing experiment runs (a
metric definition, a split strategy, a default hyperparameter, the capability
suite) are marked **[research-affecting]**. Releases 0.2.0 to 0.4.0 were
backfilled from the commit history on 2026-10-07; each lists its commit range.
The research record itself is [docs/experiments.md](docs/experiments.md).

## [Unreleased]

Documentation rewrite for public release, and removal of a model family that
no KLEOS model uses.

### Added

- `CITATION.cff` (CFF 1.2.0) and a citation section in the README.
- `.claude/` ignored in `.gitignore`.
- Documentation for outside readers: an index (`docs/README.md`), model pages
  (`docs/hermes.md`, `docs/logos.md`), runbooks under `docs/runbooks/`,
  `docs/kaggle.md`, and an index of the frozen research records
  (`docs/experiments/README.md`).

### Changed

- The commit history was rewritten to remove assistant co-author trailers from seven
  2026-09 commit messages; ids cited in the records map to the new ids in
  [docs/experiments/README.md](docs/experiments/README.md#commit-identifiers-after-the-history-rewrite).
- **Documentation rewritten** for outside readers: README, CONTRIBUTING,
  SECURITY, this changelog and the reference pages under `docs/`.
  `docs/kleos-hermes-integration.md` is now `docs/serving-api.md`. The serving
  verification records moved from `docs/deployment.md` to
  `docs/experiments/serving-verification-records.md`, and Logos findings L-F1
  to L-F5 to `docs/experiments/logos-findings.md`. Frozen research records keep
  every number, table, finding and decision; only private storage paths and
  account identifiers were replaced with placeholders, each marked as an
  editor's note.
- `configs/training/qlora_small.yaml` and `configs/training/debug.yaml`, the
  defaults for the tests, CI, `scripts/smoke_test.py` and the notebooks, now
  include `configs/models/mistral_nemo_12b.yaml` (Apache-2.0, ungated, Hermes'
  base).
- The `notes` field of `configs/models/ministral_8b.yaml` was reworded. `notes`
  is part of `config_hash`, so the current hash of
  `configs/training/kleos_policy_v006.yaml` and `debug_ministral.yaml` changes.
  No test pins either, and the recorded Ministral-8B run hash
  (`3fbb3f90ed9662ee`) already did not reproduce at HEAD once the base was
  pinned (see "Base-model revision" in `docs/experiments.md`). The pinned
  Hermes, Logos v0.0.1 and Logos v0.0.2 hashes are unchanged.
- `pyproject.toml`: version 0.4.0, author Tejas Naik, project URLs pointing at
  this repository.
- `LICENSE`: copyright holder Tejas Naik; the base-model note names the Mistral
  models this project uses (Mistral-Nemo, Ministral 3, Ministral-8B, Mistral
  Small).
- Script and module docstrings no longer cite spec section numbers.

### Removed

- The Qwen model configs (`qwen3_8b.yaml`, `qwen3_30b_a3b_thinking.yaml`), their
  two adapters and their tests. The family has been excluded from KLEOS since
  2026-09-15 (H5 and H6 in `docs/experiments.md`) and no recorded run used it.
  `ModelFamilyAdapter` stays, with the four Mistral adapters as its
  implementations.
- `ReasoningCapability.SWITCHABLE` and `ReasoningMode.NON_THINKING`, which only
  that family used. Enum members are not serialized defaults, so no recorded
  `config_hash` changes.
- The `kleos` console script from `pyproject.toml`; its module did not exist.

## [0.4.0] - 2026-10-07

KLEOS Logos v0.0.2: trained to think, measured better than Hermes (H9), and
live on ZeroGPU as a Beta. Commits `a17ace7` to `873fbcc`.

### Added

- **KLEOS Logos v0.0.2**, pre-registered as H9 on 2026-10-06 (`docs/experiments.md`, `docs/logos.md`):
  - `configs/models/ministral3_14b_reasoning.yaml` (Ministral 3 14B Reasoning @ `51f9210f`);
  - `configs/training/kleos_logos_v002.yaml` (config hash `d1961583…`) and `debug_logos_v002.yaml`;
  - `configs/evaluation/kleos_logos_v002.yaml`;
  - `configs/datasets/kleos_policy_v007.yaml`.
- **Schema 1.1:** an optional assistant `reasoning` field (kleos-policy-v0.0.7). It is
  serialized only when present; earlier releases keep their bytes and hashes.
- `Ministral3ReasoningTextAdapter` (`model_type: ministral3_reasoning`): the text-only
  view, thinking-only.
- **[research-affecting, thinking models only] Thinking split:** in thinking mode a
  completion is split at the `[/THINK]` token before decoding, and only the answer
  is graded (`split_thinking`). Non-thinking runs decode exactly as before.
- **Kaggle notebooks** for Logos v0.0.2 (`notebooks/kaggle`,
  `scripts/build_kaggle_notebooks.py`).
- **Two-GPU support** (`device_map: auto`); single-GPU estimates and manifests
  stay byte-identical:
  - a per-GPU memory probe that gates on the tighter GPU;
  - a refusal to train a split model unless the Trainer runs it model parallel;
  - per-GPU feasibility estimates (`--simulate-gpu NAME:GIB:CC:COUNT`);
  - the device map recorded in the manifest.
- **Logos v0.0.2 serving** (`docs/deployment.md`):
  - **Serving profiles** (`serving/profile.py`): a deployment record's optional
    `serving` block names the model's secrets prefix, key header, Space folder,
    baked base files and reply options. Without one, every value is Hermes'.
  - `configs/deployment/kleos_logos_v002.yaml` and `deploy/zerogpu-space-logos/`:
    a private ZeroGPU Space for Logos (`LOGOS_*`, `X-Logos-Key`, 1,024 tokens).
  - **Reply contract version 2:** adds `reasoning`, the trace beside the answer.
    A Logos request must start with a system message. Hermes stays on version 1.
  - `scripts/fill_deployment_record.py`: completes a record from its frozen
    training run, after checking the run is the one the record describes. The
    Logos record was completed from the run on 2026-10-06 (adapter
    `e49724f6554db662…`, `checkpoint-175`).
  - The reference client serves either model (`from_env(prefix=, key_header=,
    contract_version=)`); its own refusals name the model and carry its version.
  - The smoke test compares each trace as well as each answer, passes a thinking
    model only when every trace was compared, and refuses a reference from another
    evaluation before it contacts the Space.
  - The deployment package README states the research revision.
- **Logos v0.0.2 deployed** on a private ZeroGPU Space, 2026-10-07: the smoke test
  reproduced 8 of 9 answers byte for byte, answer and trace; one diverged late in
  its trace on the Space's GPU and changed its decision. Accepted and documented
  as a Beta.

### Changed

- **`finish_reason` in a served reply honors the backend's never-closed signal:**
  "length" when the budget filled or the thinking never closed (L-F7). Hermes'
  replies are unchanged.
- **Upload, Space staging and the smoke test use the stored `hf auth login` token**
  when `HF_TOKEN` is unset.
- **Results schema 3.** Each record adds `finish_reason` and,
  when the model thought, its `reasoning`. `generation_stats` adds
  `thinking_truncated` and `reasoning_chars`. Nothing existing changes.
- **A reasoning-supervised run refuses to start** if truncation would cut a target.
  Other runs drop and count the `reasoning` field.
- **Leakage detection ignores reasoning traces.** Only train and validation carry
  them, so comparing with them would hide a train copy of a test conversation.
  Content and dataset hashes still cover them.
- **The thinking-only error message no longer suggests a switchable-thinking
  checkpoint.**
- Comments and docstrings trimmed across the code base: banners, narration and
  anecdotes removed, the reasons kept.

## [0.3.0] - 2026-10-01

KLEOS Logos v0.0.1 and corrected evaluation measures. Commits `ec4f9e3` to
`c4c3a41`.

### Added

- **KLEOS Logos v0.0.1**, pre-registered as H8 on 2026-09-24, before any Logos
  training (`docs/logos.md`):
  - `configs/models/ministral3_14b.yaml` (Ministral 3 14B Instruct 2512 BF16 @ `3cea74c1`);
  - `configs/training/kleos_logos_v001.yaml` (config hash `18008c67…`) and `debug_logos.yaml`;
  - `configs/evaluation/kleos_logos_v001.yaml`.
- `Ministral3TextAdapter`: loads a Ministral 3 checkpoint's text tower without its
  vision encoder.
- **[research-affecting] Corrected measures** (`evaluation/corrections.py`,
  findings H-F11 to H-F14), reported beside the originals: a paired cluster
  bootstrap by `group_id`, consistency by `group_id` with its oracle ceiling,
  separate answerable and should-decline subsets, and the citation heuristic's
  gold floor. From H8 on, every interval resamples groups (protocol amendment,
  2026-09-24); no earlier result is re-decided.
- `scripts/rescore.py --mode annotate`: re-reports stored results with the
  corrected measures, leaving the source files untouched.
- `compare.py --cross-model --primary-subset --equivalence-margin`: compares two
  models on one arm and decides a pre-registered verdict (better, worse,
  equivalent or inconclusive) on a subset's cluster interval.
- **Resumable evaluation** (`evaluation/resume.py`, `evaluate.py --resume`): finished
  generations are kept in a partial file, and a resume is refused if anything
  that could change a generation differs.
- **Smoke gate** (`scripts/check_smoke_gate.py`): a GO / NO-GO decision for a full
  training run from a smoke run's manifest.
- **Memory probe:** a training step measured on the longest micro-batch before
  step 1. The rewritten feasibility estimator predicted Logos v0.0.1's training
  peak within 0.5%.
- `docs/datasets/kleos-policy-v0.0.7-repair-spec.md`: the dataset repair
  specification behind kleos-policy-v0.0.7.
- `tests/test_vlm_targeting_finding.py`: finding L-F1 (the vision-language
  adapter's LoRA scoping on transformers 5) held as strict xfails.
- **Results recorded:** H8b inconclusive (2026-09-29) and H8a supported
  (2026-10-01), with the full run report and findings L-F1 to L-F5.

### Fixed

- **The best checkpoint is protected from pruning** (finding H-F9).
- **Evaluation resume on a Colab Drive mount** (L-F3): the partial file is closed
  after every record, because the mount uploads only closed files.
- Measured sequence lengths corrected in docs, comments and tests (L-F2).

## [0.2.0] - 2026-09-23

The first KLEOS results and KLEOS Hermes v0.0.6, from training to free hosting.
Commits `12361d5` to `c8b9825`.

### Added

- **[research-affecting] The `kleos_policy` grader**, scoring ranking, deciding
  factor and confidence together, with `format_valid` reported separately and
  excluded from the score (deviation D6); `scripts/build_benchmark.py`, which
  derives the benchmark from a sealed test split; and the
  `kleos_policy_v006` dataset, evaluation and training configs with
  `debug_ministral.yaml`.
- **First KLEOS result** (`kleos-v006-ministral8b-run1`, Ministral-8B,
  2026-09-15): H1 supported, H2 not measurable, H3 improved but still poor,
  recorded with its deviations in `docs/experiments.md`.
- **Ministral-8B artifact audit**
  (`docs/experiments/kleos-v006-ministral8b-run1-artifact-audit.md`): hashes,
  sizes, the adapter configuration and an independent check of the H1 numbers
  against the stored evaluation files; findings F1 to F3.
- `scripts/rescore.py`: re-grades stored evaluation results offline from their
  saved responses. It refuses to write to the source and re-hashes it afterwards.
- **Base-revision pinning** (finding F3): `configs/models/ministral_8b.yaml` and
  `configs/deployment/kleos_v006_ministral8b.yaml` pin `2f494a19…`, and
  `tests/test_revision_pinning.py` keeps them in step. Generated model cards
  record the serving revision.
- **KLEOS Hermes v0.0.6** (`kleos-v006-mistralnemo12b-run1`, Mistral-Nemo-Instruct-2407
  @ `04d8a905`, Apache-2.0): `configs/models/mistral_nemo_12b.yaml`,
  `configs/training/kleos_hermes_v006.yaml` and `debug_nemo.yaml`. H1 replicated
  on 2026-09-22: 7/7 tasks improved at p < 0.001, with the run report and findings
  H-F1 to H-F10.
- **Deployment package** (`scripts/build_deployment_package.py`,
  `verify_deployment_package.py`, `serving/manifest.py`): pins the base revision,
  freezes and hashes the tokenizer files and the adapter, and is verified before
  anything is served. Package schema v2 adds the runtime contract (NF4, double
  quantization, float16 compute stated; a bfloat16 package is refused).
  `configs/deployment/kleos_hermes_v006.yaml` is Hermes' record.
- **Tokenizer contract:** `fix_mistral_regex` is stated in the package and passed
  explicitly by the loader.
- **Inference service:** `scripts/serve_hermes.py` (FastAPI, API-key
  authentication) and `scripts/hermes_smoke.py`, which compares served responses
  with the frozen evaluation.
- **Container:** `docker/hermes.Dockerfile` and `docker/compose.yaml`; runs as a
  non-root user, takes `_FILE` secrets, and builds from an allowlisted context.
- **ZeroGPU hosting:** `deploy/zerogpu-space/`, a serving path that holds the GPU
  only for generation (`serving/zerogpu.py`), the status contract
  (`serving/status.py`), a reference client that returns a status instead of
  raising (`serving/client.py`), `scripts/stage_zerogpu_space.py`,
  `upload_deployment_package.py` and `zerogpu_smoke.py`, and the integration
  contract for the KLEOS backend.
- **Hermes verified** on 2026-09-23: 9/9 responses byte-identical to the frozen
  evaluation on Colab, and again on a private ZeroGPU Space.
- `requirements.txt`.

### Changed

- **[research-affecting] nDCG is bounded** so repeated items cannot score above a
  perfect ranking (finding F1). Both Ministral-8B arms were re-graded offline:
  the baseline went from 0.5231 to 0.4744, the fine-tuned score stayed 0.8015,
  and 7/7 tasks became significant.
- The `kleos_policy_v006` run evaluates and checkpoints every 50 steps instead of
  25, from measured T4 timings.
- `configs/models/ministral_8b.yaml` declares `model_type: ministral`, as
  transformers 5 reports it.
- A published adapter leaves the tokenizer out by name, where before the scan's
  5 MB size cap refused it by accident (finding F2); the model card says to load
  the tokenizer from the pinned base.

### Fixed

- **[research-affecting] The system prompt is no longer dropped in training on
  Mistral templates** (deviation D5): when the template drops the system turn,
  `ConversationFormatter` folds it into the first user turn, which also restores
  assistant-only masking. Found by the smoke test and fixed before the first run.
- `MistralDenseAdapter` resolves both `mistral` and `ministral` model types.
- `extract_ranking` resolves prose lines to candidate names (deviation D4),
  before any result was produced.
- Test collection under bare `pytest` in CI, and the notebook CLI-flag test uses
  `sys.executable`.
- The ZeroGPU Space pins pydantic 2.12.5, the version Gradio's `mcp` extra allows.
- The Hermes adapter is read on the CPU at ZeroGPU startup.

## [0.1.0] - 2026-08-14

Initial release: the complete research infrastructure. No KLEOS fine-tuning
result exists yet.

### Added

**Configuration**
- Typed, composable YAML configuration with `extends` and `includes`
- Stable `config_hash` over the fully resolved configuration
- CLI overrides via `--set key.path=value`

**Data layer** (no torch required)
- Versioned schema for training and evaluation examples
- Variation axes with marginal, joint and gap coverage reporting
- Validation: duplicates, coverage, quality gate, placeholder text,
  sensitive-content patterns, policy-vs-fact heuristics
- Six split strategies: random, group, entity/domain/format holdout,
  scenario-family holdout
- Leakage detection: exact, normalized, near-duplicate (MinHash + LSH),
  id collisions, scenario repeats, entity leakage
- Assistant-only loss masking via incremental chat-template application

**Model layer**
- `ModelFamilyAdapter` abstraction with four implementations
- Mistral dense and Mistral3 vision-language adapters, plus a dense and a
  mixture-of-experts adapter for a second family (removed in [Unreleased](#unreleased))
- Reasoning modeled as a validated capability, not a prompt string
- LoRA target validation against the loaded model's real modules
- Memory estimation and feasibility tiers without a GPU
- 4-bit NF4 quantization with automatic compute-dtype selection

**Training**
- Real QLoRA pipeline on Hugging Face `Trainer` + PEFT
- Pre-flight report: GPU, VRAM, CUDA, versions, model, LoRA, memory estimate
- Gradient verification before the training loop
- Checkpoint discovery, validation and `--resume-from-checkpoint auto`
- Retention with a floor so it can never leave zero checkpoints
- Structured JSONL logging that never records example text

**Evaluation**
- Four research arms behind a `Backend` protocol
- Classification, ranking (nDCG, Kendall τ, footrule), set, rubric metrics
- Consistency testing with separate agreement and correct-agreement rates
- OOD reporting with in-distribution, OOD and gap kept separate
- Faithfulness heuristics: evidence coverage, citation precision,
  unsupported-claim rate
- Capability-preservation framework with a versioned fixed suite
- Paired bootstrap significance testing
- Comparison reports that decline to claim a non-significant win

**Experiments**
- Mandatory manifest per run, including failed runs
- Registry with comparability checking
- Full environment and git capture, degrading gracefully

**Tooling**
- 15 CLI scripts
- Four Colab notebooks, generated from reviewable Python
- `colab_setup.py` that installs without clobbering Colab's torch
- Private-data scanner with a pre-commit hook
- CI: privacy scan, lint, types, tests on 3.11-3.13, tests with CPU torch,
  config/schema/fixture/notebook drift checks, smoke tests

**Compatibility**
- transformers 4.56 to 5.x shim covering `warmup_ratio`/`warmup_steps`,
  `tokenizer`/`processing_class`, `torch_dtype`/`dtype`, and removed arguments

**Documentation**
- Nine documents including pre-registered hypotheses in `docs/experiments.md`

### Verification

- 564 tests passing
- Real LoRA training verified end to end on tiny randomly initialized models of
  both families (Mistral and the family removed in [Unreleased](#unreleased)) on CPU,
  including a loss-decrease assertion and a base-weight freeze check
- Verified against transformers 5.15.0, torch 2.13.0, peft 0.20.0
- **GPU training was not executed.** The 4-bit path is implemented for CUDA and
  is launched from Colab.

### Known limitations

- The bundled dataset is synthetic development fixtures, not research data
- Hyperparameters are engineering defaults, not tuned values
- The heuristic rubric grader is coarse and not a substitute for human judging
- Frontier reference arms are interface-only
- Mistral-Small-24B and the 30B mixture-of-experts config (removed in
  [Unreleased](#unreleased)) cannot be trained on a free-tier T4
