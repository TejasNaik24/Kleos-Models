# Changelog

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes that can invalidate comparisons between existing experiment runs — a
metric definition, a split strategy, a default hyperparameter, the capability
suite — are marked **[research-affecting]**.

## [Unreleased]

### Added

- **KLEOS Logos v0.0.2**, pre-registered as H9 (`docs/experiments.md`, `docs/logos.md` §11):
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
- **Kaggle notebooks** for Logos v0.0.2 (`notebooks/kaggle/`,
  `scripts/build_kaggle_notebooks.py`).
- **Two-GPU support** (`device_map: auto`):
  - a per-GPU memory probe that gates on the tighter GPU;
  - a refusal to train a split model unless the Trainer runs it model parallel;
  - per-GPU feasibility estimates (`--simulate-gpu NAME:GIB:CC:COUNT`);
  - the device map recorded in the manifest.
- **Logos v0.0.2 serving** (built and tested, not deployed; `docs/deployment.md`):
  - **Serving profiles** (`serving/profile.py`): a deployment record's optional
    `serving` block names the model's secrets prefix, key header, Space folder,
    baked base files and reply options. Without one, every value is Hermes'.
  - `configs/deployment/kleos_logos_v002.yaml` and `deploy/zerogpu-space-logos/`:
    a private ZeroGPU Space for Logos (`LOGOS_*`, `X-Logos-Key`, 1,024 tokens).
  - **Reply contract version 2:** adds `reasoning`, the trace beside the answer.
    A Logos request must start with a system message. Hermes stays on version 1.
  - `scripts/fill_deployment_record.py`: completes a record from its frozen
    training run, after checking the run is the one the record describes.
  - The reference client serves either model (`from_env(prefix=, key_header=,
    contract_version=)`); its own refusals name the model and carry its version.
  - The smoke test compares each trace as well as each answer, passes a thinking
    model only when every trace was compared, and refuses a reference from another
    evaluation before it contacts the Space.

### Changed

- **`finish_reason` in a served reply honours the backend's never-closed signal:**
  "length" when the budget filled or the thinking never closed (L-F7). Hermes'
  replies are unchanged.
- **Upload, Space staging and the smoke test use the stored `hf auth login` token**
  when `HF_TOKEN` is unset.

- **Results schema 3.** Each record adds `finish_reason` and,
  when the model thought, its `reasoning`. `generation_stats` adds
  `thinking_truncated` and `reasoning_chars`. Nothing existing changes, and
  single-GPU estimates and manifests are byte-identical.
- **A reasoning-supervised run refuses to start** if truncation would cut a target.
  Other runs drop and count the `reasoning` field.
- **Leakage detection ignores reasoning traces.** Only train and validation carry
  them, so comparing with them would hide a train copy of a test conversation.
  Content and dataset hashes still cover them.
- **The thinking-only error message no longer suggests a Qwen model.**

## [0.1.0] — 2026-08-14

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
- Qwen3 dense, Qwen3 MoE, Mistral dense, Mistral3 vision-language
- Reasoning modelled as a validated capability, not a prompt string
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
- transformers 4.56 ↔ 5.x shim covering `warmup_ratio`/`warmup_steps`,
  `tokenizer`/`processing_class`, `torch_dtype`/`dtype`, and removed arguments

**Documentation**
- Nine documents including pre-registered hypotheses in `docs/experiments.md`

### Verification

- 564 tests passing
- Real LoRA training verified end to end on tiny randomly-initialized Qwen3 and
  Mistral models on CPU, including a loss-decrease assertion and a base-weight
  freeze check
- Verified against transformers 5.15.0, torch 2.13.0, peft 0.20.0
- **GPU training was not executed.** The 4-bit path is implemented for CUDA and
  is launched from Colab.

### Known limitations

- The bundled dataset is synthetic development fixtures, not research data
- Hyperparameters are engineering defaults, not tuned values
- The heuristic rubric grader is coarse and not a substitute for human judging
- Frontier reference arms are interface-only
- Mistral-Small-24B and Qwen3-30B-A3B cannot be trained on a free-tier T4
