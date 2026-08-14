# Changelog

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes that can invalidate comparisons between existing experiment runs — a
metric definition, a split strategy, a default hyperparameter, the capability
suite — are marked **[research-affecting]**.

## [Unreleased]

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
