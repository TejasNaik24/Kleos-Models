# KLEOS Logos v0.0.2 serving (sub-project 3a): design

**Status:** design approved section by section, 2026-10-06. Next: the implementation plan.
**Part B** (the KLEOS app uses Logos) gets its own spec after Logos is live and smoke-tested.

## Goal

Serve Logos v0.0.2 from its own private Hugging Face ZeroGPU Space, exactly as it was
evaluated under H9, using the same serving code that runs Hermes. Hermes is untouched:
- its code paths, replies and tests stay byte for byte the same;
- its live Space keeps running at its pinned commit.

## Decisions (the owner's, 2026-10-06)

- **Visibility.** A private package repo and a private Space, as for Hermes.
- **Served as evaluated.** Greedy decoding, a 1,024-token budget, and the trace split off at
  `[/THINK]`. There is no repetition guard. A cut or looping answer is marked, and KLEOS
  falls back to the default model.
- **The trace.** It is returned beside the answer. KLEOS shows it as a collapsible
  "How Logos decided" (Part B).
- **Quota.** Stay on the free ZeroGPU tier. KLEOS caps Logos per day (Part B).
- **Order.** Part A (this spec), then Part B.

## Non-goals

- The KLEOS app (Part B).
- Any public repository, model card or demo.
- Redeploying or changing Hermes.
- A Docker image for Logos: the Docker path stays Hermes-only.
- Changing what the model generates. That includes repetition penalties, stop strings and
  sampling.

---

## 1. Architecture: a serving profile in each deployment record

The serving code is already driven by a deployment record (`configs/deployment/*.yaml`)
and the packaged model config. What is Hermes-only today is a set of names and constants:
- the `HERMES_*` variable names and the `x-hermes-key` header;
- the Space folder and the record file name;
- the base files to preload;
- the status messages ("Hermes v0.0.6 is available");
- the scripts' default record, and the build script's report path and README text.

**Design.** The record gains an optional `serving` block that holds these per model. When
the block is absent, every default equals today's Hermes constant, so the Hermes record
needs no edit and Hermes behaves exactly as before.

```yaml
serving:
  display_name: "Logos v0.0.2"          # status messages, startup line, Space title
  env_prefix: LOGOS                     # LOGOS_API_KEY, LOGOS_PACKAGE_REPO, LOGOS_GPU_*, ...
  key_header: x-logos-key
  space_dir: deploy/zerogpu-space-logos
  record_file: logos_record.yaml        # the record's file name inside the Space
  research_report: docs/experiments/kleos-v007-ministral314breasoning-run1-report.md
  base_files:                           # preloaded into the Space image at the pinned revision
    - config.json
    - generation_config.json
    - model.safetensors.index.json
    - model-00001-of-00006.safetensors  # ... through model-00006-of-00006.safetensors
  reply:
    reasoning: true                     # return the trace; contract_version 2
    require_system_message: true
  gpu:                                  # defaults for the duration rule; LOGOS_GPU_* still override
    base_seconds: 10
    tokens_per_second: 12               # calibrated in the smoke test (section 5)
    min_seconds: 15
    max_seconds: 60
```

**Hermes' defaults:**
- prefix `HERMES` and header `x-hermes-key`;
- `deploy/zerogpu-space` and `hermes_record.yaml`;
- Nemo's five shards plus its three config files;
- `reasoning: false`, `require_system_message: false`, and today's GPU defaults.

**Where the profile is read:**
- `ZeroGPUSettings.from_env`, `load_space_deployment` and startup take the prefix from the
  profile.
- Status messages take the display name from it.
- `stage_zerogpu_space.py` writes the Space `README.md` `preload_from_hub` line and the
  record file from it.
- `check_space_readme` compares against the record's `base_files`, not a module constant.
- The build, verify, upload and smoke scripts take `--deployment-config` and read their
  defaults and text from the record.

**The Logos Space folder** is `deploy/zerogpu-space-logos/`, holding `README.md`, `app.py`
and `requirements.txt.template`. Its `app.py` is the Hermes `app.py` with the record name and
title taken from the profile. `deploy/zerogpu-space/` is unchanged.

## 2. The Logos reply (contract version 2)

Version 2 is version 1 plus one field. Hermes, with `reasoning: false`, keeps
`contract_version: 1` and exactly its current fields.

| Field | Logos |
| --- | --- |
| `text` | The answer: the text after `[/THINK]`, decoded without special tokens |
| `reasoning` | **New.** The trace, or `null` when the completion had no thinking span |
| `finish_reason` | `"length"` when the token budget filled **or** the thinking never closed; `"stop"` otherwise |
| everything else | As in v1: `status`, `usage`, `model`, `request_id`, `timings`, `diagnostics` |

**Rules:**
- **A cut answer is still a reply.** `finish_reason: "length"` keeps `status: "ready"`, as for
  Hermes. KLEOS treats `length` as cut and falls back (Part B).
  - This fixes finding L-F7 in serving. Today serving overwrites the backend's
    "never closed" signal with a budget-only rule, and the loops evaluation recorded as
    `stop` become `length`.
  - An answer that is empty because the thinking never closed is `length`.
- **System message required** (`require_system_message`). A request whose first message is
  not a system message is refused with `status: "invalid_request"`.
  - **Why:** every training and benchmark prompt had a system message. Without one, the
    Reasoning template adds its own default "how you should think" prompt, which Logos was
    never evaluated with.
  - Like every other refusal, it is cheap and uses no GPU.
- **`/status`** also reports `reasoning: true`, the served `max_new_tokens`, and the
  display name.
- **Clients.** The reference client (`serving/client.py`) passes `reasoning` through on both
  transports. Today the HTTP path rebuilds the reply and would drop it. `validate_contract`
  accepts versions 1 and 2.

## 3. Budgets and GPU time

- **Tokens:**
  - `generation.max_new_tokens` and `limits.max_new_tokens` are both 1,024, as evaluated.
    The ZeroGPU path reads `limits`.
  - The prompt cap stays at 2,048 tokens; the longest benchmark prompt is far below it.
- **GPU time per call:** `min(max(ceil(base + budget / tokens_per_second), min), max)` with
  the profile's values. At 1,024 tokens and 12 tokens/s that is 96 s, clamped to **60 s**.
  - Normal answers fit: 293 tokens on average and 413 at the 95th percentile in the
    evaluation.
  - A looping answer runs out of GPU time. ZeroGPU aborts it, the reply is `model_error`,
    and KLEOS falls back. That is the same outcome as a cut answer, and it costs at most
    60 s of quota.
  - **Why not request the full 1,024 tokens' worth:** ZeroGPU admits a call only when 1.5 ×
    the requested time is left in the calling account's 300 s daily quota. Requesting 60 s
    allows about 7 Logos answers a day (ESTIMATED, at about 30 GPU seconds each).
    Requesting 96 s allows about 5.
- **Calibration.** `tokens_per_second` is set from the smoke test's measured throughput, as
  for Hermes. The measured value is written into the record and `docs/deployment.md`.

## 4. The Logos deployment record

`configs/deployment/kleos_logos_v002.yaml` is the identity the package and the Space must
match. Every value comes from the run's own files, never from the Hub.

**Values already known:**
- `name: kleos-logos`, `version: v0.0.2`,
  `experiment_id: kleos-v007-ministral314breasoning-run1`;
- `base_model: mistralai/Ministral-3-14B-Reasoning-2512` at
  `51f9210f3cd20f3452a80d5819d15dc61cc50630`; `auto_class: AutoModelForCausalLM`;
- `source_checkpoint: checkpoint-175`, `selection_metric: eval_loss`;
- `trainable_parameters: 60948480`;
- `dataset_version: kleos-policy-v0.0.7`; `dataset_sha256` is the bundle's dataset hash
  `5d02a5afd7be18e57f13b7d86fa8c31a1b008dd9e1927f4fa506356649699c48`;
  `split_strategy: format_holdout`;
- `training_config_hash: d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9`;
- `tokenizer.source: frozen_package`, `fix_mistral_regex: true`, `padding_side: right`;
- `runtime`: nf4, double quant, `compute_dtype: float16` (what `auto` resolved to on the
  T4s), sdpa, `max_seq_length` 1024;
- `generation`: greedy, 1,024, repetition penalty 1.0; `limits` with 1,024 tokens;
- the `serving` block of section 1;
- `notes`: the H9 result, the 6 loops, prose output rather than JSON, the trace, and the
  Apache-2.0 attribution duty.

**Values read from the downloaded training output** (Part A's plan has a task that
computes them):
- `adapter_sha256` of `adapter/adapter_model.safetensors`;
- `selection_value` from `trainer_state.json`, with `best_model_checkpoint` confirmed as
  `checkpoint-175`;
- `tokenizer.files_sha256` of the run's `tokenizer/` files: `save_pretrained` may have
  rewritten them, so they can differ from the Hub's;
- `pad_token`, as the frozen tokenizer loads it.

**What the build script refuses:** the build already requires `adapter_sha256` to verify
identity, so it refuses until the record is complete.

## 5. Getting it onto Hugging Face

**The owner runs every step.** No credential enters any file or command line here.

1. **Download from Kaggle.** From the training notebook's output, take `adapter/`,
   `tokenizer/`, `manifest.json`, `config.yaml` and `checkpoint-175/trainer_state.json`.
   Put them in a private folder outside both repositories.
2. **Complete the record.** A small script prints the hashes and values of section 4 from
   that folder, and they are written into the record.
3. **Build and verify the package on the Mac, CPU only.**
   `build_deployment_package.py` then `verify_deployment_package.py`, both with
   `--deployment-config configs/deployment/kleos_logos_v002.yaml`.
4. **Upload.**
   - Log in once with `hf auth login`, which stores the token in Hugging Face's own
     location.
   - Run `upload_deployment_package.py --dry-run`, then `--create`.
   - The script refuses a public repository, uploads only the files in the manifest, and
     prints the commit to pin.
5. **Create the Space.** A private ZeroGPU Space with four secrets: `HF_TOKEN` (read-only,
   that repository only), `LOGOS_API_KEY`, `LOGOS_PACKAGE_REPO` and
   `LOGOS_PACKAGE_REVISION`. Stage and push its three files.
6. **Watch startup.** It ends with
   `Logos ready: kleos-logos v0.0.2 adapter=… base=…@51f9210f…`.
   - Quantizing a 14B base on the Space's CPU is **unverified**. Hermes' 12B worked, but its
     peak RAM was never recorded.
   - An out-of-memory crash at startup is a **stop**. The fallback, loading inside the GPU
     call, spends quota on every cold start, so it is the owner's decision.
7. **Smoke test.** `zerogpu_smoke.py --deployment-config …logos…` checks identity first: the
   adapter, base and runtime, plus the served `max_new_tokens` and `reasoning`. It then sends
   the same nine benchmark prompts as Hermes' smoke test.
   - **Expected:** for each, the **answer and the trace** match the Kaggle evaluation's
     `response` and `reasoning` byte for byte, along with the token counts.
   - **A mismatch is a stop for the owner,** not a relaxed rule. The GPU differs (a T4 then,
     a Blackwell now), and a 300-token greedy generation can diverge. Hermes reproduced 9/9,
     which is why matching is the expectation.
   - **An empty answer** (thinking never closed) is reported as the model's behaviour and
     compared like any other, not counted as a serving failure.
   - **About 7 calls fit one day's quota**, so the nine prompts run over two days with
     `--only`.
   - The measured throughput sets `tokens_per_second`.

## 6. Testing

Test-first throughout.

- **Hermes invariance.** Every existing serving, Space, startup, client, status and
  manifest test passes unchanged. A test also checks that the Hermes record, with no
  `serving` block, resolves to today's constants. Hermes' reply has no `reasoning` key and
  `contract_version` 1.
- **The profile.**
  - Defaults versus the Logos values.
  - The prefixed variable names.
  - The staged Space's `README.md` lists exactly the record's `base_files` at the pinned
    revision.
  - The Logos `app.py` reads `logos_record.yaml`.
- **The reply.**
  - `reasoning` is present for Logos.
  - `finish_reason` is `length` when the budget fills and when the thinking never closes,
    and `stop` otherwise.
  - `invalid_request` without a system message, with no GPU call made.
  - The HTTP client keeps `reasoning`.
  - The GPU duration from the profile: 1,024 tokens give 60 s.
- **The record.** It loads and verifies against a package built from fixture files, and
  `load_expected_identity` requires its adapter hash.
- **Docker** (torch): a full thinking reply on a tiny model through `LoadedDeployment` and
  `ZeroGPUService`, with a real fast tokenizer carrying `[THINK]`/`[/THINK]`.
- **The smoke judge:** it compares `reasoning` as well as `text`, and treats an empty answer
  as behaviour.

## 7. Docs

- **`docs/deployment.md`:** a "Logos v0.0.2 on ZeroGPU" section covering the profile, the
  owner's steps, the duration and quota arithmetic, the stop conditions, and the
  verification record, which stays empty until the smoke test runs.
- **`docs/kleos-hermes-integration.md`:** the v2 reply (`reasoning`, the cut rules, the
  system-message rule), as the contract Part B builds on.
- **`docs/logos.md` §7 and §11:** the serving status, and L-F7 resolved in serving.
- **`CHANGELOG.md`.**

## 8. Risks

| Risk | Handling |
| --- | --- |
| The Space's CPU cannot quantize the 14B base at startup | Stop at step 6; the owner decides |
| Greedy output diverges on Blackwell versus the T4 | Stop at step 7; the owner decides |
| ~7 answers a day is too few for KLEOS | Part B's daily cap and fallback; measured during the smoke test |
| A profile default drifts from today's Hermes constant | The Hermes-invariance tests pin every default |

## Success criteria

1. **Hermes:** the full test suite passes with every Hermes test unchanged.
2. **Package:** the Logos package builds and verifies against
   `configs/deployment/kleos_logos_v002.yaml`, and uploads to a private repository.
3. **Startup:** the Logos Space starts and reports its identity.
4. **Smoke test:** the nine prompts reproduce the evaluation's answers and traces, or a
   mismatch is reported and decided by the owner.
5. **Contract:** the v2 reply contract is documented for Part B.
