# Artifact audit — `kleos-v006-ministral8b-run1`

**Audited:** 2026-09-15 · **Verdict:** PASS with three recorded findings, none of
which invalidate the H1 result.

Purpose: freeze the first successful KLEOS fine-tuning run as a verified,
reproducible source artifact before anything is exported, published, hosted or
integrated. Nothing was exported, uploaded or deployed in producing this file.

The artifacts live on private storage outside this repository
(`MyDrive/kleos-private/outputs/kleos-v006-ministral8b-run1/`). They are **not**
in version control and must not be. Hashes below are the record.

---

## 1. Adapter directory

`outputs/kleos-v006-ministral8b-run1/adapter/`

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `adapter_model.safetensors` | 174,655,536 | `6d4a0a93d274160baeea65843f48d57df759cddc9d423f55a20d7c88659e473c` |
| `adapter_config.json` | 1,204 | `267af1f42cd9e6a13b356362f8498b2729bb6a15bb0d6241c25af7d005cdb41f` |
| `README.md` | 5,226 | `ef50be0b034a7cb1509fe9086ea87956194fe3263204914692f3c61ae3abf6e2` |

Adapter weights: **174,655,536 B ≈ 166.6 MiB.**

**Size corroborates the parameter count independently of the manifest.** At fp32,
43,646,976 × 4 = 174,587,904 B; the file exceeds that by 67,632 B, which is the
safetensors header. The fp16 hypothesis is off by 87 MB and is excluded. The
adapter is therefore fp32 and contains exactly the claimed 43,646,976 trainable
parameters.

## 2. Adapter configuration

| Field | Value | Matches `configs/models/ministral_8b.yaml` |
| --- | --- | --- |
| `base_model_name_or_path` | `mistralai/Ministral-8B-Instruct-2410` | ✅ |
| `peft_type` / `task_type` | `LORA` / `CAUSAL_LM` | ✅ |
| `r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | ✅ |
| `bias` | `none` | ✅ |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (7) | ✅ resolved from `auto` |
| `exclude_modules` | `embed_tokens`, `lm_head` | ✅ `MistralDenseAdapter.excluded_module_patterns` |
| `use_dora` / `use_rslora` / `modules_to_save` | false / false / null | ✅ plain LoRA, no full-tune leakage |
| `peft_version` | 0.20.0 | — |
| `revision` | **`null`** | ⚠️ see finding F3 |

## 3. Reproducibility chain (manifest)

| Field | Value |
| --- | --- |
| `experiment_id` | `kleos-v006-ministral8b-run1` |
| `status` | **`completed`** |
| `config_hash` | `3fbb3f90ed9662eef801363820ddaade0c24e591468447b030ef50d12b7806d3` |
| `seed` | 42 |
| `dataset_version` | `kleos-policy-v0.0.6` |
| `dataset_hash` | `c7340f1d9cb31b1d1d7f0ade65d83740159a887f5d6cb4955820065076ffa875` |
| `dataset_counts` | train 820 / validation 181 / test 349 |
| `split_strategy` | `format_holdout` |
| `task` | `null` (multi-task; the manifest does **not** falsely claim a single task) |
| `git.commit` | `2ad7c9ae0ca63d0a1d189edc69eb5c945d37bc9a` |
| `git.dirty` | **`false`** |
| `adjustments` | **`[]`** |
| `model` | `mistralai/Ministral-8B-Instruct-2410`, revision `main`, `model_type: ministral`, `MistralForCausalLM`, `AutoModelForCausalLM` |
| `lora` | 43,646,976 / 2,854,129,664 trainable = 1.5293% |
| `quantization` | nf4, double_quant, bitsandbytes 0.50.2, cc 7.5, bf16 unsupported → fp16 |
| `environment` | Python 3.13.15, Linux, Colab, hostname hashed (`3c6a1040…`), no raw host identifiers |
| `started_at` | 2026-09-09T19:00:24Z |

Two entries carry disproportionate weight:

- **`git.dirty: false`** — the working tree was clean at commit `2ad7c9ae`, so the
  run is reproducible from a specific commit rather than from an unrecorded local
  state.
- **`adjustments: []`** — no memory-driven fallback silently altered the
  configuration. What the config specified is what ran.

Training metrics (`metrics.json`): `eval_loss` 0.03987148, `train_loss` 0.0015548,
`epoch` 3.0, `peak_memory_gb` 9.67, `train_runtime` 1970.8 s. The `eval_loss`
matches `best_metric` from `trainer_state.json`, confirming that
`load_best_model_at_end` restored **checkpoint-200 (epoch 1.95)** rather than the
final epoch-3 weights.

## 4. H1 verified against stored evaluation artifacts

Read from `eval_arm1_full.json` and `eval_arm2_full.json`, **not** from
documentation or session logs:

| Arm | n | Mean | 95% CI | Benchmark |
| --- | ---: | ---: | --- | --- |
| `arm1_base_orchestrated` | 349 | **0.5231** | 0.4979 – 0.5483 | `benchmark.jsonl` |
| `arm2_finetuned` | 349 | **0.8015** | 0.7768 – 0.8262 | `benchmark.jsonl` |

Both arms ran the same benchmark at the same n, and the intervals do not overlap.
The figures documented at audit time match the stored artifacts exactly.
**H1 is verified.**

> **Superseded by the F1 re-grade.** The table above records what the *original*
> artifacts contained, which is what this audit verified. F1 was subsequently
> fixed and both arms re-graded: `arm1` is now **0.4744** and `arm2` is unchanged
> at **0.8015**. The corrected figures live in
> [`../experiments.md`](../experiments.md); the original artifacts are retained.

## 5. Private-data containment

`collect_upload_files(adapter_dir, run_dir)` on the real run directory:

**Allowed (8):** `adapter_model.safetensors`, `adapter_config.json`, `README.md`,
`config.yaml`, `manifest.json`, `metrics.json`, `chat_template.jinja`,
`tokenizer_config.json`

**Refused (1):** `tokenizer.json` — see F2.

No `.jsonl`, no `checkpoint-*`, no `.log`, no `events.jsonl`, no `.env` appears in
the allowed set. The allowlist is closed (anything not named is refused) and the
forbidden patterns are checked first, so **no training data can reach a published
adapter by construction.** The adapter directory itself contains only three files,
none of them data.

**Step 5: PASS.**

---

## Findings

### F1 — `ndcg` can exceed 1.0 on degenerate rankings — **RESOLVED 2026-09-15**

> **Outcome.** Fixed in `evaluation/metrics.py` (an item is credited once; a
> repeat holds its slot at zero relevance), bounded-verified exhaustively and
> randomly, and pinned by 10 regression tests. Both arms were re-graded offline
> from stored responses via `scripts/rescore.py`; the frozen source artifacts were
> re-hashed afterwards and are unchanged.
>
> **arm1 0.5231 → 0.4744 (210/349 examples changed). arm2 0.8015 → 0.8015 (0
> changed).** The gap widened from +0.2784 to +0.3271 and
> `recommendation_generation` crossed into significance, taking the result from
> 6/7 to **7/7 tasks at p<0.05**. No task changed direction. The prediction below
> — that the bias ran against the claim — held.
>
> Corrected figures are now the reported ones in
> [`../experiments.md`](../experiments.md). The original artifacts are retained
> alongside the re-scored ones.

The original finding, as written at audit time:

Scores are documented as bounded in [0, 1], but `arm1` reports `max: 1.0685`.
Reproduced locally:

```
ndcg(["A","A","A"], ["A","B","C"]) = 1.3425
kleos_policy score on ["A","A","B"] = 1.0475
```

DCG accumulates gain for a repeated item while IDCG does not, so a model that
repeats its top answer can score above perfect.

**Impact on H1: none that threatens it — the bias runs the wrong way for the
claim.** `arm1` (base) hit 1.0685; `arm2` (fine-tuned) maxes at exactly 1.0, so
only the *baseline* was inflated. The true gap is therefore ≥ the reported
+0.2784, and the result is conservative.

**Not fixed here, deliberately.** Changing the grader after results exist would
invalidate the comparison unless both arms were re-scored. Both evaluation JSONs
retain model responses, so re-grading is possible offline with no GPU. That is the
correct remedy and it belongs in its own change.

### F2 — `tokenizer.json` (17.1 MB) refused by the 5 MB scan cap

`_MAX_SCAN_BYTES` is 5 MB; files above it are refused rather than skipped — a
deliberate and correct safety posture. But Ministral's legitimate fast tokenizer
is 17.1 MB, so a real export would omit it.

This is a **completeness** problem, not a privacy one. Consumers loading the
tokenizer from the base model repository are unaffected; a self-contained export
would be incomplete. Must be resolved before `export_adapter.py` or
`publish_adapter.py` is used for real. Out of scope for this audit.

### F3 — Base-model revision is not pinned

`configs/models/ministral_8b.yaml` and the manifest both record `revision: main`,
a moving pointer; `adapter_config.json` records `revision: null`. If upstream
republishes the checkpoint, this adapter cannot be reproduced against the exact
base weights it was trained on.

The config file already warns: *"Pin a commit sha for a real research run so the
checkpoint cannot move."* That was not done. Not corrected here because changing a
config alters `config_hash` and would no longer describe the run that happened.
Pin it for the **next** run.

---

## Verification performed

| Check | Result |
| --- | --- |
| Test suite | 608 passed, 22 skipped (torch-only) |
| `ruff check` / `ruff format` | clean |
| `mypy` | no issues, 43 source files |
| `validate_configs.py` | all configs valid |
| Privacy scan (`src`, `scripts`, `configs`, `docs`, `README.md`) | clean |
| Sealed v0.0.6 release vs `RELEASE.lock` | 5/5 files byte-identical |
| Benchmark reproducibility | rebuilds to `a11ffad75f5147f9` |

The sealed dataset was re-verified and **not modified**. No training, evaluation,
export, upload or deployment was performed.
