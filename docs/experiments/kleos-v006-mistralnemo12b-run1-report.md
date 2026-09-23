# Training and evaluation report — `kleos-v006-mistralnemo12b-run1` (KLEOS Hermes)

**Trained:** 2026-09-17 → 2026-09-18 · **Evaluated:** 2026-09-19 → 2026-09-22 ·
**Verdict:** H1 replicated. 7/7 tasks improved at p < 0.001, none regressed. Ten
findings recorded; none invalidates the result.

Nothing was exported, published, uploaded or deployed in producing this report.
The artifacts live on private storage outside this repository
(`MyDrive/kleos-private/outputs/kleos-v006-mistralnemo12b-run1/`). They are **not**
in version control and must not be. The hashes in §10 are the record.

---

## Summary

|                 |                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| KLEOS model     | **Hermes** (the smaller/faster model; Logos is the larger one)                                            |
| Base            | `mistralai/Mistral-Nemo-Instruct-2407` @ `04d8a90549d23fc6bd7f642064003592df51e9b3` — Apache-2.0, ungated |
| Method          | QLoRA: NF4 double-quant, r 16 / α 32 / dropout 0.05, 280 modules, 57,016,320 trainable parameters         |
| Data            | `kleos-policy-v0.0.6`, sealed and read-only; train 820 / validation 181 / test 349                        |
| Headline        | `arm1_base_orchestrated` **0.4755 → `arm2_finetuned` 0.8051**, +0.3295 (95% CI 0.3068–0.3521, p < 0.001)  |
| Tasks           | **7/7 improved, 7/7 significant, 0 regressed**                                                            |
| Consistency     | correct agreement 0.000 → 0.333; 10 of 15 scenario families still flip                                    |
| Shipped adapter | `checkpoint-200` (epoch 1.946), validation loss 0.04343                                                   |

What this run establishes:

1. **The licence-driven base swap cost nothing.** Ministral-8B is under the Mistral
   Research Licence and cannot back a product. Hermes, on an Apache-2.0 base,
   matches the Ministral result (0.4744 → 0.8015) on both arms to within 0.004.
2. **The behavioural signal transfers across Mistral base checkpoints.** Same
   sealed data, benchmark, graders, decoding and hyperparameters; different base;
   the same outcome.
3. **The limitations transfer just as exactly.** Correct agreement 0.333, 10/15
   families flipping, and an abstention table identical case for case (§8.2). A
   base with 50% more parameters did not move any of them. They are properties of
   the v0.0.6 data and training recipe, not of the base model — so the next lever
   is the dataset, not a bigger model.

---

## 1. Run identity and provenance

| Field                | Value                                                                      |
| -------------------- | -------------------------------------------------------------------------- |
| `experiment_id`      | `kleos-v006-mistralnemo12b-run1`                                           |
| `name`               | `kleos-hermes-v006-mistralnemo12b`                                         |
| `status`             | **`completed`**                                                            |
| `config_hash`        | `b2328857c6026dd7c904a6acadd00ec55a6ce5289bf6210e743615fce6d8d3e8`         |
| `seed`               | 42 (`deterministic=True`)                                                  |
| `dataset_version`    | `kleos-policy-v0.0.6`                                                      |
| `dataset_hash`       | `c7340f1d9cb31b1d1d7f0ade65d83740159a887f5d6cb4955820065076ffa875`         |
| Release content hash | `3cc9a74486c42e8e` (RELEASE.lock verified on every benchmark build)        |
| `split_strategy`     | `format_holdout`                                                           |
| `task`               | `null` (multi-task; the manifest does not claim a single task)             |
| Git                  | `4cd76c42b18228922a57127494bb7dcf6c5dc00f` on `main`, `dirty: false`       |
| `adjustments`        | **0** — no feasibility fallback; the requested configuration ran unchanged |

**Software:** torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, accelerate
1.14.0, bitsandbytes 0.50.2, datasets 4.8.5.
**Hardware:** Tesla T4 (14.56 GB, compute capability 7.5, CUDA 12.8) for both
training sessions and every evaluation. No bfloat16, so fp16 throughout.

The manifest records three compatibility notes. They are recorded, not silent:

- `warmup_ratio: 0.03` is passed through as a ratio, which transformers ≥ 5
  resolves to **10 warmup steps** (confirmed from the learning-rate trace, §3).
- `group_by_length=False` was dropped because the installed `TrainingArguments`
  does not accept it. It was false anyway.
- `Resumed from checkpoint-250 at step 250.`

---

## 2. Training timeline

|                            | Session 1                            | Session 2                 | Total                                     |
| -------------------------- | ------------------------------------ | ------------------------- | ----------------------------------------- |
| Wall clock (UTC)           | 2026-09-17 22:38:15 → 09-18 01:14:06 | 09-18 19:16:33 → 19:47:44 | 18.03 h gap between                       |
| Steps                      | 1 → 250 durable (300 attempted)      | 251 → 309                 | 309                                       |
| Trainer time               | 9,351 s (2 h 35 m 51 s)              | 1,871 s (31 m 11 s)       | 11,222 s                                  |
| + post-training validation | —                                    | 174 s                     | **11,396 s (3 h 09 m 56 s)**              |
| Pace, excluding evaluation | 27.5 s/step                          | 25.8 s/step               |                                           |
| Validation passes          | 6 × ~187 s                           | 2 × ~174 s + final        | 9 passes, 1,645 s (14.4% of trainer time) |
| Model load                 | not instrumented                     | 702.5 s                   |                                           |

Session 1 died after writing checkpoint-300's optimizer state but before its
adapter weights reached Drive (finding H-F4). Session 2 rejected that checkpoint
and resumed from checkpoint-250. Steps 251–300 and one validation pass were
recomputed: about **24 minutes of lost work**.

**The resume has no bearing on the shipped weights.** The selected adapter,
checkpoint-200, was written at 2026-09-18 00:22:18 UTC — in session 1, before the
interruption. The resume affected only checkpoints 300 and 309, neither of which
was selected.

The 14.4% evaluation overhead supports the `save_steps: 50` cadence chosen in
`configs/training/kleos_hermes_v006.yaml`. At the rejected 25-step cadence it
would have been roughly 26%.

---

## 3. Training dynamics

| Quantity                       | Value                                                            |
| ------------------------------ | ---------------------------------------------------------------- |
| Optimizer steps                | **309** (820 ÷ 8 = 102.5 → 103 per epoch × 3)                    |
| Warmup                         | 10 steps                                                         |
| fp16-skipped optimizer steps   | **4** through step 305 (method below)                            |
| Effective updates              | 305 of 309, if none were skipped in the unobserved steps 306–309 |
| Final logged train loss        | **0.00476** (step 305)                                           |
| Mean train loss, steps 251–309 | 0.00459                                                          |
| Reported `train_loss`          | 0.0008755 — **invalid, do not cite** (H-F8)                      |
| Best validation loss           | **0.04343** at step 200                                          |
| Final validation loss          | 0.04515 at step 309                                              |

**How skipped steps were counted.** When the fp16 GradScaler overflows, the
optimizer step is skipped, and transformers skips the scheduler step with it. The
scheduler therefore lags `global_step` by exactly the number of skipped steps.
Inverting the cosine-with-warmup schedule at every logged step gives integer lags:
1 at step 1, 2 at steps 5–20, and 4 at every logged step from 255 to 305. So:
step 1 (the one `nan` grad norm in the logs), one more in steps 2–5, two between
steps 21 and 250, and **none at all in session 2**. Only step 1 shows up in the
logs directly because loss and gradient norm are logged every 5 steps. The lag
holding at 4 across the resume, rather than resetting to 0, independently shows
that scheduler state was restored faithfully.

**Validation curve**

| Step | Epoch |   Validation loss |                               |
| ---: | ----: | ----------------: | ----------------------------- |
|   50 |  0.49 |           0.09527 |                               |
|  100 |  0.98 |           0.05465 |                               |
|  150 |  1.46 |           0.04975 |                               |
|  200 |  1.95 |       **0.04343** | best, selected                |
|  250 |  2.43 |           0.04437 |                               |
|  300 |  2.92 | 0.04511 / 0.04512 | measured once in each session |
|  309 |  3.00 |           0.04515 |                               |

Validation loss bottoms at epoch 1.95 and rises monotonically through epoch 3 —
the onset of overfitting. The final train loss (~0.005) sits about 10× below
validation loss: the model is memorising the 820 training targets in the third
epoch, and best-model selection is what keeps that epoch out of the shipped
adapter. Ministral's run bottomed at the **same step, 200 / epoch 1.95**.

The two independent evaluations of step 300 agree to 1.4 × 10⁻⁵ (0.045107 vs
0.045121), consistent with fp16 non-determinism across T4 instances.

**Determinism.** The pre-training gradient check returned loss
`2.1058197021484375` in both sessions — 18 hours apart, on different T4
instances, identical to the last bit. It reports 560/560 LoRA tensors receiving a
gradient and 280 with a non-zero one. That is expected, not a fault: with
`init_lora_weights: true` every `lora_B` starts at zero, so at step 0 only the 280
`lora_B` tensors have a non-zero gradient.

---

## 4. Memory

|                     |                                                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Peak allocated      | **13.09 GB of 14.56 GB** (session 2, manifest); 13.03 GB observed live in session 1                                                                |
| Pre-flight estimate | 7.96–8.12 GB — about 5 GB too low (H-F10)                                                                                                          |
| Memory warnings     | **1**, at step 299 of session 2: 93% reserved (allocated 10.63 GB, reserved 13.58 GB, free 0.84 GB), "OOM likely at evaluation or checkpoint time" |
| OOM                 | **None**, in either session, across 9 validation passes and 8 checkpoint writes                                                                    |

The step-299 warning did not materialise: the step-300 validation pass and
checkpoint that followed it completed. The free-T4 VRAM risk accepted before the
run did not cost the run.

---

## 5. Checkpoints and best-model selection

`save_total_limit: 3` left three checkpoints on disk; 50, 100, 150 and 250 were
pruned.

| Checkpoint                | Written (UTC)       | `adapter_model.safetensors` SHA-256                                    |
| ------------------------- | ------------------- | ---------------------------------------------------------------------- |
| `checkpoint-200`          | 2026-09-18 00:22:18 | `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32`     |
| `checkpoint-300`          | 2026-09-18 19:40:58 | `535b44ac3e7b5a9b39f1ac36e3e95a246250e77e4a2904c9b996a2fd14a79d11`     |
| `checkpoint-309`          | 2026-09-18 19:47:33 | `f9ada5b1fcb2c717f17bb06c7a60e076fd4179461b4f6c8fe02960b63c9176b8`     |
| **`adapter/` (exported)** | 2026-09-18 19:50:40 | **`dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32`** |

**Best-model selection is proven, not inferred.** The exported adapter is
byte-identical to checkpoint-200 and differs from the final step-309 weights.
Independently, the post-training validation pass in `metrics.json` reports
`eval_loss` `0.04343021661043167`, bit-identical to the trainer's `best_metric`
and not the step-309 value of 0.04515. `trainer_state.json` names checkpoint-200
as `best_model_checkpoint` in all three surviving checkpoints.

Checkpoint-200 survived by one checkpoint's margin (H-F9).

The exported adapter was written after a Google Drive storage-quota event and is
byte-identical to a file written before it. Two independent writes producing the
same SHA-256 is the strongest available evidence that neither is truncated.

---

## 6. Adapter artifact

`outputs/kleos-v006-mistralnemo12b-run1/adapter/`

| File                        | Size (bytes) | SHA-256                                                            |
| --------------------------- | -----------: | ------------------------------------------------------------------ |
| `adapter_model.safetensors` |  228,140,600 | `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32` |
| `adapter_config.json`       |        1,204 | `bbc55cf50bd330f81b52bb675d438d2a688cdbcd50e479e0da852b62940eab6b` |
| `README.md`                 |        5,226 | `0e3a47a58db63dbd9c8aac324f44f5c4642944b28a22367a556612054559aee2` |

`outputs/kleos-v006-mistralnemo12b-run1/tokenizer/`

| File                    | Size (bytes) | SHA-256                                                            |
| ----------------------- | -----------: | ------------------------------------------------------------------ |
| `tokenizer.json`        |   17,078,292 | `b0240ce510f08e6c2041724e9043e33be9d251d1e4a4d94eb68cd47b954b61d2` |
| `tokenizer_config.json` |          407 | `a3cd297b36e6a26dae570290548b9da0d5f9ee15363465cfea1056e4d9a78faa` |
| `chat_template.jinja`   |        3,945 | `e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272` |

**Size corroborates the parameter count independently of the manifest.** At fp32,
57,016,320 × 4 = 228,065,280 B; the file exceeds that by 75,320 B of safetensors
header, which is 134.5 B per tensor across 560 tensors (Ministral: 67,632 B over
504 tensors, 134.2 B per tensor). fp16 would give roughly 114 MB and is excluded.

**The parameter count reconciles per layer:** q 147,456 + k 98,304 + v 98,304 +
o 147,456 + gate 311,296 + up 311,296 + down 311,296 = 1,425,408, × 40 layers =
57,016,320.

| Field                                         | Value                                  | Matches `configs/models/mistral_nemo_12b.yaml`                 |
| --------------------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `base_model_name_or_path`                     | `mistralai/Mistral-Nemo-Instruct-2407` | ✅                                                             |
| `peft_type` / `task_type`                     | `LORA` / `CAUSAL_LM`                   | ✅                                                             |
| `r` / `lora_alpha` / `lora_dropout`           | 16 / 32 / 0.05                         | ✅                                                             |
| `bias`                                        | `none`                                 | ✅                                                             |
| `target_modules`                              | q, k, v, o, gate, up, down (7)         | ✅ resolved from `auto`; 280 modules matched                   |
| `exclude_modules`                             | `embed_tokens`, `lm_head`              | ✅                                                             |
| `use_dora` / `use_rslora` / `modules_to_save` | false / false / null                   | ✅ plain LoRA, no full-tune leakage                            |
| `peft_version`                                | 0.20.0                                 | —                                                              |
| `revision`                                    | **`null`**                             | ⚠️ the config and manifest are pinned; this file is not (H-F1) |

---

## 7. Evaluation

**Protocol.** Benchmark `benchmark.jsonl`, sha256 `a11ffad75f5147f9…`, 349 examples
derived from the sealed test split by `scripts/build_benchmark.py`. Greedy decoding
(`temperature 0`, `do_sample: false`), `max_new_tokens` 512, seed 42, grader
`kleos_policy`, batch size 1. Evaluations ran in separate Colab sessions, so the
benchmark was rebuilt each time. Every rebuild reproduced the same hash, and a
fingerprint of the graded targets (example id, task, family, perturbation,
reference decision) is identical across both result files (see H-F5 for why that
had to be checked by hand).

| Arm                      |               Score | 95% CI        | Duration | Mean / median completion tokens | Hit 512 cap | Mean latency |
| ------------------------ | ------------------: | ------------- | -------: | ------------------------------- | ----------: | -----------: |
| `arm1_base_orchestrated` |     0.4755 ± 0.0114 | 0.4532–0.4978 |  8,620 s | 226 / 217                       |           1 |       24.7 s |
| `arm2_finetuned`         | **0.8051 ± 0.0121** | 0.7813–0.8288 |  5,634 s | 94 / 91                         |           0 |       16.1 s |

**Pre-registered comparison — paired bootstrap, 2,000 iterations, two-sided**

| Task                               |   n |       arm1 |       arm2 |           Δ | Verdict                                       |
| ---------------------------------- | --: | ---------: | ---------: | ----------: | --------------------------------------------- |
| `workspace_reasoning`              |  46 |     0.5190 |     0.9512 |     +0.4323 | improved, p < 0.001                           |
| `context_prioritization`           |   8 |     0.5935 |     1.0000 |     +0.4065 | improved, p < 0.001 (CI 0.2612–0.5352; n = 8) |
| `mission_control_briefing`         |  43 |     0.5681 |     0.9716 |     +0.4035 | improved, p < 0.001                           |
| `memory_conflict_resolution`       | 111 |     0.4756 |     0.8553 |     +0.3798 | improved, p < 0.001                           |
| `tool_routing`                     |  62 |     0.3929 |     0.6679 |     +0.2750 | improved, p < 0.001                           |
| `recommendation_generation`        |  41 |     0.2974 |     0.4946 |     +0.1971 | improved, p < 0.001                           |
| `notification_prioritization`      |  38 |     0.6201 |     0.8105 |     +0.1904 | improved, p < 0.001                           |
| **Overall** (requested explicitly) | 349 | **0.4755** | **0.8051** | **+0.3295** | CI 0.3068–0.3521, p < 0.001                   |

`compare.py` prints `p=0.0`. With 2,000 two-sided resamples that means no
resample crossed zero, so the correct statement is **p < 0.001**, not p = 0.
Per-task confidence intervals are in `report_hermes_v006/metrics.json` (§10).
`context_prioritization` has n = 8 and its interval should not be leaned on.
Per-task values here come from `compare.py` and can differ in the fourth decimal
from the evaluator's printout, which rounds before averaging.

**Consistency** (15 scenario families)

|                                                                                                |                      arm1 |                      arm2 |
| ---------------------------------------------------------------------------------------------- | ------------------------: | ------------------------: |
| `agreement_rate`                                                                               |                     0.133 |                     0.333 |
| mean majority share                                                                            |                     0.582 |                     0.678 |
| `correct_agreement_rate`                                                                       |                 **0.000** |                 **0.333** |
| Families flipping under an irrelevant perturbation                                             |                   13 / 15 |                   10 / 15 |
| Flips: paraphrase / unspecified / evidence order / irrelevant context / length / context order | 53 / 39 / 36 / 24 / 5 / 1 | 44 / 28 / 21 / 21 / 3 / 0 |

Paraphrase — the most obviously irrelevant perturbation — causes the most flips in
both arms.

**Faithfulness** (text-level heuristics, not entailment checks)

|                                      |   arm1 |   arm2 |
| ------------------------------------ | -----: | -----: |
| Mean score                           | 0.7513 | 0.8531 |
| Evidence coverage                    | 1.0000 | 1.0000 |
| Citation precision                   | 0.5938 | 0.7192 |
| Unsupported claim rate               | 0.3400 | 0.1600 |
| Responses with a fabricated citation |    147 |     98 |

**Grader sub-scores** (`judgment` is the mean of ranking, deciding factor and
confidence; `format_valid` is reported separately and excluded from the score)

| Sub-score              |       arm1 |       arm2 |
| ---------------------- | ---------: | ---------: |
| `judgment` (the score) |     0.4755 |     0.8051 |
| `ranking`              |     0.5985 |     0.8822 |
| `ranking_ndcg`         |     0.6174 |     0.9527 |
| `ranking_top_1`        |     0.5702 |     0.7765 |
| `ranking_kendall_tau`  |     0.5449 |     0.8479 |
| `deciding_factor`      |     0.0860 |     0.6132 |
| `confidence`           |     0.7421 |     0.9198 |
| `format_valid`         | **0.0000** | **0.0000** |
| Parse failures         |          0 |          0 |

---

## 8. Behavioural analysis

### 8.1 Output format

`format_valid` is **0.0000 in both arms** and neither arm has a parse failure.
Every test prompt contains JSON input and neither model answers in JSON — exactly
as in the Ministral run. The decision policy crossed the format boundary; the
output format did not. H1 is therefore a measurement of judgment expressed in
prose, in both arms.

### 8.2 Abstention

The benchmark's 78 "should decline" cases, split by scenario family. Grader
`details` are not saved in the result files (H-F7), so correctness was
reconstructed from the `confidence` sub-score joined on `example_id` to the
benchmark's `reference.confident`.

| Family                           | Training signal                 | Test cases | Hermes arm1 | Hermes arm2 | Ministral arm2 |
| -------------------------------- | ------------------------------- | ---------: | ----------: | ----------: | -------------: |
| `mem.stale_explicit_conflict`    | 32/32 abstain (unconditional)   |         20 |        0/20 |   **20/20** |          20/20 |
| `route.ask_when_underspecified`  | 45/45 abstain (unconditional)   |         30 |        0/30 |   **30/30** |          30/30 |
| `rec.verify_before_recommending` | 25/55 abstain (**conditional**) |         25 |        3/25 |    **0/25** |           0/25 |
| `rec.abstain_without_evidence`   | 3/9 abstain (**conditional**)   |          3 |         0/3 |     **0/3** |            0/3 |
| Should commit                    | —                               |        271 |     256/271 | **271/271** |        271/271 |

Identical to Ministral, case for case. The fine-tune added exactly the two
families whose training examples always abstain, and nothing where abstention
depends on the evidence. It learned _"this kind of question → decline"_, not
_"the evidence is insufficient → decline"_. It never wrongly abstains; the
remaining error is one-sided overconfidence. The baseline over-commits almost
everywhere (3/78).

This replication across base models is the strongest evidence in either run that
the shortcut is in the data. `rec.verify_before_recommending` trains on 25
abstaining and 30 committing examples, and two different base models both failed
to learn what separates them.

### 8.3 Deciding factor

|                | Overall (n = 349) | Answerable (n = 271) | Correct on answerable |
| -------------- | ----------------: | -------------------: | --------------------: |
| Hermes arm1    |            0.0860 |               0.1107 |                    30 |
| Hermes arm2    |            0.6132 |               0.7897 |                   214 |
| Ministral arm2 |            0.6332 |                0.816 |                   221 |

Both fine-tunes score 0/78 on the abstention cases, whose labels never appear in
training (deviation D3). On answerable cases Hermes identifies seven fewer
deciding factors than Ministral.

### 8.4 Verbosity and latency

The base model writes answers about 2.4× longer than the fine-tune (226 vs 94
completion tokens on average) and runs 53% longer end to end. The fine-tune cuts
mean per-example latency from 24.7 s to 16.1 s on a T4 — relevant for a model whose
role is to be the fast, cheap one.

---

## 9. Comparison with `kleos-v006-ministral8b-run1`

**Status: descriptive, not a tested claim.** Both runs share the sealed dataset,
benchmark, graders, decoding settings and hyperparameters. They differ in base
model — 8B Ministral against 12B Nemo, with different architecture details and
pretraining — so scale, architecture and pretraining are confounded and no
difference below may be attributed to scale. No paired test across runs was
performed. One is possible, since both runs share `example_id`s, but it would be an
exploratory analysis, not a pre-registered one. `assert_comparable` does not flag
this pairing at all (H-F6). Validation losses are not comparable in magnitude
across different tokenizers; only their shape is.

|                                                        |          Ministral-8B |     Hermes (Nemo-12B) |
| ------------------------------------------------------ | --------------------: | --------------------: |
| arm1 (base + orchestration)                            |                0.4744 |                0.4755 |
| arm2 (fine-tuned)                                      |                0.8015 |                0.8051 |
| Gap                                                    |               +0.3271 |               +0.3295 |
| Tasks significant / regressed                          |                 7 / 0 |                 7 / 0 |
| arm2 `correct_agreement_rate`                          |                 0.333 |                 0.333 |
| arm2 families flipping                                 |               10 / 15 |               10 / 15 |
| arm1 `correct_agreement_rate` / flipping               |            0.000 / 14 |            0.000 / 13 |
| arm2 abstention (unconditional / conditional families) |          50/50 · 0/28 |          50/50 · 0/28 |
| arm2 faithfulness / citation precision                 |       0.8331 / 0.6676 |       0.8531 / 0.7192 |
| Fabricated citations, arm1 → arm2                      |             194 → 116 |              147 → 98 |
| Parse failures, arm1 → arm2                            |                13 → 0 |                 0 → 0 |
| `format_valid`, both arms                              |                0.0000 |                0.0000 |
| Best checkpoint                                        |  step 200, epoch 1.95 |  step 200, epoch 1.95 |
| Trainable parameters / adapter size                    | 43,646,976 / 174.7 MB | 57,016,320 / 228.1 MB |

**Per task**

| Task                                   | Ministral arm1 → arm2 (Δ) | Hermes arm1 → arm2 (Δ)    | arm2 difference |
| -------------------------------------- | ------------------------- | ------------------------- | --------------: |
| `context_prioritization` (n = 8)       | 0.5054 → 0.8924 (+0.3870) | 0.5935 → 1.0000 (+0.4065) |         +0.1076 |
| `memory_conflict_resolution` (n = 111) | 0.4781 → 0.8192 (+0.3412) | 0.4756 → 0.8553 (+0.3798) |         +0.0361 |
| `mission_control_briefing` (n = 43)    | 0.5323 → 0.9708 (+0.4385) | 0.5681 → 0.9716 (+0.4035) |         +0.0008 |
| `workspace_reasoning` (n = 46)         | 0.5228 → 0.9336 (+0.4108) | 0.5190 → 0.9512 (+0.4323) |         +0.0176 |
| `recommendation_generation` (n = 41)   | 0.3285 → 0.4569 (+0.1284) | 0.2974 → 0.4946 (+0.1971) |         +0.0377 |
| `notification_prioritization` (n = 38) | 0.6253 → 0.8871 (+0.2617) | 0.6201 → 0.8105 (+0.1904) |     **−0.0766** |
| `tool_routing` (n = 62)                | 0.3919 → 0.7181 (+0.3262) | 0.3929 → 0.6679 (+0.2750) |     **−0.0502** |

**Behavioural differences, read descriptively:**

- **Where Hermes is ahead:** memory conflict resolution, workspace reasoning and
  recommendation generation; more faithful output, with fewer fabricated citations
  (98 vs 116) and higher citation precision (0.7192 vs 0.6676).
- **Where Hermes is behind:** notification prioritization (−0.077) and tool
  routing (−0.050), and seven fewer deciding factors on answerable cases.
- **Where they are identical:** abstention behaviour, consistency, output format,
  and the epoch at which overfitting begins.
- **The larger base is not a better prompt-engineered baseline either:** 0.4755 vs
  0.4744. It does produce cleaner baseline output (no parse failures, 147 vs 194
  fabricated citations).

At n = 38–62 per task, task-level differences of this size may not survive a paired
test. Treat them as leads, not results.

**For the product decision:** Hermes delivers the Ministral result on a base that
can ship. It does not fix anything Ministral got wrong.

---

## 10. Artifact hash register

`outputs/kleos-v006-mistralnemo12b-run1/`

| File                                       | Size (bytes) | SHA-256                                                            |
| ------------------------------------------ | -----------: | ------------------------------------------------------------------ |
| `adapter/adapter_model.safetensors`        |  228,140,600 | `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32` |
| `adapter/adapter_config.json`              |        1,204 | `bbc55cf50bd330f81b52bb675d438d2a688cdbcd50e479e0da852b62940eab6b` |
| `adapter/README.md`                        |        5,226 | `0e3a47a58db63dbd9c8aac324f44f5c4642944b28a22367a556612054559aee2` |
| `tokenizer/tokenizer.json`                 |   17,078,292 | `b0240ce510f08e6c2041724e9043e33be9d251d1e4a4d94eb68cd47b954b61d2` |
| `tokenizer/tokenizer_config.json`          |          407 | `a3cd297b36e6a26dae570290548b9da0d5f9ee15363465cfea1056e4d9a78faa` |
| `tokenizer/chat_template.jinja`            |        3,945 | `e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272` |
| `checkpoint-200/adapter_model.safetensors` |  228,140,600 | `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32` |
| `checkpoint-300/adapter_model.safetensors` |  228,140,600 | `535b44ac3e7b5a9b39f1ac36e3e95a246250e77e4a2904c9b996a2fd14a79d11` |
| `checkpoint-309/adapter_model.safetensors` |  228,140,600 | `f9ada5b1fcb2c717f17bb06c7a60e076fd4179461b4f6c8fe02960b63c9176b8` |
| `manifest.json`                            |       16,134 | `1387d3569c1e57f189e177698acfaa6409f393cb1ce2caaa618a0ab57032e3d2` |
| `config.yaml`                              |        4,079 | `cb13027df6b950c4c70026b736a997a90ec3236d5c76a5ccd5ecb9aa4a25acab` |
| `metrics.json`                             |          357 | `35ff3b9aaf1ca1409719c45e5a55d4593db4a27cbd9c47caeca536a04ddb69d1` |
| `arm1_base_orchestrated.json`              |      851,836 | `71dee4c086b512972e124c3369daca4516da5ef0eeb2aac71fac905eba2b0cb1` |
| `arm2_finetuned.json`                      |      595,858 | `428400f520f7d370c996b8d3665f7672b069e7a4ff4eb3c0cab0cdde20a6e7c6` |
| `report_hermes_v006/summary.md`            |        2,193 | `8d56601c9d6de4c5c7f1ad9b7839af0d4998df0c713efebc1c262c8d39020f9e` |
| `report_hermes_v006/metrics.json`          |        8,377 | `54a2ed41709eb2cdfd6de3c9d01d1dae52f195e5fa7566503832e51e1f6c8331` |

---

## 11. Findings

None is fixed in this change. Each is tooling or record-keeping; none alters a
reported number.

### Findings that affect how the record is read

**H-F6 — `assert_comparable` never compares base models.** It raises only on a
different dataset, task or model family, and warns about the scale/architecture
confound only across families. Ministral-8B and Nemo-12B are both family
`mistral`, so the pair passes with no warning at all. The Hermes config comment and
`docs/experiments.md` both stated that it "will block pooling" these runs; that
was wrong. The caveat is methodological and is not enforced by code.

**H-F8 — `train_loss` is invalid after a resume.** transformers divides the loss
accumulated in the _current_ session (steps 251–309) by the _total_ `global_step`
(309), giving 0.0008755. The true mean over those 59 steps is 0.0008755 × 309 / 59
= 0.00459. Any resumed run's `metrics.json` and `run_completed` event carry the
same distortion.

**H-F2 — The manifest contradicts itself on trainable parameters.**
`model.trainable_parameter_count` reads 1,342,592,000 (33%) against
`lora.trainable_parameters` 57,016,320 (1.38%). The first is a load-time snapshot
taken before LoRA attachment and k-bit freezing: exactly `embed_tokens` + `lm_head`

- every layer norm (671,088,640 × 2 + 414,720). The LoRA block is authoritative.
  The stale figure is copied into the comparison report's `metrics.json` as well.

**H-F10 — The memory estimator is about 5 GB optimistic for large-vocabulary
models.** It predicted 7.96–8.12 GB against a measured 13.09 GB. Nemo's
131,072-token vocabulary leaves `embed_tokens` and `lm_head` unquantized at
roughly 2 × 671M × 2 B ≈ 2.7 GB, where the estimator budgets 0.49 GB. Its two
pre-flight LoRA parameter estimates (13,631,488 and 41,943,040) also disagree with
each other and with the true 57,016,320.

### Provenance and artifact findings

**H-F1 — `adapter_config.json` records `revision: null`.** F3's fix pinned the
model config, manifest and model card, but not this file. Loaded on its own, the
adapter resolves the base at whatever `main` points to, pairing it with weights it
may never have seen. Must be addressed before any export.

**H-F3 — A stale Ministral note is frozen into this run's records.**
`MistralDenseAdapter.capabilities` hardcodes _"Uses interleaved sliding-window
attention; very long contexts behave differently from Qwen3"_ for every Mistral
model. Nemo's config has `sliding_window: null`, and Qwen is excluded from KLEOS.
The note appears in the manifest and in both model blocks of the comparison
report.

**H-F7 — Result files drop grader `details`.** `ExampleResult.to_dict()` saves
`sub_scores` but not predicted/expected label or confidence. The abstention
analysis in §8.2 had to be reconstructed by joining the benchmark back in.

### Robustness findings

**H-F4 — `checkpoint_saved` is not a durability guarantee on Drive.** The event
fired for step 300 at 01:14:06 UTC, yet 18 hours later that directory held
optimizer, scheduler, RNG and scaler state but no `adapter_model.safetensors`. The
write had returned to the process without reaching Drive. `validate_checkpoint` is
what caught it. Without it, the resume would have paired a missing model with a
step-250 optimizer state.

**H-F9 — The KLEOS retention pass does not protect the best checkpoint.**
`CheckpointMetadataCallback` calls `prune_checkpoints`, which keeps the newest N by
recency, after transformers' own rotation, which does exempt
`best_model_checkpoint`. Here transformers deleted checkpoint-250 and kept 200,
leaving exactly three, so the KLEOS pass deleted nothing. Had the best checkpoint
been fourth-newest, it would have been deleted and `load_best_model_at_end` would
have silently shipped the final weights.

**H-F5 — `compare.py` checks benchmark identity by path, not content.** Both
result files record `/content/benchmark/benchmark.jsonl`, rebuilt in different
sessions, and no content hash is stored. Identity was verified by hand here (hash
plus target fingerprint); the tool would have accepted a different file at the
same path.

---

## 12. Deviations and run events

- **D1 carries over.** The comparison is `arm1_base_orchestrated` vs
  `arm2_finetuned`, not the `arm0` pairing H1 pre-registers. The Hermes plan
  declared this comparison before the run.
- **D2–D6 carry over unchanged:** same pipeline, same sealed release. D3 in
  particular caps `deciding_factor` on the 78 abstention cases in both arms.
- **D7 does not apply.** The base is pinned to a verified commit.
- **D8 (new).** Training was interrupted after step 250 and resumed from
  checkpoint-250 eighteen hours later on a different T4 instance. Consequence for
  the result: none — the shipped adapter was written before the interruption.
- **Evaluation attempts.** Two `arm1` attempts were lost to runtime disconnects
  (one about 1 h 20 m into generation, one during model loading) before a complete
  run. Decoding is greedy and seeded, so a rerun reproduces the same outputs rather
  than drawing new ones; repeated attempts cannot select a favourable result.

---

## 13. Not done

- **`arm0_base` and `arm3_finetuned_orchestrated` were not run.** H7 remains
  incomplete for Hermes as well as Ministral.
- **H2** is not measurable: the benchmark is 100% out-of-distribution, with no
  in-distribution population to compute a gap against.
- **H4** is untested: no fixed general-capability suite exists.
- **H5** is not tested by this run. Its pre-registered arms are Ministral-8B vs
  Mistral Small 3.2. The 8B → 12B observation in §9 is exploratory only.
- **No cross-run paired test** was performed.
- **Nothing was exported, published or deployed.** `export_adapter.py` and
  `publish_adapter.py` were not run.

## 14. Open items before serving

**Everything in this report describes the frozen research artifact, which is
unchanged.** The deployment work below happened separately, on a *copy*; see
[../deployment.md](../deployment.md) for the distinction and
`configs/deployment/kleos_hermes_v006.yaml` for the serving record.

1. **Tokenizer regex — RESOLVED for serving, 2026-09-22.** Every load of this
   tokenizer, in training and in all evaluations, warns that it has "an incorrect
   regex pattern" and should be loaded with `fix_mistral_regex=True`. The flag
   defaults to `False`, and it changes how roughly 1% of tokens split. Every arm
   and the training run used the same default, so the comparison in this report
   is internally valid, and **none of its numbers change**.

   The serving contract now pins that behaviour rather than inheriting it: the
   deployment package carries the frozen tokenizer files, hashed, and the loader
   passes `fix_mistral_regex=False` explicitly. Adopting the corrected regex
   would be a retrain, not a configuration change.

2. **H-F1 — RESOLVED for serving, 2026-09-22.** The research
   `adapter_config.json` still records `revision: null` and is **not** backfilled;
   the run did not pin it and the record says so. The deployment package carries
   its own `adapter_config.json` with the base revision pinned, the manifest
   records both the pin and the original absence, and the loader refuses to serve
   an unpinned base.

3. **Exact-artifact reproduction — VERIFIED, 2026-09-23.** The deployment
   package, loaded through the serving path on a Colab T4 with the base revision
   confirmed against the Hub, reproduced the frozen evaluation's responses **9/9,
   byte for byte**, across all seven task families plus two should-decline cases.
   This is a reproducibility check on the artifact, not a new score; the numbers
   in this report stand as they are. Details in
   [../deployment.md](../deployment.md#verification-record--hermes-v006).

4. **Abstention and consistency** are not fixed by a larger base, and are not
   addressed by any of the above. They need the next dataset revision:
   conditional-abstention families with a learnable evidence cue, and the 78
   abstention labels present in training (D3).

5. **Findings H-F2 through H-F10 remain open.** They are tooling and
   record-keeping issues; none alters a reported number, and none is fixed here.
