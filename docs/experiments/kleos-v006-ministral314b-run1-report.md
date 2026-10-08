# Training and evaluation report — `kleos-v006-ministral314b-run1` (KLEOS Logos v0.0.1)

> This is a frozen research record, written while the work was done and
> preserved as written. Editorial changes for public release are limited to
> replacing private storage paths and account identifiers with placeholders
> (marked as editor's notes) and retargeting links to documents that moved. No
> number, table, finding or decision was changed. Conventions:
> [docs/experiments/README.md](README.md).

**Trained:** 2026-09-24 → 2026-09-27 · **Evaluated:** 2026-09-27 → 2026-10-01 ·
**Verdict:** H8a supported: fine-tuning helps Logos, 5/7 tasks improved and none
regressed. **H8b (primary) inconclusive:** Logos is not measurably better than
Hermes.

Nothing was exported, published, uploaded or deployed in producing this report.
The artifacts live on private storage outside this repository
(`<private storage>/outputs/kleos-v006-ministral314b-run1/` *[editor's note: private storage path replaced]*). They are **not**
in version control and must not be. The hashes in §10 are the record.
Pre-registration: [H8 in ../experiments.md](../experiments.md#h8--does-a-stronger-base-make-a-better-kleos-model).
Model selection and the Colab procedure: [../logos.md](../logos.md), [../runbooks/logos-v001-colab.md](../runbooks/logos-v001-colab.md).

---

## Summary

|                 |                                                                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| KLEOS model     | **Logos v0.0.1** (the deeper model; Hermes is the fast one)                                                                             |
| Base            | `mistralai/Ministral-3-14B-Instruct-2512-BF16` @ `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`, text tower only. Apache-2.0, ungated       |
| Method          | QLoRA: NF4 double-quant, r 16 / α 32 / dropout 0.05, 280 modules, 60,948,480 trainable parameters                                       |
| Data            | `kleos-policy-v0.0.6`, sealed and read-only; train 820 / validation 181 / test 349                                                      |
| Headline (H8a)  | `arm1_base_orchestrated` **0.4469 → `arm2_finetuned` 0.7896**; answerable subset +0.3952 (cluster 95% CI +0.3443 to +0.4438)           |
| Tasks (H8a)     | **5/7 improved by group intervals, 0 regressed.** `context_prioritization` (8 items) not estimable by groups; `recommendation_generation` +0.0643, n.s. |
| Primary (H8b)   | Logos − Hermes on the answerable subset **−0.0168**, cluster 95% CI −0.0546 to +0.0177: **inconclusive**                                |
| Shipped adapter | `checkpoint-175` (epoch 1.70), validation loss 0.03824                                                                                  |

What this run establishes:

1. **A 14B Mistral trains and evaluates on a free T4.** Peak training memory was
   13.70 GB of 14.56 GB. The pre-flight estimate was 13.77 GB and the measured
   longest-batch probe 13.60 GB. The feasibility work held, with no OOM in four
   sessions.
2. **Fine-tuning works on the stronger base (H8a).** The overall gain (+0.3427)
   is close to Hermes' (+0.3295) and Ministral-8B's (+0.3271).
3. **The stronger base did not make a measurably better KLEOS model (H8b).**
   - The interval cannot separate Logos from Hermes, and it does not support a
     Logos advantage as large as 0.02.
   - Read descriptively, three Mistral bases from 8B to 14B finish within 0.016 of
     each other after fine-tuning: 0.8015, 0.8051, 0.7896.
   - This points at the v0.0.6 data, not the base, as what limits the score. It is
     a reading, not a tested claim.

---

## 1. Run identity and provenance

| Field                | Value                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `experiment_id`      | `kleos-v006-ministral314b-run1`                                                               |
| `name`               | `kleos-logos-v001-ministral3-14b`                                                             |
| `status`             | **`completed`**                                                                               |
| `config_hash`        | `18008c6716a58afc284c64eb7e1e96c9bfce34b8da2b8c489b6d59c0c45b6f71`, as pre-registered          |
| `seed`               | 42 (`deterministic=True`)                                                                     |
| `dataset_version`    | `kleos-policy-v0.0.6` (dataset hash `c7340f1d9cb31b1d…`, the same as Hermes')                  |
| Release content hash | `3cc9a74486c42e8e` (RELEASE.lock verified on every benchmark build)                           |
| `strict_config`      | `true`: any automatic adjustment of the requested configuration would have stopped the run   |
| Git, training        | `ec4f9e3a03958d7c864b6fb1430fd5780b3d469d` on `main`, `dirty: false`                           |
| Git, evaluation      | `arm2_finetuned` at `ec4f9e3` (`ed5a987` was pushed only after this arm finished); `arm1_base_orchestrated` at `ed5a987c1a171de9289eba85c4824f3b78484935`, `dirty: false` (declared as D11) |

**Software:** torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, accelerate
1.14.0, bitsandbytes 0.50.2, tokenizers 0.23.2, datasets 4.8.5. These are
Hermes' versions, pinned in the runbook.

**Hardware:** Tesla T4 (14.56 GB, compute capability 7.5, CUDA 12.8) for every
session. There is no bfloat16 on a T4, so fp16 throughout.

The manifest records three compatibility notes, the same as Hermes':

- `warmup_ratio: 0.03` is passed through as a ratio. transformers ≥ 5 resolves it
  to **10 warmup steps**, confirmed from the learning-rate trace (§3).
- `group_by_length=False` was dropped because the installed `TrainingArguments`
  does not accept it. It was false anyway.
- `Resumed from checkpoint-250 at step 250.`

The manifest is rewritten by each session, so it names only the last resume.
`events.jsonl`, appended across sessions, records both (§2).

---

## 2. Training timeline

|                    | Session 1                                  | Session 2                                     | Session 3                   | Session 4                               |
| ------------------ | ------------------------------------------ | --------------------------------------------- | --------------------------- | --------------------------------------- |
| When (UTC)         | first event 2026-09-24 21:21:23            | resumed 2026-09-25 20:20:30                   | —                           | 2026-09-27 02:26:17 → 03:27:45          |
| Resumed from       | —                                          | `checkpoint-100`; `checkpoint-125` rejected   | —                           | `checkpoint-250`                        |
| Steps made durable | 1 → 100 (trained past 125)                 | 101 → 250 (trained past 250, not to 275)      | none                        | 251 → 309                               |
| Ended by           | runtime disconnect after the step-125 validation | Colab's free GPU usage limit           | usage limit before training | completion                              |

- **Session 1's `checkpoint-125` has no weights.** Its `checkpoint_saved` event
  fired, but the weights never reached Drive (H-F4 again, and the mechanism behind
  L-F3). `validate_checkpoint` rejected it, so session 2 resumed from
  `checkpoint-100`. The events log shows 14 `checkpoint_saved` events for 13
  durable checkpoints.
- **Session 2 hit the Drive storage quota.** Drive reported it at 4:34:55 PM local
  time. Rotated checkpoints go to Drive's Trash and keep counting against the quota
  until it is emptied. The checkpoints left after session 2 (175, 225 and 250) were
  checked complete, file by file.
- **Session 3 never reached training.** The events log has 3 `train_begin` events
  for 4 sessions.
- **Session 4:**
  - model load 1,262.8 s, including a 19-minute unauthenticated download
  - trainer 2,157.6 s for 59 steps and three validation passes, about 27 s per step
    excluding validation
  - final validation 180.9 s
  - manifest duration 3,626.98 s
- **Lost work:**
  - steps 101–125 and one validation pass, recomputed
  - fewer than 25 steps after step 250 in session 2
  - all of session 3
  - three full model downloads

**The interruptions have no bearing on the shipped weights.**

- `checkpoint-175` was written in session 2, after its resume was verified.
- The two resume points were checked differently. `checkpoint-100` was checked by
  size (368 MB, a full checkpoint). `checkpoint-250` was checked file by file:
  weights, optimizer, scheduler, RNG and scaler state.
- The scheduler lag held at 2 across the session-4 resume (§3), which
  independently shows the scheduler state was restored.

**Validation passes:** 15 `evaluate` events at about 180 s each, roughly 45 minutes:

- 13 at the 25-step cadence; step 125 was evaluated in both sessions 1 and 2
- one at the final step
- one final pass on the restored best weights

The 25-step cadence was pre-registered. It needs about twice as many validation
passes as Hermes' 50-step cadence, and in exchange it could select step 175.

---

## 3. Training dynamics

| Quantity                       | Value                                                              |
| ------------------------------ | ------------------------------------------------------------------ |
| Optimizer steps                | **309** (820 ÷ 8 = 102.5 → 103 per epoch × 3)                      |
| Warmup                         | 10 steps                                                           |
| fp16-skipped optimizer steps   | **2**, both before step 250; none in session 4 (method below)      |
| Final logged train loss        | **0.00322** (step 305)                                             |
| Mean train loss, steps 251–309 | 0.00366                                                            |
| Reported `train_loss`          | 0.0006982: **invalid, do not cite** (H-F8 recurs)                  |
| Best validation loss           | **0.03824** at step 175                                            |
| Final validation loss          | 0.04969 at step 309                                                |

**Skipped steps.** This uses the method of Hermes' report §3:

- An fp16 overflow skips both the optimizer step and the scheduler step, so the
  scheduler lags `global_step` by the number of skipped steps.
- With 10 warmup steps, inverting the cosine schedule at every logged step from
  255 to 305 gives a lag of 2.00 at all eleven. With 9 warmup steps the lags are
  not integers, which is what pins warmup at 10.
- So two steps were skipped before step 250 and none after.
- The lag held across the resume rather than resetting to 0, which shows the
  scheduler state was restored.

The mean train loss is the reported `train_loss` corrected as in H-F8: 0.0006982 ×
309 / 59. transformers divides the loss summed over session 4's 59 steps by all
309.

**Validation curve**

| Step | Epoch | Validation loss |                |
| ---: | ----: | --------------: | -------------- |
|   25 |  0.24 |         0.16277 |                |
|   50 |  0.49 |         0.05541 |                |
|   75 |  0.73 |         0.06231 |                |
|  100 |  0.98 |         0.05292 |                |
|  125 |  1.21 |         0.05057 | session 2's measurement |
|  150 |  1.46 |         0.03911 |                |
|  175 |  1.70 |     **0.03824** | best, selected |
|  200 |  1.95 |         0.04156 |                |
|  225 |  2.19 |         0.04605 |                |
|  250 |  2.43 |         0.04933 |                |
|  275 |  2.67 |         0.04968 |                |
|  300 |  2.92 |         0.04973 |                |
|  309 |  3.00 |         0.04969 |                |

- **Validation loss bottoms at epoch 1.70 and rises through epoch 3.** Overfitting
  sets in a little earlier than in Hermes and Ministral-8B, which both bottomed at
  step 200 / epoch 1.95.
- **The 25-step grid could select step 175; Hermes' 50-step grid could not.** This
  confound was declared in advance.
- **Train loss ends about 15× below validation loss.** The third epoch memorises
  the 820 targets, and best-model selection keeps it out of the shipped adapter.
- **Validation losses are not comparable in size with Hermes'.** The tokenizers
  differ, so only the curves' shapes compare.

**Pre-training gradient check (session 4):**

- loss `1.774794340133667`
- 560/560 LoRA tensors receive a gradient; 280 receive a non-zero one, the
  `lora_B` tensors, as expected at initialisation
- the values from all three training sessions are in `events.jsonl` and are not
  compared here

---

## 4. Memory

|                     |                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Peak allocated      | **13.70 GB of 14.56 GB** (manifest, session 4)                                                                       |
| Memory probe        | Longest batch (1 × 448 tokens, fp16): 13.60 GB allocated, 13.97 GB reserved, **0.35 GB spare** once optimizer state exists |
| Pre-flight estimate | 13.77 GB allocated + 0.50 GB allocator reserve = 14.27 GB against a 14.35 GB budget: tier MARGINAL, headroom +0.08 GB |
| Estimate error      | +1.3% against the probe, +0.5% against the training peak                                                             |
| Memory warnings     | **3**, at steps 49, 149 and 299, each at 97–98% reserved. Step 299: allocated 11.21 GB, reserved 14.26 GB, free 0.17 GB |
| OOM                 | **None** in any session, across 15 validation passes and 14 checkpoint writes                                        |

The estimator built for this phase predicted the peak within 0.5%. Its
predecessor was about 5 GB optimistic for Hermes (H-F10). Logos runs closer to
the T4's limit than Hermes did (13.70 vs 13.09 GB). The warnings did not turn
into an OOM, but there is little room for a longer sequence or a larger batch.

---

## 5. Checkpoints and best-model selection

`save_total_limit: 3` left `checkpoint-175`, `checkpoint-300` and
`checkpoint-309`. Every other checkpoint was pruned.

**Best-model selection is proven, not inferred.**

- **The exported adapter is byte-identical to `checkpoint-175`'s:** sha256
  `f3e8dcdc…70a7`, checked on Drive on 2026-09-27.
- **The final validation pass reproduces the best score exactly.** It ran on the
  restored weights and reports `eval_loss` `0.03824465721845627`, bit-identical
  to the trainer's `best_metric` and not the step-309 value of 0.04969.
- **Every surviving checkpoint agrees.** `trainer_state.json` names
  `checkpoint-175` as `best_model_checkpoint` in all three.

`checkpoint-175` survived six later saves. Both transformers' rotation and the
H-F9 fix protect it. L-F4 records the one way it could still have been lost.

---

## 6. Adapter artifact

`outputs/kleos-v006-ministral314b-run1/adapter/`

| File                        | Size (bytes) | SHA-256                                                            |
| --------------------------- | -----------: | ------------------------------------------------------------------ |
| `adapter_model.safetensors` |  243,869,280 | `f3e8dcdc09a3f3bb9f6fc4f64f867ff8b97834c955a03fcbde98a8068d5570a7` |
| `adapter_config.json`       |        1,299 | `410176a9c583585bf4b2954e26714acec2b4404645a2f67f028194c2cb7ae1e6` |

`outputs/kleos-v006-ministral314b-run1/tokenizer/`

| File                    | Size (bytes) | SHA-256                                                            | Against the pinned Hub revision |
| ----------------------- | -----------: | ------------------------------------------------------------------ | ------------------------------- |
| `tokenizer.json`        |   17,078,128 | `d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135` | identical                       |
| `chat_template.jinja`   |       11,913 | `2f545122222db8bb43ca0ea0c49e9185320a8670f7d35575b0da0eb48b1e8970` | identical                       |
| `tokenizer_config.json` |          465 | `ce7ea8d28bb4ba4350064186dcf3a6dcef497b728f55b1ceee379412a8af36b2` | rewritten by `save_pretrained`; the Hub file is `f59f7294…` |

**The file size confirms the parameter count independently of the manifest.** At
fp32, 60,948,480 × 4 = 243,793,920 B. The file is 75,360 B larger, which is the
safetensors header: 134.6 B per tensor across 560 tensors. Hermes' was 134.5 B.

**The parameter count reconciles per layer:**

- each layer: q 147,456 + k 98,304 + v 98,304 + o 147,456 + gate 344,064 + up
  344,064 + down 344,064 = 1,523,712
- × 40 layers = 60,948,480
- The difference from Hermes is the wider MLP: 16,384 against Nemo's 14,336.

**Not inspected here: whether `adapter_config.json` records the base revision.**
This phase writes the revision into new adapters (the H-F1 fix). The file must be
checked before any serving work (§14).

---

## 7. Evaluation

**Protocol.** Identical to Hermes':

- **Benchmark:** `benchmark.jsonl`, sha256 `a11ffad75f5147f9…`, 349 examples,
  rebuilt and verified in every session.
- **Decoding:** greedy, `max_new_tokens` 512, seed 42.
- **Grader:** `kleos_policy`, with batch size 1.
- **Intervals:** from H8 on, they resample `group_id` clusters (protocol
  amendment, 2026-09-24).

| Arm                      |   Score | 95% CI (examples) | Answerable (cluster CI) | Should-decline | Mean / median completion tokens | Hit 512 cap | Mean latency |
| ------------------------ | ------: | ----------------- | ----------------------- | -------------- | ------------------------------- | ----------: | -----------: |
| `arm1_base_orchestrated` |  0.4469 | 0.4219–0.4720     | 0.4857 (0.4445–0.5261)  | 0.3123         | 348 / 343                       |      **41** |       36.7 s |
| `arm2_finetuned`         | **0.7896** | 0.7644–0.8147  | 0.8808 (0.8381–0.9191)  | 0.4724         | 95 / 91                         |           0 |       17.2 s |

**Generation cost:**

- Summed generation time: 12,804 s for arm1 across its two sessions, and 5,996 s
  for arm2.
- Peak VRAM while evaluating: 8.68 GB and 8.87 GB.
- Neither arm had a parse failure or an empty response.

**H8a, the pre-registered comparison.** It is a paired bootstrap with 2,000
iterations. A task improves only if its group-resampled interval excludes zero.

| Task                          |   n |   arm1 |   arm2 |       Δ | By groups               | By examples          |
| ----------------------------- | --: | -----: | -----: | ------: | ----------------------- | -------------------- |
| `memory_conflict_resolution`  | 111 | 0.3763 | 0.8350 | +0.4586 | improved, p < 0.0005    | improved, p < 0.0005 |
| `mission_control_briefing`    |  43 | 0.5552 | 0.9646 | +0.4094 | improved, p < 0.0005    | improved, p < 0.0005 |
| `workspace_reasoning`         |  46 | 0.5366 | 0.9370 | +0.4005 | improved, p < 0.0005    | improved, p < 0.0005 |
| `context_prioritization`      |   8 | 0.6204 | 0.9995 | +0.3792 | not estimable (8 items) | improved, p < 0.0005 |
| `notification_prioritization` |  38 | 0.5888 | 0.8942 | +0.3054 | improved, p < 0.0005    | improved, p < 0.0005 |
| `tool_routing`                |  62 | 0.4021 | 0.6501 | +0.2479 | improved, p < 0.0005    | improved, p < 0.0005 |
| `recommendation_generation`   |  41 | 0.3263 | 0.3906 | +0.0643 | not significant         | not significant      |

- **p < 0.0005 means no resample crossed zero.** With 2,000 resamples, that is the
  smallest p the bootstrap can show.
- **`compare.py` counts "6 significant".** It counts by examples. The
  pre-registered count, by groups, is 5.

**The 512-token cap affects the base arm, not the fine-tune.**

- 41 base answers (11.7%) were cut off at the cap, against 1 for Hermes' base.
  The cap is the pre-registered one.
- A cut-off answer can lose credit for a decision it never reached. So arm1 partly
  measures verbosity under the cap, which may inflate the arm1 → arm2 gains.
- Which tasks the 41 cut-off answers fall in was not checked (§13).

**Consistency and faithfulness**

|                                                    |  arm1 |  arm2 |
| -------------------------------------------------- | ----: | ----: |
| Consistency by `group_id`, all 78 groups (oracle 1.000)  | 0.154 | 0.833 |
| Consistency by `group_id`, 61 answerable groups    | 0.164 | 0.853 |
| Family-level `agreement_rate` (continuity; oracle 0.600) | 0.067 | 0.333 |
| Family-level `correct_agreement_rate`              | 0.000 | 0.333 |
| Responses with a "fabricated" citation (gold floor 167 of 349; UNCALIBRATED, H-F12) | 216 | 94 |
| Evidence coverage                                  | vacuous (H-F13) | vacuous (H-F13) |

**Grader sub-scores.** `judgment` is the mean of ranking, deciding factor and
confidence. `format_valid` is reported separately and excluded from the score.

| Sub-score              |       arm1 |       arm2 | Hermes arm2 |
| ---------------------- | ---------: | ---------: | ----------: |
| `judgment` (the score) |     0.4469 |     0.7896 |      0.8051 |
| `ranking`              |     0.4468 |     0.8529 |      0.8822 |
| `ranking_ndcg`         |     0.4505 |     0.9459 |      0.9527 |
| `ranking_top_1`        |     0.4413 |     0.7135 |      0.7765 |
| `ranking_kendall_tau`  |     0.5350 |     0.8345 |      0.8479 |
| `deciding_factor`      |     0.1805 |     0.5989 |      0.6132 |
| `confidence`           |     0.7135 |     0.9169 |      0.9198 |
| `format_valid`         | **0.0000** | **0.0000** |  **0.0000** |

---

## 8. Behavioural analysis

**Output format.** `format_valid` is 0.0000 in both arms, as in every KLEOS run.
The test prompts carry JSON input, and no model answers in JSON (D2).

**Verbosity and latency.**

- **The base writes about 3.7× more than the fine-tune:** 348 tokens against 95.
  For Hermes the ratio was 2.4×, at 226 against 94 tokens.
- **Base Logos is slower than base Hermes:** 36.7 s per answer, against 24.7 s.
- **Fine-tuning brings Logos to 17.2 s per answer,** close to Hermes' 16.1 s.
- **For serving, the two fine-tuned models cost about the same per answer on a
  T4.** Logos' extra size buys nothing measurable on this benchmark (H8b).

**Where fine-tuning did less for Logos.** It is `recommendation_generation`.

- **Logos' base starts higher than Hermes'** (0.3263 against 0.2974) but gains only
  +0.0643, against Hermes' +0.1971.
- **The fine-tuned Logos ends 0.1040 below fine-tuned Hermes** on this task. That
  is H8b's one secondary regression significant in both bootstraps.
- **The sub-scores locate the gap in ranking.** `ranking_top_1` is 0.7135 against
  Hermes' 0.7765, and `ranking` 0.8529 against 0.8822.

**Abstention.** Abstention is the should-decline subset (78 items).

- **Base Logos scores 0.3123 there, against base Hermes' 0.1923.**
- **After fine-tuning the two converge,** at 0.4724 against 0.4835.
- **The decline labels never occur in training** (D3), so neither fine-tune can
  learn them from v0.0.6.
- **Not redone here:** the per-family abstention table of Hermes' report §8.2.
  The grader details it needs are now saved (H-F7 fixed), so it can be built from
  the stored results.

---

## 9. Comparison with KLEOS Hermes (`kleos-v006-mistralnemo12b-run1`)

The pre-registered comparison is H8b, which is **inconclusive**:

| Arm (answerable subset, 271 items, 61 groups) | Mean   | Cluster 95% CI       |
| --------------------------------------------- | -----: | -------------------- |
| Hermes `arm2_finetuned`, re-reported          | 0.8976 | 0.8634–0.9301        |
| Logos `arm2_finetuned`                        | 0.8808 | 0.8381–0.9191        |
| **Logos − Hermes, paired**                    | **−0.0168** | **−0.0546 to +0.0177** |

- **The rule's verdict:** the interval spans zero and extends past −0.02, so Logos
  is neither better, worse nor equivalent.
- **What it rules out:** the upper end, +0.0177, is below the +0.02 margin. These
  data do not support Logos being better by the margin or more.
- **How Hermes' side was read:** from its frozen results. The source sha256
  `428400f5…` matches Hermes' register, stored scores reproduced with drift 0, and
  the file was not modified.

**Descriptive context. These are not tested claims; the bases differ in more than
size (H8's confounds).**

|                                                | Hermes (Nemo 12B)     | Logos (Ministral 3 14B) |
| ---------------------------------------------- | --------------------: | ----------------------: |
| arm1 → arm2, overall                           | 0.4755 → 0.8051       | 0.4469 → 0.7896         |
| Gain                                           | +0.3295               | +0.3427                 |
| Answerable, arm1 → arm2                        | 0.5570 → 0.8976       | 0.4857 → 0.8808         |
| Should-decline, arm1 → arm2                    | 0.1923 → 0.4835       | 0.3123 → 0.4724         |
| Consistency by `group_id`, arm1 → arm2         | 0.346 → 0.769         | 0.154 → 0.833           |
| Mean completion tokens, arm1 / arm2            | 226 / 94              | 348 / 95                |
| Base answers cut off at 512 tokens             | 1                     | 41                      |
| Mean latency, arm1 / arm2                      | 24.7 s / 16.1 s       | 36.7 s / 17.2 s         |
| "Fabricated" citations, arm1 → arm2 (gold floor 167) | 147 → 98        | 216 → 94                |
| Best checkpoint                                | step 200, epoch 1.95 (50-step grid) | step 175, epoch 1.70 (25-step grid) |
| Trainable parameters / adapter size            | 57,016,320 / 228.1 MB | 60,948,480 / 243.9 MB   |
| Peak training memory                           | 13.09 GB              | 13.70 GB                |

**Per task, arm2 (secondary rows of H8b):**

| Task                          |   n | Hermes | Logos  |       Δ | By groups              | By examples           |
| ----------------------------- | --: | -----: | -----: | ------: | ---------------------- | --------------------- |
| `notification_prioritization` |  38 | 0.8105 | 0.8942 | +0.0837 | n.s.                   | improved, p = 0.007   |
| `context_prioritization`      |   8 | 1.0000 | 0.9995 | −0.0005 | n/a                    | no change             |
| `mission_control_briefing`    |  43 | 0.9716 | 0.9646 | −0.0070 | no change              | no change             |
| `workspace_reasoning`         |  46 | 0.9512 | 0.9370 | −0.0142 | regressed, p = 0.014   | n.s.                  |
| `tool_routing`                |  62 | 0.6679 | 0.6501 | −0.0178 | n.s.                   | n.s.                  |
| `memory_conflict_resolution`  | 111 | 0.8553 | 0.8350 | −0.0204 | n.s.                   | n.s.                  |
| `recommendation_generation`   |  41 | 0.4946 | 0.3906 | −0.1040 | regressed, p = 0.006   | regressed, p < 0.0005 |

These seven rows are not corrected for multiple comparisons. Treat them as leads.

**For the product decision:**

- **On v0.0.6, Logos does not earn a place beside Hermes as the "deeper" model.**
  It is not measurably better, it is weaker at recommendations, and its fine-tuned
  latency is about the same.
- **Its one advantage on the record is consistency across reworded cases** (0.833
  against 0.769, no interval). It also has a better base-model prior on
  abstention, which fine-tuning erases.
- **Whether the larger base pays off with better data is the open question.** The
  v0.0.7 repair spec targets exactly the limits both models share.

---

## 10. Artifact hash register

`outputs/kleos-v006-ministral314b-run1/`

| File                                         | Size (bytes) | SHA-256                                                            |
| -------------------------------------------- | -----------: | ------------------------------------------------------------------ |
| `adapter/adapter_model.safetensors`          |  243,869,280 | `f3e8dcdc09a3f3bb9f6fc4f64f867ff8b97834c955a03fcbde98a8068d5570a7` |
| `adapter/adapter_config.json`                |        1,299 | `410176a9c583585bf4b2954e26714acec2b4404645a2f67f028194c2cb7ae1e6` |
| `checkpoint-175/adapter_model.safetensors`   |  243,869,280 | `f3e8dcdc09a3f3bb9f6fc4f64f867ff8b97834c955a03fcbde98a8068d5570a7` (verified identical, 2026-09-27) |
| `tokenizer/tokenizer.json`                   |   17,078,128 | `d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135` |
| `tokenizer/tokenizer_config.json`            |          465 | `ce7ea8d28bb4ba4350064186dcf3a6dcef497b728f55b1ceee379412a8af36b2` |
| `tokenizer/chat_template.jinja`              |       11,913 | `2f545122222db8bb43ca0ea0c49e9185320a8670f7d35575b0da0eb48b1e8970` |
| `config.yaml`                                |        4,094 | `279e4e7f9aab574eb6776f741d0cb6be5886db6bc8985c9db01d196ce67c459d` |
| `manifest.json`                              |       19,320 | `4e97f576c6fb0d8220fb0c683c4d719b2eba1cd895698ec009326b8ab648ff41` |
| `metrics.json`                               |          358 | `e91a9940b5d4b7b3e9d64770dbdb549e1d4f5555772a4adb7c616737f3aa7658` |
| `events.jsonl`                               |       54,029 | `d94eafe198a231ae0015b36dae041b17102813bb0107f9ebf8750e9d47dbc6dc` |
| `training.log`                               |       31,290 | `02cc1703a46f2333cdbe17ce0b664717e69f73416f34ff2138e2f95fd50516dc` (may lack killed sessions' lines, L-F3) |
| `arm1_base_orchestrated.json`                |    1,544,480 | `a0b048b156760457670c2a061c167c383ced0f72f90a12f3b87a2ef711ac1bfd` |
| `arm2_finetuned.json`                        |      812,056 | `09243630e6dba31a3a28adf3e6492f48fbc78935d82ad69852d333016112ef1f` |
| `manifest_eval_arm1_base_orchestrated.json`  |       11,077 | `ef50f7831a87c3237a853f39a15132bd58a9b17b1ce1072f9fd2dabc20c6b306` |
| `manifest_eval_arm2_finetuned.json`          |       11,073 | `269def0c1b1d0c42e5b4d273893f344f921abc2e392f4e21f4ad4b2482e5bbbd` |
| `report_logos_v001/summary.md` (H8a)         |        3,316 | `b6fcf243334ce0d9a95ef873afa06580b7609e9c5d04ce1bf6ba0215cd8a1c16` |
| `report_logos_v001/metrics.json` (H8a)       |       15,603 | `1ed9f514f9bf26c4e0d68aa4414aa0872b416075e4eaaffb0e0fca5c1272bbef` |

`outputs/report_h8b_hermes_vs_logos/` (H8b)

| File           | Size (bytes) | SHA-256                                                            |
| -------------- | -----------: | ------------------------------------------------------------------ |
| `summary.md`   |        3,892 | `2f49765f463352588d957d1be4f82d3ba629aab76e80a797e6428540183c29b7` |
| `metrics.json` |       15,295 | `84d8c3e9d4f44ed3fd127c22ed4767818506eb32aa1b156b07855e68ce26f493` |

`outputs/kleos-v006-mistralnemo12b-run1/`: new files written beside Hermes' frozen
results by `rescore.py --mode annotate`. The originals are unmodified.

| File                                     | Size (bytes) | SHA-256                                                            |
| ---------------------------------------- | -----------: | ------------------------------------------------------------------ |
| `arm1_base_orchestrated.annotated.json`  |    1,098,607 | `7f1416cff9680d43220da2b7911ff26eb177c3323d51a33bf944d2ca3a9bb387` |
| `arm1_base_orchestrated.annotated.md`    |        1,668 | `dcdb8011538c1e1e16d0f62d29a1277f6fbe3c4ed089d014921c938f0dc8b8fc` |
| `arm2_finetuned.annotated.json`          |      795,645 | `99fdcdc3bb418853e0708cf54205949bc836229bb094edc1ed391dbc25a1fc9d` |
| `arm2_finetuned.annotated.md`            |        1,651 | `4e9c9326ffd061e2e4dff1933b28d8881b1e495dc04d2c3446bb34084dcfeaa5` |

---

## 11. Findings

The Logos phase's findings are recorded with their evidence in
[logos-findings.md](logos-findings.md). In short:

| Finding | What                                                                                     | Status |
| ------- | ---------------------------------------------------------------------------------------- | ------ |
| L-F1    | `Mistral3VLMAdapter`'s LoRA scoping does not work on transformers 5                      | open; held by strict xfails |
| L-F2    | The tokenizer class depends on `config.json` being present; serving must load `TokenizersBackend` | open for serving |
| L-F3    | Evaluation resume lost killed sessions' work on Colab's Drive mount                      | **fixed** (`ed5a987`), confirmed on Colab |
| L-F4    | A fresh start into a folder with checkpoints would delete the best checkpoint            | open |
| L-F5    | `validate_checkpoint` does not check optimizer or scheduler state                        | open |

Hermes findings that recurred in this run: H-F4, a save event that did not reach
Drive (session 1's `checkpoint-125`), and H-F8, `train_loss` after a resume. Both
are tooling issues. Neither alters a reported number.

---

## 12. Deviations and run events

- **D1–D6 carry over** (same pipeline, same sealed release). D3 caps
  `deciding_factor` on the 78 should-decline items in every arm.
- **D7 does not apply:** the base is pinned.
- **D9:** training ran in four sessions with two resumes (§2). It has no effect on
  the selected adapter.
- **D10:** `arm2_finetuned` needed three sessions. L-F3 lost the first two, and
  the third generated all 349 answers in one pass.
- **D11, declared before it ran:** `arm1_base_orchestrated` ran at `ed5a987`,
  which contains the L-F3 fix. It ran across two sessions, the second replaying
  172 recorded generations and generating 177.
- **Re-attempts cannot select a result.** Decoding is greedy and seeded, so a
  re-run reproduces the same outputs rather than drawing new ones.

---

## 13. Not done

- **`arm0_base` and `arm3_finetuned_orchestrated` were not run.** H7 remains
  incomplete.
- **H2** is not measurable on this benchmark, and **H4** is untested, as for
  Hermes.
- **No second training seed.** H8b's interval covers evaluation noise only.
- **The 41 cut-off base answers were not broken down by task** (§7). The data for
  it is in `arm1_base_orchestrated.json`.
- **Hermes' §8.2 per-family abstention table was not rebuilt for Logos.**
- **Nothing was exported, published or deployed.**

## 14. Open items before any serving

1. **Whether to serve Logos at all.** On v0.0.6 the record does not support Logos
   as a better model than Hermes (H8b). Serving it would add a model that is not
   measurably better, is weaker on recommendations, and is no cheaper.
2. **If it is served:**
   - Generalize the Hermes-coupled serving path ([../logos.md, serving](../logos.md#serving)).
   - Load the tokenizer as `TokenizersBackend` (L-F2).
   - Verify that `adapter_config.json` records base revision
     `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` (the H-F1 fix).
   - Reproduce the frozen evaluation's responses byte for byte through the serving
     path, as Hermes did.
3. **The next lever is the data, not the base.** Abstention, consistency and the
   unlearnable decline labels are shared by all three runs. The v0.0.7 repair spec
   ([../datasets/kleos-policy-v0.0.7-repair-spec.md](../datasets/kleos-policy-v0.0.7-repair-spec.md))
   addresses them.
4. **L-F4 and L-F5 are open.** Fix both before the next training run.
