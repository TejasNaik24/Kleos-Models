# Training and evaluation report — `kleos-v007-ministral314breasoning-run1` (KLEOS Logos v0.0.2)

> This is a frozen research record, written while the work was done and
> preserved as written. Editorial changes for public release are limited to
> replacing private storage paths and account identifiers with placeholders
> (marked as editor's notes) and retargeting links to documents that moved. No
> number, table, finding or decision was changed. Conventions:
> [docs/experiments/README.md](README.md).

**Trained:** 2026-10-06 · **Evaluated:** 2026-10-06 · **Verdict: H9 supported.**
Logos v0.0.2 is measurably better than Hermes on the pre-registered population.

Nothing was exported, published, uploaded or deployed in producing this report.

- **Where the artifacts are:** the adapter, checkpoints and manifests are in the
  private Kaggle notebook outputs named below. The result files were copied to a
  local machine outside this repository *[editor's note: location wording generalized]*.
- **Version control:** none of it is in this repository, and it must not be. The
  hashes in §9 are the record.

Pre-registration: [H9 in ../experiments.md](../experiments.md#h9--does-a-logos-trained-to-think-beat-hermes).
Design and Kaggle runbook: [../logos.md](../logos.md#what-changed-in-v002), [../runbooks/logos-v002-kaggle.md](../runbooks/logos-v002-kaggle.md).

---

## Summary

|                 |                                                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| KLEOS model     | **Logos v0.0.2**: Logos, trained to think before it decides                                                                      |
| Base            | `mistralai/Ministral-3-14B-Reasoning-2512` @ `51f9210f3cd20f3452a80d5819d15dc61cc50630`, text tower only. Apache-2.0, ungated     |
| Method          | QLoRA: NF4 double-quant, r 16 / α 32 / dropout 0.05, 280 modules, 60,948,480 trainable parameters; v0.0.1's recipe, value for value |
| Data            | `kleos-policy-v0.0.7`, sealed; train 820 / validation 181 / test 349; train and validation answers carry policy-derived reasoning traces |
| Hardware        | Kaggle, 2 × Tesla T4, the model's layers spread over both GPUs                                                                   |
| Primary (H9)    | Logos v0.0.2 − Hermes on the answerable subset **+0.0409**, cluster 95% CI **+0.0040 to +0.0780**: **better**                    |
| Overall         | Hermes 0.8051 · Logos v0.0.1 0.7896 · **Logos v0.0.2 0.8596** (descriptive)                                                     |
| Shipped adapter | `checkpoint-175` (epoch 1.70), validation loss 0.0293                                                                            |

What this run establishes:

1. **Logos is now measurably better than Hermes**, under a decision rule fixed
   before training. Logos v0.0.1, on the same benchmark and rule, was not (H8b,
   −0.0168, inconclusive).
2. **The thinking works as built.**
   - Every one of the 349 answers was preceded by a trace.
   - None ran out of budget while thinking.
   - The graders saw only the text after `[/THINK]`.
3. **A free 2 × T4 trains a 14B model on 736-token sequences**, with 4.0 GB to spare
   on the fuller GPU.

What it does not establish:

1. **Which change did it.** The base model and the data changed together, and
   Hermes was not retrained on v0.0.7 (declared under H9).
2. **How large the gain is.** The interval's lower end, +0.004, is close to zero,
   and this is one training run per model. "Measurably better" is supported;
   "clearly better" would overstate it.

---

## 1. Run identity and provenance

| Field                | Value                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| `experiment_id`      | `kleos-v007-ministral314breasoning-run1`                                                                |
| `name`               | `kleos-logos-v002-ministral3-14b-reasoning`                                                             |
| `config_hash`        | `d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9`: the notebook checked it against the pre-registered value before training, and the run printed it again at the end |
| Git                  | `a17ace7fd42a5d91796f6bbca3aa60bcc659d90d` (the commit that pre-registered H9), cloned fresh by both notebooks |
| `seed`               | 42                                                                                                      |
| `dataset_version`    | `kleos-policy-v0.0.7` (RELEASE.lock content hash `b53afa4216bf6973…`; `test.jsonl` `a4decaaf…`, checked by the notebook) |
| `strict_config`      | `true`                                                                                                  |
| Libraries            | transformers 5.16.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.2, tokenizers 0.23.2, installed with `--no-deps` over Kaggle's image. torch is Kaggle's own, recorded in the run's `environment.txt` |
| Kaggle environment   | "Latest Container Image": the runbook's "Pin to original environment" was not set (D12)             |
| Training notebook    | `<kaggle notebook>` *[editor's note: Kaggle notebook id replaced]*, **Version 2**. Version 1 stopped in its first cell because `COMMIT` was unset (D13) |
| Evaluation notebook  | `<kaggle notebook>` *[editor's note: Kaggle notebook id replaced]*, Version 2, with the training notebook's output attached |
| Tokenizer            | `fix_mistral_regex: true`, from the config, recorded in the load metadata                              |

---

## 2. Training

| Measure                 | Value                                                                    |
| ----------------------- | ------------------------------------------------------------------------ |
| Steps                   | 309 of 309 (3 epochs, effective batch 8), one session, no resume          |
| `train_runtime`         | 15,951 s (4.43 h); the notebook ran 17,746 s in total, including install, model download and the smoke run |
| Speed                   | about 38.5 s per training step (tqdm's rate); 51.6 s per step averaged over the run, including the validation passes every 25 steps |
| `train_loss`            | 0.0721: a real single-session mean, unlike v0.0.1's resume artifact       |
| Validation loss         | 0.0311 at step 309; **0.0293** on the loaded best weights                 |
| Smoke run and gate      | Passed (10 steps at the real settings; the memory gate is below)         |
| Model-parallel check    | Passed. The Trainer ran the split model in place                          |

The validation loss now mostly measures the trace: 58% of the validation targets
by characters. It selected the shipped checkpoint, as declared under H9.

---

## 3. Memory

Measured by the memory probe on the longest batch (1 × 736 tokens, fp16) before
step 1. The fuller GPU decides the gate.

| GPU | Peak allocated | Peak reserved | Spare after optimizer state | Estimated beforehand |
| --- | -------------: | ------------: | --------------------------: | -------------------: |
| 0   | 4.75 GB        | 4.96 GB       | 9.43 GB                     | 6.38 GB              |
| 1   | **10.16 GB**   | 10.31 GB      | **4.02 GB**                 | 8.53 GB              |

- **Peak over the whole run:** 10.21 GB (`peak_memory_gb`).
- **Gate:** passed by a wide margin (≥ 0.15 GB required).
- **The estimate was off per GPU, though the total was right** (finding L-F6).
  `device_map: auto` did not split the layers evenly:
  - GPU 0 held the embedding and layers 0–8;
  - GPU 1 held layers 9–39, the final norm and `lm_head`.

  The estimator assumed 20 layers each.
- **Evaluation peak:** 6.23 GB on the fuller GPU.

---

## 4. Checkpoints and best-model selection

- **What was kept:** `checkpoint-175`, `checkpoint-300` and `checkpoint-309`.
- **Why that identifies the best one:** `save_total_limit` is 3, and the best
  checkpoint is always kept on top of the newest (H-F9). If the best had been any
  checkpoint from 275 onward, the kept set would have been 275, 300 and 309.
- **So the selected adapter is `checkpoint-175`, epoch 1.70,** the same step Logos
  v0.0.1 selected.
- **Confirmation:** `trainer_state.json` (`best_model_checkpoint`) in the training
  output; see §11.

---

## 5. Evaluation

| Measure                   | Value                                                                  |
| ------------------------- | ---------------------------------------------------------------------- |
| Arm                       | `arm2_finetuned`, 349 items, benchmark sha256 `a11ffad7…`, seed 42, greedy, `max_new_tokens` 1024 |
| Run                       | One pass: 349 new generations, 0 replayed; 18,791 s (5.2 h) plus 229 s to load the model |
| Answers with a trace      | **349 of 349**                                                         |
| Thinking never closed     | **0** (`thinking_truncated`)                                           |
| Empty answers / parse failures | 0 / 0                                                             |
| Hit the 1,024-token budget | 6, all after a closed trace (§7)                                      |
| Completion tokens         | mean 293.5, median 285, p95 413                                        |
| Trace length              | mean 549 characters, median 540, max 840 (training traces: 327–878)    |
| Latency                   | mean 53.8 s per answer, p95 76.3 s                                     |
| `format_valid`            | 0.0, as for Hermes and Logos v0.0.1. No model answers the JSON-format test in JSON (D2); excluded from the score by design (D6) |

---

## 6. H9: comparison with KLEOS Hermes (primary)

Run with `compare.py --cross-model --primary-subset answerable --equivalence-margin 0.02`
at `a17ace7`, as pre-registered.

| Arm (answerable subset, 271 items, 61 groups) | Mean   | Cluster 95% CI |
| --------------------------------------------- | -----: | -------------- |
| Hermes `arm2_finetuned` (annotated)           | 0.8976 | 0.8634–0.9301  |
| Logos v0.0.2 `arm2_finetuned`                 | 0.9385 | 0.9083–0.9663  |
| **Logos v0.0.2 − Hermes, paired**             | **+0.0409** | **+0.0040 to +0.0780** (p = 0.031) |

**Better.** The interval lies entirely above zero.

**Secondary rows.** They are reported, not decisive, and the seven tasks are not
corrected for multiple comparisons.

- **Should-decline subset** (78 items, 17 groups): 0.4835 → 0.5855 (+0.1020).
  Cluster CI −0.0018 to +0.2062 (p = 0.054): not significant by groups.
- **Per task, by groups:**
  - **improved:** `notification_prioritization` +0.1324 (p < 0.0005),
    `recommendation_generation` +0.0893 (p = 0.002) and
    `memory_conflict_resolution` +0.0737 (p = 0.048);
  - **not significant:** `tool_routing` +0.0330, `workspace_reasoning` +0.0172 and
    `mission_control_briefing` −0.0153;
  - **not estimable by groups:** `context_prioritization` (8 items), −0.0002.
- **Consistency by `group_id`** (oracle 1.000): 0.769 → 0.833 over all groups, and
  0.770 → 0.869 over the answerable groups.
- **Overall means** (descriptive): 0.8051 → 0.8596.

The comparison reports that the decoding settings differ (`max_new_tokens` 1024
against 512). That difference is declared under H9.

---

## 7. Behavioural analysis

- **Traces look like their training targets.** They average 549 characters against
  575 in validation, and every one is closed before the answer.
- **Greedy loops: 6 of 349** (finding L-F7). All six are answerable
  `memory_conflict_resolution` items. The trace closed normally, then the answer
  repeated a phrase ("— unverified —" or "— inferred —") until the 1,024-token
  budget ran out.
  - **Scores:** 0.59–0.65 each, as given; they count against Logos.
  - **How they're reported:** `finish_reason` says "stop" for these, because it
    tracks only whether the thinking closed. `hit_max_new_tokens` counts them.
  - **This is the confound H9 named:** greedy decoding on a reasoning model.
- **The answers keep v0.0.6's style,** now with the "What decided it" line on every
  answer, including declines.

---

## 8. Secondary: Logos v0.0.1 → v0.0.2

Same benchmark, same arm.

| Subset          | v0.0.1 | v0.0.2 | Difference | Cluster 95% CI |
| --------------- | -----: | -----: | ---------: | -------------- |
| Answerable      | 0.8808 | 0.9385 | +0.0577    | +0.0234 to +0.0952 (p < 0.0005) |
| Should-decline  | 0.4724 | 0.5855 | +0.1131    | +0.0028 to +0.2296 (p = 0.046) |

- **Per task:** `recommendation_generation` +0.1933 (p < 0.0005 by groups) and
  `memory_conflict_resolution` +0.0941 (p = 0.005). No task regressed.
- **Confounded:** this comparison also changes the base release and the data
  together.

---

## 9. Artifact hash register

| Artifact | sha256 |
| --- | --- |
| Logos v0.0.2 `arm2_finetuned.json` | `0dd74661e1c5a9b1a5cd4884a2d1e7aafed500dadb0b553a4a1a0ba821c30eb5`, as saved from the browser; content verified by its `benchmark_fingerprint` `c3d0d1f15601821070b9876afafbd379add14dd437cdc283ce87296dda7237d9` |
| Logos v0.0.2 evaluation resume identity | `e07bf283c847c18d629a401804795d3cde6d93044f79355fb96df9f9ad5b4ab9` |
| Hermes `arm2_finetuned.annotated.json` | `99fdcdc3bb418853e0708cf54205949bc836229bb094edc1ed391dbc25a1fc9d` |
| Logos v0.0.1 `arm2_finetuned.json` | `09243630e6dba31a3a28adf3e6492f48fbc78935d82ad69852d333016112ef1f` |
| H9 report `summary.md` | `74203855305d3cd8d5a0a70ff622cc2c2a92aea1334d468879c103aefc62935f` |
| v0.0.1 → v0.0.2 report `summary.md` | `eaa4a5e7975c62f2452ba1e5244a9b79f5ce37ec118951c5a7019b0bc5b6b514` |
| Benchmark | `a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266` |
| Adapter (`adapter_model.safetensors`) | **not yet recorded** (§11) |

---

## 10. Findings

**L-F6 — `device_map: auto` did not balance the layers across two T4s.** VERIFIED
from the evaluation's load metadata.

- **What happened:**
  - GPU 0 took the embedding and 9 layers;
  - GPU 1 took 31 layers, the norm and `lm_head`.
- **The estimator's assumption:** an even split, which put GPU 1 at 8.53 GB.
  It measured 10.16 GB.
- **Effect here:** none; 4.0 GB stayed spare.
- **Why it matters later:** for a model closer to the limit, the estimate would
  pass a run that fails.
- **The fix:** model the map accelerate actually produces, or record this run as
  an empirical anchor for two GPUs.

**L-F7 — `finish_reason` misses an answer cut by the budget after a closed trace.**
VERIFIED: 6 answers, §7.

- **What happens:** `finish_reason` reads "stop" whenever the trace closed, so the
  only record of the cut is `hit_max_new_tokens`.
- **Why it matters for serving:** a served answer needs a reason that says it
  was cut.
- **The fix:** set "length" whenever the completion ends without an end-of-sequence
  token. Deferred from the code review as a minor; it is now measured.

**L-F8 — the evaluation reports no progress while it runs.**

- **What happens:** a Kaggle commit run shows only its log, and `evaluate.py`
  prints nothing per answer. A 5-hour evaluation was opaque until it finished.
- **The fix:** log a line every N answers, with the count and the mean latency.

---

## 11. Not done, and open items before any serving

1. **Record the adapter's sha256** (`adapter_model.safetensors` from
   `outputs/kleos-v007-ministral314breasoning-run1/adapter`) and
   `trainer_state.json`'s `best_model_checkpoint`, from the training notebook's
   output.
2. **Serving a thinking model is new.** The Space needs:
   - the token-level split;
   - a decision on whether to show or return the trace;
   - L-F7's `finish_reason`;
   - a time budget for about 290 generated tokens per answer.
3. **The greedy loops** (L-F7) would reach users. Decide whether serving keeps
   greedy decoding, or adds a repetition guard (a declared serving difference).
4. **Not run:** `arm1` (declared), a second seed, and Hermes retrained on v0.0.7.
