# KLEOS Hermes

Hermes is the fast KLEOS model: a QLoRA fine-tune of Mistral AI's
Mistral-Nemo-Instruct-2407 (12B, Apache-2.0) on the sealed `kleos-policy-v0.0.6`
release. Its one release, v0.0.6, is frozen. [Logos](logos.md) is the deeper
model. The hypotheses are in [experiments.md](experiments.md); ids and evidence
labels are explained in [docs/experiments/README.md](experiments/README.md).

## Status

| | |
| --- | --- |
| Release | **Hermes v0.0.6**, experiment `kleos-v006-mistralnemo12b-run1` |
| Trained | 2026-09-17 to 2026-09-18, Colab, 1 × Tesla T4, two sessions |
| Evaluated | 2026-09-19 to 2026-09-22 |
| Result | H1 replicated: `arm1_base_orchestrated` 0.4755 → `arm2_finetuned` 0.8051, +0.3295 (95% CI 0.3068–0.3521, p < 0.001); 7 of 7 tasks improved, none regressed |
| Served | Since 2026-09-23, on a private Hugging Face ZeroGPU Space. The served package reproduced the frozen evaluation 9 of 9, byte for byte, on a Colab T4 and on the Space |
| Full report | [kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md) |

## Role

KLEOS uses Hermes as its fast model: fine-tuning cut its mean answer from 226 to
94 completion tokens and its mean latency on a T4 from 24.7 s to 16.1 s.

Hermes exists because of a license. The first KLEOS run,
`kleos-v006-ministral8b-run1` (2026-09-15), fine-tuned Ministral-8B, whose base
is under the Mistral AI Research License and cannot back a product. Hermes
repeats that experiment on an Apache-2.0 base, and Mistral-Nemo resolves to the
same `MistralDenseAdapter` with no code changes. Hermes is a separate
experiment, not a re-run: its valid comparison is within its own run, base
against fine-tuned on the same benchmark.

## Base and pins

| | |
| --- | --- |
| Base | `mistralai/Mistral-Nemo-Instruct-2407` @ `04d8a90549d23fc6bd7f642064003592df51e9b3`, pinned and audited 2026-09-15. Apache-2.0, ungated |
| Architecture | 40 layers, hidden 5120, MLP 14336, vocabulary 131,072 |
| Configs | [`configs/models/mistral_nemo_12b.yaml`](../configs/models/mistral_nemo_12b.yaml), [`configs/training/kleos_hermes_v006.yaml`](../configs/training/kleos_hermes_v006.yaml) |
| `config_hash` | `b2328857c6026dd7c904a6acadd00ec55a6ce5289bf6210e743615fce6d8d3e8` |
| Dataset | `kleos-policy-v0.0.6`, manifest `dataset_hash` `c7340f1d…` (release content hash `3cc9a744…`); train 820 / validation 181 / test 349, `format_holdout` split |
| Adapter | `checkpoint-200`, `adapter_model.safetensors` sha256 `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32`, 228,140,600 bytes |
| Tokenizer | The run's three files, frozen and hashed in the deployment package, loaded with `fix_mistral_regex=False` stated (the run trained and evaluated with the flag unset, which is `False`) |
| Serving runtime | NF4, double quantization, float16 compute, SDPA, `max_seq_length` 1024; greedy decoding, 512 new tokens |
| Serving record | [`configs/deployment/kleos_hermes_v006.yaml`](../configs/deployment/kleos_hermes_v006.yaml) |

The serving runtime states float16 rather than `auto`. The run was trained and
evaluated on a T4 (compute capability 7.5), where `auto` resolved to float16; on
any GPU with compute capability 8.0 or higher, including the one behind ZeroGPU,
`auto` resolves to bfloat16 and the served model would compute differently from
the one measured.

## Recipe

Every KLEOS model was trained with this recipe; Logos validated and saved every
25 steps instead of 50.

| Setting | Value |
| --- | --- |
| Quantization | QLoRA, NF4 with double quantization; fp16 compute (a T4 has no bfloat16) |
| LoRA | r 16, α 32, dropout 0.05, on all seven projections (q, k, v, o, gate, up, down); `embed_tokens` and `lm_head` excluded |
| Trainable parameters | 57,016,320 in 280 modules (1,425,408 per layer × 40) |
| Loss | Assistant tokens only |
| Optimizer | Paged AdamW 8-bit, learning rate 2e-4, cosine schedule, warmup ratio 0.03 (10 steps), max grad norm 1.0 |
| Batch | 1 per device × 8 accumulation steps = effective batch 8 |
| Length | `max_seq_length` 1024; the longest training example is 440 tokens |
| Schedule | 3 epochs, 309 optimizer steps; validation and checkpoint every 50 steps, 3 checkpoints kept |
| Selection | Best checkpoint by validation loss (`load_best_model_at_end`) |
| Seed | 42, deterministic |

**The run, measured.** Training took 11,396 s (3 h 09 m 56 s) over two sessions.
The Colab runtime died after step 250, and training resumed from
`checkpoint-250` eighteen hours later on a different T4 (deviation D8,
[deviations log](experiments.md#deviations-log)). The selected adapter was
written before the interruption. Peak memory was 13.09 GB of 14.56 GB, with no
out-of-memory error; 4 optimizer steps were skipped by the fp16 scaler.
Validation loss bottomed at step 200 (epoch 1.95, 0.04343) and rose through
epoch 3, so best-model selection kept the overfitting third epoch out of the
adapter. The exported adapter is byte-identical to `checkpoint-200`.

## H1 result

[H1](experiments.md#h1--primary-hypothesis) asks whether behavioral fine-tuning
on KLEOS policy data beats the prompt-engineered orchestration baseline. Hermes
was evaluated on the 349-item held-out benchmark (sha256 `a11ffad7…`) with
greedy decoding, `max_new_tokens` 512, seed 42 and the `kleos_policy` grader.
The comparison is a paired bootstrap over examples, 2,000 iterations, two-sided.

| Task | n | `arm1_base_orchestrated` | `arm2_finetuned` | Δ | Verdict |
| --- | --: | --: | --: | --: | --- |
| `workspace_reasoning` | 46 | 0.5190 | 0.9512 | +0.4323 | improved, p < 0.001 |
| `context_prioritization` | 8 | 0.5935 | 1.0000 | +0.4065 | improved, p < 0.001; n = 8 |
| `mission_control_briefing` | 43 | 0.5681 | 0.9716 | +0.4035 | improved, p < 0.001 |
| `memory_conflict_resolution` | 111 | 0.4756 | 0.8553 | +0.3798 | improved, p < 0.001 |
| `tool_routing` | 62 | 0.3929 | 0.6679 | +0.2750 | improved, p < 0.001 |
| `recommendation_generation` | 41 | 0.2974 | 0.4946 | +0.1971 | improved, p < 0.001 |
| `notification_prioritization` | 38 | 0.6201 | 0.8105 | +0.1904 | improved, p < 0.001 |
| **Overall** | 349 | **0.4755** | **0.8051** | **+0.3295** | 95% CI 0.3068–0.3521 |

How to read it:

- **The baseline is `arm1`, not `arm0`.** The research question names the
  prompt-engineered orchestration baseline (deviation D1). The split is a format
  holdout, not the pre-registered entity holdout (D2).
- **p < 0.001** means no resample of 2,000 crossed zero; `compare.py` prints it as
  `p=0.0`.
- **These intervals resample examples.** The 349 items are 78 groups of
  perturbations of one case, so example-level intervals are too narrow (finding
  H-F14, [Hermes run report, section 11](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)).
  Re-reported by group for the Logos comparisons, Hermes' answerable subset
  scores 0.8976 (cluster 95% CI 0.8634–0.9301), from 0.5570 for its base.
- **One training run.** No second seed was run.
- **Secondary measures**, base to fine-tuned: faithfulness 0.7513 → 0.8531,
  citation precision 0.5938 → 0.7192, parse failures 0 → 0. Responses flagged
  for a fabricated citation fell from 147 to 98, but the heuristic also flags 48%
  of the gold test answers, so it is uncalibrated (finding H-F12).

Beside the earlier Ministral-8B run (0.4744 → 0.8015) both arms land within
0.004, on a different base. That comparison is descriptive: scale, architecture
and pretraining are confounded.

## What it learned and did not

**Consistency** ([H3](experiments.md#h3--does-behaviour-survive-format-changes)):
improved, still poor, and identical to Ministral-8B.

| Metric (15 scenario families) | `arm1` | `arm2` |
| --- | --: | --: |
| `agreement_rate` | 0.133 | 0.333 |
| `correct_agreement_rate` | 0.000 | 0.333 |
| Families flipping under an irrelevant perturbation | 13 / 15 | 10 / 15 |

Scenario families mix cases with different correct answers, so a perfect
model's `agreement_rate` would be 0.600 (finding H-F11). By `group_id`, where a
perfect model scores 1.000, Hermes' fine-tuned consistency is 0.769, from 0.346
for its base (re-reported during the Logos v0.0.1 work).

**Abstention.** Of the 78 items where the reference declines:

| Scenario family | Training signal | Test cases | `arm1` | `arm2` |
| --- | --- | --: | --: | --: |
| `mem.stale_explicit_conflict` | 32/32 abstain (unconditional) | 20 | 0/20 | **20/20** |
| `route.ask_when_underspecified` | 45/45 abstain (unconditional) | 30 | 0/30 | **30/30** |
| `rec.verify_before_recommending` | 25/55 abstain (conditional) | 25 | 3/25 | **0/25** |
| `rec.abstain_without_evidence` | 3/9 abstain (conditional) | 3 | 0/3 | **0/3** |
| Should commit | | 271 | 256/271 | **271/271** |

Hermes declines where a whole family always declines in training and never where
declining depends on the evidence. It learned "this kind of question → decline",
not "the evidence is insufficient → decline", and it never wrongly declines: the
remaining error is one-sided overconfidence. The table matches Ministral-8B case
for case, which points to the v0.0.6 data and recipe rather than the base model.
The four decline labels of those 78 items never occur in v0.0.6 training
(deviation D3), so `deciding_factor` scores 0/78 there.

**Output format.** `format_valid` is 0.0000 in both arms. Every test prompt
carries JSON input and neither arm answers in JSON; H1 measures judgment
expressed in prose.

**Against Logos.** On the 271 answerable items, Logos v0.0.1 was not measurably
different from Hermes (H8b: −0.0168, cluster 95% CI −0.0546 to +0.0177), and
Logos v0.0.2 is measurably better (H9: +0.0409, cluster 95% CI +0.0040 to
+0.0780). Hermes was not retrained on `kleos-policy-v0.0.7`.

## Serving

The research artifact is frozen and is not edited. Serving uses a separate
deployment package built from it, and its manifest records what was added:

- **Base revision.** The research `adapter_config.json` records `revision: null`
  (finding H-F1) and stays that way. The package carries its own copy pinned to
  `04d8a905…`, and the loader refuses an unpinned base.
- **Tokenizer.** The package carries the run's frozen tokenizer files, hashed,
  and states `fix_mistral_regex=False`. transformers 5 can otherwise change how
  roughly 1% of tokens split.
- **Runtime and decoding.** Float16, NF4 and greedy decoding, as evaluated.

| Check | Date | Result |
| --- | --- | --- |
| Colab, Tesla T4, through the deployment loader | 2026-09-23 | 9 of 9 responses identical to the frozen evaluation, byte for byte: all seven task families plus two should-decline cases |
| Hugging Face ZeroGPU, RTX PRO 6000 Blackwell (MIG 2g.48gb slice) | 2026-09-23 | 9 of 9 byte-identical, prompt token counts equal on 9 of 9, decisions agree on 9 of 9, and a repeated call identical |
| Docker image (`linux/amd64`) | 2026-09-23 | Built and tested on CPU: preflight refuses a foreign or bfloat16 package. Not tested on a GPU |

Measured on ZeroGPU over 10 calls on 2026-09-23 (answers of 69–128 tokens):
peak VRAM 8.34 GiB; about 12 tokens/s; a warm call's median 7.9 s end to end,
a cold call's 12.9 s; 7–13 GPU seconds per call. A free account's 5 minutes of
daily GPU time therefore covers roughly **20–30 answers a day** (estimated),
shared by every KLEOS user behind one calling account. KLEOS treats Hermes as
opportunistic and falls back whenever it is not `ready`.

- Records: [serving verification records](experiments/serving-verification-records.md#verification-record--hermes-v006).
- Reference: [deployment.md](deployment.md), [serving-api.md](serving-api.md)
  (Hermes replies use contract version 1).
- Steps: [runbooks/deploy-hermes-zerogpu.md](runbooks/deploy-hermes-zerogpu.md).

## Limitations

- **Prose, not JSON.** A consumer that expects structured output must parse
  prose; `kleos_models.evaluation.graders` has tested extractors.
- **Abstains by scenario family, not by evidence.** It declined 50 of the 78
  should-decline items: none of the 28 where declining depends on the evidence.
- **Unstable under rephrasing.** 10 of 15 scenario families changed decision
  under a logically irrelevant perturbation; consistency by `group_id` is 0.769.
- **Citation heuristics.** 98 of 349 responses were flagged for a fabricated
  citation by a text-level heuristic that also flags gold answers.
- **What was not measured.** H2 is not measurable on the v0.0.6 benchmark, which
  is entirely out of distribution. H4 (general capability) was not run. `arm0`
  and `arm3` were not run, so H7 is incomplete.
- **One seed, untuned hyperparameters.** The recipe uses engineering defaults.
- **Small serving capacity.** About 20–30 answers a day per calling account on
  free ZeroGPU (estimated).
- **License.** The base is Apache-2.0: redistributing a derivative requires
  attribution and a statement of modification.

## Findings

Fourteen findings were recorded with the run, H-F1 to H-F14; none alters a
reported number. H-F11 to H-F14 were found on 2026-09-24 while Logos' comparison
was designed. Evidence and detail:
[Hermes run report, section 11](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings).

| Finding | Subject | Status, 2026-10-07 |
| --- | --- | --- |
| H-F1 | `adapter_config.json` records `revision: null` | Research copy kept as recorded; the deployment package pins the revision; newly trained adapters record it |
| H-F2 | The manifest contradicts itself on trainable parameters | Open |
| H-F3 | A stale note about sliding-window attention is frozen into the run's records | Open |
| H-F4 | `checkpoint_saved` is not a durability guarantee on Drive | Open; `validate_checkpoint` rejects a checkpoint without weights |
| H-F5 | `compare.py` checks benchmark identity by path, not content | Fixed for later runs |
| H-F6 | `assert_comparable` never compares base models | Open; a methodological caveat, not enforced by code |
| H-F7 | Result files drop grader `details` | Fixed for later runs |
| H-F8 | `train_loss` is invalid after a resume | Open |
| H-F9 | The retention pass does not protect the best checkpoint | Fixed |
| H-F10 | The memory estimator underestimated the peak by about 5 GB | Fixed: the rebuilt estimator reproduces the 13.09 GB peak |
| H-F11 | Consistency is grouped by scenario family | Consistency by `group_id` reported beside it from H8 on |
| H-F12 | The fabricated-citation heuristic flags gold answers | Reported beside its gold floor (`rescore.py --gold-targets`) |
| H-F13 | Evidence coverage is vacuous on this benchmark | Reported as vacuous |
| H-F14 | The confidence intervals treat 349 examples as independent | Cluster intervals by `group_id` from H8 on |
