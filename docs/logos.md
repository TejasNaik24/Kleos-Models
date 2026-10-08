# KLEOS Logos

Logos is the deeper of the two KLEOS models: a QLoRA fine-tune of the text tower
of Mistral AI's Ministral 3 14B. [Hermes](hermes.md), on Mistral-Nemo 12B, is the
fast one. The hypotheses are pre-registered as H8 and H9 in
[experiments.md](experiments.md); ids, evidence labels and the index of run
records are in [docs/experiments/README.md](experiments/README.md).

## Status

| Release | Base | Training data | Trained | Pre-registered result | Serving |
| --- | --- | --- | --- | --- | --- |
| **v0.0.1** | `mistralai/Ministral-3-14B-Instruct-2512-BF16` @ `3cea74c1` | `kleos-policy-v0.0.6` | 2026-09-24 to 2026-09-27, Colab, 1 × T4 | H8a supported; H8b, the comparison with Hermes, inconclusive | Not served. Research only |
| **v0.0.2** | `mistralai/Ministral-3-14B-Reasoning-2512` @ `51f9210f` | `kleos-policy-v0.0.7` | 2026-10-06, Kaggle, 2 × T4 | H9 supported: better than Hermes | Live as a Beta since 2026-10-07; 8 of 9 smoke-test answers reproduced byte for byte |

- **Logos v0.0.1** (`kleos-v006-ministral314b-run1`) was not measurably better
  than Hermes on v0.0.6: answerable subset −0.0168, cluster 95% CI −0.0546 to
  +0.0177. Report:
  [kleos-v006-ministral314b-run1-report.md](experiments/kleos-v006-ministral314b-run1-report.md).
- **Logos v0.0.2** (`kleos-v007-ministral314breasoning-run1`) is the Reasoning
  release of the same model, trained to write the policy's reasoning before it
  answers. It is measurably better than Hermes: answerable subset +0.0409,
  cluster 95% CI +0.0040 to +0.0780. The lower bound is close to zero and each
  model is one training run. Report:
  [kleos-v007-ministral314breasoning-run1-report.md](experiments/kleos-v007-ministral314breasoning-run1-report.md).

## Why Ministral 3

Decision (2026-09-24): Logos uses `mistralai/Ministral-3-14B-Instruct-2512-BF16`
@ `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`, text tower only. Logos v0.0.2 keeps
the architecture and moves to the Reasoning release of the same model (below).

| | |
| --- | --- |
| License | Apache-2.0, ungated. No token is needed to download it |
| Released | 2025-12-02 ([card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16), [announcement](https://mistral.ai/news/mistral-3)) |
| Checkpoint | `Mistral3ForConditionalGeneration` (`mistral3`) with a `ministral3` text tower, BF16, 6 shards |
| Parameters | 13,506,073,600 text + 438,958,080 vision (tower 403,305,472, projector 35,652,608), counted on the meta device |
| Text tower | 40 layers, hidden 5120, MLP 16384, 32 query / 8 KV heads, head_dim 128, vocabulary 131,072, untied `lm_head`, YaRN to 262,144 tokens, read from `config.json` |
| LoRA r=16 on 7 projections | 280 modules, 60,948,480 trainable parameters, computed; the same formula gives Hermes' measured 57,016,320 |
| Configs | [`configs/models/ministral3_14b.yaml`](../configs/models/ministral3_14b.yaml), [`configs/training/kleos_logos_v001.yaml`](../configs/training/kleos_logos_v001.yaml) |

The reasons, in order:

- **It is the largest Mistral open-weight model that trains on a free Colab T4.**
  Every stronger candidate needs more memory than a 16 GB T4 has, even in 4-bit.
- **Its base is more capable than Hermes' base.** MMLU (5-shot, base models) is
  79.4 against Nemo's 68.0; Mistral Small 24B scores 81.0. Ministral 3 is
  distilled from Mistral Small 3.1
  ([Ministral 3 paper, Table 3](https://arxiv.org/html/2601.08584);
  [Nemo card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)).
  These figures come from different harnesses and years, so they are indicative.
- **It keeps Hermes' architecture.** The geometry (40 × 5120), the attention and
  the vocabulary are the same; only the MLP is wider (16384 against 14336). The
  training pipeline, the deployment path and the comparison with Hermes carry
  over.
- **It serves on free ZeroGPU** as Hermes does ([Serving](#serving)).

The difference in size is modest: 13.5B text parameters against Hermes' 12.2B.
What Logos adds is base capability, and on v0.0.6 that capability can only show
where the data allows it ([Dataset](#dataset)).

**The -BF16 repository.** The Instruct model's default repository, without the
`-BF16` suffix, is published in FP8, which needs compute capability 9 or higher
([transformers FP8 docs](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)).
A T4 is 7.5, and the loader refuses a pre-quantized container. The Reasoning
release is published in BF16 directly.

### Candidates

Survey of all 75 `mistralai` repositories on Hugging Face, 2026-09-24. Parameter
counts, gating, files and revisions come from the Hub API
(`https://huggingface.co/api/models/<repo>`), release dates from each
repository's commit history.

| Model | Size | License | QLoRA on a free T4 | Verdict | Sources |
| --- | --- | --- | --- | --- | --- |
| **Ministral-3-14B-Instruct-2512-BF16** | 13.9B (13.5B text + 0.44B vision) | Apache-2.0 | Yes, with a small margin | **Selected** | [card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16), [docs](https://docs.mistral.ai/models/ministral-3-14b-25-12) |
| Mistral-Small-3.1-24B-Instruct-2503 | 24.0B | Apache-2.0 | No | Strongest candidate; needs Kaggle's free 2 × T4 ([Unsloth](https://unsloth.ai/docs/models/tutorials/magistral-how-to-run-and-fine-tune): a 24B model "slightly exceeds the memory limits of a 16GB VRAM") | [card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| Mistral-Small-3.2-24B-Instruct-2506 | 24.0B | Apache-2.0 | No | Also ships no Hugging Face tokenizer or chat template (`tekken.json` only) | [card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| Mistral-Small-24B-Instruct-2501 | 23.6B, text-only | Apache-2.0 | No | Older; 32k context; also needs Kaggle | [card](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| Magistral-Small-2509 | 24.0B, reasoning | Apache-2.0 | No | No Hugging Face tokenizer; the v0.0.6 data had no reasoning traces | [card](https://huggingface.co/mistralai/Magistral-Small-2509) |
| Mistral Medium 3 / 3.1 | | API only | | No open weights | [announcement](https://mistral.ai/news/mistral-medium-3) |
| Mistral-Medium-3.5-128B | 127.7B | Modified MIT (revenue cap) | No | Too large; restrictive license | [license](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE) |
| Mistral-Small-4-119B, Mistral-Large-3-675B, Devstral-2-123B | 119B–675B | Apache / modified MIT | No | Far too large for free hardware | [Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), [Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) |
| Large 2 / 2.1, Pixtral Large, Small-2409, Codestral-22B | 22B–124B | MRL / MNPL | | Non-commercial: disqualified | [MRL](https://mistral.ai/licenses/MRL-0.1.md), [MNPL](https://mistral.ai/licences/MNPL-0.1.md) |

Benchmarks, as each source reports them (MMLU 5-shot on base models): Nemo 68.0
([card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)), Ministral
3 14B 79.4 and Small 3.1 81.0 ([paper, Table 3](https://arxiv.org/html/2601.08584)),
Small 2501 80.73 ([card](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)).

Decision (2026-09-24): the 24B candidates were not used. They are the stronger
bases, but none fits a free Colab T4, and the workflow they need (Kaggle's 2 ×
T4, a second free platform with its own quota and session rules) was not adopted
for v0.0.1. Logos v0.0.2 later moved to Kaggle's 2 × T4 for a different reason:
its longest training example does not fit one T4 (estimated,
[Measured fits](#measured-fits)).

## The text-only view

KLEOS is text-only, and the 0.44B-parameter vision tower would cost memory a T4
does not have. `model_type: ministral3` makes `Ministral3TextAdapter` load only
the text tower, as `Ministral3ForCausalLM`, from the official checkpoint, with
its keys renamed (`language_model.model.*` → `model.*`,
`language_model.lm_head.*` → `lm_head.*`). The load refuses to continue if any
text weight is missing or mismatched, or if any key outside the vision tower and
projector goes unused. The vision weights stay on disk.

Logos v0.0.2 uses the same view through `Ministral3ReasoningTextAdapter`
(`model_type: ministral3_reasoning`), which adds a thinking-only capability and
refuses `standard` mode.

What was verified for v0.0.1, against the pinned revision, with the Docker
image's pinned transformers 5.16.1:

| Check | Result |
| --- | --- |
| Text-only loading | On a tiny checkpoint in the official key layout, the text view loads every language weight exactly; `lm_head` stays untied; vision and projector keys are the only ones unused (`tests/test_ministral3_text_view.py`) |
| Tokenizer files | `tokenizer.json` `d5f60467…8135`, `tokenizer_config.json` `f59f7294…0d6d`, `chat_template.jinja` `2f545122…8970`, pinned in `tests/test_revision_pinning.py` |
| Chat template | Renders the system prompt in place (`[SYSTEM_PROMPT]…`), so Hermes' system-prompt merge (deviation D5, [deviations log](experiments.md#deviations-log)) does not apply. Its default system prompt is used only when a conversation has none, and every KLEOS example has one |
| `fix_mistral_regex` | transformers flags this tokenizer's pre-tokenizer regex as incorrect. Setting the flag to true changes the token ids of 0 of 1,350 formatted v0.0.6 examples. Logos sets it explicitly so training and serving agree on user text. Re-measured 2026-09-24 with the tokenizer class the Hub gives (finding L-F2, [Logos findings](experiments/logos-findings.md)) |
| Sequence lengths | Longest training example 442 tokens (448 after the collator pads to a multiple of 8), validation 436, test 612, through the repository's own formatter. Hermes' tokenizer gives 440 for training. `max_seq_length` 1024 truncates nothing. Measured, and confirmed by the Colab smoke run's own measurement |

## Measured fits

### Logos v0.0.1 on one T4

The smoke run (`kleos-logos-smoke-001`, Tesla T4, 2026-09-24) probed the
longest batch (1 × 448 tokens, fp16): **13.60 GiB allocated, 13.96 GiB
reserved** (smoke run), leaving 0.34 GiB once the optimizer state exists. That
passes the 0.15 GiB gate. The ten training steps peaked at 13.55 GiB. The full
run's own probe, before step 1, recorded 13.60 GB allocated, 13.97 GB reserved
and 0.35 GB spare. The full run peaked at 13.70 GB of 14.56 GB (session 4's
manifest), with no out-of-memory error in any of its four sessions
([run report, section 4](experiments/kleos-v006-ministral314b-run1-report.md#4-memory)).

The estimate came first and decided that the smoke run was worth running. The
estimator (`src/kleos_models/models/feasibility.py`) was rebuilt for Logos after
finding H-F10 ([Hermes run report, section 11](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)).
It counts what a QLoRA step on these models holds: the embeddings and `lm_head`
upcast to fp32 by k-bit preparation, the 16-bit autocast copy of `lm_head`, fp32
LoRA weights and gradients, logits at the loss, and one layer's activations under
gradient checkpointing. Checked against the peaks measured on earlier T4 runs:

| Run | Measured peak | Estimated | Error |
| --- | ---: | ---: | ---: |
| Hermes (`kleos-v006-mistralnemo12b-run1`) | 13.09 GB | 13.09 GB | −0.03% |
| `kleos-v006-ministral8b-run1` | 9.67 GB | 9.66 GB | −0.10% |

For Logos at its longest batch (estimated):

```text
$ python scripts/plan_run.py --config configs/training/kleos_logos_v001.yaml \
      --simulate-gpu t4-colab --seq-length 448

✓ ministral3_14b (mistralai/Ministral-3-14B-Instruct-2512-BF16): ADAPTER TRAIN (MARGINAL)
  base weights               10.85 GB (of which unquantized 5.00 GB)
  LoRA weights               0.227 GB (60,948,480 trainable params)
  gradients                  0.227 GB
  optimizer state            0.114 GB (paged, outside the allocator)
  activations and logits      2.47 GB (batch 1 x seq 448)
  ----------------------------------------
  estimated peak allocated   13.77 GB
  + allocator reserve         0.50 GB
  required                   14.27 GB
  budget:   14.41 GB
  headroom: +0.14 GB
```

The estimate was 1.3% pessimistic against the smoke run's probe. Reserved memory
was 0.36 GiB above allocated under expandable segments, against Hermes' 0.49. The
budget is the 14.56 GiB torch reports on Colab's T4, less the 0.14 GiB Hermes'
run used outside PyTorch's allocator. The 0.5 GiB reserve is the fragmentation
slack Hermes' run showed (13.58 GB reserved at a 13.09 GB peak).

Logos' peak is about 0.5 GiB above Hermes': 0.60 GiB of extra 4-bit MLP weights
and 0.03 GiB of extra LoRA state, with sequences of the same length (442 against
440 tokens). It fits only if the allocator fragments no more than it did for
Hermes, which led to two measures:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set for every Logos run.
   It changes how memory is allocated, not what is computed. It is recorded in
   the manifest's environment block and declared under H8.
2. Before its first step, the trainer runs two forward and backward passes on the
   longest batch and records the measured peak (`manifest.metrics.memory_probe`).
   A ten-step smoke run would otherwise report whatever ten random batches
   happened to need.

### The go/no-go gate

After the smoke run:

```bash
python scripts/check_smoke_gate.py --manifest <run>/manifest.json \
    --expect-modules 280 --expect-trainable 60948480 --min-spare-gb 0.15
```

GO requires all of the following: the run completed with no adjustment; the text
view loaded with 0 missing and 0 mismatched weights; exactly 280 LoRA modules and
60,948,480 trainable parameters; gradients reached the adapter; and at least 0.15
GiB spare at the longest batch once the optimizer state exists. The 0.15 GiB is a
judgment, not a measurement: room for validation passes and checkpoint writes,
which Hermes survived with 0.84 GiB free.

On NO-GO the full run does not start, and no hyperparameter changes to make it
fit. The options recorded on 2026-09-24, none implemented: keep `lm_head` in
16-bit instead of letting k-bit preparation upcast it to fp32 (under fp16 autocast
its matmul runs in 16-bit either way, so the arithmetic should be identical, and
it saves about 2.5 GiB; a code change with a bit-identity test and a declared
deviation), or move to Kaggle's free 2 × T4 or P100.

### Logos v0.0.2 on two T4s

Estimated with the same estimator: seven LoRA targets, batch 1, the longest
example (736 tokens), paged 8-bit AdamW.

| Hardware | Tier | Peak allocated |
| --- | --- | --- |
| 1 × T4 (Colab) | Smoke: does not fit | 14.46 GB against a 14.41 GB budget |
| 2 × T4 (Kaggle) | Full research | 6.38 GB on GPU 0, 8.53 GB on GPU 1 (`lm_head`, its 16-bit copy and the logits) |

```bash
python scripts/plan_run.py --config configs/training/kleos_logos_v002.yaml \
    --seq-length 736 --simulate-gpu T4:14.56:7.5:2
```

Measured by the memory probe on the longest batch (1 × 736 tokens, fp16) on
2026-10-06:

| | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Peak allocated | 4.75 GB | **10.16 GB** |
| Spare after optimizer state | 9.43 GB | **4.02 GB** |
| Estimated beforehand | 6.38 GB | 8.53 GB |

The gate passed by a wide margin. The total matched the estimate, but the split
did not: `device_map: auto` put the embedding and layers 0–8 on GPU 0, and
layers 9–39, the final norm and `lm_head` on GPU 1, where the estimator assumed
20 layers each (finding L-F6,
[Logos v0.0.2 run report, section 10](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings)).
The peak over the whole run was 10.21 GB, and the evaluation peaked at 6.23 GB
on the fuller GPU.

## H8 and H9 in brief

Both hypotheses were pre-registered before the run that tests them, with a
decision rule fixed in advance. Each compares two `arm2_finetuned` models on the
**271 answerable** benchmark items (61 groups), by a paired cluster bootstrap over
`group_id` with 2,000 iterations. The verdict is better (95% CI entirely above 0),
worse (entirely below), equivalent (entirely within ±0.02) or inconclusive. The
other 78 items should be declined, and their four decline labels never occur in
v0.0.6 training (deviation D3, [deviations log](experiments.md#deviations-log)),
so they are reported as a secondary row.

### H8: a stronger base (Logos v0.0.1)

Same sealed data, recipe (value for value), benchmark (sha256 `a11ffad7…`),
grader, greedy decoding and seed as Hermes; the declared differences are listed
below. `config_hash`
`18008c6716a58afc284c64eb7e1e96c9bfce34b8da2b8c489b6d59c0c45b6f71`.
[Pre-registration and full result](experiments.md#h8--does-a-stronger-base-make-a-better-kleos-model).

| | Result | Verdict |
| --- | --- | --- |
| **H8a**: Logos base vs Logos fine-tuned (2026-10-01) | 5 of 7 tasks improved by group intervals, none regressed. Overall 0.4469 → 0.7896; answerable +0.3952 (cluster CI +0.3443 to +0.4438) | Supported |
| **H8b** (primary): Hermes vs Logos v0.0.1 (2026-09-29) | Answerable 0.8976 → 0.8808, **−0.0168**, cluster 95% CI −0.0546 to +0.0177 | Inconclusive |

- **What H8b rules out:** the upper end, +0.0177, is below the +0.02 margin, so
  these data do not support a Logos advantage as large as the margin. A Hermes
  lead of up to about 0.055 is not ruled out either.
- **Secondary rows** (not corrected for multiple comparisons):
  `recommendation_generation` regressed (0.4946 → 0.3906) in both bootstraps;
  consistency by `group_id` was 0.833 for Logos against 0.769 for Hermes (no
  interval).
- **Declared differences from Hermes' run**, none in the training arithmetic:
  the base model, tokenizer and chat template; `fix_mistral_regex: true`;
  evaluation and checkpoint cadence 25 steps instead of 50; `strict_config: true`;
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the best checkpoint
  protected from pruning (finding H-F9,
  [Hermes run report, section 11](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings));
  a gradient check at the configured batch size and the memory probe, both
  before the Trainer re-seeds.

### H9: trained to think (Logos v0.0.2)

The training section is v0.0.1's, value for value, as
`tests/test_logos_v002_config.py` checks; the base release, the data, the
evaluation token budget and the hardware differ (next section). `config_hash`
`d1961583546b5761dd854cfe845838ca91a87ea6189d54be617de21085e8cfa9`.
[Pre-registration and full result](experiments.md#h9--does-a-logos-trained-to-think-beat-hermes).

| Arm (answerable subset, 271 items, 61 groups) | Mean | Cluster 95% CI |
| --- | ---: | --- |
| Hermes `arm2_finetuned` (annotated) | 0.8976 | 0.8634–0.9301 |
| Logos v0.0.2 `arm2_finetuned` | 0.9385 | 0.9083–0.9663 |
| **Logos v0.0.2 − Hermes, paired** | **+0.0409** | **+0.0040 to +0.0780** (p = 0.031) |

- **Verdict: better** (2026-10-06). The lower end, +0.004, is close to zero, and
  each model is a single training run, so training noise is not in the interval.
  The base release, the data and the generation budget (1,024 against 512 new
  tokens) changed together, and Hermes was not retrained on v0.0.7, so H9 does
  not say which change produced the gain.
- **Should-decline subset** (78 items, 17 groups): 0.4835 → 0.5855 (+0.1020),
  cluster CI −0.0018 to +0.2062: not significant by groups.
- **Per task, by groups:** `notification_prioritization` +0.1324,
  `recommendation_generation` +0.0893 and `memory_conflict_resolution` +0.0737
  improved; `tool_routing`, `workspace_reasoning` and `mission_control_briefing`
  were not significant; `context_prioritization` (8 items) is not estimable by
  groups.
- **Against Logos v0.0.1:** answerable +0.0577 (cluster CI +0.0234 to +0.0952),
  should-decline +0.1131 (+0.0028 to +0.2296), no task regressed. This also
  changes the base release, the data and the generation budget together.
- **Overall means** (descriptive): Hermes 0.8051, Logos v0.0.1 0.7896, Logos
  v0.0.2 0.8596.

## Dataset

Decision (2026-09-24): Logos v0.0.1 trains on the sealed `kleos-policy-v0.0.6`,
unchanged, so that any difference from Hermes is the base and not the data. The
cost is that v0.0.6's limits cap Logos as they cap Hermes:

- The four decline labels are absent from training, so judgment is capped at
  0.9255 and `deciding_factor` at 0.7765 on the full benchmark.
- Abstention in training is conditional on the scenario family, not on the
  evidence, which is the shortcut both Hermes and Ministral-8B learned.
- The test split is 100% JSON-formatted inputs, a format no training example uses.

The repair belongs upstream, in the private data repository. The
[v0.0.7 repair specification](datasets/kleos-policy-v0.0.7-repair-spec.md) keeps
`test.jsonl` byte-identical, so every v0.0.7 result stays comparable with every
v0.0.6 number. Logos v0.0.2 trains on `kleos-policy-v0.0.7` (RELEASE.lock content
hash `b53afa4216bf6973…`): v0.0.6 plus a policy-derived reasoning trace on every
train and validation answer, a "What decided it" line on every answer, and the
four decline labels in training.

## What changed in v0.0.2

| | Logos v0.0.1 | Logos v0.0.2 |
| --- | --- | --- |
| Base | Ministral 3 14B Instruct (BF16) @ `3cea74c1` | Ministral 3 14B **Reasoning** @ `51f9210f` |
| Data | `kleos-policy-v0.0.6` | **`kleos-policy-v0.0.7`**: v0.0.6 plus a policy-derived trace on every train and validation answer, and a "What decided it" line |
| What is trained | The answer | `[THINK]trace[/THINK]answer</s>`, all supervised |
| Longest example | 448 tokens | 736 tokens |
| Hardware | Colab, 1 × T4 | Kaggle, **2 × T4**, layers spread over both |
| Evaluation | `max_new_tokens` 512 | 1024; the trace is split off at `[/THINK]` and only the answer is graded |
| Recipe | Hermes' | v0.0.1's, value for value |
| Adapter selection | Lowest validation loss on the answer | Lowest validation loss on trace and answer together; the trace is 58% of the validation targets by characters |

The base, checked at the pinned revision
`51f9210f3cd20f3452a80d5819d15dc61cc50630`:

- the same architecture and text-tower shape as v0.0.1: 13,506,073,600 text
  parameters, untied `lm_head`, and the same 280 LoRA modules with 60,948,480
  trainable parameters;
- its own tokenizer files and chat template: `tokenizer.json` `577575…`,
  `chat_template.jinja` `6b5044…`, `tokenizer_config.json` `f3a437…` (the Hub
  file at the pinned revision; the copy the run saved and the package carries
  is `ce7ea8d2…`, see
  [`configs/deployment/kleos_logos_v002.yaml`](../configs/deployment/kleos_logos_v002.yaml));
  `fix_mistral_regex: true` changes 0 of 1,350 v0.0.7 examples (measured).

The template renders an assistant message's `reasoning` field as the thinking
span. It adds its default "how you should think" system prompt only to a
conversation without a system message; every v0.0.7 example and every benchmark
prompt has one, so it never appears. The
[model card](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512)
reports AIME 2025 0.850 and GPQA Diamond 0.712 for this release, and no
reasoning scores for the Instruct release. It recommends sampling at
temperature 1; KLEOS evaluates greedily, as for every model, and H9 declares
the difference.

What the code does for a model trained to think:

- **Data contract.** Schema 1.1 adds an optional assistant `reasoning` field. It
  is written only when present, so every earlier release keeps its bytes and
  hashes.
- **Formatting.** With `strip_thinking_from_targets: false` the field goes to the
  chat template and is supervised. A run that would cut any supervised token
  refuses to start. For any other model the field is dropped and counted.
- **Evaluation.** In thinking mode the completion is split on token 35
  (`[/THINK]`) before decoding, so the graders never read the trace. A completion
  whose thinking never closes has an empty answer and `finish_reason: length`;
  `generation_stats.thinking_truncated` counts them. Results schema 3 stores each
  trace beside its answer.
- **Two GPUs.** The memory probe measures every GPU and gates on the tighter one.
  The run refuses to train a model spread over GPUs unless the Trainer runs it
  model parallel; transformers 5.16.1 does (checked in its source), and the check
  stops the run if that ever changes. The manifest records where each part of the
  model landed.

## Recorded result

The v0.0.2 result as recorded on 2026-10-06, kept as written. Its references to
sections that have since moved were retargeted to the documents that now hold
them.

> Measured on the run, with ESTIMATED figures from [the Kaggle runbook](runbooks/logos-v002-kaggle.md) alongside:
> *[editor's note: the ESTIMATED figures are in the
> [run report](experiments/kleos-v007-ministral314breasoning-run1-report.md),
> not reproduced here.]*
>
> - **Training:** 4.4 hours (about 4.9 with setup), one session. `checkpoint-175` was
>   selected (epoch 1.70, validation loss 0.0293).
> - **Evaluation:** 5.2 hours, 53.8 s per answer. Every answer thought first, and
>   none ran out of budget while thinking.
> - **[H9](experiments.md#h9-result--2026-10-06): better than Hermes.** Answerable
>   subset 0.8976 → 0.9385 (+0.0409, cluster 95% CI +0.0040 to +0.0780). Overall
>   0.8051 → 0.8596.
> - **Against Logos v0.0.1:** answerable +0.0577, should-decline +0.1131, and no task
>   regressed.
> - **Kaggle quota used:** about 10.5 GPU hours of the weekly 30.
>
> Three findings, L-F6 to L-F8, are recorded in the
> [run report](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings):
>
> - **L-F6:** the two-GPU split was uneven.
> - **L-F7:** 6 answers looped after their trace until the token budget ran out, yet
>   `finish_reason` still reads "stop". **Resolved in serving:** a served reply
>   says "length" whenever the budget filled or the thinking never closed. The
>   evaluation code and its stored results are unchanged.
> - **L-F8:** the evaluation shows no progress while it runs.
>
> Its [section 11](experiments/kleos-v007-ministral314breasoning-run1-report.md#11-not-done-and-open-items-before-any-serving)
> lists what serving a thinking model still needs. That serving is now
> built: [deployment.md](deployment.md#serving-profiles).

## Serving

Logos v0.0.2 is served on a private Hugging Face ZeroGPU Space
(`<owner>/<space>`), from a private deployment package (`<owner>/<package-repo>`),
by the same serving code as Hermes. A per-model profile in
[`configs/deployment/kleos_logos_v002.yaml`](../configs/deployment/kleos_logos_v002.yaml)
sets what differs ([serving profiles](deployment.md#serving-profiles)). There is
no Docker service for Logos.

| | |
| --- | --- |
| Adapter | `e49724f6554db662769fbe9b9431105dbc12f8f75cecd6e0e00728ea3f04dc4f`, from `checkpoint-175` (`eval_loss` 0.0293) |
| Base | `mistralai/Ministral-3-14B-Reasoning-2512` @ `51f9210f3cd20f3452a80d5819d15dc61cc50630`, 6 shards baked into the Space image; the vision weights stay on disk |
| Runtime | NF4, double quantization, float16 compute, SDPA; greedy decoding; 1,024 new tokens, as evaluated |
| Reply | Contract version 2: `text` is the answer, `reasoning` the trace written before it (or `null`). Only the answer was graded in H9 |
| Cut replies | `finish_reason: "length"` whenever the 1,024-token budget filled or the thinking never closed (then `text` is empty). KLEOS then falls back |
| Requests | The first message must be a system message, as in every training and benchmark prompt; otherwise the request is refused as `invalid_request` before any GPU work |

**Reproduction, 2026-10-07.** The smoke test compared 9 served answers with the
T4 evaluation, over two quota windows on 2026-10-07:

- 8 of 9 answers and traces were byte-identical; prompt tokens matched on 9 of 9;
  decisions agreed on 8 of 9.
- The ninth (`kx-mcb-066bff4fa33aa29b`, a briefing) kept an identical trace for
  568 of its 761 characters, then took a different sentence. The served answer
  declined and asked before looking in another workspace, where the evaluation
  ranked the items correctly. The evidence is consistent with GPU arithmetic
  rather than setup: equal prompt tokens and a late divergence. Rounding in fp16
  on the Space's Blackwell GPU differs slightly from the T4's, and a thinking
  model has a few hundred trace tokens in which a near-tie between two next
  tokens can flip. That Hermes, which does not think, matched 9 of 9 on the same
  GPU type supports this reading without proving it.
- Decision (2026-10-07): accept and document. Logos is served as a Beta,
  numerically different from the T4 evaluation. Precision and decoding stay as
  evaluated, because changing them to force a match would change the model.

**Measured on the Space, 2026-10-07:** model load 57.1 s at startup, host peak
memory 30.6 GiB; GPU peak 8.86 GiB on an RTX PRO 6000 Blackwell MIG 2g.48gb
slice; 11.0 tokens/s warm, a warm median of 19.7 s per call; 24–26 GPU seconds
per answer. **About 9 answers fit a fresh daily window** (estimated: ZeroGPU
admits a call only while 1.5 × the requested 60 s, 90 s, of the 300 s quota
remain). On day 1, 7 were admitted and the 8th refused; that window had already
been partly used before the test.

The full deployment record and the smoke-test log are in the
[serving verification records](experiments/serving-verification-records.md#verification-record--logos).
Deployment steps: [runbooks/deploy-logos-zerogpu.md](runbooks/deploy-logos-zerogpu.md).
The request and reply contract: [serving-api.md](serving-api.md).

Logos v0.0.1 is not served. On v0.0.6 it was not measurably better than Hermes,
it was weaker on recommendations, and it cost about the same per answer (17.2 s
against Hermes' 16.1 s on a T4). The serving path was generalized from Hermes for
v0.0.2 instead: one codebase that reads a per-model profile from the deployment
record.

## Limitations

- **One training run per model.** Every interval covers evaluation noise, not
  training noise. No second seed was run.
- **H9 confounds three changes.** The base release, the data and the generation
  budget (1,024 against 512 new tokens) changed together, and Hermes was not
  retrained on v0.0.7 (decided 2026-10-06). H9's lower bound, +0.0040, is close
  to zero.
- **No `arm1` for v0.0.2.** Fine-tuning's own effect on v0.0.2 was not measured;
  H8a answered that question for v0.0.1's architecture.
- **The traces come from the policies.** They are written from the same policies
  that define the benchmark's reference answers, so H9 measures how well a model
  learns those policies, not general reasoning.
- **Adapter selection rewards the trace.** v0.0.2's validation loss covers trace
  and answer together, and the trace is 58% of the targets by characters. No
  answer-only loss is recorded.
- **Should-decline items.** v0.0.2's gain there (+0.1020) is not significant by
  groups. v0.0.1 cannot learn the four decline labels at all. Whether Logos
  v0.0.2 declines by evidence rather than by scenario family (the shortcut found
  in H3 for Hermes) was not analyzed.
- **Prose, not JSON.** `format_valid` is 0.0 for every KLEOS model: no model
  answers the JSON-format test inputs in JSON (deviation D2,
  [deviations log](experiments.md#deviations-log)).
- **Greedy decoding loops.** 6 of 349 v0.0.2 answers repeated a phrase after a
  closed trace until the 1,024-token budget ran out (L-F7). They were scored as
  given, and a served reply is marked `length`.
- **The served GPU is not the evaluated GPU.** H9 measured the T4 outputs. The
  Space reproduced 8 of 9 checked answers, and its outputs are not re-measured:
  the full 349 would take over five weeks at about 9 answers a day.
- **Small serving capacity.** About 9 answers a day per calling account
  (estimated; 7 observed on day 1), shared with Hermes when one token calls both.
- **Open training gaps.** Findings L-F4 (a fresh start into a folder with
  checkpoints would delete the best one) and L-F5 (`validate_checkpoint` does
  not check optimizer or scheduler state) are not fixed in `scripts/train.py`
  ([Logos findings](experiments/logos-findings.md)). The Kaggle training notebook
  works around both.

## Runbooks

- [Logos v0.0.1 on Colab](runbooks/logos-v001-colab.md): the smoke run, gate,
  training, both evaluations and the H8a and H8b comparisons, as run between
  2026-09-24 and 2026-10-01.
- [Logos v0.0.2 on Kaggle](runbooks/logos-v002-kaggle.md): the two unattended
  notebooks, resume, and the H9 comparison, as run on 2026-10-06.
- [Deploying Logos on ZeroGPU](runbooks/deploy-logos-zerogpu.md): package,
  Space and the smoke test over two quota windows.

## Findings

Eight findings came out of the Logos work. L-F1 to L-F5 were found during
v0.0.1 and L-F6 to L-F8 during v0.0.2; all eight, with their evidence and status,
are in [Logos findings](experiments/logos-findings.md).

| Finding | Subject | Status |
| --- | --- | --- |
| L-F1 | `Mistral3VLMAdapter`'s LoRA scoping does not work on transformers 5 | Open; held by strict xfails. No KLEOS run uses that adapter |
| L-F2 | The tokenizer class depends on `config.json` being present | Served prompts tokenize as in the evaluation: prompt tokens equal on 9 of 9 smoke-test prompts (2026-10-07) |
| L-F3 | Evaluation resume lost a killed session's work on Colab's Drive mount | Fixed (`ed5a987`), confirmed on Colab 2026-10-01 |
| L-F4 | A fresh start into a folder with checkpoints would delete the best one | Open |
| L-F5 | `validate_checkpoint` does not check optimizer or scheduler state | Open |
| L-F6 | `device_map: auto` did not balance the layers across two T4s | Open |
| L-F7 | `finish_reason` misses an answer cut by the budget after a closed trace | Resolved in serving |
| L-F8 | The evaluation reports no progress while it runs | Open |
