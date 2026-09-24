# KLEOS Logos v0.0.1

**Status (2026-09-24): selected, implemented, pre-registered. NOT TRAINED.**

Logos is the deeper of the two KLEOS models. Hermes (Mistral-Nemo 12B) is frozen and
deployed; nothing on this page changes it. This page records which base Logos uses
and why, what was verified, whether it fits the free hardware, how the experiment
is designed, and the exact Colab procedure. The hypotheses themselves are
pre-registered as **H8** in [experiments.md](experiments.md#h8--does-a-stronger-base-make-a-better-kleos-model).

Labels used below: **VERIFIED** means checked in this repository against the
pinned artifacts. **ESTIMATED** means computed, not measured. **SOURCE** means
quoted from a published source, linked.

---

## 1. Decision

**Base: `mistralai/Ministral-3-14B-Instruct-2512-BF16` @ `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`, text tower only.**

| | |
| --- | --- |
| Licence | Apache-2.0, ungated. No token needed to download |
| Released | 2025-12-02 ([card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16), [announcement](https://mistral.ai/news/mistral-3)) |
| Checkpoint | `Mistral3ForConditionalGeneration` (`mistral3`) with a `ministral3` text tower, BF16, 6 shards |
| Parameters | 13,506,073,600 text + 438,958,080 vision (tower 403,305,472, projector 35,652,608). **VERIFIED** on the meta device |
| Text tower | 40 layers, hidden 5120, MLP 16384, 32 query / 8 KV heads, head_dim 128, vocab 131,072, untied `lm_head`, YaRN to 262,144 tokens. **VERIFIED** from `config.json` |
| LoRA r=16, 7 projections | 280 modules, **60,948,480** trainable parameters (**VERIFIED** arithmetic; the same formula gives Hermes' measured 57,016,320) |
| Config | [`configs/models/ministral3_14b.yaml`](../configs/models/ministral3_14b.yaml), [`configs/training/kleos_logos_v001.yaml`](../configs/training/kleos_logos_v001.yaml) |

**Why this model.**

- **It is the largest Mistral open-weight model that trains on a free Colab T4.**
  Every stronger candidate needs more memory than a 16 GB T4 has, even in 4-bit
  (section 2).
- **It is much more capable than Hermes' base.** MMLU (5-shot, base models) 79.4
  against Nemo's 68.0; Mistral Small 24B scores 81.0. It is distilled from Mistral
  Small 3.1 (SOURCE: [Ministral 3 paper, Table 3](https://arxiv.org/html/2601.08584);
  [Nemo card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)).
  Mistral's own numbers come from different harnesses and years, so treat them as
  indicative.
- **It keeps Hermes' architecture.** Same 40 × 5120 geometry, same attention, same
  vocabulary; only the MLP is wider (16384 against 14336). The training pipeline,
  the deployment path and the comparison with Hermes all carry over.
- **It serves on free ZeroGPU the way Hermes does** (section 7).

**The -BF16 repository, not the default one.** `mistralai/Ministral-3-14B-Instruct-2512`
is published in FP8, which needs compute capability ≥ 9 (SOURCE:
[transformers FP8 docs](https://huggingface.co/docs/transformers/quantization/finegrained_fp8));
a T4 is 7.5. The loader refuses a pre-quantized container.

**Text-only view.** KLEOS is text-only, and the 0.44B-parameter vision tower would
cost memory a T4 does not have. `model_type: ministral3` makes
`Ministral3TextAdapter` load only the text tower, as `Ministral3ForCausalLM`, from
the official checkpoint with its keys renamed (`language_model.model.*` →
`model.*`, `language_model.lm_head.*` → `lm_head.*`). The load then refuses to
continue if any text weight is missing or mismatched, or if any key outside the
vision tower and projector goes unused.

**Honest caveat.** Logos is only modestly *larger* than Hermes: 13.5B text
parameters against 12.2B. What it adds is base capability. On v0.0.6 that capability
can only show where the data allows it (section 6).

---

## 2. Candidates

Survey of all 75 `mistralai` repositories on Hugging Face, 2026-09-24. Parameter
counts, gating, files and revisions come from the Hub API
(`https://huggingface.co/api/models/<repo>`), release dates from each repository's
commit history.

| Model | Size | Licence | QLoRA on a free T4 | Verdict | Sources |
| --- | --- | --- | --- | --- | --- |
| **Ministral-3-14B-Instruct-2512-BF16** | 13.9B (13.5B text + 0.44B vision) | Apache-2.0 | **Yes, marginal** (section 4) | **Selected** | [card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16), [docs](https://docs.mistral.ai/models/ministral-3-14b-25-12) |
| Mistral-Small-3.1-24B-Instruct-2503 | 24.0B | Apache-2.0 | No | Strongest candidate; needs Kaggle's free 2×T4 ([Unsloth](https://unsloth.ai/docs/models/tutorials/magistral-how-to-run-and-fine-tune): a 24B model "slightly exceeds the memory limits of a 16GB VRAM") | [card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| Mistral-Small-3.2-24B-Instruct-2506 | 24.0B | Apache-2.0 | No | Also ships no Hugging Face tokenizer or chat template (`tekken.json` only) | [card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| Mistral-Small-24B-Instruct-2501 | 23.6B, text-only | Apache-2.0 | No | Older; 32k context; also needs Kaggle | [card](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| Magistral-Small-2509 | 24.0B, reasoning | Apache-2.0 | No | No HF tokenizer; KLEOS data has no reasoning traces | [card](https://huggingface.co/mistralai/Magistral-Small-2509) |
| Mistral Medium 3 / 3.1 | — | API only | — | No open weights | [announcement](https://mistral.ai/news/mistral-medium-3) |
| Mistral-Medium-3.5-128B | 127.7B | Modified MIT (revenue cap) | No | Too large; restrictive licence | [licence](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE) |
| Mistral-Small-4-119B, Mistral-Large-3-675B, Devstral-2-123B | 119B–675B | Apache / modified MIT | No | Far too large for $0 | [Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), [Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) |
| Large 2 / 2.1, Pixtral Large, Small-2409, Codestral-22B | 22B–124B | MRL / MNPL | — | Non-commercial: disqualified | [MRL](https://mistral.ai/licenses/MRL-0.1.md), [MNPL](https://mistral.ai/licences/MNPL-0.1.md) |

**Benchmarks, as each source reports them** (MMLU 5-shot on base models):
Nemo 68.0 ([card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)),
Ministral 3 14B 79.4 and Small 3.1 81.0 ([paper, Table 3](https://arxiv.org/html/2601.08584)),
Small 2501 80.73 ([card](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)).

**Why not the 24B.** It is the stronger base, but it does not fit a free Colab T4,
and the workflow it needs (Kaggle's 2×T4, a second free platform with its own
quota and session rules) was not chosen. That was a deliberate trade, confirmed
by the user on 2026-09-24.

---

## 3. What was verified locally

All against the pinned revision, with the Docker image's pinned transformers 5.16.1.

| Check | Result |
| --- | --- |
| Text-only loading | On a tiny checkpoint in the official key layout, the text view loads **every** language weight exactly; `lm_head` stays untied; vision and projector keys are the only ones unused. **VERIFIED** (`tests/test_ministral3_text_view.py`) |
| Tokenizer files | `tokenizer.json` `d5f60467…8135`, `tokenizer_config.json` `f59f7294…0d6d`, `chat_template.jinja` `2f545122…8970`. **VERIFIED**, pinned in `tests/test_revision_pinning.py` |
| Chat template | Renders the system prompt in place (`[SYSTEM_PROMPT]…`), so Hermes' system-prompt merge (deviation D5) does not apply. Its default system prompt is used only when a conversation has none; every KLEOS example has one. **VERIFIED** |
| `fix_mistral_regex` | transformers flags this tokenizer's pre-tokenizer regex as incorrect. Setting it to true changes tokenization in **0 of 1,350** v0.0.6 examples. Logos sets it explicitly so training and serving agree on user text. **VERIFIED** |
| Sequence lengths | Longest training example **492 tokens** (496 after the collator pads to a multiple of 8), validation 480, test 503, through the repository's own formatter. Hermes' tokenizer gives 440. `max_seq_length` 1024 truncates nothing. **VERIFIED** |

---

## 4. Does it fit a free T4?

**Short answer: probably, by a hair. Run the smoke test before committing a
session.**

The memory estimator (`src/kleos_models/models/feasibility.py`) was rebuilt for
this phase (finding H-F10). It now counts what a QLoRA step on these models
actually holds: the embeddings and `lm_head` upcast to fp32 by k-bit preparation,
the 16-bit autocast copy of `lm_head`, fp32 LoRA weights and gradients, logits at
the loss, and one layer's activations under gradient checkpointing. Checked against
the peaks measured on this project's own T4 runs:

| Run | Measured peak | Estimated | Error |
| --- | ---: | ---: | ---: |
| Hermes (`kleos-v006-mistralnemo12b-run1`) | 13.09 GiB | 13.09 GiB | −0.03% |
| `kleos-v006-ministral8b-run1` | 9.67 GiB | 9.66 GiB | −0.10% |

For Logos at its longest batch (ESTIMATED):

```text
$ python scripts/plan_run.py --config configs/training/kleos_logos_v001.yaml \
      --simulate-gpu t4-colab --seq-length 496

✓ ministral3_14b (mistralai/Ministral-3-14B-Instruct-2512-BF16): ADAPTER TRAIN (MARGINAL)
  base weights               10.85 GB (of which unquantized 5.00 GB)
  LoRA weights               0.227 GB (60,948,480 trainable params)
  gradients                  0.227 GB
  optimizer state            0.114 GB (paged, outside the allocator)
  activations and logits      2.59 GB (batch 1 x seq 496)
  ----------------------------------------
  estimated peak allocated   13.89 GB
  + allocator reserve         0.50 GB
  required                   14.39 GB
  budget:   14.41 GB
  headroom: +0.03 GB
```

The budget is the 14.56 GiB torch reports on Colab's T4, less the 0.14 GiB Hermes'
run used outside PyTorch's allocator. The 0.5 GiB reserve is the fragmentation
slack Hermes' run showed (13.58 GiB reserved at a 13.09 GiB peak).

**What that means.** Logos' peak is 0.80 GiB above Hermes': 0.60 GiB of extra 4-bit
MLP weights, 0.03 GiB of extra LoRA state, and longer sequences under its tokenizer.
It fits only if the allocator fragments no more than it did for Hermes. Two
things follow:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set in the runbook.
   It changes how memory is allocated, not what is computed. It is recorded in the
   manifest's environment block and declared under H8.
2. **The smoke run decides, not this estimate.** Before its first step, the
   trainer runs two forward/backward passes on the *longest* batch and records the
   measured peak (`manifest.metrics.memory_probe`). A ten-step smoke run would
   otherwise report whatever ten random batches happened to need.

### Go / no-go gate

After the smoke run (runbook step 8):

```bash
python scripts/check_smoke_gate.py --manifest <run>/manifest.json \
    --expect-modules 280 --expect-trainable 60948480 --min-spare-gb 0.15
```

**GO** requires all of: the run completed with no adjustment; the text view loaded
with 0 missing and 0 mismatched weights; exactly 280 LoRA modules and 60,948,480
trainable parameters; gradients reached the adapter; and at least 0.15 GiB spare at
the longest batch once the optimizer state exists. The 0.15 GiB is a judgement,
not a measurement: room for validation passes and checkpoint writes, which Hermes
survived with 0.84 GiB free.

**NO-GO**, in order (no hyperparameter changes silently, ever):

1. Stop and report the probe's numbers. Do not start the full run.
2. Candidate fix, **not implemented**: keep `lm_head` in 16-bit instead of letting
   k-bit preparation upcast it to fp32. Under fp16 autocast its matmul runs in
   16-bit either way, so the arithmetic should be identical, and it saves about
   2.5 GiB. It would be a code change with a bit-identity test and a declared
   deviation from Hermes' recipe.
3. Kaggle's free 2×T4 or P100 16 GB. Unverified for this stack.

---

## 5. Experiment design

Pre-registered in full as **H8** in [experiments.md](experiments.md#h8--does-a-stronger-base-make-a-better-kleos-model),
before any training. In short:

- **Same everything except the base.** The same sealed dataset, the same recipe
  value for value, and the same benchmark (sha256 `a11ffad7…`), grader, greedy
  decoding and seed as Hermes. The declared differences are in section 8.
- **H8a:** Logos `arm1_base_orchestrated` vs Logos `arm2_finetuned`: does
  fine-tuning help this base, as it helped Hermes'?
- **H8b (primary):** Hermes `arm2_finetuned` vs Logos `arm2_finetuned` on the
  **271 answerable** examples (61 groups), where the data leaves headroom. Paired
  cluster bootstrap by `group_id`, 2,000 iterations; the verdict is better, worse,
  equivalent (95% CI within ±0.02) or inconclusive.
- **Why the answerable subset.** The 78 should-decline cases carry four decline
  labels that never occur in training, so no model trained on v0.0.6 can learn
  them (deviation D3). They are reported, but they cannot tell two bases apart.

---

## 6. Dataset plan

**Logos v0.0.1 trains on the sealed `kleos-policy-v0.0.6`, unchanged** (user
decision, 2026-09-24). That isolates the base model: any difference from Hermes is
the base, not the data. The price is that v0.0.6's limits cap Logos exactly as they
cap Hermes:

- The four decline labels are absent from training, so judgment is capped at 0.9255
  and `deciding_factor` at 0.7765 on the full benchmark.
- Abstention in training is conditional on the scenario family, not on the
  evidence, which is the shortcut both Hermes and Ministral-8B learned.
- The test split is 100% JSON-formatted inputs, a format no training example uses.

The repair belongs upstream, in the private `kleos-training-data` repository:
[datasets/kleos-policy-v0.0.7-repair-spec.md](datasets/kleos-policy-v0.0.7-repair-spec.md)
specifies it, and keeps `test.jsonl` byte-identical so v0.0.7 results stay
comparable with every v0.0.6 number. Logos v0.0.2 would train on it.

---

## 7. Deployment (later: only after Logos is validated)

Nothing below is built yet. The model must first pass H8, and nothing spends
ZeroGPU quota during development.

- **Where:** a private ZeroGPU Space, like Hermes'. ESTIMATED VRAM about 9 GiB
  in NF4: Hermes measured 8.34 GiB, and Logos adds 0.60 GiB of 4-bit weights. A
  48 GB `large` slice holds it easily.
- **Speed and quota:** ESTIMATED slightly slower than Hermes (about 11% more
  linear weights), so a few fewer answers from each account's 5 free GPU-minutes a
  day.
- **What is Hermes-specific and must be generalised first:**
  - `serving/space.py` (`BASE_PRELOAD_FILES`, `RECORD_NAME`) and `deploy/zerogpu-space`
  - `build_deployment_package` defaults and its hard-coded research report
  - `serving/startup.py` (`HERMES_*` settings, `MIN_VRAM_GIB`) and `serving/status.py` messages
  - the Docker files
  - publishing's model-card snippet, which loads with `AutoModelForCausalLM`; a
    text-view adapter needs the key mapping
- **Then:** a frozen Logos deployment record and package, a smoke suite reproduced
  byte for byte, and KLEOS wiring. The router that picks Hermes (fast) or Logos
  (deep) is later still, and only if Logos proves worth it.

---

## 8. Risks and declared differences from Hermes

| Risk | Handling |
| --- | --- |
| T4 memory (section 4) | Memory probe on the longest batch; go/no-go gate; `strict_config: true` so nothing shrinks silently |
| fp16 on a T4 for a model released in BF16 | Hermes had 4 fp16-skipped optimizer steps. The count is recorded; many skipped steps would be reported, not hidden |
| transformers ≥ 5 required (`ministral3` first appears in 5.0.0) | Runbook pins transformers 5.16.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.2 and tokenizers 0.23.2, Hermes' versions. Colab's torch is not reinstalled, so it may differ from Hermes' 2.11.0 |
| One seed per model | Stated as a limitation of H8 |
| Dataset ceiling | Primary comparison on the answerable subset |

**Declared differences from Hermes' run**, none of them in the training arithmetic:

1. Base model, tokenizer and chat template: the object of study.
2. `fix_mistral_regex: true` (0 of 1,350 examples tokenized differently).
3. Evaluation and checkpoint cadence 25 steps, not 50: finer best-checkpoint
   selection around Hermes' step-200 minimum.
4. `strict_config: true`.
5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: allocation only.
6. Pipeline fixes made in this phase: best checkpoint protected (H-F9), gradient
   check at the configured batch size, and the memory probe. Neither the probe nor
   the gradient check changes training: both run before the Trainer, which re-seeds
   when it is constructed.

---

## 9. Colab runbook

Prerequisite: the code from this phase is committed and **pushed** (Colab clones it
from GitHub). Use a **T4 GPU** runtime. Run one cell at a time; if a cell errors,
stop there.

Drive needs about 3 GB free for Logos' checkpoints and results. Model weights go
to `/content`, never Drive: the checkpoint is about 28 GB.

**1. Mount Drive**
```python
from google.colab import drive

drive.mount("/content/drive")
```
**2. Clone**
```
!rm -rf /content/kleos-models && git clone https://github.com/TejasNaik24/Kleos-Models.git /content/kleos-models
```
**3. Enter it**
```
%cd /content/kleos-models
```
**4. Install, without touching Colab's torch**
```
!python scripts/colab_setup.py
```
**5. Pin Hermes' library versions**
```
!pip install -q --no-deps transformers==5.16.1 peft==0.20.0 accelerate==1.14.0 bitsandbytes==0.50.2 tokenizers==0.23.2
```
**6. Environment.** Weights cached on `/content`; allocation tuned; no `KLEOS_*` path
variables, because they change `config_hash`.
```
%env HF_HOME=/content/hf_cache
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
```
!python -c "import os, torch, transformers, peft, bitsandbytes; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__, bitsandbytes.__version__); print([k for k in os.environ if k.startswith('KLEOS_')] or 'no KLEOS_ variables')"
```
Expect `Tesla T4`, `5.16.1`, `0.20.0`, `0.50.2` and `no KLEOS_ variables`.

**7. Plan against the live GPU**
```
!python scripts/plan_run.py --config configs/training/kleos_logos_v001.yaml --seq-length 496
```

**8. Smoke run** (10 steps at the real settings, scratch space on `/content`)
```
!python scripts/train.py --config configs/training/debug_logos.yaml --dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output-dir /content/logos-smoke --experiment-id kleos-logos-smoke-001
```
**9. Gate** (section 4). Stop here on NO-GO and report the output.
```
!python scripts/check_smoke_gate.py --manifest /content/logos-smoke/kleos-logos-smoke-001/manifest.json --expect-modules 280 --expect-trainable 60948480 --min-spare-gb 0.15
```
**10. Resume drill.** Remove the last checkpoint, then resume: it must continue from
step 5 and finish.
```
!rm -rf /content/logos-smoke/kleos-logos-smoke-001/checkpoint-10 && python scripts/train.py --config configs/training/debug_logos.yaml --dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output-dir /content/logos-smoke --experiment-id kleos-logos-smoke-001 --resume-from-checkpoint auto
```

**11. Train Logos** (about 3.5 hours on a T4, ESTIMATED from Hermes' 3.2)
```
!python scripts/train.py --config configs/training/kleos_logos_v001.yaml --dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output-dir /content/drive/MyDrive/kleos-private/outputs --experiment-id kleos-v006-ministral314b-run1
```
The config hash it prints must be `18008c6716a58afc` (pre-registered under H8).

**12. If the runtime died:** repeat cells 1–6, then:
```
!python scripts/train.py --config configs/training/kleos_logos_v001.yaml --dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output-dir /content/drive/MyDrive/kleos-private/outputs --experiment-id kleos-v006-ministral314b-run1 --resume-from-checkpoint auto
```

**13. Build and check the benchmark** (the sha256 must start `a11ffad75f5147f9`)
```
!python scripts/build_benchmark.py --dataset /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6 --output /content/benchmark && sha256sum /content/benchmark/benchmark.jsonl
```

**14. Evaluate Logos arm1** (base, orchestrated; 2–3 hours; re-run the same cell to
resume after a disconnect)
```
!python scripts/evaluate.py --config configs/training/kleos_logos_v001.yaml --arm arm1_base_orchestrated --benchmark /content/benchmark/benchmark.jsonl --output /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/arm1_base_orchestrated.json --resume
```
**15. Evaluate Logos arm2** (fine-tuned)
```
!python scripts/evaluate.py --config configs/training/kleos_logos_v001.yaml --arm arm2_finetuned --adapter /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/adapter --benchmark /content/benchmark/benchmark.jsonl --output /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --resume
```

**16. H8a: Logos arm1 vs arm2**
```
!python scripts/compare.py --base /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/arm1_base_orchestrated.json --finetuned /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --report /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/report_logos_v001 --experiment-id kleos-v006-ministral314b-run1 --dataset-version kleos-policy-v0.0.6
```

**17. Re-report Hermes with the corrected measures** (CPU work; Hermes' files are
read, never written)
```
!python scripts/rescore.py --mode annotate --results /content/drive/MyDrive/kleos-private/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.json --benchmark /content/benchmark/benchmark.jsonl --output /content/drive/MyDrive/kleos-private/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json --gold-targets /content/drive/MyDrive/kleos-private/kleos-policy-v0.0.6/test.jsonl --summary /content/drive/MyDrive/kleos-private/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.md
```
(Repeat with `arm1_base_orchestrated` for the baseline.)

**18. H8b, the primary comparison**
```
!python scripts/compare.py --cross-model --base /content/drive/MyDrive/kleos-private/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json --finetuned /content/drive/MyDrive/kleos-private/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --primary-subset answerable --equivalence-margin 0.02 --report /content/drive/MyDrive/kleos-private/outputs/report_h8b_hermes_vs_logos
```

---

## 10. Findings from this phase

**L-F1 — `Mistral3VLMAdapter`'s LoRA scoping does not work on transformers 5.**
VERIFIED with transformers 5.16.1 and peft 0.20.0 on a tiny
`Mistral3ForConditionalGeneration`. Two separate misses:

- Target validation keeps only modules whose names *start with* `language_model`.
  transformers 5 names them `model.language_model.*`, so nothing matches and
  attaching LoRA raises (loudly, at least).
- The vision tower is kept out of LoRA by listing `"vision_tower"` in PEFT's
  `exclude_modules`. PEFT matches list entries by exact name or `.suffix`, which no
  projection inside the tower has. With the adapter's own LoRA config, 7 of 14 LoRA
  modules landed in the vision tower.

Not fixed: the adapter serves only `configs/models/mistral_small_3_2.yaml`, which
no KLEOS run uses, and Logos does not touch it (its text view has no vision
modules). `tests/test_vlm_targeting_finding.py` holds both as strict xfails, so a
fix makes them fail until this entry is closed. A fix would match on a
`language_model.` path segment rather than a prefix, and pass PEFT a regex (or
explicit module names) instead of suffix lists.

The other findings of this phase, H-F11 to H-F14, are about how the evaluation
measures. They are recorded with the run they were found in:
[experiments/kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings).
