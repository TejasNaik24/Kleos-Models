# Serving verification records

This is a frozen research record, written while the work was done and preserved
as written. Editorial changes for public release are limited to replacing private
storage paths and account identifiers with placeholders (marked as editor's
notes) and retargeting links to documents that moved. No number, table, finding
or decision was changed. Conventions:
[docs/experiments/README.md](README.md).

## Verification record — Hermes v0.0.6

| Check | Result — 2026-09-23, Colab, Tesla T4 |
| --- | --- |
| Package integrity | Verified. Adapter `dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32` — the frozen checkpoint-200 weights |
| `adapter_config.json` | Research copy `revision: null`; package copy pinned to `04d8a905…` |
| Base revision | `mistralai/Mistral-Nemo-Instruct-2407@04d8a90549d23fc6bd7f642064003592df51e9b3`, confirmed with the Hub |
| Tokenizer | Loaded from the frozen package with `fix_mistral_regex=False` stated |
| Smoke suite | **9/9 responses identical to the frozen v0.0.6 evaluation**, byte for byte: all seven task families plus two should-decline cases |

That reproduction held across a different Colab session, a different code path
(the deployment loader rather than `scripts/evaluate.py`), and a tokenizer loaded
from frozen files rather than resolved from the Hub.

Two lines missing from the log corroborate the tokenizer contract independently:

- **No regex warning.** transformers warns only when `fix_mistral_regex` is left
  unset. Stating it removes the warning; the identical outputs show it did not
  change tokenization.
- **No `Tokenizer had no pad token` message.** Every training and evaluation load
  logged it, because the Hub's tokenizer has no pad token. The packaged
  `tokenizer_config.json` already records `</s>`, so its absence shows the
  tokenizer came from the package.

This is a reproducibility result, **not a score**. Nine examples say nothing about
quality that the 349-example benchmark does not already say better.

### Container

**Status, 2026-09-23: built, not GPU-tested.** The image was built for
`linux/amd64` (6.31 GB) and exercised without a GPU. It was rebuilt the same day
with the package schema v2 and ZeroGPU changes, and the preflight re-run:

| Check | Result |
| --- | --- |
| `docker build --check` | No warnings |
| Build context | Exactly the 57 allowlisted files; `.env`, `data/`, `outputs/`, `.git` excluded |
| Installed versions | torch 2.11.0+cu128, transformers 5.16.1, tokenizers 0.23.2, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.2 — as pinned |
| Model stack imports | torch (CUDA 12.8 build), transformers, peft, bitsandbytes and the serving loader all import |
| Runs as | uid 10001, non-root |
| Foreign package mounted | Refused, exit 1: adapter and all three tokenizer hashes differ from the baked-in record |
| Matching package mounted | Preflight passed, exit 0 |
| No GPU visible | Stopped with the `--gpus all` fix, exit 1, before any download |
| No API key | Stopped, exit 1 |
| Named cache volume | Detected as persistent; writable by uid 10001 |
| Healthcheck, nothing listening | Unhealthy, exit 1 |
| Secret values in output | Never printed |
| `docker compose config` | Valid |
| Schema v2 package, self-consistent but built with `compute_dtype: bfloat16` | Refused, exit 1: `runtime.compute_dtype: package has 'bfloat16', expected 'float16'` |
| Serving and ZeroGPU test suites inside the image, CPU | 257 passed, including the backend split (`prepare` → `generate_ids` → `finish`) generating token-for-token what the previous `generate` did, on a tiny Mistral with torch 2.11.0+cu128 and transformers 5.16.1 |

**Not verified:** loading the base model, attaching the adapter, or serving a
request inside this image on a GPU. The first GPU start closes that gap — see
[First GPU start](../deployment.md#first-gpu-start-re-establish-the-99-match).

## Free hosting on Hugging Face ZeroGPU

**Status, 2026-09-23: deployed and verified.** The private Space
`<owner>/<space>` [editor's note: account identifier replaced] serves the frozen artifact from the private package
repository `<owner>/<package-repo>` [editor's note: account identifier replaced] (commit `5d179599…`). It reproduces
the frozen evaluation's responses **9/9, byte for byte**, on an RTX PRO 6000
Blackwell, at $0. See the [verification record](#verification-record--zerogpu)
for what is verified, what is estimated, and what is not tested.

One non-model pin differs: `pydantic 2.12.5`, not 2.13.5. Hugging Face installs
`gradio[oauth,mcp]==6.28.0` next to the Space's requirements, and its `mcp` extra
caps pydantic at 2.12.5. The first build, on 2026-09-23, failed on exactly that
conflict. pydantic only validates configs and requests, and the full test suite
passes on 2.12.5 under Python 3.12. To check a requirements change before pushing
it, dry-run the install in the platform's base image (`python:3.12.12`), adding
`gradio[oauth,mcp]==6.28.0 spaces==0.51.3 "torch<=2.13.0"` as the build log shows.

### Free-tier facts

Verified 2026-09-23 against the Hugging Face ZeroGPU documentation and the
source of `spaces` 0.51.3:

| | |
| --- | --- |
| GPU | Half an NVIDIA RTX Pro 6000 Blackwell (`large`, the default): 48 GB, compute capability 12.0 |
| Daily quota | Free account **5 minutes** of GPU time; unauthenticated caller 2 minutes |
| Who pays | The **calling** account, identified by the token the caller sends, not the Space owner |
| Reset | Exactly 24 hours after that account's first GPU use |
| Admission | A call is admitted only if the remaining quota covers its **requested** duration ×1.5; the time actually used is what is charged |
| Beyond the quota | Only PRO, Team and Enterprise accounts can buy more. A free account cannot be charged: it is refused |
| Queue | A call waits up to 60 s for a GPU, then fails "No GPU was available". Less remaining quota means lower queue priority |
| Hosting | A free account in good standing (verified email, older than 30 days) may host up to 2 ZeroGPU Spaces |
| Loading | Models load at startup on the CPU under "CUDA emulation", then move to a real GPU for each call. bitsandbytes ≥ 0.46 needs no ZeroGPU patch |
| Storage | No persistent storage; the package is re-downloaded at each start |

### What that means for KLEOS

The figures below are **observed ZeroGPU smoke-test measurements** from 10
calls on 2026-09-23: the nine frozen prompts and one repeat. Answers were 69–128
tokens long. Ten calls describe this deployment, not the platform in general.

| | Observed |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 2g.48gb slice (sm 12.0), CUDA 12.8 |
| Peak VRAM | 8.34 GiB of 48 |
| Generation speed | ~12 tokens/s, cold and warm alike (NF4, float16 compute, batch 1) |
| Warm call | median 7.9 s end to end; 0.15 s to get the GPU |
| Cold call (new GPU worker) | median 12.9 s end to end; 4.1 s to get the GPU and move the weights onto it |
| Slowest call | 34 s end to end, most of it waiting for a GPU |
| Cold vs warm | 7 of 10 calls got a new worker, even seconds apart. A cold call costs only about 4 s more |
| GPU time for the run | 73–131 s of the 300-second daily quota: 7–13 s per call |
| Space start | Package download 2 s, base load and NF4 quantization 41 s, packing 8.54 GB for the GPU workers 3 s, plus container start (not timed) |

### Verification record — ZeroGPU

| | Status |
| --- | --- |
| Serving path splits CPU and GPU work without changing generation | **VERIFIED** locally: in the Docker image on CPU, the split generates token-for-token what the previous `generate` did (tiny Mistral, torch 2.11.0+cu128, transformers 5.16.1) |
| Runtime contract pins float16; a bfloat16 package is refused | **VERIFIED**: unit tests and the container preflight |
| Auth, validation, bounds, error → status mapping, no content in logs | **VERIFIED** by unit tests with a fake GPU call; ZeroGPU errors use the exact `spaces` 0.51.3 messages |
| Reference client never raises; sleeping Space → `starting` | **VERIFIED** by unit tests with fake transports |
| Space files: pins equal Docker's, correct preload, no secrets or data | **VERIFIED** by static tests |
| Package upload refuses unlisted files and checks remote bytes | **VERIFIED** by unit tests, and on the Hub on 2026-09-23: 9 files, every size and hash checked remotely |
| Free-tier facts above | **VERIFIED** against the HF documentation and `spaces` source, 2026-09-23 |
| Latency, throughput, VRAM | **Observed ZeroGPU smoke-test measurements** (10 calls; table above) |
| Daily capacity | **ESTIMATED** from the observed GPU time per call |
| Peak CPU RAM at startup | **Not recorded.** The first deployment did not log the startup report; see the next build's `Hermes ready:` line |
| The Space builds with these requirements | **VERIFIED** 2026-09-23, second attempt. The first failed on the platform's pydantic cap (see above) |
| Startup fits the host's CPU RAM | **VERIFIED**: the 12B base loads and quantizes to NF4 on the CPU (363 weights in 41 s) and the Space starts |
| NF4 quantization under CUDA emulation | **VERIFIED** on the Space |
| Attaching the adapter under emulation | **VERIFIED** on the Space, after a fix. The first start failed when PEFT read the adapter file straight onto the emulated GPU ("No CUDA GPUs are available"). The Space path now reads it on the CPU, a fix proven first under `spaces` 0.51.3's emulation with a tiny stand-in model |
| Generation on Blackwell | **VERIFIED** |
| 9/9 exact match on ZeroGPU | **VERIFIED** 2026-09-23: 9/9 byte-identical to the frozen evaluation, 9/9 prompt token counts equal, 9/9 decisions agree, and a repeated call identical |
| `X-Hermes-Key` reaches the app through Hugging Face's proxy | **VERIFIED**: every smoke-test call authenticated with it |
| A free account may run a private ZeroGPU Space | **VERIFIED**: `<owner>/<space>` [editor's note: account identifier replaced] |

## Verification record — Logos

**Deployed 2026-10-07; smoke test complete 2026-10-07: 8 of 9 identical.**

| | |
| --- | --- |
| Space | `<owner>/<space>` [editor's note: account identifier replaced] (private, ZeroGPU), staged from kleos-models `c0db1e0f1f08` |
| Package | `<owner>/<package-repo>` [editor's note: account identifier replaced] (private) at `8b48069ad46f`, 9 files verified on the Hub |
| Adapter | `e49724f6554db662…`, from `checkpoint-175` (eval_loss 0.0293) |
| Startup | `Logos ready`: `download_s=1.9 load_s=57.1`, host `process_peak_gib` 30.6 of 2,000; the 4-bit model packed to 9.20 GB for the GPU. No out-of-memory: quantizing the 14B text tower on the Space's CPU works |
| Runtime | torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, gradio 6.28.0, spaces 0.51.3; NF4, double quant, float16 compute, SDPA |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb (sm 12.0, CUDA 12.8); peak 8.86 GiB |
| Benchmark | rebuilt from v0.0.7, sha256 `a11ffad7…b266` |
| Reference | Logos v0.0.2 `arm2_finetuned.json`, sha256 `0dd74661…0eb5` |

**Day 1: 7 of the 9 prompts ran** before the quota refused the 8th.

- **6 of 7 reproduced exactly:** answer and thinking trace, byte for byte.
- **Prompt tokens equal the evaluation's on all 7.**
- **1 of 7 differs, `kx-mcb-066bff4fa33aa29b`** (a briefing, answerable).
  - The trace is identical for 568 of its 761 characters (75%). Then one
    sentence goes another way: "Startup has…" where the evaluation wrote
    "Startup is the active workspace…".
  - From there the reasoning changes, and so does the answer. The evaluation
    ranked the three items in the active workspace (deciding factor `scope`,
    scored 1.0). The Space's answer declines and asks before looking in another
    workspace (`ask_before_crossing`).
- **Diagnosis: GPU arithmetic, not setup** (the three checks in
  [If the outputs differ](../deployment.md#diagnosing-a-reproduction-mismatch)):
  1. the prompt tokens are equal;
  2. the text diverges late, after an identical 75% of the trace;
  3. the decision does **not** agree on this item.

  A different GPU rounds fp16 slightly differently. When two next tokens are
  nearly tied, that can pick the other one, and a thinking model has a few
  hundred tokens of trace in which it can happen before the answer. Hermes,
  which does not think, matched 9/9.
- **Timings:** warm median 19.7 s per call (19.4 s generating, 11.0 tokens/s);
  cold median 29.9 s (5.5 s to acquire a GPU, 10.9 tokens/s). 161–174 GPU
  seconds for the 7 answers.

**Day 2 (evening of 2026-10-07): the remaining 2 prompts.**

- **2 of 2 reproduced exactly** (`kx-trt-05103645026eb390`,
  `kx-wsr-0761ffe45f4fe530`), answer and trace; prompt tokens equal; decisions
  agree.
- **Timings:** cold 25.2 s (2.1 s to acquire, 10.2 tokens/s, 236 tokens); warm
  33.6 s (12.1 tokens/s, a 406-token answer). 56.5–58.8 GPU seconds for the 2.
  Peak 8.86 GiB.
- The reference was re-downloaded from the evaluation notebook's output after
  the browser-saved copy was lost; its sha256 is the same `0dd74661…0eb5`.

**All 9:** 8 identical, answer and trace. Prompt tokens equal on 9 of 9;
decisions agree on 8 of 9. 217.8–232.3 GPU seconds for the 9 (24–26 per
answer).

**Decision (owner, 2026-10-07) [editor's note: the project author]: accept and document.** Logos is served on the
Space as a Beta, numerically different from the T4 evaluation: 8 of 9 checked
answers identical, 1 diverged late in its thinking and changed its decision.

- **What H9 measured** is the T4 outputs. The Space's outputs are not
  re-measured: the full 349 would take about a month of the free quota.
- **Serving on T4-class hardware**, to match exactly, needs a paid GPU host. It
  is declined for now.
- **Precision and decoding stay as evaluated.** Changing them to force a match
  would change the model.
- **Claims about Logos** say "measured better than Hermes on the benchmark",
  not that the live Space reproduces that score.
