# Deploying a KLEOS model

This repository trains and evaluates. It does not host. What it produces for a
host is a **deployment package**: a verifiable directory that says exactly which
base weights, tokenizer, adapter and decoding settings make up one model, and
refuses to load if any of them has moved.

---

## Research artifact vs deployment artifact

These are two different things and the distinction is load-bearing.

| | Research artifact | Deployment artifact |
| --- | --- | --- |
| What it is | What the run produced | A copy, prepared for serving |
| Where | `outputs/<experiment-id>/` on private storage | A package built from it |
| Purpose | Evidence | Operation |
| Mutable | **Never** | Rebuilt whenever serving needs change |
| `adapter_config.json` `revision` | `null` — finding H-F1 | The pinned base commit |
| Tokenizer | Whatever the run loaded | The frozen files, with the flag stated |
| Identity record | `manifest.json` (the run) | `manifest.json` (the package) |

The research artifact is **not edited to look correct in hindsight**. Hermes
v0.0.6 was trained with PEFT writing `revision: null`, and that file still says
`null`. The deployment package carries the pin instead, and its manifest records
both the pin and the fact that the original lacked it. Anyone comparing the two
can see exactly what was added and when.

The evidence for Hermes v0.0.6 is in
[experiments/kleos-v006-mistralnemo12b-run1-report.md](experiments/kleos-v006-mistralnemo12b-run1-report.md).
Nothing in this document changes a number in it.

---

## Why pinning matters more than it looks

A LoRA adapter is not a model. It is a set of deltas against *specific* base
weights. Attach it to a different revision of the same checkpoint and:

- no exception is raised,
- the shapes still match,
- the model still answers,
- the answers are quietly worse.

There is no test at inference time that catches this. The only defence is to
record the exact commit and refuse to serve without it, which is what
`DeploymentManifest` and `verify_base_revision` do.

The same argument applies to the tokenizer, for the same reason: different
tokenization produces a different prompt, and a different prompt produces a
different answer, with nothing raising an error.

---

## The tokenizer contract

Every load of Mistral-Nemo's tokenizer under transformers ≥ 5 prints:

> The tokenizer you are loading from 'mistralai/Mistral-Nemo-Instruct-2407' with
> an incorrect regex pattern… You should set the `fix_mistral_regex=True` flag.

**What the flag actually does.** transformers ≥ 5 can replace the pre-tokenizer's
`Split` regex with the one `mistral-common` uses. The two disagree on how some
sequences split — the documented example is `'The'` becoming
`["'", "T", "he", "'"]` under the old pattern and `["'", "The", "'"]` under the
new one — affecting roughly 1% of tokens. The flag defaults to **`False`**.

**What v0.0.6 used.** `load_tokenizer` never passed the flag, so training and all
four evaluation loads used the default, `False`. Because every arm used the same
tokenizer, the comparison in the research report is internally valid.

**What serving does.** Two things, belt and braces:

1. **The tokenizer files are packaged.** `tokenizer/` in the deployment package
   holds the exact `tokenizer.json`, `tokenizer_config.json` and
   `chat_template.jinja` the run saved, hashed in the manifest. The loader points
   at that directory, not at the Hub.
2. **The flag is stated, not inherited.** The manifest records
   `fix_mistral_regex: false` and the loader passes it explicitly, so a change to
   the library default cannot move tokenization under a frozen adapter.

`mistral_regex_kwarg` handles versions that predate the flag: there is nothing to
patch there, so the behaviour is already the unpatched one and nothing is passed.
Asking for `True` on such a version raises rather than silently giving you
something else.

> **This differs from the Ministral deployment record on purpose.** That one
> takes the tokenizer from the base repository, which is the right default when
> no library flag can change it. Hermes packages it because one can.

**If you ever want the corrected regex**, that is a retrain, not a config change.
Serving with `fix_mistral_regex=True` against weights trained with `False` is
train/serve skew — precisely what the pin exists to prevent.

---

## The runtime contract

How the base model is instantiated is part of what was measured, so it is part
of the identity a server checks (package schema v2):

| Field | Hermes v0.0.6 | Why it is pinned |
| --- | --- | --- |
| `quantization_mode` | `nf4` | 4-bit NF4 is what trained and evaluated |
| `double_quant` | `true` | Changes the quantized weights |
| `compute_dtype` | **`float16`** | See below |
| `attn_implementation` | `sdpa` | Different kernels, different arithmetic |
| `max_seq_length` | `1024` | The trained context |

**Why `compute_dtype` is stated rather than `auto`.** The training config says
`auto`, which resolves to float16 below compute capability 8.0 and to bfloat16
at or above it. v0.0.6 trained and evaluated on a T4 (7.5), so it was measured
in **float16**, which the research manifest records as
`resolved_compute_dtype: float16`. Every newer GPU a server is likely to have,
including ZeroGPU's RTX Pro 6000 (Blackwell, 12.0), would have resolved `auto`
to **bfloat16** and served a model computing differently from the one measured,
with no error. Stating float16 does not change Hermes. It stops the hardware
from changing it.

The contract lives in `configs/deployment/kleos_hermes_v006.yaml` under
`runtime`. The builder applies it to the packaged `deployment/model_config.yaml`,
which is itself hashed in the manifest. The loader refuses a package whose
model config disagrees with the contract, and `verify_identity` refuses one
whose contract differs from the serving record. A version-1 package, built
before the contract existed, is refused: rebuild it.

---

## Building a package

```bash
python scripts/build_deployment_package.py \
    --run /path/to/outputs/kleos-v006-mistralnemo12b-run1 \
    --deployment-config configs/deployment/kleos_hermes_v006.yaml \
    --output /path/to/packages/hermes-v0.0.6
```

The run directory is read-only throughout. The build refuses to continue if the
adapter's sha256 is not the frozen one recorded in the deployment config, so a
package can never be built from weights nobody measured.

```
hermes-v0.0.6/
├── manifest.json            identity + every file's sha256
├── adapter/                 adapter_model.safetensors, adapter_config.json (pinned), README.md
├── tokenizer/               tokenizer.json, tokenizer_config.json, chat_template.jinja
└── deployment/              README.md, model_config.yaml
```

**Base weights are not in the package.** They are ~24 GB and they belong to
Mistral; the package names a repository and a commit, and the loader fetches and
pins that. Never commit a package to git: `.gitignore` covers `outputs/`, and a
package belongs on the serving host or in an artifact store.

## Verifying one

```bash
python scripts/verify_deployment_package.py --package /path/to/hermes-v0.0.6 \
    --expect-deployment-config configs/deployment/kleos_hermes_v006.yaml
```

Re-hashes every recorded file, checks `adapter_config.json` pins the revision the
manifest names, and prints the model's identity and measured limitations. Needs
no GPU, no torch and no network, so it runs in CI and as a pre-start check. Add
`--check-base-revision` to confirm the commit with the Hub as well.

`--expect-deployment-config` is what makes this an identity check rather than a
consistency check. Without it, a package is only compared against its own
manifest, and a *different* adapter packaged just as carefully would pass. With
it, the package must also match the serving record: the frozen adapter hash, the
base revision, the three tokenizer files by hash, the tokenizer flag and the
greedy decoding settings. The container always runs with it.

It exits non-zero and prints `DO NOT SERVE THIS PACKAGE` on any mismatch.

## Proving the artifact actually runs

```bash
python scripts/hermes_smoke.py \
    --package /path/to/hermes-v0.0.6 \
    --benchmark /path/to/benchmark.jsonl \
    --reference /path/to/arm2_finetuned.json
```

Loads the real thing — pinned base, frozen tokenizer, frozen adapter — and runs a
small deterministic suite spanning every task family plus abstention cases. This
produces **no score**; it is too small to mean anything next to the 349-example
benchmark, and reporting a number from it would be misleading.

What it proves is reproducibility: with `--reference`, it compares each response
against what the frozen v0.0.6 evaluation recorded for the same example id.
Decoding is greedy and seeded, so they should match exactly.

A mismatch is a **deployment** problem, never a reason to retrain. Check, in
order: the GPU (fp16 arithmetic is not bit-identical across devices),
quantization settings, tokenizer behaviour, then prompt assembly.

### Verification record — Hermes v0.0.6

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

---

## The inference service

```
KLEOS Web
    ↓
KLEOS FastAPI backend
    ↓ HTTPS
Hermes inference service      ← scripts/serve_hermes.py
    ↓
Mistral-Nemo base @ pinned revision + Hermes LoRA adapter
```

```bash
export HERMES_PACKAGE_DIR=/srv/hermes-v0.0.6
export HERMES_API_KEY=...          # never commit one
python scripts/serve_hermes.py --host 127.0.0.1 --port 8000
```

| Variable | Meaning |
| --- | --- |
| `HERMES_PACKAGE_DIR` | Package to serve. Required. |
| `HERMES_API_KEY` | Shared secret(s), comma-separated for rotation. Required. |
| `HERMES_DEVICE_MAP` | e.g. `cuda:0`. Optional. |
| `HERMES_REQUIRE_REMOTE_REVISION` | `1` to confirm the base commit with the Hub at startup. |
| `HERMES_REQUEST_TIMEOUT_SECONDS` | May only *tighten* the manifest's limit. |
| `HERMES_MAX_INPUT_CHARS` | May only *tighten* the manifest's limit. |

The service refuses to start without `HERMES_API_KEY` unless
`HERMES_ALLOW_UNAUTHENTICATED=1` is set explicitly, which is for a local
experiment on loopback and nothing else.

### API

| | |
| --- | --- |
| `GET /health` | Liveness. No credentials. Reports `ready` and, if not, why. |
| `GET /ready` | Readiness plus the identity of exactly what is loaded. Authenticated. |
| `POST /v1/generate` | Generate one response. Authenticated. |

```jsonc
// POST /v1/generate
{
  "messages": [{"role": "user", "content": "..."}],
  "max_new_tokens": 256,        // optional; may lower the ceiling, never raise it
  "request_id": "..."           // optional; echoed back, used in logs
}
```

```jsonc
{
  "text": "...",
  "finish_reason": "stop",
  "prompt_tokens": 412,
  "completion_tokens": 96,
  "request_id": "...",
  "model": {
    "name": "kleos-hermes", "version": "v0.0.6",
    "adapter_sha256": "dc121fa3…", "base_model": "...", "base_revision": "04d8a905…"
  }
}
```

Every response says which adapter produced it. A caller that logs
`model.adapter_sha256` can tell later exactly which weights answered.

| Status | Meaning |
| --- | --- |
| 401 | Missing or wrong bearer token. No detail about which. |
| 413 | Too many messages, or input over `max_input_chars`. Checked before any GPU work. |
| 422 | Malformed body, unknown role, or an unknown field. |
| 500 | Generation failed. No internal detail is returned. |
| 503 | Model not loaded, or startup verification failed. |
| 504 | Generation exceeded the timeout. |

### What the service is not

Hermes is a **model service**. KLEOS remains the source of truth for users,
workspaces, memories, projects, skills, the career graph, missions,
conversations, documents, jobs, research and notifications.

Enforced here rather than left to convention:

- **No persistence.** Nothing is written between requests, so there is no
  cross-user or cross-workspace state to leak.
- **A narrow request body.** `messages`, `max_new_tokens`, `request_id` — and
  unknown fields are rejected, so a caller cannot quietly start sending workspace
  state or tool definitions.
- **No tools and no fetching.** The only outbound traffic is to the model
  registry, at startup, for the pinned base weights.
- **No prompt or response logging.** Logs carry ids, counts and timings.
- **No API key in logs**, and no key echoed in an error.
- **No published schema.** `/docs`, `/redoc` and `/openapi.json` are off.
- **One generation at a time**, held by a lock inside the worker thread, so a
  timed-out request that is still running cannot overlap the next one.

### Connecting KLEOS

The KLEOS backend lives in a **different repository**, so its provider wiring is
not implemented here. The handoff — environment variables, the request and
response contract, the status vocabulary shared with the ZeroGPU Space, fallback
rules and frontend states — is
[kleos-hermes-integration.md](kleos-hermes-integration.md), with a tested
reference client in `kleos_models.serving.client`.

Send only the messages the model needs. Anything else is data handed to a process
that does not need it.

---

## Container

`docker/hermes.Dockerfile` runs the serving path above on a generic NVIDIA GPU
host.

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
[First GPU start](#first-gpu-start-re-establish-the-99-match).

What goes where:

| | Where it lives | Why |
| --- | --- | --- |
| Code and pinned dependencies | In the image | Reproducible builds |
| Serving record (`kleos_hermes_v006.yaml`) | In the image | The identity the image may serve |
| Deployment package (~245 MB) | Mounted read-only at `/models/hermes-v0.0.6` | Private weights stay out of the build context and the image |
| Base model (~24.5 GB) | Persistent volume at `/cache/huggingface` | Downloaded once, at the pinned revision |
| `HERMES_API_KEY`, `HF_TOKEN` | Supplied at run time | Never in git, the image or the manifest |

### Runtime assumptions

| | |
| --- | --- |
| Architecture | `linux/amd64` (the base image also exists for `linux/arm64`, but only `amd64` has been built) |
| GPU | NVIDIA, **≥ 16 GB VRAM** specified. Evaluation and the 9/9 smoke test ran on a 14.6 GiB T4. Inference peak was **not** measured; estimated ~9–10 GB (4-bit base ~5.4 GB, the unquantized 131k-vocab embeddings and `lm_head` ~2.7 GB, adapter 0.23 GB, CUDA context and cache). Training peaked at 13.09 GB. |
| Compute capability | ≥ 7.5 (T4 and newer). fp16 is used below 8.0, bf16 is not required. |
| Host driver | Must support CUDA 12.8: driver 570 or newer on most hosts. The base image declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, which the NVIDIA runtime checks at container start. |
| Host software | Docker with the **NVIDIA Container Toolkit**. |
| Disk | ~30 GB free for the cache volume. |
| Network | Outbound to the Hugging Face Hub on the first start. None afterwards if `HF_HUB_OFFLINE=1`. |
| Replicas | One process per GPU, one worker per process. Each worker would load its own copy of the model. |

### Build

```bash
docker build --platform linux/amd64 -f docker/hermes.Dockerfile -t kleos-hermes:v0.0.6 .
```

The build context is an **allowlist** (`.dockerignore`): only `pyproject.toml`,
`README.md`, `LICENSE`, `src/`, four scripts, the serving record and the pinned
requirements reach the Docker daemon. `.env`, `data/`, `outputs/`, `.git` and
everything else stay behind. `tests/test_container.py` enforces this.

Dependencies are pinned exactly in `docker/requirements-hermes.txt`. The model
runtime matches the research record: transformers 5.16.1, peft 0.20.0,
accelerate 1.14.0, bitsandbytes 0.50.2, and torch 2.11.0+cu128 from the PyTorch
CUDA 12.8 index. The image writes its complete installed set to
`/app/requirements.lock`.

### Secrets

| Variable | Required | Notes |
| --- | --- | --- |
| `HERMES_API_KEY` or `HERMES_API_KEY_FILE` | **Yes** | The bearer token the KLEOS backend sends. Comma-separate several to rotate. |
| `HF_TOKEN` or `HF_TOKEN_FILE` | No | The base is ungated; a token raises Hub rate limits for the first download. |

Prefer the `_FILE` form. The value then lives in a mounted secret file and never
appears in `docker inspect`. Setting both forms of one secret is refused. The
container runs as uid 10001, so a secret file must be readable by that user.
`docker/hermes.env.example` lists every setting, with secret values left empty
on purpose: an unreplaced placeholder would become a real, guessable key, and an
empty one stops the container instead.

### Run

```bash
# One-time: a persistent cache volume, the package on the host, and the secrets.
docker volume create hermes-hf-cache
install -d -m 0755 /srv/kleos/secrets
openssl rand -hex 32 > /srv/kleos/secrets/hermes_api_key   # also give it to the KLEOS backend
chmod 0444 /srv/kleos/secrets/hermes_api_key

docker run -d --name hermes --gpus all --init --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v hermes-hf-cache:/cache/huggingface \
  -v /srv/kleos/hermes-v0.0.6:/models/hermes-v0.0.6:ro \
  -v /srv/kleos/secrets:/run/secrets:ro \
  -e HERMES_API_KEY_FILE=/run/secrets/hermes_api_key \
  kleos-hermes:v0.0.6
```

`docker/compose.yaml` expresses the same thing, including the GPU reservation.

| | |
| --- | --- |
| Port | **8000** inside the container. Publish it on loopback or a private network, behind TLS. |
| Health | `GET /health`, unauthenticated. `{"ready": true}` only once the model is loaded and verified. The image's `HEALTHCHECK` polls it. |
| Readiness | `GET /ready`, authenticated. Returns 503 until loaded, then the loaded identity. |
| Inference | `POST /v1/generate`, authenticated. See [API](#api). |

### Startup sequence

`python -m kleos_models.serving.startup` is the entrypoint. Every check that can
fail cheaply runs **before** the base-model download, and each one fails closed —
exit 1, nothing served:

1. **Secrets.** An API key must be present, from env or file.
2. **Package integrity.** Every file in the mounted package is re-hashed against
   its manifest. `adapter_config.json` must pin the base revision.
3. **Package identity.** The package must be the frozen Hermes artifact: adapter
   `dc121fa3…`, base `04d8a905…`, the three tokenizer files by hash,
   `fix_mistral_regex=false`, greedy decoding. These are compared against the
   serving record baked into the image, so an intact but *different* package is
   refused.
4. **Model cache.** `HF_HOME` must be writable. It warns if it is not on a
   mounted volume.
5. **GPU.** A CUDA device must be visible. It warns below 14 GiB.
6. **Serve.** uvicorn starts. The app loads the base at the pinned revision, the
   frozen tokenizer and the adapter, re-verifying identity as it goes, and checks
   the loaded tokenizer and adapter against the manifest. With
   `HERMES_EXIT_ON_LOAD_FAILURE=1`, set by the image, a load failure exits the
   process instead of leaving it up but never ready.

Warnings, not failures: no `HF_TOKEN`, a cache that is not on a volume, and a GPU
below 14 GiB.

The startup banner records the base, adapter, tokenizer flag, secret *sources*
(never values), cache state, GPU and the versions of the libraries that decide
the output.

To check a package on a host without a GPU, or before committing GPU time:

```bash
docker run --rm -v /srv/kleos/hermes-v0.0.6:/models/hermes-v0.0.6:ro \
  -e HERMES_ALLOW_UNAUTHENTICATED=1 kleos-hermes:v0.0.6 --check-only --no-gpu-check
```

### First start and later starts

**First start** on a new cache volume:

1. Preflight takes seconds.
2. The base model downloads into `/cache/huggingface`: about 24.5 GB on disk,
   ~21 GB transferred. On Colab this took 11.5 minutes unauthenticated; your
   network will differ.
3. The weights load in about 2 minutes.
4. `/health` reports `ready: true`.

The `HEALTHCHECK` start period is 45 minutes so this counts as starting, not
failing.

**Every later start** with the same volume:

1. Preflight, then the banner shows `base in cache : yes`.
2. The weights load from the cache in about 2 minutes, with no download.

With `HF_HUB_OFFLINE=1` a later start needs no network at all. The revision pin
still applies, because the snapshot is looked up by commit.

**Without a persistent volume**, every new container re-downloads the base. The
startup banner warns when this is about to happen.

### First GPU start: re-establish the 9/9 match

The 9/9 exact-match result was produced in a Colab session, not in this image.
The model-runtime pins match the research record, but the Colab session's full
package set was not recorded. So before the service takes traffic, run the smoke
test in the image on the GPU host. It shares the cache volume, so it also does
the first-start download:

```bash
docker run --rm --gpus all \
  -v hermes-hf-cache:/cache/huggingface \
  -v /srv/kleos/hermes-v0.0.6:/models/hermes-v0.0.6:ro \
  -v /srv/kleos/eval:/eval:ro \
  --entrypoint python kleos-hermes:v0.0.6 /app/scripts/hermes_smoke.py \
    --package /models/hermes-v0.0.6 \
    --benchmark /eval/benchmark.jsonl --reference /eval/arm2_finetuned.json
```

Run it **before** starting the server, not with `docker exec` alongside it: a
second process loads a second copy of the model and would not fit a 16 GB GPU.

`benchmark.jsonl` and `arm2_finetuned.json` are derived from the private
dataset. Copy them only to a host you control, keep them out of the image, and
remove them afterwards.

A mismatch here is a serving-environment difference. It is never a reason to
retrain.

---

## Free hosting on Hugging Face ZeroGPU

A second host for the same frozen artifact, at $0: a Gradio Space on
Hugging Face's shared ZeroGPU hardware. It is an **adapter**, not a second model
implementation. It loads the package through the same `load_deployment`,
verifies the same identity, and generates through the same `HuggingFaceBackend`
as the Docker service. The only difference is that generation is split, so the
GPU is held for the one step that needs it. The Docker path above stays the
canonical, portable deployment.

**Status, 2026-09-23: implemented and tested locally; not deployed.** Nothing
has been uploaded to Hugging Face, no Space exists, and Hermes has not yet run
on ZeroGPU. See the [verification record](#verification-record--zerogpu) for
exactly what is verified, what is estimated, and what is not tested.

### How it fits together

```
KLEOS backend (server only: holds HERMES_HF_TOKEN and HERMES_API_KEY)
   │  kleos_models.serving.client → gradio_client, header X-Hermes-Key
   ▼
ZeroGPU Space (Gradio 6.28.0, Python 3.12)        deploy/zerogpu-space/app.py
   startup, CPU  download package @ pinned commit → verify hashes + identity
                 → load base (preloaded shards) + adapter, NF4, float16
   /generate     auth → validate → render + tokenize          CPU, no quota
                 → @spaces.GPU  generate_ids                   GPU, quota
                 → decode → status-contract JSON               CPU, no quota
   /status       identity, runtime, limits                     CPU, no quota
```

| What | Where it lives |
| --- | --- |
| Space code | `deploy/zerogpu-space/`: `README.md` (Space config), `app.py` (wiring), `requirements.txt.template` |
| Request logic | `kleos_models.serving.zerogpu`, `status`, `smoke`, installed in the Space from a pinned commit of this repository |
| Base weights (24.5 GB) | Baked into the Space image at build time by `preload_from_hub`: the five HF shards and three configs at `04d8a905…`. Not `consolidated.safetensors` (a second 24.5 GB copy) and not the Hub's tokenizer |
| Deployment package (~245 MB) | A **private** model repository, downloaded at startup at a pinned commit |
| Secrets | Space secrets: `HF_TOKEN`, `HERMES_API_KEY`, `HERMES_PACKAGE_REPO`, `HERMES_PACKAGE_REVISION` |
| Never in the Space | Model weights, datasets, benchmark, evaluation outputs, experiment logs, user data, tokens |

The model runtime in the Space is the Docker image's, pin for pin:
`torch 2.11.0+cu128` (the exact wheel, by URL and sha256), `transformers 5.16.1`,
`peft 0.20.0`, `accelerate 1.14.0`, `bitsandbytes 0.50.2`, `tokenizers 0.23.2`,
`Jinja2 3.1.6`. `tests/test_zerogpu_space.py` fails if any of them drift apart.

### Lifecycle

From the `spaces` 0.51.3 source, not assumed:

1. **Space start** (a build, a restart, or waking from sleep). `app.py` runs
   once: it downloads the package, verifies it, and loads the base plus adapter
   into CPU memory under CUDA emulation, quantizing to NF4 on the way. This is
   the **cold model load**. It uses no GPU quota, and it happens once per Space
   process, never per request.
2. **First call to a GPU worker (cold).** ZeroGPU schedules a GPU, forks a
   worker process and moves the model's tensors onto it. The model is copied,
   not reloaded or re-quantized.
3. **Warm inference.** When a call finishes, the GPU allocation is released,
   but the worker process and its GPU-resident weights are kept. If the next
   call lands on the same GPU while it is still assigned to this Space and
   idle, that worker is reused and only generation runs. The response's
   `diagnostics.cold_start` and `worker_call_index` say which case occurred.
4. **Worker release and the next cold start.** If the GPU was reassigned or the
   worker died, the next call forks a new worker and moves the weights again,
   as in step 2. How long an idle GPU stays assigned to a Space is not
   documented and is **NOT VERIFIED**; the smoke test observes it.
5. **Sleep.** An idle free Space goes to sleep, dropping everything. The next
   request wakes it, and step 1 runs again.

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

All estimates below are **ESTIMATED**: no Hermes request has run on ZeroGPU yet.
`scripts/zerogpu_smoke.py` replaces them with observed measurements.

**Capacity is small.** If KLEOS calls with one service account's token, every
KLEOS user shares that account's 5 minutes a day. A Hermes response is roughly
100–400 tokens. At an estimated 10–30 tokens/s on half a Blackwell with NF4, that
is about 5–40 s of GPU per call, or on the order of **5–20 generations per day**.
This host suits a beta or a demo, not production traffic. KLEOS must treat
Hermes as opportunistic and fall back every time it is not `ready`.

**Requested duration matters.** Each call requests
`ceil(10 + max_new_tokens / HERMES_GPU_TOKENS_PER_SECOND)` seconds, clamped to
15–60 s. The throughput default of 12 tokens/s is deliberately T4-conservative:
a 512-token budget requests 53 s and needs 80 s of quota remaining to be
admitted. After the smoke test, set `HERMES_GPU_TOKENS_PER_SECOND` to the
measured warm throughput, so the last minute or so of each day's quota is not
refused needlessly.

| Latency | Estimate | Why |
| --- | --- | --- |
| Space start (after a build, a restart or sleep) | 3–10 min | Download the package, read 24.5 GB of shards, quantize 12B parameters to NF4 on the CPU |
| First call in a new GPU worker (cold) | +5–30 s | Fork a worker and move ~8 GB of quantized weights onto the GPU |
| Warm call | 5–40 s | Generation only: the worker and its weights are reused while it stays assigned |
| Waiting for a GPU | up to 60 s | Then `queue_unavailable` |

**When something is unavailable,** KLEOS gets a status and falls back. Never an
error:

| Situation | Status KLEOS sees |
| --- | --- |
| The account's daily quota is spent | `quota_exhausted`, with Hugging Face's stated wait when it gives one |
| No GPU within 60 s, or the Space's queue of 8 is full | `queue_unavailable` |
| The Space is asleep, building or loading the model | `starting` |
| The Space is paused, failed to build, or crashed (including a refused package at startup) | `disabled` |
| Hugging Face unreachable, or the connection drops mid-call | `starting` while connecting, `queue_unavailable` mid-call |
| The package repository unreachable at startup | The Space does not start, so `starting` and then `disabled` |
| A call runs past `HERMES_TIMEOUT` | `queue_unavailable`; the queued job is cancelled |
| A cold call runs past its requested GPU duration and ZeroGPU aborts it | `model_error`. If this shows up in the smoke test, raise `HERMES_GPU_BASE_SECONDS` |

### Request bounds

| Bound | Value | Enforced |
| --- | --- | --- |
| Fields | `messages`, `max_new_tokens`, `request_id` only; roles `system`, `user`, `assistant` | Before any GPU work |
| Messages | ≤ 64 | Before any GPU work |
| Characters | ≤ 24,000 | Before any GPU work |
| Prompt tokens | ≤ 2,048 (`HERMES_MAX_INPUT_TOKENS`) | After tokenizing, before the GPU |
| Output tokens | ≤ 512; a request may lower it, never raise it | Always |
| Concurrency | One generation at a time; queue of 8, then `queue_unavailable` | Gradio queue |

Decoding is the frozen contract (greedy, `max_new_tokens` 512) and cannot be
changed per request.

### Security model

- **Authentication.** Every endpoint requires the `X-Hermes-Key` header, compared
  in constant time against `HERMES_API_KEY` (comma-separated keys allow
  rotation). A custom header, because Hugging Face uses `Authorization` for its
  own token. The Space refuses to start without the key. An unauthenticated
  request is answered `unauthorized` before any validation, tokenization or GPU
  scheduling, so it cannot spend anyone's quota.
- **Two tokens, two jobs.** The Space's `HF_TOKEN` is a fine-grained,
  **read-only** token scoped to the package repository: it downloads the
  package, nothing else. KLEOS's `HERMES_HF_TOKEN` opens a private Space and is
  the account ZeroGPU charges. Neither is ever sent to a browser.
- **Private Space preferred.** A private Space is visible only to its owner and
  callable only with a token that can read it. Whether a free account can run a
  **private** ZeroGPU Space is **NOT VERIFIED**: the documentation lists
  visibility options without saying. Create it private if the form allows.
- **If it must be public** (a free-tier restriction), then anyone can see the
  Space's four files and the API's parameter names. They are the same files
  that are public in this repository, and none holds a secret, weight or
  dataset. Anyone can call the endpoints, but gets `unauthorized` without the
  key. The residual risk is flooding: a burst of refused requests can briefly
  fill the queue of 8, and KLEOS sees `queue_unavailable` and falls back. It
  cannot reach the model or spend quota. Protected Spaces, which would close
  this, need PRO. This is not hidden: it is the trade-off of $0.
- **Duplicating the Space does not copy Hermes.** A copy gets the four public
  files but none of the secret values, and the package lives in a private
  repository the copy cannot read.
- **Logs.** Request id, status, counts and timings only: never prompt or
  response text, headers or keys. An exception's type is logged, not its
  message, unless it is a ZeroGPU scheduling error, whose title carries no
  input. `spaces` prints a worker's traceback when generation crashes: code
  locations and the exception, not prompts.
- **No state.** Nothing is persisted between requests, as in the Docker service.

### Deploying it

Every step below is run by a person, in order. None of them costs money. **If
any screen asks for payment details, stop and report it. Do not proceed.**

1. **Commit and push this repository.** The Space installs `kleos-models` from
   GitHub at an exact commit, so the code it runs must be pushed.
2. **Build the schema-v2 package** where the frozen run lives. Colab, on a CPU
   runtime (no GPU needed), with Drive mounted:
   ```bash
   python scripts/build_deployment_package.py \
       --run /content/drive/MyDrive/<runs>/kleos-v006-mistralnemo12b-run1 \
       --output /content/hermes-v0.0.6
   python scripts/verify_deployment_package.py --package /content/hermes-v0.0.6 \
       --expect-deployment-config configs/deployment/kleos_hermes_v006.yaml
   ```
3. **Upload it to a private repository**, dry run first. `HF_TOKEN` must be a
   write token, from a Colab secret, never pasted into a cell:
   ```bash
   python scripts/upload_deployment_package.py --package /content/hermes-v0.0.6 \
       --repo YOUR_USERNAME/kleos-hermes-v006-package --dry-run
   python scripts/upload_deployment_package.py --package /content/hermes-v0.0.6 \
       --repo YOUR_USERNAME/kleos-hermes-v006-package --create
   ```
   It refuses a public repository, uploads only manifest-listed files, then
   re-lists the repository and checks every file's size and hash. It prints the
   commit to pin.
4. **Create the Space** at huggingface.co/new-space: SDK Gradio, hardware
   **ZeroGPU**, **private** if offered.
5. **Set four secrets** in the Space's settings:

   | Secret | Value |
   | --- | --- |
   | `HF_TOKEN` | A new fine-grained token: read access to the package repository only |
   | `HERMES_API_KEY` | A new random secret, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`. KLEOS gets the same value |
   | `HERMES_PACKAGE_REPO` | `YOUR_USERNAME/kleos-hermes-v006-package` |
   | `HERMES_PACKAGE_REVISION` | The commit step 3 printed (40 hex characters) |

6. **Stage and push the Space.** Stage it, read the four files, then push:
   ```bash
   python scripts/stage_zerogpu_space.py --out /tmp/hermes-space
   HF_TOKEN=... python scripts/stage_zerogpu_space.py --out /tmp/hermes-space-push \
       --push --space YOUR_USERNAME/kleos-hermes
   ```
   Or upload the four staged files in the Space's Files tab.
7. **Watch the startup log.** Success ends with one line:
   `Hermes ready: kleos-hermes v0.0.6 adapter=dc121fa36ce1409c… base=…@04d8a90549d2
   compute_dtype=float16 download_s=… load_s=… memory={…} versions={…}`. Record
   `memory` and `load_s`. **A crash with an out-of-memory error is a STOP**: the
   host could not quantize the base on its CPU. The only fallback, loading
   inside the GPU call, spends quota on every cold start, so it is a decision
   for the project owner, not a quiet fix.
8. **Run the frozen-output smoke test** from Colab, where the benchmark and the
   frozen reference already are:
   ```bash
   pip install gradio_client==2.7.1
   HERMES_API_KEY=... HF_TOKEN=... python scripts/zerogpu_smoke.py \
       --space YOUR_USERNAME/kleos-hermes \
       --benchmark /content/drive/MyDrive/<eval>/benchmark.jsonl \
       --reference /content/drive/MyDrive/<eval>/arm2_finetuned.json \
       --output /content/drive/MyDrive/<eval>/zerogpu_smoke.json
   ```
   It checks the Space's identity before spending any GPU time, then runs the
   same nine examples as the Colab 9/9, plus one warm repeat. That is about 10
   GPU calls, sized to fit one day's free quota; it stops at the first quota
   refusal and keeps what it measured.
9. **Calibrate.** Set `HERMES_GPU_TOKENS_PER_SECOND` (a Space variable, not a
   secret) to the smoke test's warm `median_tokens_per_second`, rounded down.

### If the outputs differ: stop

Exact reproduction on ZeroGPU is **uncertain in advance**. Three things differ
from the T4 that produced the frozen outputs, none of them a model change:

- the GPU architecture (Blackwell, not Turing), so different fp16 matmul and
  SDPA kernels;
- the NF4 quantization runs on the CPU under emulation at startup, not on the
  GPU at load time;
- the CUDA libraries inside the same torch build choose kernels per
  architecture.

A mismatch is a **STOP**: do not retrain, and do not change the model, the
tokenizer, the decoding or the precision to make outputs match. Record, from
`zerogpu_smoke.json`:

1. **Prompt tokens equal to the evaluation's?** If not, the difference is
   tokenization or prompt assembly, and it is a bug to fix before anything else.
2. **If they are equal, where the text first diverges.** An identical opening
   and a late one-token flip points to arithmetic. A difference from the first
   token points to setup.
3. **Whether the extracted decision still agrees.** It says whether the
   difference changes what Hermes decides.

Then decide, as the project owner: accept and document "numerically different,
decision-equivalent on N/9", or serve from the Docker image on a GPU host
instead. The evidence for either is in the smoke output.

### Verification record — ZeroGPU

| | Status |
| --- | --- |
| Serving path splits CPU and GPU work without changing generation | **VERIFIED** locally: in the Docker image on CPU, the split generates token-for-token what the previous `generate` did (tiny Mistral, torch 2.11.0+cu128, transformers 5.16.1) |
| Runtime contract pins float16; a bfloat16 package is refused | **VERIFIED**: unit tests and the container preflight |
| Auth, validation, bounds, error → status mapping, no content in logs | **VERIFIED** by unit tests with a fake GPU call; ZeroGPU errors use the exact `spaces` 0.51.3 messages |
| Reference client never raises; sleeping Space → `starting` | **VERIFIED** by unit tests with fake transports |
| Space files: pins equal Docker's, correct preload, no secrets or data | **VERIFIED** by static tests |
| Package upload refuses unlisted files and checks remote bytes | **VERIFIED** by unit tests; never run against the Hub |
| Free-tier facts above | **VERIFIED** against the HF documentation and `spaces` source, 2026-09-23 |
| Latency, throughput, capacity, VRAM, startup RAM | **ESTIMATED** |
| The Space builds with these requirements | **NOT TESTED** |
| Startup fits the host's CPU RAM | **NOT TESTED**; estimated peak 13–16 GB |
| NF4 quantization under CUDA emulation, then generation on Blackwell | **NOT TESTED** |
| 9/9 exact match on ZeroGPU | **NOT TESTED** |
| `X-Hermes-Key` reaches the app through Hugging Face's proxy | **NOT TESTED** |
| A free account may run a private ZeroGPU Space | **NOT VERIFIED** |

---

## Checklist before serving

1. `verify_deployment_package.py --expect-deployment-config …` exits 0.
2. `hermes_smoke.py` reproduces the frozen responses **inside the container on
   the GPU host**, or the difference is understood and recorded.
3. `HERMES_API_KEY` is set from a secret store, not a file in the repository.
4. The service is bound to loopback or a private network, behind TLS.
5. The model cache is a persistent volume.
6. The consumer knows the measured limitations in `manifest.json`: prose not
   JSON, one-sided overconfidence, and instability under rephrasing.

On ZeroGPU, in place of 2–5:

- `zerogpu_smoke.py` reports 9/9, or the difference is recorded and the owner
  has decided what to do about it.
- The package repository is private, and `HERMES_PACKAGE_REVISION` is a commit,
  not a branch.
- `HERMES_API_KEY` and a read-only, package-scoped `HF_TOKEN` are Space secrets.
- KLEOS falls back on every status other than `ready`
  ([integration contract](kleos-hermes-integration.md)).
