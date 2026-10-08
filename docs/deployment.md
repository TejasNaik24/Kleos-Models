# Deploying a KLEOS model

This repository trains and evaluates models; it does not host them. What it
produces for a host is a deployment package: a verifiable directory that names
the base weights, tokenizer, adapter and decoding settings that make up one
model, and refuses to load if any of them has changed.

Two models are served from packages built this way:

| Model | Base, pinned revision | Hosts | Status |
| --- | --- | --- | --- |
| Hermes v0.0.6 | `mistralai/Mistral-Nemo-Instruct-2407` @ `04d8a905…` | ZeroGPU Space; Docker image | Live on a private ZeroGPU Space since 2026-09-23, 9 of 9 frozen outputs reproduced. Docker image built, not GPU-tested |
| Logos v0.0.2 | `mistralai/Ministral-3-14B-Reasoning-2512` @ `51f9210f…` | ZeroGPU Space only | Live on a private ZeroGPU Space as a Beta since 2026-10-07, 8 of 9 frozen outputs reproduced |

This page is the reference: what a package contains and guarantees, the
serving profile of each model, the inference API, the container, the ZeroGPU
host, and how to diagnose a reproduction mismatch. Related pages:

- Operator procedures: [Deploy Hermes v0.0.6 to ZeroGPU](runbooks/deploy-hermes-zerogpu.md)
  and [Deploy Logos v0.0.2 to ZeroGPU](runbooks/deploy-logos-zerogpu.md).
- The contract the KLEOS backend integrates against: [serving-api.md](serving-api.md).
- The dated verification records:
  [experiments/serving-verification-records.md](experiments/serving-verification-records.md).
  Finding and decision ids follow the conventions in
  [experiments/README.md](experiments/README.md).

Placeholders used below: `<owner>/<space>` is a Hugging Face Space id,
`<owner>/<package-repo>` the private model repository that holds a package,
`<private storage>` the private location of training and evaluation outputs,
`<path to release>` a local copy of a dataset release, and `<commit sha>` a
40-character commit of this repository.

---

## Research artifact and deployment artifact

| | Research artifact | Deployment artifact |
| --- | --- | --- |
| What it is | What the training run produced | A copy, prepared for serving |
| Where | `<private storage>/outputs/<experiment-id>/` | A package built from it |
| Purpose | Evidence | Operation |
| Mutable | Never | Rebuilt whenever serving needs change |
| `adapter_config.json` `revision` | `null` for Hermes v0.0.6 (finding H-F1) | The pinned base commit |
| Tokenizer | Whatever the run loaded | The frozen files, with the regex flag stated |
| Identity record | `manifest.json` (the run) | `manifest.json` (the package) |

The research artifact is never edited to look correct in hindsight. Hermes
v0.0.6 was trained with PEFT writing `revision: null` into
`adapter_config.json` (finding H-F1, in section 11 of the
[Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)),
and that file still says `null`. The deployment package carries the pin
instead, and its manifest records both the pin and the fact that the original
lacked it, so a comparison of the two shows what was added and when.

The evidence for each model is in its run report:
[Hermes v0.0.6](experiments/kleos-v006-mistralnemo12b-run1-report.md) and
[Logos v0.0.2](experiments/kleos-v007-ministral314breasoning-run1-report.md).

---

## Why the base revision is pinned

A LoRA adapter is a set of deltas against specific base weights. Attached to a
different revision of the same checkpoint:

- no exception is raised,
- the shapes still match,
- the model still answers,
- the answers are quietly worse.

No test at inference time catches this. The defense is to record the exact
commit and refuse to serve without it, which is what `DeploymentManifest` and
`verify_base_revision` do.

The same argument applies to the tokenizer: different tokenization produces a
different prompt, and a different prompt produces a different answer, with no
error raised.

---

## The tokenizer contract

Under transformers ≥ 5, loading a Mistral tokenizer, such as
`mistralai/Mistral-Nemo-Instruct-2407`'s, without stating `fix_mistral_regex`
prints a warning that it has "an incorrect regex pattern" and should be loaded
with `fix_mistral_regex=True`.

The flag replaces the pre-tokenizer's `Split` regex with the one
`mistral-common` uses. The two patterns disagree on how some sequences split.
The documented example is `'The'`, which becomes `["'", "T", "he", "'"]` under
the old pattern and `["'", "The", "'"]` under the new one; roughly 1% of tokens
are affected. The flag defaults to `False`.

Each model is served with the value it was trained and evaluated with:

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Value in training and evaluation | `False`: `load_tokenizer` never passed the flag, so every load used the default | `True`, set in `configs/models/ministral3_14b_reasoning.yaml`; it changes 0 of 1,350 kleos-policy-v0.0.7 examples (measured) |
| Value the package records and the loader passes | `fix_mistral_regex: false` | `fix_mistral_regex: true` |
| Pad token | `</s>` | `<pad>` |

Every Hermes arm used the same tokenizer, so the comparison in the Hermes run
report is internally valid.

Serving pins tokenization in two ways:

1. **The tokenizer files are packaged.** `tokenizer/` in the deployment package
   holds the exact `tokenizer.json`, `tokenizer_config.json` and
   `chat_template.jinja` the run saved, hashed in the manifest and in the
   deployment record. The loader points at that directory, not at the Hub.
2. **The flag is stated.** The manifest records `fix_mistral_regex` and the
   loader passes it explicitly, so a change to the library default cannot move
   tokenization under a frozen adapter.

`mistral_regex_kwarg` handles transformers versions that predate the flag.
Such a version already uses the unpatched pattern, so nothing is passed, and a
request for `True` raises an error instead of silently doing something else.

The Ministral-8B deployment record
(`configs/deployment/kleos_v006_ministral8b.yaml`) takes the tokenizer from the
base repository instead. That is the right default when no library flag can
change tokenization; Hermes and Logos package theirs because one can.

Moving Hermes to the corrected regex requires a retrain. Serving with
`fix_mistral_regex=True` against weights trained with `False` is train/serve
skew, which the pin exists to prevent.

---

## The runtime contract

How the base model is instantiated is part of what was measured, so it is part
of the identity a server checks (package schema v2). Both records state the
same contract:

| Field | Hermes v0.0.6 | Logos v0.0.2 | Why it is pinned |
| --- | --- | --- | --- |
| `quantization_mode` | `nf4` | `nf4` | 4-bit NF4 is what was trained and evaluated |
| `double_quant` | `true` | `true` | Changes the quantized weights |
| `compute_dtype` | `float16` | `float16` | See below |
| `attn_implementation` | `sdpa` | `sdpa` | Different kernels, different arithmetic |
| `max_seq_length` | `1024` | `1024` | The trained context |

`compute_dtype` is stated rather than left at `auto`. The training configs say
`auto`, which resolves to float16 below compute capability 8.0 and to bfloat16
at or above it. Both models were trained and evaluated on T4s (compute
capability 7.5), so both were measured in float16; the Hermes research manifest
records `resolved_compute_dtype: float16`. Nearly every newer GPU a server is
likely to have, including the RTX Pro 6000 (Blackwell, 12.0) behind ZeroGPU,
would resolve `auto` to bfloat16 and serve a model that computes differently
from the one measured, with no error. Stating float16 keeps the hardware from
changing the model.

The contract lives under `runtime` in each deployment record. The builder
applies it to the packaged `deployment/model_config.yaml`, which is itself
hashed in the manifest. The loader refuses a package whose model config
disagrees with the contract, and `verify_identity` refuses one whose contract
differs from the serving record. A version-1 package, built before the contract
existed, is refused and must be rebuilt.

---

## Serving profiles

One serving codebase serves both models. A deployment record may carry a
`serving` block, read by `src/kleos_models/serving/profile.py`; a record
without one gets Hermes' values. The Logos record sets its own:

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Deployment record | `configs/deployment/kleos_hermes_v006.yaml` | `configs/deployment/kleos_logos_v002.yaml` |
| Settings prefix | `HERMES_` | `LOGOS_` |
| Space secrets | `HF_TOKEN`, `HERMES_API_KEY`, `HERMES_PACKAGE_REPO`, `HERMES_PACKAGE_REVISION` | `HF_TOKEN`, `LOGOS_API_KEY`, `LOGOS_PACKAGE_REPO`, `LOGOS_PACKAGE_REVISION` |
| Optional Space variables | `HERMES_GPU_BASE_SECONDS`, `HERMES_GPU_TOKENS_PER_SECOND`, `HERMES_GPU_MIN_SECONDS`, `HERMES_GPU_MAX_SECONDS`, `HERMES_MAX_INPUT_TOKENS`, `HERMES_MAX_NEW_TOKENS` | The same names with the `LOGOS_` prefix |
| Key header | `X-Hermes-Key` | `X-Logos-Key` |
| Space folder | `deploy/zerogpu-space` | `deploy/zerogpu-space-logos` |
| Record file in the Space | `hermes_record.yaml` | `logos_record.yaml` |
| Base files baked into the Space | Mistral-Nemo: 5 shards and 3 configs | Ministral 3 14B Reasoning: 6 shards and 3 configs (the text-only view leaves the vision weights on disk) |
| Reply | Contract version 1 | Contract version 2, which adds `reasoning` |
| System message | Optional | Required as the first message |
| Output-token ceiling | 512 | 1,024, as evaluated |
| GPU duration defaults | 10 s + tokens ÷ 12 tokens/s, clamped to 15–60 s | The same |
| Docker service | Yes | No (`space_only: true`) |
| Research report | [Hermes run report](experiments/kleos-v006-mistralnemo12b-run1-report.md) | [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md) |

The secrets are set in the Space's settings as secrets. The `*_GPU_*`,
`*_MAX_INPUT_TOKENS` and `*_MAX_NEW_TOKENS` settings carry no secret and are set
as Space variables; `*_MAX_NEW_TOKENS` may only lower the record's ceiling.
`<PREFIX>_ALLOW_UNAUTHENTICATED=1` turns the key check off for local tests and
is never set on a Space. The KLEOS backend's own settings, `<PREFIX>_SPACE`,
`<PREFIX>_HF_TOKEN` and the rest, are listed in
[serving-api.md](serving-api.md#configuration).

---

## Building a package

```bash
python scripts/build_deployment_package.py \
    --run <private storage>/outputs/kleos-v006-mistralnemo12b-run1 \
    --deployment-config configs/deployment/kleos_hermes_v006.yaml \
    --output <private storage>/packages/hermes-v0.0.6
```

Logos takes its record and its model config:

```bash
python scripts/build_deployment_package.py \
    --run <private storage>/outputs/kleos-v007-ministral314breasoning-run1 \
    --deployment-config configs/deployment/kleos_logos_v002.yaml \
    --model-config configs/models/ministral3_14b_reasoning.yaml \
    --output <private storage>/packages/logos-v0.0.2
```

The run directory is read-only throughout. The build refuses to continue if the
adapter's sha256 is not the frozen one recorded in the deployment record, so a
package cannot be built from weights that were never measured. Nor can one be
built from a record whose identity fields are still `null` (adapter hash,
selection value, pad token, tokenizer file hashes);
`scripts/fill_deployment_record.py` completes those from the run first.

```
hermes-v0.0.6/
├── manifest.json            identity + every file's sha256
├── adapter/                 adapter_model.safetensors, adapter_config.json (pinned), README.md
├── tokenizer/               tokenizer.json, tokenizer_config.json, chat_template.jinja
└── deployment/              README.md, model_config.yaml
```

Base weights are not in the package. Mistral-Nemo alone is about 24 GB, and
the weights belong to Mistral AI; the package names a repository and a commit,
and the loader fetches and pins that. A package is never committed to git:
`.gitignore` covers `outputs/`, and a package belongs on the serving host, in an
artifact store, or in a private model repository.

## Verifying a package

```bash
python scripts/verify_deployment_package.py \
    --package <private storage>/packages/hermes-v0.0.6 \
    --expect-deployment-config configs/deployment/kleos_hermes_v006.yaml
```

The script re-hashes every recorded file, checks that `adapter_config.json`
pins the revision the manifest names, and prints the model's identity and
measured limitations. It needs no GPU, no torch and no network, so it runs in CI
and as a pre-start check. `--check-base-revision` also confirms the commit with
the Hub.

`--expect-deployment-config` makes this an identity check rather than a
consistency check. Without it, a package is compared only against its own
manifest, and a different adapter packaged as carefully would pass. With it,
the package must also match the serving record: the frozen adapter hash, the
base revision, the three tokenizer files by hash, the tokenizer flag, the
runtime contract and the greedy decoding settings. The container and both
Spaces always check against the record baked into them.

On any mismatch the script exits non-zero and prints `DO NOT SERVE THIS PACKAGE`.

## Reproducing the frozen outputs

A package that verifies holds the right files. Whether it runs as evaluated is
checked by a smoke test: one example per task family plus two should-decline
cases, nine in all, each compared with the output the frozen evaluation recorded
for the same example id. Decoding is greedy and seeded, so on the evaluation's
hardware the outputs should match byte for byte.

The smoke test produces no score. Nine examples are too few to say anything
about quality next to the 349-example benchmark, and reporting a number from
them would mislead.

| Script | Runs | Covers |
| --- | --- | --- |
| `scripts/hermes_smoke.py` | In-process, through the deployment loader, on a GPU host or inside the container | Hermes |
| `scripts/zerogpu_smoke.py` | Against a Space, through the reference client; the benchmark and reference stay local | Hermes, or Logos with `--record configs/deployment/kleos_logos_v002.yaml` |

```bash
python scripts/hermes_smoke.py \
    --package <private storage>/packages/hermes-v0.0.6 \
    --benchmark <private storage>/<eval>/benchmark.jsonl \
    --reference <private storage>/<eval>/arm2_finetuned.json
```

`benchmark.jsonl` rebuilds from the dataset release with
`scripts/build_benchmark.py` (sha256 `a11ffad75f51…b266`, the same file for both
models). The reference is the model's frozen `arm2_finetuned.json`: Hermes
v0.0.6's evaluation results (sha256 `428400f520f7d370…`) or Logos v0.0.2's
(sha256 `0dd74661e1c5a9b1…c30eb5`). Both are derived from the private dataset.

`zerogpu_smoke.py` first checks the Space's identity and runtime with `/status`,
which spends no GPU time. For a thinking model it compares the trace as well as
the answer, refuses a reference generated with another token budget or without
traces before it contacts the Space, and reports a run that could not compare
every trace as verifying nothing. It stops at the first quota refusal and keeps
what it measured; `--only <ids>` finishes an incomplete run later. Exit status:
0 when every output reproduces, 1 when a Hermes response is empty, 2 for a
difference or an incomplete run.

---

## Diagnosing a reproduction mismatch

A mismatch between a served output and the frozen evaluation output is a
deployment problem, never a reason to retrain. The model, tokenizer, decoding
and precision are not changed to force a match: each of those changes would
serve a model other than the one that was measured.

Sources of difference that do not change the model:

- **The GPU.** fp16 arithmetic is not bit-identical across devices, and the CUDA
  libraries inside the same torch build choose matmul and SDPA kernels per
  architecture. Both models were evaluated on a T4 (Turing); ZeroGPU runs
  Blackwell.
- **Where quantization runs.** On ZeroGPU, NF4 quantization runs on the CPU under
  CUDA emulation at startup, not on the GPU at load time.
- **Setup.** Quantization settings, tokenizer behavior and prompt assembly. The
  package and the runtime contract pin all three, so a difference here points
  to a changed loader or environment.

Procedure, using the smoke-test output (`zerogpu_smoke.py --output`, or the
`hermes_smoke.py` report):

1. **Compare prompt token counts with the evaluation's.** If they differ, the
   cause is tokenization or prompt assembly. Fix that bug before anything else.
2. **If they are equal, find where the text first diverges.** An identical
   opening followed by a late one-token flip points to arithmetic. A difference
   from the first token points to setup: check quantization, dtype, attention
   implementation and the tokenizer, in that order.
3. **Check whether the extracted decision still agrees.** This says whether the
   difference changes what the model decides.
4. **Decide and record.** Either accept and document the host as numerically
   different, naming each item that differs and whether its decision changed,
   or serve from hardware that matches the evaluation (a T4-class GPU, for
   example with the Docker image). Record the decision with its date beside the
   verification record.

Logos v0.0.2 went through this procedure on 2026-10-07. One of nine answers,
`kx-mcb-066bff4fa33aa29b`, diverged on ZeroGPU: its prompt tokens matched, its
trace was identical for the first 568 of 761 characters, and the different
sentence that followed led to a different decision. The diagnosis was GPU
arithmetic. A thinking model writes a few hundred tokens of trace before its
answer, which gives a near-tied next token more chances to flip; Hermes, which
does not think, matched 9 of 9. Decision (2026-10-07): accept and document,
and serve Logos as a Beta. The full record and the consequences of that
decision are in the
[Logos verification record](experiments/serving-verification-records.md#verification-record--logos).

---

## The inference service

The Docker host runs a FastAPI service for Hermes. Logos has none.

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
export HERMES_PACKAGE_DIR=/srv/kleos/hermes-v0.0.6
export HERMES_API_KEY=...          # from a secret store; never committed
python scripts/serve_hermes.py --host 127.0.0.1 --port 8000
```

| Variable | Meaning |
| --- | --- |
| `HERMES_PACKAGE_DIR` | Package to serve. Required |
| `HERMES_API_KEY` | Shared secret or secrets, comma-separated for rotation. Required |
| `HERMES_DEVICE_MAP` | For example `cuda:0`. Optional |
| `HERMES_REQUIRE_REMOTE_REVISION` | `1` to confirm the base commit with the Hub at startup |
| `HERMES_REQUEST_TIMEOUT_SECONDS` | May only tighten the manifest's limit |
| `HERMES_MAX_INPUT_CHARS` | May only tighten the manifest's limit |
| `HERMES_EXPECTED_DEPLOYMENT_CONFIG` | Serving record the package must match. Set by the container |
| `HERMES_EXIT_ON_LOAD_FAILURE` | `1` to exit when the model fails to load instead of answering 503. Set by the container |

The service refuses to start without `HERMES_API_KEY` unless
`HERMES_ALLOW_UNAUTHENTICATED=1` is set explicitly, which is for a local
experiment on loopback only.

### API

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | Liveness. No credentials. Reports `ready` and, if not ready, why |
| `GET /ready` | Readiness plus the identity of what is loaded. Authenticated |
| `POST /v1/generate` | Generate one response. Authenticated |

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

Every response names the adapter that produced it, so a caller that logs
`model.adapter_sha256` can later tell exactly which weights answered.

| HTTP status | Meaning |
| --- | --- |
| 401 | Missing or wrong bearer token. No detail about which |
| 413 | Too many messages, or input over `max_input_chars`. Checked before any GPU work |
| 422 | Malformed body, unknown role, or an unknown field |
| 500 | Generation failed. No internal detail is returned |
| 503 | Model not loaded, or startup verification failed |
| 504 | Generation exceeded the timeout |

The reference client maps these codes to the status contract
([serving-api.md](serving-api.md#where-statuses-come-from)).

### Scope of the service

Hermes is a model service. KLEOS remains the source of truth for users,
workspaces, memories, projects, skills, the career graph, missions,
conversations, documents, jobs, research and notifications. The service
enforces that boundary:

- **No persistence.** Nothing is written between requests, so there is no
  cross-user or cross-workspace state to leak.
- **A narrow request body.** `messages`, `max_new_tokens` and `request_id`
  only. Unknown fields are rejected, so a caller cannot quietly start sending
  workspace state or tool definitions.
- **No tools and no fetching.** The only outbound traffic is to the model
  registry, at startup, for the pinned base weights.
- **No prompt or response logging.** Logs carry ids, counts and timings.
- **No API key in logs**, and no key echoed in an error.
- **No published schema.** `/docs`, `/redoc` and `/openapi.json` are off.
- **One generation at a time**, held by a lock inside the worker thread, so a
  timed-out request that is still running cannot overlap the next one.

The KLEOS backend lives in a separate repository, so its provider wiring is not
implemented here. Its contract (configuration, request and reply, statuses,
fallback rules and client states) is [serving-api.md](serving-api.md), with a
tested reference client in `kleos_models.serving.client`. Send only the
messages the model needs: anything else is data handed to a process that does
not need it.

---

## Container

`docker/hermes.Dockerfile` runs the inference service on a generic NVIDIA GPU
host.

Status: built on 2026-09-23 for `linux/amd64` (6.31 GB) and preflight-tested
without a GPU; the results are in the
[container record](experiments/serving-verification-records.md#container).
Loading the base model, attaching the adapter and serving a request inside the
image on a GPU are not tested. The first GPU start closes that gap
([First GPU start](#first-gpu-start-re-establish-the-99-match)).

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
| Architecture | `linux/amd64`. The base image also exists for `linux/arm64`, but only `amd64` has been built |
| GPU | NVIDIA, ≥ 16 GB VRAM specified. Evaluation and the 9 of 9 Colab smoke test ran on a 14.6 GiB T4. Inference peak on a T4: not measured, estimated ~9–10 GB (4-bit base ~5.4 GB, the unquantized 131k-vocabulary embeddings and `lm_head` ~2.7 GB, adapter 0.23 GB, CUDA context and cache). Measured on ZeroGPU: 8.34 GiB. Training peaked at 13.09 GB |
| Compute capability | ≥ 7.5 (T4 and newer). float16 compute; bfloat16 is not required |
| Host driver | Must support CUDA 12.8: driver 570 or newer on most hosts. The base image declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, which the NVIDIA runtime checks at container start |
| Host software | Docker with the NVIDIA Container Toolkit |
| Disk | ~30 GB free for the cache volume |
| Network | Outbound to the Hugging Face Hub on the first start. None afterwards with `HF_HUB_OFFLINE=1` |
| Replicas | One process per GPU, one worker per process. Each worker would load its own copy of the model |

### Build

```bash
docker build --platform linux/amd64 -f docker/hermes.Dockerfile -t kleos-hermes:v0.0.6 .
```

The build context is an allowlist (`.dockerignore`): only `pyproject.toml`,
`README.md`, `LICENSE`, `src/`, four scripts, the serving record and the pinned
requirements reach the Docker daemon. `.env`, `data/`, `outputs/`, `.git` and
everything else stay behind, and `tests/test_container.py` enforces this.

Dependencies are pinned exactly in `docker/requirements-hermes.txt`. The model
runtime matches the research record: transformers 5.16.1, peft 0.20.0,
accelerate 1.14.0, bitsandbytes 0.50.2, and torch 2.11.0+cu128 from the PyTorch
CUDA 12.8 index. The image writes its complete installed set to
`/app/requirements.lock`.

### Secrets

| Variable | Required | Notes |
| --- | --- | --- |
| `HERMES_API_KEY` or `HERMES_API_KEY_FILE` | Yes | The bearer token the KLEOS backend sends. Comma-separate several to rotate |
| `HF_TOKEN` or `HF_TOKEN_FILE` | No | The base is ungated; a token raises Hub rate limits for the first download |

The `_FILE` form is preferred: the value then lives in a mounted secret file and
never appears in `docker inspect`. Setting both forms of one secret is refused.
The container runs as uid 10001, so a secret file must be readable by that user.
`docker/hermes.env.example` lists every setting with the secret values left
empty on purpose: an unreplaced placeholder would become a real, guessable key,
and an empty one stops the container instead.

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

`docker/compose.yaml` expresses the same configuration, including the GPU
reservation.

| | |
| --- | --- |
| Port | 8000 inside the container. Publish it on loopback or a private network, behind TLS |
| Health | `GET /health`, unauthenticated. `{"ready": true}` only once the model is loaded and verified. The image's `HEALTHCHECK` polls it |
| Readiness | `GET /ready`, authenticated. Returns 503 until loaded, then the loaded identity |
| Inference | `POST /v1/generate`, authenticated. See [API](#api) |

### Startup sequence

`python -m kleos_models.serving.startup` is the entrypoint. Every check that can
fail cheaply runs before the base-model download, and each one fails closed
(exit 1, nothing served):

1. **Secrets.** An API key must be present, from the environment or a file.
2. **Package integrity.** Every file in the mounted package is re-hashed against
   its manifest. `adapter_config.json` must pin the base revision.
3. **Package identity.** The package must be the frozen Hermes artifact: adapter
   `dc121fa3…`, base `04d8a905…`, the three tokenizer files by hash,
   `fix_mistral_regex=false`, greedy decoding. These are compared against the
   serving record baked into the image, so an intact but different package is
   refused.
4. **Model cache.** `HF_HOME` must be writable. A cache that is not on a mounted
   volume produces a warning.
5. **GPU.** A CUDA device must be visible. Less than 14 GiB produces a warning.
6. **Serve.** uvicorn starts. The app loads the base at the pinned revision, the
   frozen tokenizer and the adapter, re-verifying identity as it goes, and
   checks the loaded tokenizer and adapter against the manifest. With
   `HERMES_EXIT_ON_LOAD_FAILURE=1`, set by the image, a load failure exits the
   process instead of leaving it up but never ready.

Warnings, not failures: no `HF_TOKEN`, a cache that is not on a volume, and a
GPU with less than 14 GiB.

The startup banner records the base, adapter, tokenizer flag, secret sources
(never values), cache state, GPU and the versions of the libraries that decide
the output.

To check a package on a host without a GPU, or before committing GPU time:

```bash
docker run --rm -v /srv/kleos/hermes-v0.0.6:/models/hermes-v0.0.6:ro \
  -e HERMES_ALLOW_UNAUTHENTICATED=1 kleos-hermes:v0.0.6 --check-only --no-gpu-check
```

### First start and later starts

First start on a new cache volume:

1. Preflight takes seconds.
2. The base model downloads into `/cache/huggingface`: about 24.5 GB on disk,
   ~21 GB transferred. On Colab on 2026-09-23 this took 11.5 minutes without a
   token; other networks will differ.
3. The weights load in about 2 minutes.
4. `/health` reports `ready: true`.

The `HEALTHCHECK` start period is 45 minutes, so this counts as starting, not
failing.

Every later start with the same volume:

1. Preflight, then the banner shows `base in cache : yes`.
2. The weights load from the cache in about 2 minutes, with no download.

With `HF_HUB_OFFLINE=1` a later start needs no network at all. The revision pin
still applies, because the snapshot is looked up by commit.

Without a persistent volume, every new container downloads the base again. The
startup banner warns when this is about to happen.

### First GPU start: re-establish the 9/9 match

The 9 of 9 exact match was produced in a Colab session, not in this image. The
model-runtime pins match the research record, but the Colab session's full
package set was not recorded. Before the service takes traffic, run the smoke
test in the image on the GPU host. It shares the cache volume, so it also
performs the first-start download:

```bash
docker run --rm --gpus all \
  -v hermes-hf-cache:/cache/huggingface \
  -v /srv/kleos/hermes-v0.0.6:/models/hermes-v0.0.6:ro \
  -v /srv/kleos/eval:/eval:ro \
  --entrypoint python kleos-hermes:v0.0.6 /app/scripts/hermes_smoke.py \
    --package /models/hermes-v0.0.6 \
    --benchmark /eval/benchmark.jsonl --reference /eval/arm2_finetuned.json
```

Run it before starting the server, not with `docker exec` alongside it: a second
process loads a second copy of the model and would not fit a 16 GB GPU.

`benchmark.jsonl` and `arm2_finetuned.json` are derived from the private
dataset. Copy them only to a host under the operator's control, keep them out of
the image, and remove them afterwards. A mismatch here is a serving-environment
difference ([Diagnosing a reproduction mismatch](#diagnosing-a-reproduction-mismatch)).

---

## Hugging Face ZeroGPU

The second host serves both models at $0: one private Gradio Space per model, on
Hugging Face's shared ZeroGPU hardware. Each Space loads its package through the
same `load_deployment`, verifies the same identity, and generates through the
same `HuggingFaceBackend` as the Docker service. The difference is that
generation is split, so the GPU is held only for the step that needs it. The
Docker image remains the portable deployment for Hermes.

| Model | Space source | Visibility | Live since | Reproduction |
| --- | --- | --- | --- | --- |
| Hermes v0.0.6 | `deploy/zerogpu-space/` | Private | 2026-09-23 | 9 of 9 byte for byte |
| Logos v0.0.2 | `deploy/zerogpu-space-logos/` | Private | 2026-10-07, as a Beta | 8 of 9 byte for byte |

### How it fits together

```
KLEOS backend (server only: holds <PREFIX>_HF_TOKEN and <PREFIX>_API_KEY)
   │  kleos_models.serving.client → gradio_client, header X-Hermes-Key or X-Logos-Key
   ▼
ZeroGPU Space (Gradio 6.28.0, Python 3.12)        deploy/zerogpu-space*/app.py
   startup, CPU  download package @ pinned commit → verify hashes + identity
                 → load base (preloaded shards) + adapter, NF4, float16
   /generate     auth → validate → render + tokenize          CPU, no quota
                 → @spaces.GPU  generate_ids                   GPU, quota
                 → decode → status-contract JSON               CPU, no quota
   /status       identity, runtime, limits                     CPU, no quota
```

| What | Where it lives |
| --- | --- |
| Space code | `deploy/zerogpu-space/` (Hermes) and `deploy/zerogpu-space-logos/` (Logos): `README.md` (Space configuration), `app.py` (wiring), `requirements.txt.template` |
| Request logic | `kleos_models.serving.zerogpu`, `status`, `smoke` and `profile`, installed in the Space from a pinned commit of this repository |
| Base weights | Baked into the Space image at build time by `preload_from_hub`, at the pinned revision: the base's shards and three configs. For Hermes that is 24.5 GB, without `consolidated.safetensors` (a second 24.5 GB copy). Neither Space preloads the Hub's tokenizer |
| Deployment package | A private model repository, downloaded at startup at a pinned commit (about 245 MB for Hermes) |
| Serving record | `hermes_record.yaml` or `logos_record.yaml`, staged from `configs/deployment/` |
| Secrets and variables | Set in the Space's settings; see [Serving profiles](#serving-profiles) |
| Never in a Space | Adapter weights (they arrive from the private package), datasets, benchmark, evaluation outputs, experiment logs, user data, tokens |

`scripts/stage_zerogpu_space.py` renders a Space repository of exactly four
files: `README.md`, `app.py`, `requirements.txt` (the template with this
repository's commit filled in) and the record file. It refuses a README whose
`preload_from_hub` disagrees with the record and scans every staged file for
secrets.

The model runtime in a Space is the Docker image's, pin for pin:
`torch 2.11.0+cu128` (the exact wheel, by URL and sha256), `transformers 5.16.1`,
`peft 0.20.0`, `accelerate 1.14.0`, `bitsandbytes 0.50.2`, `tokenizers 0.23.2`,
`Jinja2 3.1.6`. `tests/test_zerogpu_space.py` fails if any of them drift apart.

One non-model pin differs: `pydantic 2.12.5` instead of 2.13.5. Hugging Face
installs `gradio[oauth,mcp]==6.28.0` next to a Space's requirements, and its
`mcp` extra caps pydantic at 2.12.5; the first Hermes build, on 2026-09-23,
failed on that conflict. pydantic only validates configs and requests, and the
full test suite passes on 2.12.5 under Python 3.12. A requirements change can be
checked before pushing by dry-running the install in the platform's base image
(`python:3.12.12`) with `gradio[oauth,mcp]==6.28.0 spaces==0.51.3
"torch<=2.13.0"` added, as the build log shows.

### Lifecycle

From the `spaces` 0.51.3 source:

1. **Space start** (a build, a restart, or waking from sleep). `app.py` runs
   once: it downloads the package, verifies it, and loads the base plus adapter
   into CPU memory under CUDA emulation, quantizing to NF4 on the way. This is
   the cold model load. It uses no GPU quota, and it happens once per Space
   process, never per request.
2. **First call to a GPU worker (cold).** ZeroGPU schedules a GPU, forks a
   worker process and moves the model's tensors onto it. The model is copied,
   not reloaded or re-quantized.
3. **Warm inference.** When a call finishes, the GPU allocation is released,
   but the worker process and its GPU-resident weights are kept. If the next
   call lands on the same GPU while it is still assigned to this Space and
   idle, that worker is reused and only generation runs. The reply's
   `diagnostics.cold_start` and `worker_call_index` say which case occurred.
4. **Worker release and the next cold start.** If the GPU was reassigned or the
   worker died, the next call forks a new worker and moves the weights again,
   as in step 2. How long an idle GPU stays assigned to a Space is not
   documented and not verified; in the Hermes smoke test, 7 of 10 calls got a
   new worker even seconds apart.
5. **Sleep.** An idle free Space goes to sleep, dropping everything. The next
   request wakes it, and step 1 runs again.

After starting, ZeroGPU deletes the base model's cached files from disk
("Cleaned 22.81GB of tensor files … after packing", observed for Hermes). The
packed copy is what the GPU workers use; a restart restores the files from the
Space image.

### Free-tier facts

Verified 2026-09-23 against the Hugging Face ZeroGPU documentation and the
source of `spaces` 0.51.3:

| | |
| --- | --- |
| GPU | Half an NVIDIA RTX Pro 6000 Blackwell (`large`, the default): 48 GB, compute capability 12.0 |
| Daily quota | Free account 5 minutes of GPU time; unauthenticated caller 2 minutes |
| Who pays | The calling account, identified by the token the caller sends, not the account that hosts the Space |
| Reset | 24 hours after that account's first GPU use |
| Admission | A call is admitted only if the remaining quota covers its requested duration × 1.5; the time actually used is what is charged |
| Beyond the quota | Only PRO, Team and Enterprise accounts can buy more. A free account cannot be charged: its calls are refused |
| Queue | A call waits up to 60 s for a GPU, then fails with "No GPU was available". Less remaining quota means lower queue priority |
| Hosting | A free account in good standing (verified email, older than 30 days) may host up to 2 ZeroGPU Spaces; the Hermes and Logos Spaces use both |
| Loading | Models load at startup on the CPU under CUDA emulation, then move to a real GPU for each call. bitsandbytes ≥ 0.46 needs no ZeroGPU patch |
| Storage | No persistent storage; the package is downloaded again at each start |

### Capacity and latency

Measured by the smoke tests: Hermes from 10 calls on 2026-09-23 (the nine
frozen prompts and one repeat; answers 69–128 tokens long), Logos from 9 calls
over two quota windows on 2026-10-07. Samples this small describe these
deployments, not the platform in general.

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 2g.48gb slice (sm 12.0), CUDA 12.8 | The same |
| Peak VRAM | 8.34 GiB of 48 | 8.86 GiB |
| Generation speed | ~12 tokens/s, cold and warm alike | 11.0 tokens/s warm median, 10.9 cold median (day 1); 10.2–12.1 tokens/s on day 2 |
| Warm call, end to end | Median 7.9 s; 0.15 s to get the GPU | Median 19.7 s, 19.4 s of it generating (day 1); 33.6 s for a 406-token answer (day 2) |
| Cold call, end to end | Median 12.9 s; 4.1 s to get the GPU and move the weights onto it | Median 29.9 s; 5.5 s to acquire a GPU (day 1) |
| Slow calls | 34 s end to end, most of it waiting for a GPU | Not separately recorded |
| GPU seconds per answer | 7–13 s | 24–26 s on average (217.8–232.3 s for the 9) |
| Space start | Package download 2 s, base load and NF4 quantization 41 s, packing 8.54 GB for the GPU workers 3 s, plus container start (not timed) | Package download 1.9 s, load 57.1 s, 9.20 GB packed for the GPU, host process peak 30.6 GiB, plus container start (not timed) |
| Answers per day per calling account | Roughly 20–30 (estimated from the GPU seconds per call, with the admission rule reserving the last 80 s) | About 9 (estimated from 24–26 GPU s per answer, with the admission rule reserving the last 90 s); 7 admitted on day 1, in a window already partly used |

When KLEOS calls with one service account's token, every KLEOS user shares that
account's 5 GPU minutes a day, and Hermes and Logos share them too if the same
token calls both. Longer answers cost more: at 12 tokens/s a 512-token Hermes
answer takes about 43 s of generation (estimated from the measured speed). This
host suits a beta or a demo, not production traffic. KLEOS treats both models as
opportunistic and falls back every time a call does not return `ready`
([serving-api.md](serving-api.md#fallback-rules)).

### GPU duration requested per call

Each call requests

```
min(max(ceil(BASE + max_new_tokens / TOKENS_PER_SECOND), MIN), MAX) seconds
```

from the Space variables `<PREFIX>_GPU_BASE_SECONDS` (default 10),
`<PREFIX>_GPU_TOKENS_PER_SECOND` (12), `<PREFIX>_GPU_MIN_SECONDS` (15) and
`<PREFIX>_GPU_MAX_SECONDS` (60). `max_new_tokens` is the request's budget, so a
lower budget lowers the request. Admission needs 1.5 × the requested duration
left in the calling account's quota.

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Full budget | 512 tokens | 1,024 tokens |
| Requested duration | 53 s | 60 s (the rule gives 96 s at 12 tokens/s, capped at 60) |
| Quota needed for admission | 80 s | 90 s |
| Calibration | The measured 12 tokens/s equals the default, so `HERMES_GPU_TOKENS_PER_SECOND` stays unset | At the measured 11.0 tokens/s the request is still the 60 s cap, so `LOGOS_GPU_TOKENS_PER_SECOND` stays unset (decision, 2026-10-07) |

For Hermes, a lower `HERMES_MAX_OUTPUT_TOKENS` in KLEOS lowers the admission
threshold, at the risk of answers being cut off (`finish_reason: "length"`).

For Logos, evaluation answers averaged 293 generated tokens (95th percentile
413), so a normal answer fits in 60 s. A looping answer (6 of 349 in the
evaluation) runs out of GPU time, is aborted, and becomes `model_error`, which
KLEOS treats like a cut answer, at a cost of at most 60 s. Raising
`LOGOS_GPU_MAX_SECONDS` to request the full 1,024 tokens' worth (~96 s) would
cut capacity to about 5 answers a day (estimated).

### When a call cannot be answered

KLEOS gets a status and falls back; it never sees an error:

| Situation | Status KLEOS sees |
| --- | --- |
| The calling account's daily quota is spent | `quota_exhausted`, with Hugging Face's stated wait when it gives one |
| No GPU within 60 s, or the Space's queue of 8 is full | `queue_unavailable` |
| The Space is asleep, building or loading the model | `starting` |
| The Space is paused, failed to build, or crashed (including a refused package at startup) | `disabled` |
| Hugging Face unreachable, or the connection drops mid-call | `starting` while connecting, `queue_unavailable` mid-call |
| The package repository unreachable at startup | The Space does not start: `starting`, then `disabled` |
| A call runs past `<PREFIX>_TIMEOUT` | `queue_unavailable`; the queued job is cancelled |
| A call runs past its requested GPU duration and ZeroGPU aborts it (a slow cold call, or a looping Logos answer) | `model_error`. If a Hermes smoke test shows this on cold calls, raise `HERMES_GPU_BASE_SECONDS` |

### Request bounds

| Bound | Hermes v0.0.6 | Logos v0.0.2 | Enforced |
| --- | --- | --- | --- |
| Fields | `messages`, `max_new_tokens`, `request_id` only; roles `system`, `user`, `assistant` | The same | Before any GPU work |
| First message | Any role | Must be `system` | Before any GPU work |
| Messages | ≤ 64 | ≤ 64 | Before any GPU work |
| Characters | ≤ 24,000 | ≤ 24,000 | Before any GPU work |
| Prompt tokens | ≤ 2,048 (`HERMES_MAX_INPUT_TOKENS`) | ≤ 2,048 (`LOGOS_MAX_INPUT_TOKENS`) | After tokenizing, before the GPU |
| Output tokens | ≤ 512 | ≤ 1,024 | Always; a request may lower it, never raise it |
| Concurrency | One generation at a time; queue of 8, then `queue_unavailable` | The same | Gradio queue |

Decoding is the frozen contract (greedy, at the model's token ceiling) and
cannot be changed per request.

### Security model

- **Authentication.** Every endpoint requires the model's key header
  (`X-Hermes-Key` or `X-Logos-Key`), compared in constant time against
  `<PREFIX>_API_KEY`; comma-separated keys allow rotation. The header is custom
  because Hugging Face uses `Authorization` for its own token. A Space refuses
  to start without the key. An unauthenticated request is answered
  `unauthorized` before any validation, tokenization or GPU scheduling, so it
  cannot spend anyone's quota.
- **Two tokens with separate jobs.** A Space's `HF_TOKEN` secret is a
  fine-grained, read-only token scoped to its package repository: it downloads
  the package and nothing else. The KLEOS backend's `<PREFIX>_HF_TOKEN` opens the
  private Space and is the account ZeroGPU charges. Neither token is ever sent to
  a browser.
- **Private Spaces.** A private Space is visible only to its account and
  callable only with a token that can read it. A free account can run a private
  ZeroGPU Space (verified 2026-09-23), and both KLEOS Spaces are private.
- **If a Space must be public** (a free-tier restriction), anyone can see its
  four files and the API's parameter names. They are the same files that are
  public in this repository, and none holds a secret, weight or dataset.
  Anyone can call the endpoints but gets `unauthorized` without the key. The
  residual risk is flooding: a burst of refused requests can briefly fill the
  queue of 8, and KLEOS sees `queue_unavailable` and falls back. It cannot reach
  the model or spend quota. Protected Spaces, which would remove this risk,
  require a PRO account.
- **Duplicating a Space does not copy the model.** A copy gets the four public
  files but none of the secret values, and the package lives in a private
  repository the copy cannot read.
- **Logs.** Request id, status, counts and timings only: never prompt, answer
  or trace text, headers or keys. An exception's type is logged, not its
  message, unless it is a ZeroGPU scheduling error, whose title carries no
  input. `spaces` prints a worker's traceback when generation crashes: code
  locations and the exception, not prompts.
- **No state.** Nothing is persisted between requests, as in the Docker service.

---

## Logos v0.0.2 on ZeroGPU

Logos v0.0.2 measured better than Hermes on the answerable benchmark items
(hypothesis H9, [result in experiments.md](experiments.md#h9-result--2026-10-06);
[Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md)).
It has its own private package repository and its own private Space, served by
the same code as Hermes. It is served as a Beta: on the Space's GPU, 8 of 9 checked answers reproduce the
T4 evaluation byte for byte
([Diagnosing a reproduction mismatch](#diagnosing-a-reproduction-mismatch)).

### What a Logos reply means

- **`text`** is the answer; **`reasoning`** is the trace written before it, or
  `null` if there was none. Only the answer was graded in H9.
- **`finish_reason: "length"`** when the 1,024-token budget filled or the
  thinking never closed (then `text` is empty). Either way the reply is cut and
  KLEOS falls back. Serving needs both signals: the evaluation's own
  `finish_reason` missed the 6 answers that looped after a closed trace (finding
  L-F7, in section 10 of the
  [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings)),
  and the budget alone misses a trace that ends without closing.
- **A request whose first message is not a system message** is refused with
  `invalid_request` before any GPU work. Every training and benchmark prompt had
  one; without it the Reasoning template adds its own default thinking prompt,
  which is not the model that was evaluated.
- **Nothing in the trace is logged**, as for prompts and answers.

The request and reply fields are specified in
[serving-api.md](serving-api.md#logos-contract-version-2).

---

## Verification status

What has been checked for each served model, with links to the dated records in
[serving-verification-records.md](experiments/serving-verification-records.md):

| Check | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Package integrity and identity | Verified 2026-09-23 on Colab; re-checked at every container and Space start | 9 files verified on the Hub, 2026-10-07; re-checked at every Space start |
| Exact reproduction on a T4 | 9 of 9 byte for byte, Colab, 2026-09-23 | Not run (the evaluation itself ran on T4s) |
| Exact reproduction on ZeroGPU | 9 of 9 byte for byte, 2026-09-23; prompt tokens and decisions 9 of 9; a repeated call identical | 8 of 9 byte for byte (answer and trace), 2026-10-07; prompt tokens 9 of 9; decisions 8 of 9; accepted as a Beta |
| Startup fits the Space's CPU RAM | Verified 2026-09-23; peak not recorded | Verified 2026-10-07; host process peak 30.6 GiB |
| Peak VRAM on ZeroGPU | 8.34 GiB (measured) | 8.86 GiB (measured) |
| Generation speed on ZeroGPU | ~12 tokens/s (measured) | 11.0 tokens/s warm median (measured) |
| Daily capacity per calling account | Roughly 20–30 answers (estimated) | About 9 answers (estimated); 7 admitted on day 1 |
| Docker image | Built and preflight-tested on CPU, 2026-09-23; not GPU-tested | Not applicable: Space only |
| Records | [Colab](experiments/serving-verification-records.md#verification-record--hermes-v006), [container](experiments/serving-verification-records.md#container), [ZeroGPU](experiments/serving-verification-records.md#verification-record--zerogpu) | [ZeroGPU, day 1 and day 2, decision](experiments/serving-verification-records.md#verification-record--logos) |

---

## Checklist before serving

### Every model, every host

1. `python scripts/verify_deployment_package.py --package <package> --expect-deployment-config <record>`
   exits 0.
2. The smoke test reproduces the frozen outputs on the serving hardware, or the
   difference has been diagnosed
   ([Diagnosing a reproduction mismatch](#diagnosing-a-reproduction-mismatch))
   and the decision recorded with its date.
3. The API key comes from a secret store, not from a file in the repository.
4. The consumer knows the measured limitations listed in `manifest.json`:
   - Hermes v0.0.6: prose, not JSON (`format_valid` 0.0000 on 349 held-out
     examples); abstains by scenario family rather than by evidence, so its
     error is one-sided overconfidence; unstable under rephrasing (10 of 15
     scenario groups changed decision).
   - Logos v0.0.2: prose, not JSON; greedy decoding loops on 6 of 349 evaluation
     answers; requests must start with a system message; the trace was not
     graded.
5. KLEOS falls back on every status other than `ready`
   ([serving-api.md](serving-api.md#fallback-rules)).

### Docker service (Hermes)

1. `hermes_smoke.py` reproduces the frozen responses inside the container on the
   GPU host ([First GPU start](#first-gpu-start-re-establish-the-99-match)).
2. The service is bound to loopback or a private network, behind TLS.
3. The model cache is a persistent volume.

### ZeroGPU Space (Hermes and Logos)

1. `zerogpu_smoke.py --record <record>` reports 9 of 9, or the difference has
   been diagnosed and the decision recorded. For Logos the reference is
   v0.0.2's `arm2_finetuned.json` (sha256 `0dd74661e1c5a9b1…c30eb5`).
2. The Space and its package repository are private, and
   `<PREFIX>_PACKAGE_REVISION` is a commit, not a branch.
3. `<PREFIX>_API_KEY` and a read-only, package-scoped `HF_TOKEN` are Space
   secrets; any `<PREFIX>_GPU_*` value is a Space variable.
4. The startup log ends with the model's `ready` line, and its `memory` and
   `load_s` values are recorded.
