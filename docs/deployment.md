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
python scripts/verify_deployment_package.py --package /path/to/hermes-v0.0.6
```

Re-hashes every recorded file, checks `adapter_config.json` pins the revision the
manifest names, and prints the model's identity and measured limitations. Needs
no GPU, no torch and no network, so it runs in CI and as a pre-start check. Add
`--check-base-revision` to confirm the commit with the Hub as well.

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
not implemented here. The contract it needs is above; the boundary it should keep
is configuration, not hard-coded provider logic:

```bash
MODEL_PROVIDER=hermes
HERMES_BASE_URL=https://...      # never commit the production URL
HERMES_API_KEY=...               # never commit the secret
```

Send only the messages the model needs. Anything else is data handed to a process
that does not need it.

---

## Checklist before serving

1. `verify_deployment_package.py` exits 0.
2. `hermes_smoke.py` reproduces the frozen responses, or the difference is
   understood and recorded.
3. `HERMES_API_KEY` is set from a secret store, not a file in the repository.
4. The service is bound to loopback or a private network, behind TLS.
5. The consumer knows the measured limitations in `manifest.json`: prose not
   JSON, one-sided overconfidence, and instability under rephrasing.
