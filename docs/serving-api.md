# Serving API

The contract between the KLEOS backend and the two served models, Hermes v0.0.6
and Logos v0.0.2 (Beta). The KLEOS backend and frontend live in a separate
repository. This repository defines the contract and ships a tested reference
client, `kleos_models.serving.client`, that implements the backend half. How
the models are packaged and hosted is described in
[deployment.md](deployment.md); finding and hypothesis ids follow the
conventions in [experiments/README.md](experiments/README.md).

Both models are optional. Every call returns a status. `ready` means the reply's
`text` is usable. Any other status means the default model answers that
request, and none of them is an error a KLEOS user should see: a spent free
quota, a sleeping Space or a busy GPU never becomes a KLEOS failure.

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Settings prefix | `HERMES_` | `LOGOS_` |
| Providers | `zerogpu` (ZeroGPU Space), `http` (Docker service) | `zerogpu` only |
| Key header | `X-Hermes-Key` | `X-Logos-Key` |
| Contract version | 1 | 2: adds `reasoning` |
| Output-token ceiling | 512 | 1,024 |
| First message | Any role | Must be `system` |

Sections up to [Logos: contract version 2](#logos-contract-version-2) describe
contract version 1 and Hermes; that section lists what differs for Logos.

---

## Architecture

```
Browser ──► KLEOS backend ──► HermesClient ──► ZeroGPU Space   (<PREFIX>_PROVIDER=zerogpu)
   ▲              │                        └─► Docker service  (HERMES_PROVIDER=http)
   │              ▼
   └── status + public message only        default model, whenever status != "ready"
```

The browser never talks to a model host, never sees a model credential, and
never sees a provider error. It gets what the KLEOS backend decides to show.

## Backend requirements

1. **A provider abstraction**, if the backend has none: one interface for
   "generate a reply", with the default model behind it.
2. **One provider per model** on that interface, wrapping `HermesClient` or a
   port of it, configured entirely by the variables below.
3. **A fallback provider:** the model first when it is enabled, the default
   model whenever the status is not `ready`.
4. **An availability indicator** in the frontend, fed by the backend with the
   status and nothing else ([Client states](#client-states)).

---

## Configuration

Backend environment only. None of these may reach client-side code: no
`NEXT_PUBLIC_`, `VITE_` or similar prefix, and not in any bundle. Each model
reads the same names under its own prefix, `HERMES_` or `LOGOS_`.

| Variable | Example | Secret | Meaning |
| --- | --- | --- | --- |
| `<PREFIX>_ENABLED` | `true` | No | Master switch. Any value other than `1`, `true`, `yes` or `on` makes every call return `disabled` without touching the network |
| `<PREFIX>_PROVIDER` | `zerogpu` | No | `zerogpu` (the Space) or `http` (the Docker service, Hermes only). Default `zerogpu` |
| `<PREFIX>_SPACE` | `<owner>/<space>` | No | zerogpu: the Space id. Required |
| `<PREFIX>_BASE_URL` | `https://hermes.internal` | No | http: the service's base URL. Required. The production value is never committed |
| `<PREFIX>_TIMEOUT` | `120` | No | Seconds one call may take in total, GPU queueing included. Default 120 |
| `<PREFIX>_MAX_OUTPUT_TOKENS` | `512` | No | KLEOS-side ceiling on output tokens. The host also enforces its own: 512 for Hermes, 1,024 for Logos |
| `<PREFIX>_HF_TOKEN` | (none) | Yes | zerogpu: a read token for the account KLEOS calls as. It opens the private Space, and ZeroGPU charges every call's GPU time to this account |
| `<PREFIX>_API_KEY` | (none) | Yes | The shared secret. Required. Sent in the key header to the Space and as `Authorization: Bearer` to the Docker service |

A misconfigured client (enabled, but a required variable missing or invalid)
logs the names of the problem variables once, never their values, and then
answers `disabled`. It does not crash KLEOS.

---

## Reference client

```python
from kleos_models.serving.client import HermesClient

hermes = HermesClient()  # reads HERMES_* once; create one per process, at startup

result = hermes.generate(
    [{"role": "user", "content": prompt}],
    request_id=request_id,  # optional; echoed back, used in logs on both sides
)
if result["status"] == "ready":
    answer = result["text"]
else:
    answer = default_model(prompt)  # always available; Hermes is not
```

| Method | Returns | GPU quota |
| --- | --- | --- |
| `generate(messages, *, max_new_tokens=None, request_id=None)` | A reply in the contract's shape. Never raises for availability | Spent by the calling account |
| `status()` | Readiness and identity, without generating | None |

`generate` is synchronous and blocks for up to `<PREFIX>_TIMEOUT` seconds. From
an async backend, run it on a worker thread (`anyio.to_thread.run_sync`,
`asyncio.to_thread`) so it does not stall the event loop.

A connection to a waking Space continues in the background: a call that is still
connecting after 10 s returns `starting`, and a later call reuses the connection
once it is made. Any request to a sleeping Space wakes it under standard Spaces
behavior, so `status()` should wake one; that is not tested with these Spaces.

To use the reference client as it is, install it with the provider's library:

```bash
pip install "kleos-models @ git+https://github.com/TejasNaik24/Kleos-Models.git@<commit sha>" \
    gradio_client==2.7.1   # zerogpu provider; the http provider needs httpx instead
```

Alternatively, port `serving/client.py` and `serving/status.py`. They are about
600 lines together, and `tests/test_hermes_client.py` states the behavior a port
must keep.

---

## Request

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_new_tokens": 256,
  "request_id": "kleos-7f3a..."
}
```

| Field | Rules |
| --- | --- |
| `messages` | 1–64 turns; roles `system`, `user`, `assistant`; ≤ 24,000 characters in total; ≤ 2,048 prompt tokens after the chat template |
| `max_new_tokens` | Optional, positive. May lower the 512 ceiling, never raise it. The reference client also applies `<PREFIX>_MAX_OUTPUT_TOKENS` |
| `request_id` | Optional, ≤ 128 characters. Generated when absent |

Nothing else is accepted, and an unknown field is refused. There is no user id,
workspace id, memory handle or tool list: the model needs none of them, and
anything sent is data handed to a process that does not need it. Send only the
turns the model must see.

Decoding is fixed: greedy, as evaluated. There is no temperature to set.

---

## Reply

Success:

```json
{
  "contract_version": 1,
  "ok": true,
  "status": "ready",
  "text": "...",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 412, "completion_tokens": 187},
  "model": {
    "name": "kleos-hermes",
    "version": "v0.0.6",
    "adapter_sha256": "dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32",
    "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
    "base_revision": "04d8a90549d23fc6bd7f642064003592df51e9b3"
  },
  "request_id": "kleos-7f3a...",
  "timings": {"prepare_s": 0.02, "gpu_call_s": 9.8, "gpu_generate_s": 7.9,
              "gpu_acquire_s": 1.9, "finish_s": 0.01, "total_s": 9.9},
  "diagnostics": {"cold_start": false, "worker_call_index": 4,
                  "device": "...", "compute_capability": "12.0",
                  "peak_vram_gib": 8.7, "cuda_runtime": "12.8"}
}
```

Any other outcome:

```json
{
  "contract_version": 1,
  "ok": false,
  "status": "quota_exhausted",
  "message": "Hermes' free GPU quota is currently exhausted. Please try again later.",
  "retryable": true,
  "retry_after_seconds": 49327,
  "request_id": "kleos-7f3a..."
}
```

The timing and token values above are illustrative.

| Field | Present in | Meaning |
| --- | --- | --- |
| `contract_version` | Every reply | 1 for Hermes, 2 for Logos, on success and on refusals |
| `ok` | Every reply | `true` exactly when `status` is `ready` |
| `status` | Every reply | One of the [statuses](#statuses) |
| `text` | Success | The answer. Prose, not JSON (see below) |
| `reasoning` | Success, version 2 only | The thinking trace, or `null` ([Logos](#logos-contract-version-2)) |
| `finish_reason` | Success | `stop`, or `length` when the answer used its whole token budget and was almost certainly cut off. Treat `length` as incomplete |
| `usage` | Success | `prompt_tokens` and `completion_tokens` |
| `model` | Success | `name`, `version`, `adapter_sha256`, `base_model`, `base_revision`: exactly which weights answered |
| `request_id` | Every reply | The caller's id, or a generated one |
| `timings` | Success, ZeroGPU only | `prepare_s`, `gpu_call_s`, `gpu_generate_s`, `gpu_acquire_s` (scheduling, queueing, worker start and weight transfer), `finish_s`, `total_s`. Empty through the Docker service |
| `diagnostics` | Success, ZeroGPU only | `cold_start`, `worker_call_index`, `device`, `compute_capability`, `peak_vram_gib`, `cuda_runtime`. Empty through the Docker service |
| `message` | Refusal | Safe to show a user. It never names a provider, a quota figure, an account or an internal error |
| `retryable` | Refusal | `true` for `starting`, `quota_exhausted` and `queue_unavailable` |
| `retry_after_seconds` | Refusal | Set only when Hugging Face stated a wait in its own quota message. Never estimated |

`text` is prose, not JSON. This is a measured limitation: `format_valid` was
0.0000 on all 349 held-out examples for Hermes v0.0.6, and for every model
compared in H9. Parse the text as prose; the extractors in
`kleos_models.evaluation.graders` are tested for this.

Through the Docker service, the client builds the same reply shape from the
service's JSON, with `usage` taken from its token counts.

`status()` answers, on success, with `contract_version`, `ok`, `status`,
`model`, `runtime` (quantization mode, double quant, compute dtype, attention
implementation, maximum sequence length), and, from a Space, `limits`
(`max_new_tokens`, `max_input_tokens`, `max_input_chars`, `max_messages`) and
the library `versions`. A Logos Space adds `"reasoning": true`.

---

## Statuses

The values are an API. They are pinned by tests and change only with
`contract_version`.

| `status` | Meaning | `retryable` | KLEOS backend | Client state |
| --- | --- | --- | --- | --- |
| `ready` | Answered | n/a | Use `text` | Available |
| `starting` | Space building, waking or loading the model | Yes | Fall back now; later requests try again | Waking |
| `quota_exhausted` | The calling account's free daily GPU quota is spent | Yes | Fall back, and stop calling the model until `retry_after_seconds` passes, or for a fixed back-off (30 minutes, for example) when it is absent | Quota exhausted |
| `queue_unavailable` | No GPU within the provider's wait, a full queue, or a timeout | Yes | Fall back now | Busy |
| `model_error` | Generation failed, or the reply broke the contract | No | Fall back; log `request_id`; alert if it repeats | Unavailable |
| `invalid_request` | Malformed, or over a bound | No | Fall back; it is a KLEOS bug, so log it | Unavailable |
| `unauthorized` | Wrong or missing `<PREFIX>_API_KEY` | No | Fall back; alert: configuration error | Unavailable |
| `disabled` | Switched off, misconfigured, paused or unreachable | No | Fall back silently | Unavailable |

### Fallback rules

1. Every status other than `ready` means the default model answers this
   request. Decide immediately; do not retry while the KLEOS request waits.
2. Do not retry `quota_exhausted` early. The quota resets 24 hours after the
   calling account's first GPU use, and polling does not bring that sooner.
3. `starting` and `queue_unavailable` clear on their own, and the next request
   can try again. A background `status()` call, which spends no quota, is the
   way to wake a sleeping Space.
4. Never show a raw provider message, an exception or a stack trace.

### Where statuses come from

The reference client does all of this; the tables are for anyone porting it.

ZeroGPU errors, raised by `spaces` 0.51.3 and classified inside the Space:

| ZeroGPU error | Status |
| --- | --- |
| "ZeroGPU quota exceeded": "You have exceeded your … quota … Try again in H:MM:SS" | `quota_exhausted`, with `retry_after_seconds` |
| "ZeroGPU quota exceeded": "Space app has reached its GPU limit", or "… runs limit" | `quota_exhausted` |
| "ZeroGPU queue timeout", or any "No GPU was available" | `queue_unavailable` |
| "ZeroGPU pending credits exceeded" (too much quota reserved by running tasks) | `queue_unavailable` |
| "ZeroGPU client error: Expired ZeroGPU proxy token" | `queue_unavailable` |
| "ZeroGPU API /schedule error" (the scheduler itself failed) | `queue_unavailable` |
| "ZeroGPU illegal duration", "ZeroGPU worker error", anything else | `model_error` |

Caller side, in the reference client:

| Condition | Status |
| --- | --- |
| Space paused, failed to build or crashed; not found with this token | `disabled` |
| Any other connection failure (typically a Space waking up) | `starting` |
| Still connecting after 10 s (the connection continues in the background) | `starting` |
| Call exceeded `<PREFIX>_TIMEOUT` (the job is cancelled) | `queue_unavailable` |
| Network error mid-call; Gradio "Queue is full" | `queue_unavailable` |
| A reply that is not the contract | `model_error` |

Docker service, by HTTP status:

| HTTP | Status |
| --- | --- |
| 200 | `ready` |
| 401 | `unauthorized` |
| 413, 422 | `invalid_request` |
| 500 | `model_error` |
| 503 | `starting` (model still loading) |
| 504 | `queue_unavailable` (generation timed out) |
| Unreachable | `disabled` |

---

## Client states

The frontend renders from the status the backend passes it, and nothing else.
A suggested small indicator, one state at a time:

| Indicator | Statuses | Copy |
| --- | --- | --- |
| ● Available | `ready` | "Hermes v0.0.6 available" |
| ◐ Waking GPU… | `starting` | "Hermes is starting a free GPU worker…" |
| ○ Free quota exhausted | `quota_exhausted` | "Hermes' free GPU quota has been reached for today. KLEOS will use another model until it resets." |
| ◐ Busy | `queue_unavailable` | "Hermes is temporarily busy. Try again later or continue with the default model." |
| ○ Unavailable | `model_error`, `invalid_request`, `unauthorized`, `disabled` | "Hermes is currently unavailable." |

For Logos, the copy names Logos ("Logos v0.0.2 available", "Logos' free GPU
quota has been reached for today…").

"For today" is a simplification: the quota resets 24 hours after the calling
account's first GPU use, not at midnight. The `message` field of a refusal holds
a shorter, KLEOS-neutral version of the same copy; the frontend should use the
copy above.

- **No invented countdown.** Show a time only when `retry_after_seconds` is
  present, since that is Hugging Face's own figure, and round it ("about 3
  hours"). Without it, say "later".
- **No "N requests remaining".** No provider exposes that number, and quota is
  measured in GPU seconds, not requests.
- In every state other than Available, the conversation continues on the default
  model. The state explains why the model was not used; it is not an error
  banner.

---

## Capacity and latency

Measured on the live Spaces by the smoke tests: Hermes on 2026-09-23 (10 calls,
answers 69–128 tokens long), Logos on 2026-10-07 (9 calls over two quota windows).
The samples are small, and the figures describe these deployments rather than
ZeroGPU in general. Details: [deployment.md](deployment.md#capacity-and-latency).

| | Hermes v0.0.6 | Logos v0.0.2 |
| --- | --- | --- |
| Warm call, median | 7.9 s | 19.7 s |
| Cold call (new GPU worker), median | 12.9 s | 29.9 s |
| Generation speed | ~12 tokens/s | 11.0 tokens/s (warm median) |
| GPU seconds per answer | 7–13 | 24–26 on average |
| Answers per day per calling account | Roughly 20–30 (estimated from the GPU seconds per call) | About 9 (estimated from 24–26 GPU s per answer and the 90 s admission margin); 7 admitted on day 1, in a window already partly used |

- One Hermes call in ten took 34 s, most of it waiting for a GPU.
- A long Hermes answer (512 tokens) takes about 43 s of generation at the
  measured speed (estimated).
- A Logos answer of 406 tokens took 33.6 s on a warm worker.
- A sleeping Space restarts before it answers. Measured at startup: about 46 s
  for the Hermes Space (package download, base load and packing) and about 59 s
  for the Logos Space (package download and load), plus a container start that
  was not timed in either case.

Every KLEOS request made with one `<PREFIX>_HF_TOKEN` shares that account's 5
GPU minutes a day, and Hermes and Logos share them as well when both use the
same account. On the free tier, plan the product around both models being
unavailable for much of the day, with the default model carrying the load and
KLEOS capping each model's daily use.

---

## Privacy and security

- Calls are server-to-server over HTTPS. Keys live in the backend's secret
  store; the browser never receives them.
- On ZeroGPU, prompts are processed on Hugging Face infrastructure. Treat that
  as a third-party processor in KLEOS's privacy disclosures.
- The model hosts keep nothing between requests and log no prompt, answer or
  trace text. KLEOS should do the same for model traffic and log `request_id`
  and `status` only.
- `<PREFIX>_API_KEY` rotates without downtime: set `old,new` on the Space or
  service, move KLEOS to `new`, then remove `old`.

---

## Testing KLEOS without a model

- `<PREFIX>_ENABLED=false`: every call returns `disabled`. KLEOS must behave
  normally in this state, and it is the first one to test.
- For the other statuses, inject a fake transport, as
  `tests/test_hermes_client.py` does:
  `HermesClient(settings, space_client_factory=lambda _: fake)` for the
  zerogpu provider, or `HermesClient(settings, http_client=fake)` for http.

---

## Logos: contract version 2

Logos v0.0.2 is Ministral 3 14B Reasoning fine-tuned to think before it
answers. It measured better than Hermes on the answerable benchmark items
(hypothesis H9, [result in experiments.md](experiments.md#h9-result--2026-10-06))
and is served as a Beta: on the Space's GPU, 8 of 9 checked answers reproduce
the evaluation byte for byte
([deployment.md](deployment.md#diagnosing-a-reproduction-mismatch)). It runs in
its own private ZeroGPU Space and speaks contract version 1 with one addition.
Everything above applies, with these differences.

### Configuration

The same variables with the `LOGOS_` prefix: `LOGOS_ENABLED`, `LOGOS_PROVIDER`
(`zerogpu` only: there is no Docker service for Logos), `LOGOS_SPACE`,
`LOGOS_TIMEOUT`, `LOGOS_MAX_OUTPUT_TOKENS`, `LOGOS_HF_TOKEN`, `LOGOS_API_KEY`.
The key is sent as `X-Logos-Key`. The reference client takes the prefix, the
header and the contract version, so that its own refusals (`disabled`,
`starting` and the rest) name Logos and carry version 2:

```python
from kleos_models.serving.client import HermesClient, HermesClientSettings

logos = HermesClient(
    HermesClientSettings.from_env(prefix="LOGOS", key_header="x-logos-key", contract_version=2)
)
```

Keep `LOGOS_TIMEOUT` at 120 s or more (the default is 120): one call may hold
the GPU for up to 60 s, plus queueing and waking.

### Request

- **The first message must be a system message.** Every training and benchmark
  prompt had one. Without it the Reasoning template adds its own default
  thinking prompt, which is not the model that was evaluated, so the Space
  answers `invalid_request` before any GPU work.
- **Leave `max_new_tokens` out.** The ceiling is 1,024, as evaluated. The trace
  and the answer share it, and an evaluated answer used 293 tokens on average
  (413 at the 95th percentile), so a lower budget cuts more answers.
- **Send earlier assistant turns as their answer (`text`) only.** A message has
  no field for a trace.

### Reply

`contract_version` is 2, on success and on refusals, and a success carries
`reasoning`:

```json
{
  "contract_version": 2,
  "ok": true,
  "status": "ready",
  "text": "...",
  "reasoning": "...",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 655, "completion_tokens": 284},
  "model": {
    "name": "kleos-logos",
    "version": "v0.0.2",
    "adapter_sha256": "e49724f6554db662769fbe9b9431105dbc12f8f75cecd6e0e00728ea3f04dc4f",
    "base_model": "mistralai/Ministral-3-14B-Reasoning-2512",
    "base_revision": "51f9210f3cd20f3452a80d5819d15dc61cc50630"
  },
  "request_id": "kleos-7f3a..."
}
```

`timings` and `diagnostics` are as for Hermes; the token counts are
illustrative.

- **`text` is the answer, and the only part that was graded.** Like Hermes', it
  is prose, not JSON.
- **`reasoning` is the trace the model wrote before answering,** or `null` when
  there was none. Show it, if at all, apart from the answer (for example, in a
  collapsible "How Logos decided" disclosure). Never merge it into the answer,
  and never present it as a checked explanation: it is the model's working, and
  nothing scored it.
- **`finish_reason` is `"length"`** when the 1,024-token budget filled or the
  thinking never closed. In the second case `text` is empty. Either way the
  answer is cut: fall back, as for a status other than `ready`. In the
  evaluation, 6 of 349 answers looped after their trace until the budget ran
  out (finding L-F7, in section 10 of the
  [Logos v0.0.2 run report](experiments/kleos-v007-ministral314breasoning-run1-report.md#10-findings)).
- **A looping answer may instead exceed the 60 s GPU cap.** ZeroGPU aborts it,
  and the reply is `model_error`: fall back.
- **The trace is user content.** Log neither it nor the answer, as for Hermes:
  `request_id` and `status` only.

### Messages and capacity

The refusal `message` and the client-state copy name Logos instead of Hermes
("Logos is starting a free GPU worker.", "Logos' free GPU quota is currently
exhausted. Please try again later.", and so on).

Each Logos call requests 60 s of GPU time, and ZeroGPU admits it only while
1.5 × that, 90 s, is left of the calling account's 300 s a day. On 2026-10-07, 7
answers fitted one day's quota (measured), shared with Hermes when both use the
same account. Measured latency is in [Capacity and latency](#capacity-and-latency).
