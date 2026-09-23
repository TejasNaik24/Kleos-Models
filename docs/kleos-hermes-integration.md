# Integrating Hermes into KLEOS

For the KLEOS backend and frontend. This repository does not change KLEOS. It
defines the contract KLEOS integrates against and ships a tested reference
client, `kleos_models.serving.client`, that implements the backend half.

**The one rule: Hermes is optional.** Every call returns a status. `ready` means
use the text. Anything else means use the default model for this request, and
none of it is an error a KLEOS user should see. A spent free quota, a sleeping
Space or a busy GPU must never become a KLEOS failure.

---

## Shape

```
Browser ──► KLEOS backend ──► HermesClient ──► ZeroGPU Space  (HERMES_PROVIDER=zerogpu)
   ▲              │                        └─► Docker service (HERMES_PROVIDER=http)
   │              ▼
   └── status + public message only        default model, whenever status != "ready"
```

The browser never talks to Hermes, never sees a Hermes credential, and never
sees a provider error. It gets what the KLEOS backend decides to show.

---

## What KLEOS implements

1. **A provider abstraction** in the KLEOS backend, if it does not have one:
   one interface for "generate a reply", with the default model behind it.
2. **A Hermes provider** on that interface, wrapping `HermesClient` or a port
   of it. Configured entirely by the variables below.
3. **A fallback provider:** Hermes first when enabled; the default model
   whenever the status is not `ready`.
4. **A frontend availability indicator**, fed by the backend with the status
   and nothing else ([Frontend states](#frontend-states)).

---

## Configuration

Backend environment only. None of these may reach client-side code: no
`NEXT_PUBLIC_`, `VITE_` or similar prefix, and not in any bundle.

| Variable | Example | Secret | Meaning |
| --- | --- | --- | --- |
| `HERMES_ENABLED` | `true` | no | Master switch. Anything but `1/true/yes/on` means every call returns `disabled` without touching the network |
| `HERMES_PROVIDER` | `zerogpu` | no | `zerogpu` (the free Space) or `http` (the Docker service) |
| `HERMES_SPACE` | `owner/kleos-hermes` | no | zerogpu: the Space id |
| `HERMES_BASE_URL` | `https://hermes.internal` | no | http: the service's base URL. Never commit the production value |
| `HERMES_TIMEOUT` | `120` | no | Seconds one call may take in total, GPU queueing included |
| `HERMES_MAX_OUTPUT_TOKENS` | `512` | no | KLEOS-side ceiling on output tokens. The service enforces its own, 512, as well |
| `HERMES_HF_TOKEN` | — | **yes** | zerogpu: a read token for the account KLEOS calls as. It opens a private Space, and ZeroGPU charges every call's GPU time to this account |
| `HERMES_API_KEY` | — | **yes** | The shared secret. Sent as `X-Hermes-Key` to the Space, `Authorization: Bearer` to the service |

A misconfigured client (enabled, but a required variable missing) logs the
missing variable **names** once, then answers `disabled`. It does not crash
KLEOS.

---

## Calling Hermes

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
    answer = default_model(prompt)  # always available; Hermes never is
```

`generate` is synchronous and blocks up to `HERMES_TIMEOUT` seconds. From an
async backend, run it on a worker thread (`anyio.to_thread.run_sync`,
`asyncio.to_thread`) so it does not stall the event loop.

`hermes.status()` returns readiness and identity without generating, and spends
no GPU quota. Connecting to a sleeping Space should wake it, since any request
to a Space does. That is standard Spaces behaviour, **not yet tested** with this
Space.

To use the reference client as it is, install it with the provider's library:

```bash
pip install "kleos-models @ git+https://github.com/TejasNaik24/Kleos-Models.git@<commit>" \
    gradio_client==2.7.1   # zerogpu provider; the http provider needs httpx instead
```

Or port `serving/client.py` and `serving/status.py`. They are about 600 lines
together, and `tests/test_hermes_client.py` states the behaviour to keep.

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
| `max_new_tokens` | Optional. May lower the 512 ceiling, never raise it |
| `request_id` | Optional, ≤ 128 characters. Generated when absent |

Nothing else is accepted, and an unknown field is refused. That is deliberate.
There is no user id, workspace id, memory handle or tool list, because Hermes
needs none of them, and anything sent is data handed to a process that does not
need it. Send only the turns the model must see.

Decoding is fixed: greedy, as evaluated. There is no temperature to set.

---

## Response

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

Anything else:

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

- `finish_reason` is `"length"` when the answer used its whole token budget,
  which means it was almost certainly cut off. Treat it as incomplete.
- `text` is **prose, not JSON**. This is a measured limitation of v0.0.6
  (`format_valid` 0.0000 on 349 held-out examples). Parse it as prose; the
  extractors in `kleos_models.evaluation.graders` are tested for this.
- `timings` and `diagnostics` are filled by the ZeroGPU Space only. Through the
  Docker service they are empty, and `usage` is built from its token counts.
- `retry_after_seconds` is set **only** when Hugging Face stated a wait in its
  own quota message. It is never estimated.
- `message` is safe to show a user. It never names a provider, a quota figure,
  an account or an internal error.

---

## Statuses

The values are an API. They are pinned by tests and change only with
`contract_version`.

| `status` | Meaning | `retryable` | KLEOS backend | Frontend |
| --- | --- | --- | --- | --- |
| `ready` | Answered | — | Use `text` | Available |
| `starting` | Space building, waking or loading the model | yes | Fall back now; later requests will try again | Waking |
| `quota_exhausted` | The calling account's free daily GPU quota is spent | yes | Fall back, and stop calling Hermes until `retry_after_seconds` passes, or for a fixed back-off (30 min, say) when it is absent | Quota exhausted |
| `queue_unavailable` | No GPU within the provider's wait, a full queue, or a timeout | yes | Fall back now | Busy |
| `model_error` | Generation failed, or the reply broke the contract | no | Fall back; log `request_id`; alert if it repeats | Unavailable |
| `invalid_request` | Malformed or over a bound | no | Fall back; it is a KLEOS bug, so log it | Unavailable |
| `unauthorized` | Wrong or missing `HERMES_API_KEY` | no | Fall back; alert: configuration error | Unavailable |
| `disabled` | Switched off, misconfigured, paused or unreachable | no | Fall back silently | Unavailable |

**Fallback rules.**

1. Every status other than `ready` means the default model answers this
   request. Decide immediately; do not retry inside the user's request.
2. Do not retry `quota_exhausted` early. The quota resets 24 hours after the
   account's first GPU use of the day, and polling does not bring that sooner.
3. `starting` and `queue_unavailable` clear on their own. The next request can
   try again. A background `hermes.status()` call, which spends no quota, is the
   way to nudge a sleeping Space awake.
4. Never show a raw provider message, an exception or a stack trace.

---

## Where statuses come from

The reference client does all of this; it is listed for anyone porting it.

**ZeroGPU**, raised by `spaces` 0.51.3 and classified inside the Space:

| ZeroGPU error | Status |
| --- | --- |
| "ZeroGPU quota exceeded": "You have exceeded your … quota … Try again in H:MM:SS" | `quota_exhausted`, with `retry_after_seconds` |
| "ZeroGPU quota exceeded": "Space app has reached its GPU limit", or "… runs limit" | `quota_exhausted` |
| "ZeroGPU queue timeout", or any "No GPU was available" | `queue_unavailable` |
| "ZeroGPU pending credits exceeded" (too much quota reserved by running tasks) | `queue_unavailable` |
| "ZeroGPU client error: Expired ZeroGPU proxy token" | `queue_unavailable` |
| "ZeroGPU illegal duration", "ZeroGPU worker error", anything else | `model_error` |

**Caller side**, in the reference client:

| Condition | Status |
| --- | --- |
| Space paused, failed to build or crashed; not found with this token | `disabled` |
| Any other connection failure (typically a Space waking up) | `starting` |
| Still connecting after 10 s (connection continues in the background) | `starting` |
| Call exceeded `HERMES_TIMEOUT` (the job is cancelled) | `queue_unavailable` |
| Network error mid-call; Gradio "Queue is full" | `queue_unavailable` |
| A reply that is not the contract | `model_error` |

**Docker service**, by HTTP status:

| HTTP | Status |
| --- | --- |
| 200 | `ready` |
| 401 | `unauthorized` |
| 413, 422 | `invalid_request` |
| 500 | `model_error` |
| 503 | `starting` (model still loading) |
| 504 | `queue_unavailable` (generation timed out) |
| unreachable | `disabled` |

---

## Frontend states

The frontend renders from the status the backend passes it, and nothing else. A
small indicator, one state at a time:

| Indicator | Statuses | Copy |
| --- | --- | --- |
| ● Available | `ready` | "Hermes v0.0.6 available" |
| ◐ Waking GPU… | `starting` | "Hermes is starting a free GPU worker…" |
| ○ Free quota exhausted | `quota_exhausted` | "Hermes' free GPU quota has been reached for today. KLEOS will use another model until it resets." |
| ◐ Busy | `queue_unavailable` | "Hermes is temporarily busy. Try again later or continue with the default model." |
| ○ Unavailable | `model_error`, `invalid_request`, `unauthorized`, `disabled` | "Hermes is currently unavailable." |

"For today" is a simplification. The quota resets 24 hours after the calling
account's first GPU use, not at midnight.

The `message` field in a Hermes response holds a shorter, KLEOS-neutral
version of the same copy. The frontend should use the copy above.

- **No invented countdown.** Show a time only when `retry_after_seconds` is
  present, since that is Hugging Face's own figure, and round it ("about 3
  hours"). Without it, say "later".
- **No "N requests remaining".** No provider exposes that number, and quota is
  measured in GPU seconds, not requests.
- In every non-available state the conversation continues on the default model.
  The state explains why Hermes was not used. It is not an error banner.

---

## Capacity and latency

**ESTIMATED** until `scripts/zerogpu_smoke.py` runs on the real Space; see
[deployment.md](deployment.md#what-that-means-for-kleos). On free ZeroGPU,
every KLEOS request made with one `HERMES_HF_TOKEN` shares that account's 5 GPU
minutes a day: on the order of 5–20 Hermes answers. A warm answer takes about
5–40 s, and a waking Space several minutes. Plan the product around Hermes
being usually unavailable on the free tier, with the default model carrying
the load.

---

## Privacy and security

- Calls are server-to-server over HTTPS. Keys live in the backend's secret
  store; the browser never receives them.
- On ZeroGPU, prompts are processed on Hugging Face infrastructure. Treat that
  as a third-party processor in KLEOS's privacy disclosures.
- Hermes keeps nothing between requests and logs no prompt or response text.
  KLEOS should do the same for Hermes traffic: log `request_id` and `status`.
- Rotate `HERMES_API_KEY` without downtime: set `old,new` on the Space or
  service, move KLEOS to `new`, then remove `old`.

---

## Testing KLEOS without Hermes

- `HERMES_ENABLED=false`: every call returns `disabled`. KLEOS must behave
  normally, and that is the first thing to test.
- Inject a fake transport for the other statuses, as
  `tests/test_hermes_client.py` does:
  `HermesClient(settings, space_client_factory=lambda _: fake)`.
