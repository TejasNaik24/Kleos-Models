# Privacy and the public/private boundary

**This repository is public. It must never contain real user data.**

## What lives where

| This public repository | The private `kleos-training-data` repository |
| --- | --- |
| Code, schemas, configs | Raw conversations and memory records |
| Synthetic development fixtures | Sanitized real examples |
| Dataset manifests and loaders | Supabase exports |
| Validation and evaluation code | Private resumes and documents |
| Aggregate metrics | Private evaluation traces |
| Model cards, research docs | Anything identifying a person |

## Never in this repository

- Real conversations, memory records, resumes or documents
- Supabase exports, connection strings, anon keys or service-role keys
- API keys, access tokens, cookies, JWTs, session identifiers
- Personal contact information or private user identifiers
- Production database dumps
- Private evaluation traces

## The boundary in practice

The private repository produces a versioned artifact:

```
dataset/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

This repository consumes it **by path**:

```bash
python scripts/train.py --config configs/training/qlora_small.yaml \
                        --dataset /path/to/private/dataset
```

Neither repository imports the other. The public repo does not know or care where
the dataset came from, and the private artifact is never copied here.

There is **no Supabase dependency** in the training pipeline and there will not be
one. Training operates on exported, versioned artifacts — which keeps the research
reproducible and stops the public repo becoming coupled to production
infrastructure.

## Safeguards

### 1. `.gitignore` — deny-by-default under `data/`

Everything under `data/` is ignored unless explicitly allowlisted. Only three
things are permitted: `data/README.md`, `data/schema/` (JSON Schema, no data) and
`data/examples/` (synthetic fixtures).

This is an allowlist on purpose. A blocklist — "ignore `data/raw/`, ignore
`*.jsonl`" — only blocks the filenames someone thought of in advance. A real
export dropped in as `data/my_export.jsonl` or `data/memories.json` would sail
straight into a public commit. Deny-by-default means a new file has to be
*deliberately* permitted before it can ever be published.

Also ignored: `outputs/`, `checkpoints/`, `wandb/`, `reports/`, `.env`, and key
material.

`tests/test_gitignore.py` runs git against a throwaway repository containing the
real `.gitignore` and asserts both halves — that private paths are blocked, and
that the schema and fixtures are still tracked. Two subtleties it guards against,
both of which were real bugs:

- **Trailing comments are not supported.** `!data/schema/  # keep` is a literal
  pattern matching nothing, so the negation silently does nothing.
- **A file cannot be re-included if its parent directory is excluded.**
  `outputs/` followed by `!outputs/.gitkeep` drops the `.gitkeep`; the rule must
  be `outputs/*`.

A safety net, not a substitute for judgement.

### 2. The private-data scanner

```bash
python scripts/check_no_private_data.py .
python scripts/check_no_private_data.py --staged      # pre-commit mode
python scripts/check_no_private_data.py --install-hook
```

Detects AWS/OpenAI/Anthropic/HF/GitHub/Slack/Google keys, JWTs, private key
blocks, Supabase URLs and service keys, bearer tokens, assigned secrets, email
addresses and phone numbers. Also flags files that must not exist at all
(`.env`, anything under `data/raw/`, committed outputs).

Runs in CI on every push.

### 3. Dataset validation

`validate_dataset.py` scans example content for the same patterns. A dataset that
contains an email address fails validation before it can be trained on.

### 4. Publishing allowlist

`publish_adapter.py` uploads only named artifacts — adapter weights, tokenizer,
config, manifest, metrics, model card — and scans each one before upload. Training
data, `.env`, checkpoints and logs are refused categorically.

### 5. Logging discipline

Structured logs record scalars, identifiers and configuration. **Example text is
never logged.** Helpers that accept free text redact it to a length and a digest.

`--show-masking` prints token counts, not content, so it is safe against a private
dataset.

## Policy, not private facts

The deepest privacy protection here is not a scanner — it is what the training
data teaches.

The model learns **decision policies**: weigh a nearer deadline against evidence
quality, route a request to the right tool, prefer the more reliable source when
records conflict.

It must not learn **facts about a person**: where someone works, what their resume
says, what they discussed last Tuesday. Those belong in KLEOS's retrieval and
memory layers, where they can be updated, scoped and deleted.

This matters beyond privacy: a memorized fact goes stale, cannot be corrected
without retraining, and does not generalize to anyone else.

The validator flags likely fact-teaching phrasing, but this rule is ultimately
enforced by whoever writes the data. See [data-contract.md](data-contract.md).

## Handling secrets

- Local: `.env` (git-ignored). Copy from `.env.example`.
- Colab: **Colab Secrets**, never a literal token in a cell.
- CI: repository secrets.

Nothing in the data, validation, splitting, leakage or offline evaluation
pipelines requires a secret. `HF_TOKEN` is needed only for gated model downloads
and publishing.

**Never use production credentials for a training export.** Use a scoped,
read-only, purpose-created credential and revoke it afterwards.

## If something leaks

1. **Rotate the credential immediately.** Removing it from the working tree does
   not remove it from git history, and history is public.
2. Report it — see [SECURITY.md](../SECURITY.md).
3. Rewrite history only after rotating; assume anything pushed was captured.
4. If a dataset leaked, treat every record in it as exposed.

## Before you commit

```bash
python scripts/check_no_private_data.py .
git status                     # anything unexpected staged?
git diff --cached              # read it
```

Install the hook so this happens automatically:

```bash
python scripts/check_no_private_data.py --install-hook
```
