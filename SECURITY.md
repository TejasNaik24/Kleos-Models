# Security

## Reporting a vulnerability or a data exposure

Do **not** open a public issue. Report privately to the repository maintainers.

For an **accidental data exposure** — a credential, a private dataset, or personal
information committed here — treat it as urgent:

1. **Rotate the credential immediately.** Removing a file from the working tree
   does not remove it from git history, and this repository is public. Assume
   anything pushed has been captured.
2. Notify the maintainers privately.
3. Rewrite history only *after* rotating.
4. If a dataset was exposed, treat every record in it as compromised and follow
   the KLEOS incident process.

## This repository is public

It must never contain:

- real user conversations, memories, resumes or documents
- Supabase exports, connection strings, anon keys or service-role keys
- API keys, access tokens, cookies, JWTs or session identifiers
- personal contact information or private user identifiers
- production database dumps
- private evaluation traces

See [docs/privacy.md](docs/privacy.md) for the full boundary.

## No secrets in git

Secrets belong in environment variables, never in tracked files.

- **Local:** `.env`, which is git-ignored. Copy `.env.example`.
- **Colab:** Colab Secrets (sidebar key icon). Never paste a token into a cell —
  notebooks get shared, committed and screenshotted.
- **CI:** repository secrets.

Nothing in the data, validation, splitting, leakage or offline evaluation
pipelines requires a secret. `HF_TOKEN` is needed only to download gated models or
to publish an adapter.

## Automated checks

```bash
python scripts/check_no_private_data.py .            # full scan
python scripts/check_no_private_data.py --staged     # staged files only
python scripts/check_no_private_data.py --install-hook
```

Detects AWS, OpenAI, Anthropic, Hugging Face, GitHub, Slack and Google keys; JWTs;
private key blocks; Supabase URLs and service keys; bearer tokens; assigned
secrets; emails and phone numbers. It also fails on files that must not exist at
all — `.env`, anything under `data/raw/`, committed outputs.

This runs in CI as the **first** job, so a leaked secret fails the build before
any dependency is installed.

## Token handling

### Hugging Face

- Use **fine-grained** tokens with the minimum scope needed.
- Read-only for downloading gated models; write scope only when publishing.
- Never commit a token. Never put one in a config or a notebook.
- Rotate if a token is ever printed into a log or a notebook output.

### Supabase

Supabase credentials have **no place in this repository**. The training pipeline
has no Supabase dependency and must never gain one.

Dataset exports happen in the private repository, using a scoped, read-only,
purpose-created credential that is revoked afterwards.

**Never use production credentials for a training export.**

## Publishing safety

`scripts/publish_adapter.py` uploads from an allowlist and scans every eligible
file for secrets first. Training data, `.env`, checkpoints and logs are refused
categorically. Always `--dry-run` before publishing.

## Dependencies

Optional heavy dependencies (`torch`, `transformers`, `peft`, `bitsandbytes`) are
isolated behind extras, so the default install surface is small. Pin versions for
reproducible research runs.

`trust_remote_code` defaults to `false` in every shipped config. Enable it only
for a repository you have reviewed and trust — it executes arbitrary code from the
model repo at load time.

## Before every commit

```bash
python scripts/check_no_private_data.py .
git diff --cached
```
