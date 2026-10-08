# Security policy

## Scope

This policy covers the code, configuration and documentation in this
repository, including the serving code it ships (`src/kleos_models/serving/`,
`docker/`, `deploy/`), and any private data or credential exposed through it.
The KLEOS application, the private dataset repository, and the private Hugging
Face Spaces and package repositories that serve the KLEOS models are operated
separately; a report about one of them goes through the same channel.

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. Do not open
a public issue, discussion or pull request for a vulnerability or a data
exposure.

A useful report includes what is affected (file, commit or component), how to
reproduce it, the impact, and whether it has been shared anywhere else.

| Step | Timeline |
| --- | --- |
| Acknowledgment of the report | Within 7 days |
| A remediation plan, or a reasoned decision that no change is needed | Within 30 days |

## Accidental data exposure

For a credential, a private dataset or personal information committed here,
treat it as urgent:

1. **Rotate the credential immediately.** Removing a file from the working tree
   does not remove it from git history, and this repository is public. Assume
   anything pushed has been captured.
2. **Report it privately** through the channel above.
3. **Rewrite history only after rotating.**
4. **If a dataset was exposed,** treat every record in it as compromised:
   rotate any credential it contains, remove the data from the repository,
   rewrite history if it was pushed, and record the incident in the changelog.

## What this repository must never contain

- real user conversations, memories, resumes or documents
- Supabase exports, connection strings, anon keys or service-role keys
- API keys, access tokens, cookies, JWTs or session identifiers
- personal contact information or private user identifiers
- production database dumps
- private evaluation traces, benchmark files or result files derived from the
  private dataset
- model weights, adapters or deployment packages

[docs/privacy.md](docs/privacy.md) defines the full public/private boundary.

## Secrets handling

Secrets belong in environment variables or a platform's secret store, never in
tracked files. Nothing in the data, validation, splitting, leakage or offline
evaluation pipelines needs a secret.

| Where | How secrets are supplied |
| --- | --- |
| Local machine | `.env`, which is git-ignored; copy `.env.example`. Hub logins via `hf auth login` are stored outside the repository |
| Google Colab | Colab Secrets (the key icon in the sidebar). Never paste a token into a cell: notebooks get shared, committed and screenshotted |
| Kaggle | Kaggle Secrets (Add-ons, then Secrets). The Logos v0.0.2 notebooks need no token, because the base model is ungated |
| CI | None needed; the workflow uses no secrets |
| Hermes ZeroGPU Space | Space secrets: `HF_TOKEN`, `HERMES_API_KEY`, `HERMES_PACKAGE_REPO`, `HERMES_PACKAGE_REVISION` |
| Logos ZeroGPU Space | Space secrets: `HF_TOKEN`, `LOGOS_API_KEY`, `LOGOS_PACKAGE_REPO`, `LOGOS_PACKAGE_REVISION`. `LOGOS_GPU_TOKENS_PER_SECOND` is a tuning variable, not a secret |
| Docker service | `HERMES_API_KEY_FILE` and `HF_TOKEN_FILE`, pointing at mounted secret files, so the values never appear in `docker inspect`. Setting both forms of one secret is refused |
| KLEOS backend | Holds the calling token and the model's API key server side; neither is ever sent to a browser |

`HF_TOKEN` is needed only to download a gated base model (Ministral-8B), to
upload a deployment package or publish an adapter, and inside a Space to read
its private package repository.

### Hugging Face tokens

- Use fine-grained tokens with the minimum scope needed.
- Read-only for downloads. A Space's `HF_TOKEN` is read-only and scoped to its
  package repository. Write scope only for an upload.
- Never commit a token, and never put one in a config, a notebook or a command
  line.
- Rotate a token that was ever printed into a log or a notebook output.

### Supabase

Supabase credentials have no place in this repository. The training pipeline has
no Supabase dependency and must never gain one. Dataset exports happen in the
private repository, with a scoped, read-only, purpose-created credential that is
revoked afterward. Production credentials are never used for a training export.

## Automated checks

```bash
python scripts/check_no_private_data.py .            # full scan
python scripts/check_no_private_data.py . --strict   # warnings fail too
python scripts/check_no_private_data.py --staged     # staged files only
python scripts/check_no_private_data.py --install-hook
```

The scanner detects AWS, OpenAI, Anthropic, Hugging Face, GitHub, Slack and
Google keys; JWTs; private key blocks; Supabase URLs and service keys; bearer
tokens; assigned secrets; and email addresses and phone numbers (reported as
warnings, which fail only with `--strict`). It also fails on files that must not
exist at all: `.env` and other environment files, anything under `data/raw/` or
`data/processed/`, key material (`.pem`, `.key`, `.p12`), and committed outputs.
This file and `docs/privacy.md` are exempt by name, because they describe the
patterns.

The scan runs as the first CI job and needs no dependencies, so a leaked secret
fails the build before anything is installed.

## Publishing and serving safety

- `scripts/publish_adapter.py` uploads from an allowlist and scans every eligible
  file for secrets first. Training data, `.env`, checkpoints and logs are refused
  outright. Run it with `--dry-run` first.
- `scripts/upload_deployment_package.py` refuses files the package manifest does
  not list, and checks the size and hash of every uploaded file on the Hub.
- Every serving endpoint requires a key, compared in constant time: a bearer
  token for the Docker service, and the model's key header (`X-Hermes-Key` or
  `X-Logos-Key`) on a ZeroGPU Space, because Hugging Face's proxy uses
  `Authorization` for its own token. An unauthenticated request is refused
  before validation, tokenization or any GPU work. A service without a key
  refuses to start, unless `HERMES_ALLOW_UNAUTHENTICATED=1` is set for a local
  experiment on loopback.
- Service logs carry request ids, statuses, counts and timings, never prompts,
  answers, reasoning traces, headers or keys.
- The Docker image runs as a non-root user (uid 10001) and is built from an
  allowlisted context that excludes `.env`, `data/`, `outputs/` and `.git`.

## Dependencies

Heavy dependencies (`torch`, `transformers`, `peft`, `bitsandbytes`) sit behind
extras, so the default install surface is small. The serving images pin every
package (`docker/requirements-hermes.txt`, the Space requirement templates), and
tests fail if the model-runtime pins drift apart.

`trust_remote_code` is `false` in every shipped model config. Enable it only for
a repository that has been reviewed and is trusted: it executes arbitrary code
from the model repository at load time.

## Before every commit

```bash
python scripts/check_no_private_data.py .
git diff --cached
```
