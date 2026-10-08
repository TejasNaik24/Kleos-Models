# Privacy and the public/private boundary

**This repository is public. It must never contain real user data.** This page is
the single statement of that rule; other documents link here.

## What lives where

| This public repository | Private storage, outside this repository and [Kleos-Training-Data](https://github.com/TejasNaik24/Kleos-Training-Data) |
| --- | --- |
| Code, schemas, configs | Raw conversations and memory records |
| Synthetic development fixtures | Sanitized real examples |
| Dataset manifests and loaders | Dataset releases, and any Supabase export |
| Validation and evaluation code | Private resumes and documents |
| Aggregate metrics in reports | Per-example results with model responses |
| Model cards, research documents | Trained adapters, deployment packages, anything identifying a person |

Trained adapters and deployment packages live in private storage and private
Hugging Face repositories ([publishing.md](publishing.md)). Run reports in
[experiments/README.md](experiments/README.md) cite them by hash and by
placeholder paths such as `<private storage>/outputs/<experiment-id>/`.

## Never in this repository

- Real conversations, memory records, resumes or documents.
- Supabase exports, connection strings, anon keys or service-role keys.
- API keys, access tokens, cookies, JWTs, session identifiers.
- Personal contact information or private user identifiers.
- Production database dumps.
- Private evaluation traces, including stored model responses to private
  benchmark prompts.

## The boundary in practice

The [Kleos-Training-Data](https://github.com/TejasNaik24/Kleos-Training-Data) pipeline produces a versioned release directory, kept in private storage outside both repositories:

```
<path to release>/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

This repository consumes it by path:

```bash
python scripts/train.py --config configs/training/kleos_hermes_v006.yaml \
    --dataset <path to release>
```

Neither repository imports the other. The public code does not know where the
dataset came from, and the private artifact is never copied here. On Colab the
release sits on a private Drive folder; on Kaggle it is a private Kaggle dataset,
copied to `/tmp`, which Kaggle never saves ([kaggle.md](kaggle.md)).

The training pipeline has no Supabase dependency. It operates on exported,
versioned artifacts, which keeps the research reproducible and keeps the public
code decoupled from production infrastructure.

## Safeguards

### 1. `.gitignore`: deny by default under `data/`

Everything under `data/` is ignored unless explicitly allowed. Only three things
are permitted: `data/README.md`, `data/schema/` (JSON Schema, no data) and
`data/examples/` (synthetic fixtures). Split files (`train.jsonl`,
`validation.jsonl`, `test.jsonl`) are ignored anywhere else in the tree.

The allowlist is deliberate. A blocklist ("ignore `data/raw/`, ignore `*.jsonl`")
blocks only the filenames someone thought of in advance; a real export dropped in
as `data/my_export.jsonl` or `data/memories.json` would pass into a public
commit. Deny by default means a new file has to be permitted deliberately before
it can be published.

Also ignored: `outputs/`, `checkpoints/`, `wandb/`, `reports/`, `.env`, and key
material.

`tests/test_gitignore.py` runs git against a throwaway repository containing the
real `.gitignore` and asserts both halves: private paths are blocked, and the
schema and fixtures are still tracked. It guards against two subtleties, both of
which were real bugs:

- **Trailing comments are not supported.** `!data/schema/  # keep` is a literal
  pattern matching nothing, so the negation does nothing.
- **A file cannot be re-included if its parent directory is excluded.**
  `outputs/` followed by `!outputs/.gitkeep` drops the `.gitkeep`; the rule must
  be `outputs/*`.

The ignore file is a safety net, not a substitute for judgment.

### 2. The private-data scanner

```bash
python scripts/check_no_private_data.py .
python scripts/check_no_private_data.py --strict .    # warnings fail too
python scripts/check_no_private_data.py --staged      # pre-commit mode
python scripts/check_no_private_data.py --install-hook
```

It detects AWS, OpenAI, Anthropic, Hugging Face, GitHub, Slack and Google keys,
JWTs, private key blocks, Supabase URLs and service keys, bearer tokens and
assigned secrets (errors), and email addresses and phone numbers (warnings, which
fail only with `--strict`). It also flags files that must not exist at all:
`.env`, anything under `data/raw/`, committed outputs. It runs in CI on every
push.

A short list of files is exempt (`SELF_EXEMPT`), because they define or test the
patterns or document them: the scanner itself, the dataset and upload scanners
and their tests, `.env.example`, `SECURITY.md` and this page.

### 3. Dataset validation

`validate_dataset.py` scans example content for the same patterns. A dataset
that contains an email address fails validation before it can be trained on.

### 4. Publishing allowlist

`publish_adapter.py` uploads only named artifacts: adapter weights and config,
the effective config, the manifest, metrics and the generated model card. It
scans each text file before upload. No tokenizer file is published (the tokenizer
comes from the pinned base model), and training data, `.env`, checkpoints and logs
are refused categorically ([publishing.md](publishing.md#what-gets-uploaded)).
Deployment packages go only to private repositories, and
`upload_deployment_package.py` refuses a repository that is or would be public.

### 5. Logging discipline

Structured logs record scalars, identifiers and configuration. Example text is
never logged; helpers that accept free text redact it to a length and a digest.
`--show-masking` prints token counts, not content, so it is safe against a
private dataset. The inference service is written not to log prompts, answers or
reasoning traces.

## Policy, not private facts

The deepest privacy protection is what the training data teaches.

The model learns decision policies: weigh a nearer deadline against evidence
quality, route a request to the right tool, prefer the more reliable source when
records conflict. It must not learn facts about a person: where someone works,
what their resume says, what they discussed last Tuesday. Those belong in KLEOS's
retrieval and memory layers, where they can be updated, scoped and deleted.

This matters beyond privacy: a memorized fact goes stale, cannot be corrected
without retraining, and does not generalize to anyone else. The validator flags
likely fact-teaching phrasing, but the rule is ultimately enforced by whoever
writes the data ([data-contract.md](data-contract.md#policy-not-private-facts)).

## Handling secrets

- Local: `.env` (git-ignored), copied from `.env.example`.
- Colab: Colab Secrets, never a literal token in a cell.
- Kaggle: the notebooks need no token, because their base model is ungated.
- CI: repository secrets.
- Serving: the Space's secrets (`HERMES_*`, `LOGOS_*`), or for Docker the
  `*_FILE` form, which reads a value from a mounted file.

Nothing in the data, validation, splitting, leakage or offline evaluation
pipelines needs a secret. `HF_TOKEN` is needed only for gated model downloads,
publishing and private package repositories. [SECURITY.md](../SECURITY.md) covers
secrets for serving in detail.

Never use production credentials for a training export. Use a scoped, read-only,
purpose-created credential and revoke it afterwards.

## If something leaks

1. **Rotate the credential immediately.** Removing it from the working tree does
   not remove it from git history, and history is public.
2. Report it, as [SECURITY.md](../SECURITY.md) describes.
3. Rewrite history only after rotating; assume anything pushed was captured.
4. If a dataset leaked, treat every record in it as exposed.

## Before committing

```bash
python scripts/check_no_private_data.py .
git status                     # anything unexpected staged?
git diff --cached              # read it
```

Install the hook so this happens automatically:

```bash
python scripts/check_no_private_data.py --install-hook
```
