# Deploy Logos v0.0.2 to a ZeroGPU Space

This runbook deploys the frozen Logos v0.0.2 artifact to its own private
Hugging Face ZeroGPU Space and checks, over two quota windows, that the Space
reproduces the frozen evaluation's answers and thinking traces. The operator
runs every step, in order. The last deployment ran in two quota windows on
2026-10-07; its record, including the 8 of 9 result and the decision to serve
Logos as a Beta, is in
[serving-verification-records.md](../experiments/serving-verification-records.md#verification-record--logos).
What each check means is explained in [deployment.md](../deployment.md), and
the conventions for finding and decision ids are in
[experiments/README.md](../experiments/README.md).

Logos is served by the same code as Hermes, from its own record
(`configs/deployment/kleos_logos_v002.yaml`), Space folder
(`deploy/zerogpu-space-logos`) and `LOGOS_*` settings
([Serving profiles](../deployment.md#serving-profiles)). Deploying it leaves the
Hermes Space untouched.

No step costs money. If any screen asks for payment details, stop and note
which screen asked: the free tier needs none. No token is ever typed into a
file or onto a command line.

## Before starting

| Requirement | Detail |
| --- | --- |
| Hugging Face account | In good standing: verified email, older than 30 days. A free account may host up to 2 ZeroGPU Spaces, and with Hermes deployed this is the second |
| Training output | The finished Kaggle training notebook's output for `kleos-v007-ministral314breasoning-run1` |
| Frozen evaluation results | Logos v0.0.2's `arm2_finetuned.json`, sha256 `0dd74661e1c5a9b1…c30eb5`, from the evaluation notebook's output |
| Dataset release | kleos-policy-v0.0.7 at `<path to release>`, to rebuild the benchmark |
| This repository | Installed (`pip install -e .` and `pip install huggingface_hub`), with the commit to deploy pushed to GitHub |
| Write token | A Hugging Face write token, stored once with `hf auth login` (pasted only into that prompt) |

Placeholders: `<owner>/<package-repo>` is the private package repository,
`<owner>/<space>` the Space, `<private storage>` a private folder outside this
repository and outside the dataset repository, and `<commit sha>` the commit of
this repository the Space installs.

## Build and upload the package

1. **Push the commit the Space will install.** The Space installs
   `kleos-models` from GitHub at an exact commit, and
   `scripts/stage_zerogpu_space.py` uses `HEAD` and refuses a dirty working
   tree. Commit, push, and note the commit as `<commit sha>`.

2. **Download the training output.** On Kaggle, open the training notebook's
   finished version, then Output, then Download. Unzip it into
   `<private storage>`, so that the run is at
   `<private storage>/outputs/kleos-v007-ministral314breasoning-run1`.

3. **Check the record against the run.**

   ```bash
   python scripts/fill_deployment_record.py \
       --run <private storage>/outputs/kleos-v007-ministral314breasoning-run1 \
       --record configs/deployment/kleos_logos_v002.yaml
   ```

   The script refuses a run whose best checkpoint is not `checkpoint-175`, or
   whose `config_hash`, dataset hash or experiment id differ from the record's,
   and it refuses to replace a value that is already there and different. The
   v0.0.2 record is already complete (adapter `e49724f6554db662…`), so this run
   only confirms it. For a record that still has `null` values, review the
   printed values and run the same command with `--write`; then commit and push
   the record, and use that commit as `<commit sha>`.

4. **Build and verify the package.** No GPU is needed.

   ```bash
   python scripts/build_deployment_package.py \
       --run <private storage>/outputs/kleos-v007-ministral314breasoning-run1 \
       --deployment-config configs/deployment/kleos_logos_v002.yaml \
       --model-config configs/models/ministral3_14b_reasoning.yaml \
       --output <private storage>/packages/logos-v0.0.2
   python scripts/verify_deployment_package.py \
       --package <private storage>/packages/logos-v0.0.2 \
       --expect-deployment-config configs/deployment/kleos_logos_v002.yaml
   ```

   Expected: exit 0, with adapter `e49724f6554db662…` and base
   `mistralai/Ministral-3-14B-Reasoning-2512` at `51f9210f…` in the printed
   identity. If it prints `DO NOT SERVE THIS PACKAGE`, stop: the package is not
   the frozen artifact. Rebuild it from the frozen run, and never edit the
   record to match a package.

5. **Upload the package to a private repository**, dry run first.

   ```bash
   python scripts/upload_deployment_package.py \
       --package <private storage>/packages/logos-v0.0.2 \
       --repo <owner>/<package-repo> \
       --deployment-config configs/deployment/kleos_logos_v002.yaml --dry-run
   python scripts/upload_deployment_package.py \
       --package <private storage>/packages/logos-v0.0.2 \
       --repo <owner>/<package-repo> \
       --deployment-config configs/deployment/kleos_logos_v002.yaml --create
   ```

   The script uses the stored login. It refuses a public repository, uploads
   only the files the manifest lists, then re-lists the repository and checks
   every file's size and hash. Note the commit it prints (40 hex characters):
   it becomes `LOGOS_PACKAGE_REVISION`.

## Create and push the Space

6. **Create the Space** at huggingface.co/new-space: SDK Gradio, hardware
   ZeroGPU, visibility private.

7. **Set four secrets** in the Space's settings:

   | Secret | Value |
   | --- | --- |
   | `HF_TOKEN` | A new fine-grained token with read access to `<owner>/<package-repo>` only |
   | `LOGOS_API_KEY` | A new random secret, for example from `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The KLEOS backend gets the same value |
   | `LOGOS_PACKAGE_REPO` | `<owner>/<package-repo>` |
   | `LOGOS_PACKAGE_REVISION` | The commit step 5 printed |

   Leave the optional Space variables (`LOGOS_GPU_*`, `LOGOS_MAX_INPUT_TOKENS`,
   `LOGOS_MAX_NEW_TOKENS`) unset. Never set `LOGOS_ALLOW_UNAUTHENTICATED` on a
   Space.

8. **Stage the Space and review it.**

   ```bash
   python scripts/stage_zerogpu_space.py \
       --record configs/deployment/kleos_logos_v002.yaml --out /tmp/logos-space
   ```

   Read the four staged files: `README.md`, `app.py`, `requirements.txt` and
   `logos_record.yaml`. `requirements.txt` must pin `kleos-models` to
   `<commit sha>`, and the README must preload the six Ministral 3 shards at
   `51f9210f…`; the script refuses one that does not.

9. **Push the Space.**

   ```bash
   python scripts/stage_zerogpu_space.py \
       --record configs/deployment/kleos_logos_v002.yaml --out /tmp/logos-space-push \
       --push --space <owner>/<space>
   ```

## Check the Space

10. **Watch the build and startup logs.** Startup downloads the package,
    verifies it, and quantizes the 14B text tower on the CPU; the Space allows
    up to 1 hour (`startup_duration_timeout`). Success ends with:

    ```
    Logos ready: kleos-logos v0.0.2 adapter=… base=mistralai/Ministral-3-14B-Reasoning-2512@51f9210f3cd2 …
    ```

    Record `download_s`, `load_s` and `memory`. On 2026-10-07 they were
    `download_s=1.9`, `load_s=57.1` and a host process peak of 30.6 GiB.

    - If the startup log shows the package refused (an integrity or identity
      error), stop: the Space is not serving the frozen artifact. Check
      `LOGOS_PACKAGE_REPO` and `LOGOS_PACKAGE_REVISION` first, and rebuild the
      package only if they are right.
    - If the startup log shows an out-of-memory error, stop: the host could not
      quantize the text tower on its CPU. The only fallback, loading inside the
      GPU call, spends quota on every cold start. That is a design change:
      decide and record it before any code changes.

11. **Rebuild the benchmark** and check its hash.

    ```bash
    python scripts/build_benchmark.py --dataset <path to release> --output /tmp/bench
    shasum -a 256 /tmp/bench/benchmark.jsonl      # must be a11ffad75f51…b266
    ```

    If the hash differs, stop: the release is not the one Logos was evaluated
    on.

12. **Check the reference.** Confirm that the evaluation results file is Logos
    v0.0.2's:

    ```bash
    shasum -a 256 <private storage>/<eval>/arm2_finetuned.json   # must be 0dd74661e1c5a9b1…c30eb5
    ```

    Logos v0.0.1's results have the same file name (sha256 `09243630…`). The
    smoke test refuses that file, and any reference generated with another
    token budget or without traces, before it contacts the Space.

### Smoke test over two quota windows

About 7 Logos calls fit one day's free quota, so the nine-prompt smoke test
runs over two quota windows. Each answer costs 24–26 GPU seconds on average
(measured on 2026-10-07), and a call is admitted only while 90 s of quota
remain. The calling account (`$HF_TOKEN`, or the stored login) pays.

13. **Day 1: run as many prompts as the quota admits.**

    ```bash
    pip install gradio_client==2.7.1
    read -s LOGOS_API_KEY && export LOGOS_API_KEY     # paste the Space's key; not echoed
    python scripts/zerogpu_smoke.py --space <owner>/<space> \
        --record configs/deployment/kleos_logos_v002.yaml \
        --benchmark /tmp/bench/benchmark.jsonl \
        --reference <private storage>/<eval>/arm2_finetuned.json \
        --warm-repeats 0 --output <private storage>/<eval>/logos-smoke-day1.json
    ```

    The script checks identity first (adapter, base, runtime, the served
    1,024-token budget and the reasoning reply), then compares each answer and
    trace with the evaluation, byte for byte. It stops at the first quota
    refusal, keeps what it measured, and exits 2 with "Incomplete run". On
    2026-10-07, 7 prompts ran before the 8th was refused.

14. **Day 2: run the remaining prompts** once the calling account's quota has
    reset, 24 hours after its first GPU use on day 1. Take the ids that ran
    from the `results` list in the day-1 output, and pass the others,
    comma-separated:

    ```bash
    python scripts/zerogpu_smoke.py --space <owner>/<space> \
        --record configs/deployment/kleos_logos_v002.yaml \
        --benchmark /tmp/bench/benchmark.jsonl \
        --reference <private storage>/<eval>/arm2_finetuned.json \
        --warm-repeats 0 --only <remaining ids> \
        --output <private storage>/<eval>/logos-smoke-day2.json
    ```

15. **Read the result across both days.** Expected for an exact reproduction:
    9 of 9 answers and traces identical, prompt tokens equal on 9 of 9,
    decisions agreeing on 9 of 9.

    - An empty answer (the thinking never closed) is reported as model behavior
      and graded as usual; it is not a serving failure.
    - A run that could not compare every trace verifies nothing and says so.
      Fix the reference before continuing.
    - If any answer or trace differs, stop: do not retrain, and do not change
      the model, the tokenizer, the decoding or the precision to make outputs
      match. Diagnose from the two output files with
      [Diagnosing a reproduction mismatch](../deployment.md#diagnosing-a-reproduction-mismatch),
      and record the decision before KLEOS sends traffic.

    On 2026-10-07 the result was 8 of 9 identical: one answer diverged late in
    its trace on the Space's GPU and changed its decision. It was diagnosed as
    GPU arithmetic. Decision (2026-10-07): accept and document, and serve Logos
    as a Beta.

16. **Leave the GPU request as it is.** At the measured 11.0 tokens/s (warm
    median, 2026-10-07) the requested duration is still the 60 s cap, so
    `LOGOS_GPU_TOKENS_PER_SECOND` stays unset: setting it would change nothing
    ([GPU duration requested per call](../deployment.md#gpu-duration-requested-per-call)).

## Connect KLEOS

17. **Configure the KLEOS backend** with the `LOGOS_` variables in
    [serving-api.md](../serving-api.md#logos-contract-version-2):
    `LOGOS_ENABLED=true`, `LOGOS_PROVIDER=zerogpu`,
    `LOGOS_SPACE=<owner>/<space>`, `LOGOS_API_KEY` (the value set in step 7) and
    `LOGOS_HF_TOKEN` (a read token for the account KLEOS calls as, which ZeroGPU
    charges). Keep `LOGOS_TIMEOUT` at 120 s or more. Confirm that `status()`
    returns `ready` with adapter `e49724f6554db662…` and `"reasoning": true`.

## Record the deployment

18. **Write a dated verification record** beside the existing ones: the date,
    `<commit sha>`, the package commit, the startup line's values, both days'
    smoke summaries and any decision taken. Update the
    [verification status](../deployment.md#verification-status) table.

19. **Remove the local copies** of the benchmark and the reference (for example
    `/tmp/bench`): both are derived from the private dataset.

Before KLEOS takes traffic from this Space, work through the
[checklist before serving](../deployment.md#checklist-before-serving).
