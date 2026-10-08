# Deploy Hermes v0.0.6 to a ZeroGPU Space

This runbook deploys the frozen Hermes v0.0.6 artifact to a private Hugging Face
ZeroGPU Space and checks that the Space reproduces the frozen evaluation
outputs. The operator runs every step, in order. The last deployment ran on
2026-09-23; its record is in
[serving-verification-records.md](../experiments/serving-verification-records.md#verification-record--zerogpu).
What each check means is explained in [deployment.md](../deployment.md), and
the conventions for finding and decision ids are in
[experiments/README.md](../experiments/README.md).

No step costs money. If any screen asks for payment details, stop and note
which screen asked: the free tier needs none.

## Before starting

| Requirement | Detail |
| --- | --- |
| Hugging Face account | In good standing: verified email, older than 30 days. A free account may host up to 2 ZeroGPU Spaces |
| Frozen research run | `<private storage>/outputs/kleos-v006-mistralnemo12b-run1`, read-only |
| Frozen evaluation results | Hermes v0.0.6's `arm2_finetuned.json` (sha256 `428400f520f7d370…`) at `<private storage>/<eval>/` |
| Dataset release | kleos-policy-v0.0.6 at `<path to release>`, to rebuild the benchmark |
| This repository | Installed (`pip install -e .`), with the commit to deploy pushed to GitHub |
| Write token | A Hugging Face write token for the upload and the Space push, stored with `hf auth login` or exported as `$HF_TOKEN` from a secret store. Never type it into a file, a notebook cell or a command line |

Placeholders: `<owner>/<package-repo>` is the private package repository,
`<owner>/<space>` the Space, `<private storage>` the private location of the run
and its evaluation outputs, and `<commit sha>` the commit of this repository the
Space installs.

## Build and upload the package

1. **Push the commit the Space will install.** The Space installs
   `kleos-models` from GitHub at an exact commit, and
   `scripts/stage_zerogpu_space.py` uses `HEAD` and refuses a dirty working
   tree. Commit, push, and note the commit as `<commit sha>`.

2. **Build the schema-v2 package** on any machine that can read the run. No GPU
   is needed.

   ```bash
   python scripts/build_deployment_package.py \
       --run <private storage>/outputs/kleos-v006-mistralnemo12b-run1 \
       --deployment-config configs/deployment/kleos_hermes_v006.yaml \
       --output <private storage>/packages/hermes-v0.0.6
   ```

   The run directory stays read-only. The build refuses an adapter whose sha256
   is not the frozen `dc121fa3…`.

3. **Verify the package against the serving record.**

   ```bash
   python scripts/verify_deployment_package.py \
       --package <private storage>/packages/hermes-v0.0.6 \
       --expect-deployment-config configs/deployment/kleos_hermes_v006.yaml \
       --check-base-revision
   ```

   Expected: exit 0, with adapter `dc121fa3…` and base revision `04d8a905…` in
   the printed identity. If it prints `DO NOT SERVE THIS PACKAGE`, stop: the
   package is not the frozen artifact. Rebuild it from the frozen run, and never
   edit the record to match a package.

4. **Upload the package to a private repository**, dry run first.

   ```bash
   python scripts/upload_deployment_package.py \
       --package <private storage>/packages/hermes-v0.0.6 \
       --repo <owner>/<package-repo> --dry-run
   python scripts/upload_deployment_package.py \
       --package <private storage>/packages/hermes-v0.0.6 \
       --repo <owner>/<package-repo> --create
   ```

   The script uses `$HF_TOKEN` if it is set, otherwise the token stored by
   `hf auth login`. It refuses a public repository, uploads only the files the
   manifest lists, then re-lists the repository and checks every file's size and
   hash. Note the commit it prints (40 hex characters): it becomes
   `HERMES_PACKAGE_REVISION`.

## Create and push the Space

5. **Create the Space** at huggingface.co/new-space: SDK Gradio, hardware
   ZeroGPU, visibility private.

6. **Set four secrets** in the Space's settings:

   | Secret | Value |
   | --- | --- |
   | `HF_TOKEN` | A new fine-grained token with read access to `<owner>/<package-repo>` only |
   | `HERMES_API_KEY` | A new random secret, for example from `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The KLEOS backend gets the same value |
   | `HERMES_PACKAGE_REPO` | `<owner>/<package-repo>` |
   | `HERMES_PACKAGE_REVISION` | The commit step 4 printed |

   Leave the optional Space variables (`HERMES_GPU_*`,
   `HERMES_MAX_INPUT_TOKENS`, `HERMES_MAX_NEW_TOKENS`) unset for the first
   start. Never set `HERMES_ALLOW_UNAUTHENTICATED` on a Space.

7. **Stage the Space and review it.**

   ```bash
   python scripts/stage_zerogpu_space.py --out /tmp/hermes-space
   ```

   Read the four staged files: `README.md`, `app.py`, `requirements.txt` and
   `hermes_record.yaml`. `requirements.txt` must pin `kleos-models` to
   `<commit sha>`. The script refuses a README whose `preload_from_hub`
   disagrees with the record, and any staged file that matches a secret
   pattern.

8. **Push the Space.**

   ```bash
   python scripts/stage_zerogpu_space.py --out /tmp/hermes-space-push \
       --push --space <owner>/<space>
   ```

   The push uses `$HF_TOKEN` if it is set, otherwise the stored login. The four
   staged files can instead be uploaded in the Space's Files tab.

## Check the Space

9. **Watch the build and startup logs.** The build installs the pinned
   requirements and bakes the base shards into the image. Startup downloads the
   package, verifies it, and quantizes the base on the CPU; the Space allows up
   to 1 hour (`startup_duration_timeout`). Success ends with one line:

   ```
   Hermes ready: kleos-hermes v0.0.6 adapter=dc121fa36ce1409c… base=…@04d8a90549d2
   compute_dtype=float16 download_s=… load_s=… memory={…} versions={…}
   ```

   Record `download_s`, `load_s` and `memory`.

   - If the build fails on a dependency conflict, dry-run the install in the
     platform's base image as described in
     [deployment.md](../deployment.md#how-it-fits-together), fix
     `deploy/zerogpu-space/requirements.txt.template`, and start again from
     step 1.
   - If the startup log shows the package refused (an integrity or identity
     error), stop: the Space is not serving the frozen artifact. Check
     `HERMES_PACKAGE_REPO` and `HERMES_PACKAGE_REVISION` first, and rebuild the
     package only if they are right.
   - If the startup log shows an out-of-memory error, stop: the host could not
     quantize the base on its CPU. The only fallback, loading inside the GPU
     call, spends quota on every cold start. That is a design change: decide and
     record it before any code changes.

10. **Rebuild the benchmark** and check its hash.

    ```bash
    python scripts/build_benchmark.py --dataset <path to release> --output /tmp/bench
    shasum -a 256 /tmp/bench/benchmark.jsonl      # must be a11ffad75f51…b266
    ```

    If the hash differs, stop: the release is not the one Hermes was evaluated
    on, and the comparison would be meaningless.

11. **Run the frozen-output smoke test.**

    ```bash
    pip install gradio_client==2.7.1
    read -s HERMES_API_KEY && export HERMES_API_KEY    # paste the Space's key; not echoed
    python scripts/zerogpu_smoke.py --space <owner>/<space> \
        --benchmark /tmp/bench/benchmark.jsonl \
        --reference <private storage>/<eval>/arm2_finetuned.json \
        --output <private storage>/<eval>/zerogpu_smoke.json
    ```

    The calling account (`$HF_TOKEN`, or the stored login) pays for the GPU
    time. The script checks the Space's identity with `/status` before spending
    any, then runs the same nine examples as the Colab check plus one warm
    repeat: about 10 GPU calls, sized to fit one day's free quota.

    Expected: exit 0, with 9/9 exact, 9/9 prompt token counts equal, 9/9
    decisions agreeing and the repeat identical. That was the result on
    2026-09-23.

    - If the run stops at a quota refusal (exit 2, "Incomplete run"), wait until
      the calling account's quota resets, 24 hours after its first GPU use, and
      finish with the same command plus `--only <remaining ids>`.
    - If any output differs, stop: do not retrain, and do not change the model,
      the tokenizer, the decoding or the precision to make outputs match.
      Diagnose from `zerogpu_smoke.json` with
      [Diagnosing a reproduction mismatch](../deployment.md#diagnosing-a-reproduction-mismatch),
      and record the decision before KLEOS sends traffic.

12. **Calibrate the GPU request.** If the smoke test's warm
    `median_tokens_per_second`, rounded down, differs from the default of 12,
    set `HERMES_GPU_TOKENS_PER_SECOND` to it as a Space variable (it is not a
    secret). For Hermes v0.0.6 the measured value was 12, so it stays unset. If
    cold calls ended in `model_error` because ZeroGPU aborted them, raise
    `HERMES_GPU_BASE_SECONDS`.

## Connect KLEOS

13. **Configure the KLEOS backend** with the variables in
    [serving-api.md](../serving-api.md#configuration): `HERMES_ENABLED=true`,
    `HERMES_PROVIDER=zerogpu`, `HERMES_SPACE=<owner>/<space>`, `HERMES_API_KEY`
    (the value set in step 6) and `HERMES_HF_TOKEN` (a read token for the
    account KLEOS calls as, which ZeroGPU charges). Confirm that `status()`
    returns `ready` with adapter `dc121fa3…`.

## Record the deployment

14. **Write a dated verification record** beside the existing ones: the date,
    `<commit sha>`, the package commit, the startup line's values and the smoke
    summary. Update the
    [verification status](../deployment.md#verification-status) table.

15. **Remove the local copies** of the benchmark and the reference (for example
    `/tmp/bench`): both are derived from the private dataset.

Before KLEOS takes traffic from this Space, work through the
[checklist before serving](../deployment.md#checklist-before-serving).
