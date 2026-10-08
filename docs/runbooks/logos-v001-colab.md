# Logos v0.0.1 on Colab

This runbook trains and evaluates KLEOS Logos v0.0.1
(`kleos-v006-ministral314b-run1`) on a free Colab T4, then runs the two
comparisons pre-registered as H8. The recorded run followed it between 2026-09-24
and 2026-10-01. Model background, the memory gate and the results are in
[Logos](../logos.md); ids and placeholders are explained in
[docs/experiments/README.md](../experiments/README.md).

## Before starting

- **Runtime.** A Colab notebook on a **T4 GPU** runtime. Each numbered step below
  is one cell. Run them one at a time, in order, and do not run the next cell
  after a failure until the failure is understood.
- **Code.** The commit to run is pushed to GitHub; cell 2 clones it.
- **Data and outputs.** The sealed `kleos-policy-v0.0.6` release sits in a private
  Google Drive folder, `<private storage>`, which also receives the run's outputs.
  Drive needs about 3 GB free for Logos' checkpoints and results. Model weights go
  to the runtime's local disk, never to Drive: the checkpoint is about 28 GB.
- **Hermes' results.** Cells 17 and 18 read Hermes' stored evaluation from
  `<private storage>/outputs/kleos-v006-mistralnemo12b-run1/`. They read it and
  never write it.

**The config hash depends on the paths.** H8's pre-registered `config_hash`
(`18008c6716a58afc…`) was computed with the literal `--dataset` and
`--output-dir` paths recorded under
[H8 in docs/experiments.md](../experiments.md#h8--does-a-stronger-base-make-a-better-kleos-model),
and with no `KLEOS_*` variable set. To reproduce that hash, `<private storage>`
must be the Drive folder recorded there. Any other folder trains the same model
under a different hash.

**The recorded commits.** Training and the `arm2_finetuned` evaluation ran at
`ec4f9e3a03958d7c864b6fb1430fd5780b3d469d`. `arm1_base_orchestrated` ran at
`ed5a987c1a171de9289eba85c4824f3b78484935`, which adds the fix for finding L-F3
(evaluation resume on a Drive mount, [Logos findings](../experiments/logos-findings.md));
this was declared in advance as deviation D11
([deviations log](../experiments.md#deviations-log)). To repeat the recorded run
exactly, check out the matching commit after cell 3 with
`!git checkout <commit sha>`. Before `ed5a987`, an evaluation interrupted on Drive
loses its partial work.

## Set up the runtime

1. Mount Google Drive from the Colab sidebar: **Files → Mount Drive**.

2. Clone the repository:

    ```
    !rm -rf /content/kleos-models && git clone https://github.com/TejasNaik24/Kleos-Models.git /content/kleos-models
    ```

3. Enter it:

    ```
    %cd /content/kleos-models
    ```

4. Install the package without touching Colab's torch:

    ```
    !python scripts/colab_setup.py
    ```

5. Pin the library versions Hermes ran with:

    ```
    !pip install -q --no-deps transformers==5.16.1 peft==0.20.0 accelerate==1.14.0 bitsandbytes==0.50.2 tokenizers==0.23.2
    ```

6. Set the environment and the storage location. Weights are cached on the
   runtime's disk and allocation is tuned. No `KLEOS_*` variable is set, because
   those change `config_hash`. `PRIVATE` is a notebook variable that later cells
   expand as `{PRIVATE}`; replace `<private storage>` with the Drive folder.

    ```
    %env HF_HOME=/content/hf_cache
    %env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PRIVATE = "<private storage>"
    ```

    Then check the runtime:

    ```
    !python -c "import os, torch, transformers, peft, bitsandbytes; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__, bitsandbytes.__version__); print([k for k in os.environ if k.startswith('KLEOS_')] or 'no KLEOS_ variables')"
    ```

    Expect `Tesla T4`, `5.16.1`, `0.20.0`, `0.50.2` and `no KLEOS_ variables`.

## Smoke run and gate

7. Plan against the live GPU:

    ```
    !python scripts/plan_run.py --config configs/training/kleos_logos_v001.yaml --seq-length 448
    ```

8. Run the smoke test: 10 steps at the real settings, on scratch space on the
   runtime's disk.

    ```
    !python scripts/train.py --config configs/training/debug_logos.yaml --dataset {PRIVATE}/kleos-policy-v0.0.6 --output-dir /content/logos-smoke --experiment-id kleos-logos-smoke-001
    ```

9. Apply the go/no-go gate. On NO-GO, do not start the full run and do not change
   any hyperparameter to make it fit: record the probe's numbers and see
   [the gate's options](../logos.md#the-gono-go-gate).

    ```
    !python scripts/check_smoke_gate.py --manifest /content/logos-smoke/kleos-logos-smoke-001/manifest.json --expect-modules 280 --expect-trainable 60948480 --min-spare-gb 0.15
    ```

10. Drill a resume. Remove the last checkpoint, then resume: the run must
    continue from step 5 and finish.

    ```
    !rm -rf /content/logos-smoke/kleos-logos-smoke-001/checkpoint-10 && python scripts/train.py --config configs/training/debug_logos.yaml --dataset {PRIVATE}/kleos-policy-v0.0.6 --output-dir /content/logos-smoke --experiment-id kleos-logos-smoke-001 --resume-from-checkpoint auto
    ```

## Train

11. Train Logos. ESTIMATED about 3.5 hours on a T4, from Hermes' 3.2; the recorded
    run took four Colab sessions between 2026-09-24 and 2026-09-27 (deviation D9).

    ```
    !python scripts/train.py --config configs/training/kleos_logos_v001.yaml --dataset {PRIVATE}/kleos-policy-v0.0.6 --output-dir {PRIVATE}/outputs --experiment-id kleos-v006-ministral314b-run1
    ```

    The config hash it prints must be `18008c6716a58afc`, the value pre-registered
    under H8, which requires `<private storage>` to be the recorded folder.

12. If the runtime died, repeat cells 1 to 6, then resume:

    ```
    !python scripts/train.py --config configs/training/kleos_logos_v001.yaml --dataset {PRIVATE}/kleos-policy-v0.0.6 --output-dir {PRIVATE}/outputs --experiment-id kleos-v006-ministral314b-run1 --resume-from-checkpoint auto
    ```

    Never run cell 11 again into a folder that already holds checkpoints. A fresh
    start there would delete the best checkpoint at its first save (finding L-F4).
    Before resuming, check that the latest checkpoint holds `optimizer.pt` and
    `scheduler.pt`: `validate_checkpoint` does not check them (finding L-F5), and
    without them the run would continue with a fresh optimizer and a restarted
    learning-rate schedule.

## Evaluate

13. Build and check the benchmark. Its sha256 must start `a11ffad75f5147f9`.

    ```
    !python scripts/build_benchmark.py --dataset {PRIVATE}/kleos-policy-v0.0.6 --output /content/benchmark && sha256sum /content/benchmark/benchmark.jsonl
    ```

14. Evaluate Logos `arm1_base_orchestrated` (the base, orchestrated). ESTIMATED
    2 to 3 hours; the recorded arm spent 12,804 s generating, across two sessions.
    Re-run the same cell to resume after a disconnect.

    ```
    !python scripts/evaluate.py --config configs/training/kleos_logos_v001.yaml --arm arm1_base_orchestrated --benchmark /content/benchmark/benchmark.jsonl --output {PRIVATE}/outputs/kleos-v006-ministral314b-run1/arm1_base_orchestrated.json --resume
    ```

15. Evaluate Logos `arm2_finetuned`. The recorded arm spent 5,996 s generating.

    ```
    !python scripts/evaluate.py --config configs/training/kleos_logos_v001.yaml --arm arm2_finetuned --adapter {PRIVATE}/outputs/kleos-v006-ministral314b-run1/adapter --benchmark /content/benchmark/benchmark.jsonl --output {PRIVATE}/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --resume
    ```

## Compare

16. H8a, Logos `arm1` against `arm2`:

    ```
    !python scripts/compare.py --base {PRIVATE}/outputs/kleos-v006-ministral314b-run1/arm1_base_orchestrated.json --finetuned {PRIVATE}/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --report {PRIVATE}/outputs/kleos-v006-ministral314b-run1/report_logos_v001 --experiment-id kleos-v006-ministral314b-run1 --dataset-version kleos-policy-v0.0.6
    ```

    `compare.py`'s closing line counts significant tasks by the example
    bootstrap. H8a's pre-registered count uses the group intervals.

17. Re-report Hermes with the corrected measures. This is CPU work, and it writes
    a separate `.annotated.json` beside Hermes' frozen results:

    ```
    !python scripts/rescore.py --mode annotate --results {PRIVATE}/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.json --benchmark /content/benchmark/benchmark.jsonl --output {PRIVATE}/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json --gold-targets {PRIVATE}/kleos-policy-v0.0.6/test.jsonl --summary {PRIVATE}/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.md
    ```

    Repeat with `arm1_base_orchestrated` in place of `arm2_finetuned` for the
    baseline.

18. H8b, the primary comparison:

    ```
    !python scripts/compare.py --cross-model --base {PRIVATE}/outputs/kleos-v006-mistralnemo12b-run1/arm2_finetuned.annotated.json --finetuned {PRIVATE}/outputs/kleos-v006-ministral314b-run1/arm2_finetuned.json --primary-subset answerable --equivalence-margin 0.02 --report {PRIVATE}/outputs/report_h8b_hermes_vs_logos
    ```

The recorded verdicts are in
[H8b result](../experiments.md#h8b-result--2026-09-29) and
[H8a result](../experiments.md#h8a-result--2026-10-01), with the artifact hashes in
the [run report](../experiments/kleos-v006-ministral314b-run1-report.md#10-artifact-hash-register).
