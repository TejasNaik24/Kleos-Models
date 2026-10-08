# Training configurations

Each file extends `../base.yaml` and includes one model, one dataset and one
evaluation config through `includes:`. The resolved configuration is hashed into
the run's `config_hash`, which the manifest records.

| Config | Base model | Purpose | Status |
| --- | --- | --- | --- |
| `debug.yaml` | Mistral-Nemo 12B | Ten steps on the synthetic fixtures with short sequences (seq 512, LoRA r 8). Proves the pipeline runs end to end. | Development check, not a research run |
| `qlora_small.yaml` | Mistral-Nemo 12B | The QLoRA path with the repository's engineering defaults, on the synthetic fixtures. The default config of the tests, CI, `scripts/smoke_test.py` and the Colab notebooks. | Template, not a research run |
| `kleos_policy_v006.yaml` | Ministral-8B | KLEOS v0.0.6, the first behavioral fine-tune. Research only: the base is under the Mistral AI Research License. | Trained and evaluated 2026-09-15 as `kleos-v006-ministral8b-run1` |
| `debug_ministral.yaml` | Ministral-8B | Ten-step check of the Ministral-8B load path, with shrunk model settings. | Development check, not a research run |
| `kleos_hermes_v006.yaml` | Mistral-Nemo 12B | KLEOS Hermes over kleos-policy-v0.0.6. | Trained 2026-09-17 to 2026-09-18 as `kleos-v006-mistralnemo12b-run1`. Frozen: `config_hash` `b2328857…` pinned by `tests/test_logos_config.py` |
| `debug_nemo.yaml` | Mistral-Nemo 12B | Ten steps at Hermes' real model settings: the T4 memory check. | Development check, not a research run |
| `kleos_logos_v001.yaml` | Ministral 3 14B | KLEOS Logos v0.0.1 over kleos-policy-v0.0.6: Hermes' recipe, with evaluation and checkpoints every 25 steps and `strict_config: true`. | Trained 2026-09-24 to 2026-09-27 as `kleos-v006-ministral314b-run1`. Frozen: `config_hash` `18008c67…`, pre-registered under H8, pinned by `tests/test_logos_config.py` |
| `debug_logos.yaml` | Ministral 3 14B | Ten steps at Logos v0.0.1's real model settings: the go/no-go memory gate. | Smoke run 2026-09-24 (`kleos-logos-smoke-001`): 13.60 GiB peak on a T4, gate passed |
| `kleos_logos_v002.yaml` | Ministral 3 14B Reasoning | KLEOS Logos v0.0.2 over kleos-policy-v0.0.7, trained to think: v0.0.1's recipe on Kaggle's 2 × T4. | Trained 2026-10-06 as `kleos-v007-ministral314breasoning-run1`. Frozen: `config_hash` `d1961583…`, pre-registered under H9, pinned by `tests/test_logos_v002_config.py` |
| `debug_logos_v002.yaml` | Ministral 3 14B Reasoning | Ten steps at Logos v0.0.2's real settings: the per-GPU memory gate. | Smoke run 2026-10-06 (`kleos-logos-v002-smoke-001`) inside the Kaggle training notebook, before the real run: gate passed |

Debug runs exist to observe the machinery. Their loss values are not results and
are never reported. The hypotheses, runs and their records are indexed in
[docs/experiments/README.md](../../docs/experiments/README.md).

## Frozen configs and their hashes

`kleos_hermes_v006.yaml`, `kleos_logos_v001.yaml` and `kleos_logos_v002.yaml`
are frozen. Their `config_hash` values are pinned by tests (`HERMES_HASH` and
`LOGOS_HASH` in `tests/test_logos_config.py`, `LOGOS_V002_HASH` in
`tests/test_logos_v002_config.py`), so any edit to these files, to the configs
they include, or to a serialized default fails the suite.

The hash covers the whole resolved configuration, including `description` and the
resolved `--dataset` and `--output-dir` paths. The pinned hashes reproduce only
with the literal paths recorded under H8 and H9 in
[docs/experiments.md](../../docs/experiments.md), with no `KLEOS_*` environment
variable set. The runbooks
([Logos v0.0.1 on Colab](../../docs/runbooks/logos-v001-colab.md),
[Logos v0.0.2 on Kaggle](../../docs/runbooks/logos-v002-kaggle.md)) use those
paths. The `description` fields of the three frozen configs still say "NOT YET
TRAINED": they were written before the runs and are part of the hashed
configuration, so they stay as written.

## Untuned hyperparameters

These files carry engineering defaults chosen so a first run completes reliably
on a free T4: learning rate 2e-4 with a cosine schedule, three epochs, effective
batch 8, LoRA r 16 / alpha 32. Every KLEOS run so far used them unchanged. No
search over them has been run, so results from them are not results of a tuned
configuration and are not reported as such.

## Memory settings on a constrained GPU

| Setting | Effect |
| --- | --- |
| `model.max_seq_length` | Activation memory scales linearly with it. The first thing to reduce. |
| `training.gradient_checkpointing` | A large memory saving for roughly 20-30% slower steps. |
| `per_device_train_batch_size` | Halve it and double `gradient_accumulation_steps` to keep the effective batch and the learning dynamics unchanged. |
| `training.optim` | `paged_adamw_8bit` keeps optimizer state off the critical path. |
| `model.quantization.mode` | `nf4` is the QLoRA default. |

Check the fit before launching. `--seq-length` plans for the longest sequence in
the data instead of `model.max_seq_length`, and `--simulate-gpu` plans for a GPU
that is not attached (a preset such as `t4-colab`, or `NAME:GIB:CC:COUNT` for
several GPUs):

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml
python scripts/plan_run.py --config configs/training/kleos_logos_v002.yaml \
    --seq-length 736 --simulate-gpu T4:14.56:7.5:2
```

## `strict_config`

`strict_config: false` lets the pipeline adjust a configuration that does not
fit, and records every adjustment in `manifest.adjustments[]`.

`strict_config: true` turns any required adjustment into an error. Research runs
compared against another model use it (Logos v0.0.1 and v0.0.2): the run either
uses exactly the configuration specified or refuses to start, so a memory-driven
fallback cannot quietly invalidate a comparison. Hermes ran with `false`, so that
a first fallback would be recorded instead of stopping the run; none occurred.

## Precision

`precision: auto` resolves to bf16 on compute capability 8.0 and above and to
fp16 below it. A T4 is compute capability 7.5 and cannot run bfloat16, so `auto`
is the right setting unless the GPU is known in advance. The resolved choice is
recorded in the manifest.
