# Training configurations

| Config | Purpose |
| --- | --- |
| `debug.yaml` | Ten steps, tiny sequences. Proves the pipeline runs. |
| `qlora_small.yaml` | The real QLoRA path, with engineering defaults. |

## These hyperparameters are not tuned

Both files carry engineering defaults chosen so a first run completes reliably on
a free Colab T4 (spec §47, §48). They are a starting point for the research
phase, not a result. Do not report numbers from them as if the configuration had
been optimized.

## The knobs that actually matter on a constrained GPU

| Setting | Effect |
| --- | --- |
| `model.max_seq_length` | Activation memory scales linearly. First thing to reduce. |
| `training.gradient_checkpointing` | Large memory saving for ~20-30% slower steps. |
| `per_device_train_batch_size` | Halve it and double `gradient_accumulation_steps` to keep the effective batch and the learning dynamics unchanged. |
| `training.optim` | `paged_adamw_8bit` keeps optimizer state off the critical path. |
| `model.quantization.mode` | `nf4` is the QLoRA default. |

Check before you launch:

```bash
python scripts/plan_run.py --config configs/training/qlora_small.yaml
```

## `strict_config`

`strict_config: false` lets the pipeline adjust a configuration that does not fit
— and **records every adjustment** in `manifest.adjustments[]`.

`strict_config: true` makes any required adjustment a hard error instead. Turn it
on for real research runs: it guarantees the run either used exactly the
configuration you specified or refused to start, so a comparison cannot be
quietly invalidated by a memory-driven fallback.

## Precision

`precision: auto` resolves to bf16 on compute capability 8.0+ and fp16 below it.
A Colab T4 is compute capability 7.5 and **cannot** do bfloat16, so `auto` is the
right setting unless you know exactly which GPU you will be assigned. The
resolved choice is recorded in the manifest.
