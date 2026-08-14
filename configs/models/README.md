# Model configurations

One YAML per checkpoint. The training and evaluation pipelines consume all of
them unchanged — everything family-specific lives in the model-family adapter
(`src/kleos_models/models/adapters.py`), not in the pipeline.

| Config | Checkpoint | Params | Reasoning | Free-T4 trainable? |
| --- | --- | --- | --- | --- |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 8.2B dense | switchable | yes (4-bit) |
| `ministral_8b` | `mistralai/Ministral-8B-Instruct-2410` | 8.0B dense | none | yes (4-bit) |
| `mistral_small_3_2` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B VLM | none | **no** — needs A100 |
| `qwen3_30b_a3b_thinking` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30.5B MoE / 3.3B active | always on | **no** — needs A100 |

## Two traps these configs exist to document

**Mistral Small 3.2 is not a causal LM.** It declares
`Mistral3ForConditionalGeneration` / `model_type: mistral3`, which transformers
registers only in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.
`AutoModelForCausalLM` fails on it. It is a VLM, so LoRA is scoped to
`language_model.*` and the vision tower stays frozen and unquantized.

**Qwen3-30B-A3B is MoE and thinking-only.** LoRA targets attention only —
targeting 128 experts across 48 layers would create ~18,400 adapter modules — and
the router is excluded from both LoRA and quantization. Its chat template does
not accept `enable_thinking`, so asking for non-thinking mode raises an error
rather than silently producing a broken prompt.

## Choosing between the Mistral arms

For a **family comparison** (Qwen vs Mistral), use `ministral_8b`: it is
scale-matched to `qwen3_8b`, so the result is not confounded by parameter count
or modality.

For a question about **the 24B checkpoint specifically**, use
`mistral_small_3_2` and state the scale difference as a limitation.

## Adding a model

1. Copy the closest existing config and update the checkpoint facts. Get them
   from the model's real `config.json`, not from a blog post.
2. If the architecture needs different loading, targeting or reasoning handling,
   add a `ModelFamilyAdapter` subclass and `@register_adapter` it. **The training
   pipeline needs no changes** — that is the design requirement.
3. Verify against the real architecture:
   ```bash
   python scripts/inspect_model.py --model <checkpoint-id>
   ```
4. Check it fits your hardware before launching:
   ```bash
   python scripts/plan_run.py --config configs/training/qlora_small.yaml
   ```

## Pinning revisions

`revision: main` is convenient and not reproducible — `main` moves. Pin a commit
sha for any run whose results you intend to report. The manifest records whatever
you set, so an unpinned run is at least visibly unpinned.
