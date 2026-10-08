# Model configurations

One YAML file per base checkpoint. The training and evaluation pipelines consume
every one of them unchanged. What differs between checkpoints lives in the
model-family adapters (`src/kleos_models/models/adapters.py`), selected by each
config's `model_type`.

| Config | Checkpoint | Parameters | Reasoning | License | Used by | Training memory |
| --- | --- | --- | --- | --- | --- | --- |
| `mistral_nemo_12b` | `mistralai/Mistral-Nemo-Instruct-2407` | 12.2B dense | none | Apache-2.0, ungated | KLEOS Hermes | 13.09 GB peak on one T4 (measured, `kleos-v006-mistralnemo12b-run1`) |
| `ministral3_14b` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 13.5B text tower of a 13.9B VLM | none | Apache-2.0, ungated | KLEOS Logos v0.0.1 | 13.60 GiB peak on one T4 (measured by the memory probe of the smoke run, 2026-09-24) |
| `ministral3_14b_reasoning` | `mistralai/Ministral-3-14B-Reasoning-2512` | 13.5B text tower of a 13.9B VLM | always on | Apache-2.0, ungated | KLEOS Logos v0.0.2 | 4.75 GB and 10.16 GB peak on the two GPUs of a 2 × T4 (measured, `kleos-v007-ministral314breasoning-run1`); needs two GPUs, since one T4 cannot hold its 736-token longest example (estimated) |
| `ministral_8b` | `mistralai/Ministral-8B-Instruct-2410` | 8.0B dense | none | Mistral AI Research License, gated | The first KLEOS research run | 9.67 GB peak on one T4 (measured, `kleos-v006-ministral8b-run1`) |
| `mistral_small_3_2` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B VLM | none | Apache-2.0 | Never trained | Does not fit a T4 (estimated). Kept as the example of the VLM adapter, whose LoRA scoping has a known defect (finding L-F1) |

The memory figures are peak allocated memory at each config's real training
settings (NF4 double-quant, LoRA r 16, batch 1, longest batch), in the units the
runs recorded. They come from the run records indexed in
[docs/experiments/README.md](../../docs/experiments/README.md).

## Vision-language checkpoints

Mistral Small 3.2 is not a causal LM. It declares
`Mistral3ForConditionalGeneration` (`model_type: mistral3`), which transformers
registers only in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`, so
`AutoModelForCausalLM` cannot load it. `Mistral3VLMAdapter` loads it through
`AutoModelForImageTextToText`, is meant to scope LoRA to the language tower, and
keeps the vision tower and projector frozen and unquantized. That scoping does not
work on transformers 5 (finding L-F1,
[Logos findings](../../docs/experiments/logos-findings.md)): target validation
looks for module names starting with `language_model`, which transformers 5 names
`model.language_model.*`, and the vision tower's `exclude_modules` entry matches
none of the projections inside it. `tests/test_vlm_targeting_finding.py` holds
both defects as expected failures (`xfail`). The defect is not fixed because no
KLEOS model uses this config. Its repository also ships no Hugging Face tokenizer
(only `tekken.json`, no chat template), which `load_tokenizer` requires.

Ministral 3 14B is a VLM checkpoint used text-only. Its repository holds
`Mistral3ForConditionalGeneration` (`model_type: mistral3`) with a `ministral3`
text tower. `ministral3_14b.yaml` sets `model_type: ministral3`, which makes
`Ministral3TextAdapter` load only the text tower: `Ministral3ForCausalLM` with
the checkpoint's keys renamed (`language_model.model.*` to `model.*`,
`language_model.lm_head.*` to `lm_head.*`). The 0.44B-parameter vision tower
stays on disk, and the load refuses to continue if any text weight is missing,
mismatched or unexpected. Use the `-BF16` repository: the default one is FP8,
which a T4 (compute capability 7.5) cannot run, and the loader refuses a
pre-quantized container. The Reasoning release (`ministral3_14b_reasoning.yaml`,
`model_type: ministral3_reasoning`) opens through the same view with
`Ministral3ReasoningTextAdapter`. It is published in BF16 and always thinks: it
writes `[THINK]...[/THINK]` before its answer, and evaluation grades only the
answer.

## Choosing a config

- License. Mistral-Nemo and both Ministral 3 releases are Apache-2.0 and
  ungated, so a derivative can back a product. Ministral-8B is gated under the
  Mistral AI Research License (non-commercial): accept the license on Hugging
  Face and set `$HF_TOKEN` before loading it. Its run is valid research and
  cannot ship.
- Reasoning. Only `ministral3_14b_reasoning` thinks. Requesting `thinking` from
  any other config raises an error, and requesting `standard` from it raises an
  error.
- Scale and modality. `ministral_8b` is a text-only dense model, so a comparison
  against it does not mix modality into the result. A comparison against
  `mistral_small_3_2` measures scale and modality together; state that as a
  limitation when the question is about the 24B checkpoint itself.

## Adding a model

1. Copy the closest existing config and update the checkpoint facts from the
   model's own `config.json` at the revision being pinned.
2. If the architecture needs different loading, LoRA targeting or reasoning
   handling, add a `ModelFamilyAdapter` subclass and register it with
   `@register_adapter`. The training pipeline itself needs no changes; keeping
   family-specific behavior inside adapters is a design requirement.
3. Check the config against the real architecture:
   ```bash
   python scripts/inspect_model.py --config configs/models/<new>.yaml
   ```
4. Check that it fits the hardware before launching. `--set-model` swaps the
   model into an existing training config, `--simulate-gpu` plans for a GPU that
   is not attached, and `--seq-length` sets the longest sequence in the data
   (the default is `model.max_seq_length`):
   ```bash
   python scripts/plan_run.py --config configs/training/qlora_small.yaml \
       --set-model configs/models/<new>.yaml --simulate-gpu t4-colab
   ```

## Pinning revisions

Every config behind a KLEOS model pins `revision` to an audited commit sha. A
LoRA adapter is a set of deltas against specific base weights: served or resumed
against a different revision, it is paired with weights it never saw, and
nothing raises an error. The adapter's own `adapter_config.json` records
`revision: null` (finding H-F1,
[Hermes run report, section 11](../../docs/experiments/kleos-v006-mistralnemo12b-run1-report.md#11-findings)),
so the pin in the config and the run manifest is the record.

`kleos-v006-ministral8b-run1` was trained with `revision: main`, and the commit
it resolved to is not recoverable from the preserved artifacts (deviation D7,
[deviations log](../../docs/experiments.md#deviations-log)). The sha in
`ministral_8b.yaml` applies to later runs only; see
[Base-model revision](../../docs/experiments.md#base-model-revision).
`mistral_small_3_2.yaml` is deliberately unpinned: resolve that repository's own
commit before its first real run. The manifest records whatever is set, so an
unpinned run is visibly unpinned.
