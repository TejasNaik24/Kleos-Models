#!/usr/bin/env python3
"""Inspect a model's real architecture before configuring LoRA (spec §46).

Do not copy LoRA target modules from a tutorial written for a different
architecture. This prints the actual module tree of the checkpoint, the family
adapter that will handle it, the auto class required to load it, and the target
modules that resolve from `target_modules: auto`.

Without --load it reports what can be learned from the checkpoint's config.json
alone — a few KB, no weight download. With --load it downloads weights and lists
every adaptable leaf module.

Usage::

    python scripts/inspect_model.py --model Qwen/Qwen3-8B
    python scripts/inspect_model.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
    python scripts/inspect_model.py --config configs/models/qwen3_8b.yaml --load
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.config import ModelConfig, load_model_config
from kleos_models.logging_utils import get_logger
from kleos_models.models.adapters import get_adapter

logger = get_logger("inspect_model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face checkpoint id.")
    source.add_argument("--config", type=Path, help="Model config YAML.")
    parser.add_argument(
        "--load", action="store_true", help="Download weights and list every adaptable module."
    )
    parser.add_argument("--limit", type=int, default=40, help="Max modules to list.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    if args.config:
        config = load_model_config(args.config)
    else:
        config = ModelConfig(name="adhoc", family="", base_model=args.model)

    print_header(f"Model inspection — {config.base_model}")

    # --- what the config file claims ---------------------------------------
    adapter = get_adapter(config)
    caps = adapter.capabilities
    print("Resolved by the KLEOS model-family adapter:")
    print(f"  adapter class        : {type(adapter).__name__}")
    print(f"  family               : {adapter.family}")
    print(f"  model_type           : {caps.model_type}")
    print(f"  REQUIRED auto class  : {caps.auto_class}")
    print(f"  reasoning capability : {caps.reasoning.value}")
    print(f"  MoE / multimodal     : {caps.is_moe} / {caps.is_multimodal}")
    if caps.language_model_prefix:
        print(f"  LoRA scope prefix    : {caps.language_model_prefix}.*")
    print(f"  default LoRA targets : {adapter.default_target_modules}")
    print(f"  excluded patterns    : {adapter.excluded_module_patterns}")
    print(f"  never quantized      : {adapter.modules_to_not_quantize}")
    if caps.notes:
        print("\n  Notes:")
        for note in caps.notes:
            print(f"    - {note}")

    # --- what the checkpoint actually says ---------------------------------
    print("\n── checkpoint config.json " + "─" * 37)
    try:
        from kleos_models.models.loading import load_hf_config

        hf_config = load_hf_config(config)
        architectures = list(getattr(hf_config, "architectures", None) or [])
        reported_type = getattr(hf_config, "model_type", "unknown")
        print(f"  architectures        : {architectures}")
        print(f"  model_type           : {reported_type}")

        text_config = getattr(hf_config, "text_config", hf_config)
        for field in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "num_experts",
            "num_experts_per_tok",
        ):
            value = getattr(text_config, field, None) or getattr(hf_config, field, None)
            if value is not None:
                print(f"  {field:<21}: {value}")

        if hasattr(hf_config, "vision_config"):
            print("\n  ! This checkpoint has a vision_config: it is a vision-language")
            print("    model. AutoModelForCausalLM will NOT load it. LoRA must be")
            print("    scoped to the language tower.")

        if config.model_type and reported_type != config.model_type:
            print(f"\n  ! Config declares model_type={config.model_type!r} but the")
            print(f"    checkpoint reports {reported_type!r}. Fix the config.")
    except Exception as exc:
        print(f"  Could not read the checkpoint config: {type(exc).__name__}")
        print(f"  {str(exc).splitlines()[0]}")
        print("\n  Install the training extra and set HF_TOKEN for gated repositories.")

    # --- the real module tree ----------------------------------------------
    if args.load:
        print("\n── adaptable modules (weights loaded) " + "─" * 26)
        from kleos_models.models.loading import list_candidate_modules, load_model

        loaded = load_model(config, for_training=False)
        candidates = list_candidate_modules(loaded.model)
        by_suffix = Counter(c["suffix"] for c in candidates)

        print(f"  {len(candidates)} adaptable leaf module(s), by suffix:\n")
        print(f"    {'suffix':<24} {'count':>6}  {'shape (example)':>22}")
        print(f"    {'-' * 24} {'-' * 6}  {'-' * 22}")
        for suffix, count in by_suffix.most_common(args.limit):
            example = next(c for c in candidates if c["suffix"] == suffix)
            shape = f"{example['in_features']}x{example['out_features']}"
            marker = " ←" if suffix in adapter.default_target_modules else ""
            print(f"    {suffix:<24} {count:>6}  {shape:>22}{marker}")
        print("\n    ← marks suffixes selected by the family adapter's defaults.")

        resolution = adapter.validate_target_modules(loaded.model, config.lora)
        print(
            f"\n  Target validation: {resolution.matched_module_count} module(s) matched "
            f"{resolution.matched}"
        )
    else:
        print("\n  Pass --load to download weights and list every adaptable module.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
