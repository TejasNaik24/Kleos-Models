#!/usr/bin/env python3
"""Validate every YAML config in the repository (CI gate).

Loads and type-checks each config, resolves its model-family adapter, and checks
the reasoning mode against the checkpoint's real capability. Catches a broken
config in CI rather than three hours into a GPU session.

Usage::

    python scripts/validate_configs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent

from kleos_models.config import load_config, load_model_config
from kleos_models.models.adapters import get_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Config validation")
    failures: list[tuple[str, str]] = []

    print("Model configs")
    for path in sorted((REPO_ROOT / "configs" / "models").glob("*.yaml")):
        try:
            config = load_model_config(path)
            adapter = get_adapter(config)
            mode = adapter.resolve_reasoning_mode()
            targets = adapter.resolve_target_modules(config.lora)
            print(
                f"  ✓ {path.name:<34} {adapter.family:<8} "
                f"{adapter.capabilities.auto_class:<30} reasoning={mode.value} "
                f"targets={len(targets)}"
            )
        except Exception as exc:
            print(f"  ✗ {path.name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            failures.append((path.name, str(exc)))

    print("\nExperiment configs")
    for directory in ("training", "evaluation", "datasets"):
        for path in sorted((REPO_ROOT / "configs" / directory).glob("*.yaml")):
            # Only training configs are complete experiments; the others are
            # fragments consumed via `includes:`.
            if directory != "training":
                continue
            try:
                config = load_config(path)
                print(
                    f"  ✓ {path.name:<34} model={config.model.name:<24} hash={config.short_hash()}"
                )
            except Exception as exc:
                print(f"  ✗ {path.name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                failures.append((path.name, str(exc)))

    print()
    if failures:
        print(f"✗ {len(failures)} config(s) failed validation.\n", file=sys.stderr)
        return 1
    print("✓ All configs valid.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
