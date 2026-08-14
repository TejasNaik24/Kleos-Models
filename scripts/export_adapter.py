#!/usr/bin/env python3
"""Export an adapter into a self-contained, shareable directory.

Copies the adapter weights, tokenizer, effective config, manifest and metrics
into one directory, and generates a model card. Everything is run through the
same upload safety check `publish_adapter.py` uses, so an export can be inspected
before anything is published.

Usage::

    python scripts/export_adapter.py --run outputs/<experiment-id> \
                                     --output exports/kleos-qwen3-8b
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.errors import KleosError
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.publishing import build_model_card, collect_upload_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="Training run directory.")
    parser.add_argument("--output", type=Path, required=True, help="Export destination.")
    parser.add_argument(
        "--repo-id", default="local/kleos-adapter", help="Name used in the model card."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing export.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Export adapter")

    adapter_dir = args.run / "adapter"
    if not adapter_dir.exists():
        raise KleosError(
            f"No adapter/ directory in {args.run}",
            suggestions=["Point --run at a completed training run directory."],
        )
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise KleosError(
            f"{args.output} already exists and is not empty.",
            suggestions=["Choose another --output, or pass --force."],
        )

    manifest = None
    manifest_path = args.run / "manifest.json"
    if manifest_path.exists():
        manifest = ExperimentManifest.load(manifest_path)
        print(f"  run    : {manifest.experiment_id}")
        print(f"  status : {manifest.status.value}")

    files, rejected = collect_upload_files(adapter_dir, args.run)
    args.output.mkdir(parents=True, exist_ok=True)

    for path in files:
        shutil.copy2(path, args.output / path.name)
        print(f"    + {path.name}")
    for path, reason in rejected:
        print(f"    - {path.name}: {reason}")

    card = build_model_card(repo_id=args.repo_id, manifest=manifest)
    (args.output / "README.md").write_text(card, encoding="utf-8")

    print(f"\n✓ Exported {len(files)} file(s) to {args.output}")
    print(f"\n  Review {args.output}/README.md, then publish with:")
    print(f"    python scripts/publish_adapter.py --adapter {adapter_dir} --repo-id <you>/<name>\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
