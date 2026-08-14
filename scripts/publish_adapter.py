#!/usr/bin/env python3
"""Publish a trained adapter to the Hugging Face Hub (spec §18, §35).

Publishing is never automatic. It requires this explicit command.

What is uploaded: adapter weights, the tokenizer, the effective config, the
manifest, and a generated model card.

What is refused: raw training data, `.env`, anything matching the private-data
scanner, and any file the safety check flags. The model card will not claim the
model is better unless an evaluation result is supplied that shows it.

Usage::

    python scripts/publish_adapter.py --adapter outputs/<experiment-id>/adapter \\
                                      --repo-id YOUR_USERNAME/kleos-qwen3-8b

    # Inspect exactly what would be uploaded, without uploading:
    python scripts/publish_adapter.py --adapter outputs/<id>/adapter \\
                                      --repo-id YOU/name --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.errors import PublishingError
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.logging_utils import get_logger
from kleos_models.publishing import (
    ALLOWED_UPLOAD_NAMES,
    build_model_card,
    collect_upload_files,
)

logger = get_logger("publish_adapter")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adapter", type=Path, required=True, help="Adapter directory to publish.")
    parser.add_argument(
        "--repo-id", required=True, help="Target repo, e.g. username/kleos-qwen3-8b."
    )
    parser.add_argument("--results", type=Path, help="Evaluation results JSON to cite in the card.")
    parser.add_argument("--comparison", type=Path, help="Comparison JSON from compare.py.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default="Publish KLEOS adapter")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be uploaded, then stop."
    )
    parser.add_argument(
        "--card-only", action="store_true", help="Write the model card locally and stop."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Publish KLEOS adapter")

    adapter_dir = args.adapter
    if not adapter_dir.exists():
        raise PublishingError(
            f"Adapter directory not found: {adapter_dir}",
            suggestions=["Training writes it to outputs/<experiment-id>/adapter/."],
        )

    run_dir = adapter_dir.parent
    manifest: ExperimentManifest | None = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = ExperimentManifest.load(manifest_path)
        print(f"  run          : {manifest.experiment_id}")
        print(f"  base model   : {manifest.model.get('base_model', 'unknown')}")
        print(f"  dataset      : {manifest.dataset_version}")
        print(f"  status       : {manifest.status.value}")
        if manifest.status.value != "completed":
            print("\n  ! This run did not complete successfully. Publishing an adapter")
            print("    from an incomplete run is rarely what you want.")
    else:
        print(f"  ! No manifest.json beside {adapter_dir}.")
        print("    The model card will be missing provenance, which undermines the")
        print("    reproducibility chain. Publish from a full run directory instead.")

    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else None
    comparison = (
        json.loads(args.comparison.read_text(encoding="utf-8")) if args.comparison else None
    )

    # --- safety: decide exactly what may be uploaded -----------------------
    print("\n── upload safety check " + "─" * 40)
    files, rejected = collect_upload_files(adapter_dir, run_dir)

    print(f"  {len(files)} file(s) will be uploaded:")
    for path in files:
        print(f"    + {path.relative_to(run_dir) if run_dir in path.parents else path.name}")
    if rejected:
        print(f"\n  {len(rejected)} file(s) refused:")
        for path, reason in rejected:
            print(f"    - {path.name}: {reason}")
    print(f"\n  Allowed names: {sorted(ALLOWED_UPLOAD_NAMES)}")
    print("  Raw training data, .env and checkpoints are never uploaded.")

    # --- model card ---------------------------------------------------------
    card = build_model_card(
        repo_id=args.repo_id,
        manifest=manifest,
        results=results,
        comparison=comparison,
    )
    card_path = run_dir / "README.model_card.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"\n  Model card written to {card_path}")

    if args.card_only:
        print("\n✓ Card generated. Nothing was uploaded (--card-only).\n")
        return 0

    if args.dry_run:
        print("\n✓ Dry run complete. Nothing was uploaded.")
        print(f"  Review {card_path}, then re-run without --dry-run.\n")
        return 0

    # --- upload -------------------------------------------------------------
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise PublishingError(
            "No Hugging Face token found.",
            details={"checked": "HF_TOKEN, HUGGING_FACE_HUB_TOKEN"},
            suggestions=[
                "Create a token at https://huggingface.co/settings/tokens with write scope.",
                "Set it: export HF_TOKEN=hf_...",
                "On Colab, add it as a secret named HF_TOKEN and enable notebook access.",
                "Never put a token in a config file or a notebook cell.",
            ],
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        from kleos_models.errors import MissingDependencyError

        raise MissingDependencyError(
            "huggingface_hub", extra="train", purpose="publish to the Hub"
        ) from exc

    print(f"\n── uploading to {args.repo_id} " + "─" * max(0, 34 - len(args.repo_id)))
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, repo_type="model")

    for path in files:
        destination = path.name if path.parent == adapter_dir else f"{path.parent.name}/{path.name}"
        if path.parent == adapter_dir:
            destination = path.name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=destination,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )
        print(f"    uploaded {destination}")

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )
    print("    uploaded README.md (model card)")

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n✓ Published to {url}")
    print("\n  Check the model card renders correctly and that no private data is")
    print("  present before sharing the link.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
