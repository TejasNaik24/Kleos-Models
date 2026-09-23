#!/usr/bin/env python3
"""Verify a deployment package without loading a model.

Re-hashes every recorded artifact, checks the adapter config pins the base
revision the manifest names, and prints the identity of what would be served.
Needs no GPU, no torch and no network, so it can run in CI, in a container
build, or as a pre-start check on the serving host.

Usage::

    python scripts/verify_deployment_package.py --package /path/to/hermes-v0.0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.serving.manifest import DeploymentManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True, help="Deployment package root.")
    parser.add_argument(
        "--check-base-revision",
        action="store_true",
        help="Also confirm the base commit with the model registry (needs network).",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Verify deployment package")

    manifest = DeploymentManifest.load(args.package)
    print(f"  model        : {manifest.model_name} {manifest.model_version}")
    print(f"  experiment   : {manifest.adapter.experiment_id}")
    print(f"  base         : {manifest.base_model}")
    print(f"  revision     : {manifest.base_revision}")
    print(f"  adapter      : {manifest.adapter.weights_sha256}")
    print(f"  checkpoint   : {manifest.adapter.source_checkpoint}")
    print(f"  dataset      : {manifest.dataset.version} ({manifest.dataset.sha256[:16]}…)")
    print(f"  config hash  : {manifest.training_config_hash[:16]}…")
    print(
        f"  tokenizer    : {manifest.tokenizer.source}, "
        f"fix_mistral_regex={manifest.tokenizer.fix_mistral_regex}"
    )
    print(f"  files        : {len(manifest.all_files())}")

    problems = manifest.problems(args.package)
    if problems:
        print(f"\n✗ {len(problems)} problem(s). DO NOT SERVE THIS PACKAGE.\n")
        for problem in problems:
            print(f"    - {problem}")
        print()
        return 1

    print("\n  ✓ every recorded file matches its hash")
    print("  ✓ adapter_config.json pins the manifest's base revision")

    if args.check_base_revision:
        from kleos_models.serving.loader import verify_base_revision

        verify_base_revision(manifest.base_model, manifest.base_revision, require_remote=True)
        print("  ✓ base revision confirmed with the model registry")

    if manifest.known_limitations:
        print("\n  Known limitations carried by this model:")
        for note in manifest.known_limitations:
            print(f"    · {' '.join(note.split())}")

    print("\n✓ Package verified.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
