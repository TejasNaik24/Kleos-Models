#!/usr/bin/env python3
"""Upload a verified Hermes deployment package to a PRIVATE Hugging Face repo.

The ZeroGPU Space downloads the package from there at startup, pinned to the
commit this script prints. Run it where the package is (Colab, next to the
Drive run), with a write token in HF_TOKEN.

Refuses, before anything leaves the machine:

* a package that fails verification, or is not the artifact the serving
  record describes (adapter hash, base revision, tokenizer, runtime contract);
* any file the manifest does not account for (checkpoints, logs, datasets);
* text matching the secrets / personal-data scanner;
* a repository that is, or would be, public.

After uploading, it lists the repository at the new commit and checks every
file's size and hash against the local copy, so the pinned commit is proven to
hold the verified bytes.

Usage::

    # Everything except the upload (no network):
    python scripts/upload_deployment_package.py --package /path/to/hermes-v0.0.6 \\
        --repo YOUR_USERNAME/kleos-hermes-v006-package --dry-run

    HF_TOKEN=... python scripts/upload_deployment_package.py \\
        --package /path/to/hermes-v0.0.6 --repo YOUR_USERNAME/kleos-hermes-v006-package --create
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import REPO_ROOT, add_common_arguments, print_header, run, setup_logging
from kleos_models.errors import ConfigError
from kleos_models.serving.manifest import load_expected_identity, verify_identity
from kleos_models.serving.space import RemoteFile, compare_remote_files, package_upload_files

DEFAULT_RECORD = REPO_ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True, help="Deployment package root.")
    parser.add_argument("--repo", required=True, help="Private model repo, owner/name.")
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=DEFAULT_RECORD,
        help="Serving record the package must match.",
    )
    parser.add_argument(
        "--create", action="store_true", help="Create the repository (private) if missing."
    )
    parser.add_argument("--dry-run", action="store_true", help="Check only; upload nothing.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Hermes deployment package → private Hugging Face repository")

    manifest, files = package_upload_files(args.package)
    verify_identity(manifest, load_expected_identity(args.deployment_config))
    total = sum((args.package / f).stat().st_size for f in files)
    print(f"  package  : {manifest.model_name} {manifest.model_version}")
    print(f"  adapter  : {manifest.adapter.weights_sha256[:16]}…")
    print(f"  base     : {manifest.base_model}@{manifest.base_revision[:12]}…")
    print(f"  identity : matches {args.deployment_config.name}")
    print(f"  files    : {len(files)} ({total / 1e6:.1f} MB)")
    for relative in files:
        print(f"    {relative}")

    if args.dry_run:
        print("\n✓ Dry run: the package verifies and is safe to upload. Nothing was sent.\n")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ConfigError(
            "HF_TOKEN is not set.",
            suggestions=[
                "Use a token with write access to your account, from the environment "
                "or a Colab secret. Never paste it into a notebook cell or a file.",
            ],
        )

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        info = api.repo_info(args.repo, repo_type="model")
    except RepositoryNotFoundError:
        if not args.create:
            raise ConfigError(
                f"{args.repo} does not exist.",
                suggestions=["Pass --create to create it as a private repository."],
            ) from None
        api.create_repo(args.repo, repo_type="model", private=True)
        info = api.repo_info(args.repo, repo_type="model")
        print(f"\n  created  : {args.repo} (private)")
    if info.private is not True:
        raise ConfigError(
            f"{args.repo} is not private. Refusing to upload the Hermes package to it.",
            suggestions=["Make it private in its settings, or choose another name."],
        )

    commit = api.upload_folder(
        folder_path=str(args.package),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=files,
        commit_message=(
            f"{manifest.model_name} {manifest.model_version} deployment package "
            f"(adapter {manifest.adapter.weights_sha256[:16]})"
        ),
    )
    revision = commit.oid
    print(f"\n  uploaded : commit {revision}")

    remote = {
        entry.path: RemoteFile(
            path=entry.path,
            size=entry.size,
            blob_id=getattr(entry, "blob_id", None),
            lfs_sha256=entry.lfs.sha256 if getattr(entry, "lfs", None) else None,
        )
        for entry in api.list_repo_tree(
            args.repo, recursive=True, revision=revision, repo_type="model"
        )
        if hasattr(entry, "size")
    }
    problems = compare_remote_files(args.package, files, remote)
    if problems:
        raise ConfigError(
            "The repository does not hold exactly the verified package.",
            details={"problems": problems},
            suggestions=["Do not point the Space at this commit."],
        )
    print(f"  verified : {len(files)} file(s) at {revision[:12]} match the local package")

    print("\n  Set these as Space secrets (Settings → Variables and secrets):")
    print(f"    HERMES_PACKAGE_REPO     = {args.repo}")
    print(f"    HERMES_PACKAGE_REVISION = {revision}")
    print("\n✓ Package uploaded to a private repository and verified remotely.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
