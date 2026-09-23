#!/usr/bin/env python3
"""Render the Hermes ZeroGPU Space into a directory, checked, ready to push.

The Space repository holds exactly four files:

    README.md           Space config: Gradio 6.28.0, Python 3.12, base shards
                        preloaded at the pinned revision
    app.py              thin wiring into kleos_models.serving.zerogpu
    requirements.txt    the Docker image's model runtime, plus kleos-models
                        pinned to one commit of this repository
    hermes_record.yaml  the serving record the package must match

No weights, no data, no secrets. Secrets are set in the Space settings; the
package arrives at startup from its private repository at a pinned commit.

The Space installs kleos-models from GitHub at the commit given here, so that
commit must be pushed, and it should be the one whose tests passed. By default
this is HEAD, and a dirty working tree is refused: what the Space would install
is not what is on disk.

Usage::

    python scripts/stage_zerogpu_space.py --out /tmp/hermes-space
    python scripts/stage_zerogpu_space.py --out /tmp/hermes-space \\
        --push --space YOUR_USERNAME/kleos-hermes          # needs HF_TOKEN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import REPO_ROOT, add_common_arguments, print_header, run, setup_logging
from kleos_models.data.loaders import file_sha256
from kleos_models.errors import ConfigError
from kleos_models.serving.space import SPACE_SOURCE_DIR, stage_space

DEFAULT_RECORD = REPO_ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ConfigError(f"git {' '.join(args)} failed: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def resolve_commit(requested: str | None, *, allow_dirty: bool) -> str:
    """The commit the Space will install, checked against the working tree."""
    if requested is None:
        if _git("status", "--porcelain") and not allow_dirty:
            raise ConfigError(
                "The working tree has uncommitted changes.",
                suggestions=[
                    "The Space installs a commit from GitHub, not these files. Commit "
                    "and push, then stage again.",
                    "Or pass --allow-dirty to stage HEAD anyway (for a local look only).",
                ],
            )
        return _git("rev-parse", "HEAD")
    _git("cat-file", "-e", f"{requested}^{{commit}}")
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="Empty or new directory.")
    parser.add_argument("--commit", help="kleos-models commit to install (default: HEAD).")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="Serving record.")
    parser.add_argument("--allow-dirty", action="store_true", help="Stage despite local changes.")
    parser.add_argument("--push", action="store_true", help="Upload to an existing Space.")
    parser.add_argument("--space", help="Space id, owner/name (with --push).")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Stage the Hermes ZeroGPU Space")

    commit = resolve_commit(args.commit, allow_dirty=args.allow_dirty)
    remote_branches = _git("branch", "-r", "--contains", commit)
    staged = stage_space(REPO_ROOT / SPACE_SOURCE_DIR, args.record, args.out, commit=commit)

    print(f"  kleos-models : {commit}")
    if not remote_branches:
        print(
            "  ! this commit is on no remote-tracking branch. Push it before the Space "
            "builds, or the install will fail."
        )
    print(f"  staged into  : {args.out}")
    for path in staged:
        print(f"    {path.name:<20} {path.stat().st_size:>7} B  sha256 {file_sha256(path)[:16]}…")

    if not args.push:
        print(
            "\n✓ Staged and checked. Review the files, then push with --push --space, or\n"
            "  upload them in the Space's Files tab. Nothing was sent anywhere.\n"
        )
        return 0

    if not args.space:
        raise ConfigError("--push needs --space owner/name.")
    if args.allow_dirty:
        raise ConfigError("Refusing to push a Space staged from a dirty working tree.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ConfigError("HF_TOKEN is not set.", suggestions=["Use a token with write access."])

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # The Space is created by hand, where its hardware (ZeroGPU) and visibility
    # are chosen. This script never creates one, and never changes either.
    info = api.space_info(args.space)
    hardware = getattr(getattr(info, "runtime", None), "hardware", None)
    print(f"\n  space        : {args.space} ({'private' if info.private else 'PUBLIC'})")
    print(f"  hardware     : {hardware or 'unknown'}")
    if not info.private:
        print(
            "  ! The Space is public. Anyone can see its files and call it; requests "
            "without the X-Hermes-Key header are refused. See docs/deployment.md."
        )
    result = api.upload_folder(
        folder_path=str(args.out),
        repo_id=args.space,
        repo_type="space",
        commit_message=f"Hermes v0.0.6 Space at kleos-models {commit[:12]}",
    )
    print(f"  pushed       : Space commit {result.oid}")
    print("\n✓ Pushed. The Space rebuilds now; watch its logs for the startup report.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
