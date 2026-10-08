"""Environment capture for reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.compat import library_versions
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, returning ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


@dataclass
class GitInfo:
    """Source-control state of the code that produced a run."""

    available: bool = False
    commit: str = "unavailable"
    branch: str | None = None
    dirty: bool | None = None
    remote: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "remote": self.remote,
            "reason": self.reason,
        }


def capture_git_info(root: Path | str | None = None) -> GitInfo:
    """Capture git state, tolerating a repository that has not been initialized."""
    cwd = Path(root) if root else Path.cwd()

    inside = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if inside != "true":
        return GitInfo(
            available=False,
            reason=(
                "not a git repository — run `git init` and commit so experiments "
                "are traceable to code"
            ),
        )

    commit = _run_git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return GitInfo(
            available=False,
            reason="git repository has no commits yet",
        )

    status = _run_git(["status", "--porcelain"], cwd)
    dirty = bool(status)
    if dirty:
        logger.warning(
            "The working tree has uncommitted changes. This run will not be exactly "
            "reproducible from commit %s alone.",
            commit[:8],
        )

    return GitInfo(
        available=True,
        commit=commit,
        branch=_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        dirty=dirty,
        remote=_run_git(["config", "--get", "remote.origin.url"], cwd),
    )


@dataclass
class EnvironmentSnapshot:
    """Software and hardware context for a run."""

    python_version: str
    platform: str
    machine: str
    processor: str
    hostname_hash: str
    libraries: dict[str, str | None] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    environment_flags: dict[str, str] = field(default_factory=dict)
    in_colab: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "hostname_hash": self.hostname_hash,
            "in_colab": self.in_colab,
            "libraries": self.libraries,
            "gpu": self.gpu,
            "git": self.git,
            "environment_flags": self.environment_flags,
        }

    def render(self) -> str:
        lines = [
            "Environment",
            f"  python      : {self.python_version}",
            f"  platform    : {self.platform} ({self.machine})",
            f"  colab       : {self.in_colab}",
            "",
            "Libraries",
        ]
        for name, version in self.libraries.items():
            lines.append(f"  {name:<16}: {version or 'not installed'}")
        lines.extend(["", "Hardware", f"  {self.gpu.get('name', 'unknown')}"])
        if self.gpu.get("available"):
            lines.append(
                f"  {self.gpu.get('total_memory_gb')} GB total, "
                f"compute capability {self.gpu.get('compute_capability')}, "
                f"bf16 {'yes' if self.gpu.get('bf16_supported') else 'no'}"
            )
        lines.extend(
            [
                "",
                "Git",
                f"  commit      : {self.git.get('commit', 'unavailable')}",
                f"  branch      : {self.git.get('branch') or '-'}",
                f"  dirty       : {self.git.get('dirty')}",
            ]
        )
        if self.git.get("reason"):
            lines.append(f"  note        : {self.git['reason']}")
        return "\n".join(lines)


def _hash_hostname() -> str:
    """Hash the hostname."""
    import hashlib

    return hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]


def detect_colab() -> bool:
    """Whether this process is running inside Google Colab."""
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return True
    return "google.colab" in sys.modules


#: Environment variables worth recording; never one that could hold a credential.
_TRACKED_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "HF_HOME",
    "OMP_NUM_THREADS",
    "COLAB_GPU",
)


def capture_environment(root: Path | str | None = None) -> EnvironmentSnapshot:
    """Capture the full environment snapshot recorded in every manifest."""
    from kleos_models.models.feasibility import probe_gpu

    return EnvironmentSnapshot(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        hostname_hash=_hash_hostname(),
        libraries=library_versions(),
        gpu=probe_gpu().to_dict(),
        git=capture_git_info(root).to_dict(),
        environment_flags={
            name: os.environ[name] for name in _TRACKED_ENV_VARS if name in os.environ
        },
        in_colab=detect_colab(),
    )


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG this pipeline touches."""
    import random

    random.seed(seed)
    record: dict[str, Any] = {"seed": seed, "python_random": True, "deterministic": False}

    try:
        import numpy as np

        np.random.seed(seed)
        record["numpy"] = True
    except ImportError:  # pragma: no cover
        record["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        record["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            record["torch_cuda"] = True
        if deterministic:
            # Disabling cuDNN autotuning costs throughput but makes runs repeatable.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            record["deterministic"] = True
    except ImportError:
        record["torch"] = False

    logger.info("Seeded RNGs with %d (deterministic=%s)", seed, record["deterministic"])
    return record
