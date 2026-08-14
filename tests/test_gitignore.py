"""Verify .gitignore actually keeps private data out of a public repository.

This file exists because a .gitignore is easy to get subtly wrong and impossible
to notice: the failure mode is a file being committed, which nobody sees until it
is already public and in the history.

Two mistakes it guards against, both of which were real:

1. **Blocklist thinking.** Listing `data/raw/` and `*.jsonl` only blocks the
   filenames someone thought of. `data/my_export.jsonl` sails through. The
   `data/` rules are deny-by-default for this reason.
2. **Trailing comments.** Git does not support them. `!data/schema/  # note` is a
   literal pattern that matches nothing, silently un-ignoring nothing.

The tests run git against a throwaway repository containing only the real
.gitignore, so they check git's actual behaviour rather than re-implementing its
matching rules.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from tests.conftest import REPO_ROOT

#: Paths that must never reach a public commit.
MUST_BE_IGNORED = [
    # The private artifact, wherever someone puts it.
    "data/raw/conversations.jsonl",
    "data/raw/nested/deep/export.jsonl",
    "data/processed/train.jsonl",
    "data/private/resume.pdf",
    "data/exports/dump.jsonl",
    # Loose files with names no blocklist would predict — the real gap.
    "data/kleos_real_export.jsonl",
    "data/my_private_memories.json",
    "data/supabase_dump.csv",
    "data/manifest_from_private_repo.json",
    "data/notes.txt",
    # A whole dataset version dropped in from the private repo.
    "data/kleos-policy-v0.1.0/train.jsonl",
    "data/kleos-policy-v0.1.0/manifest.json",
    # Dataset artifacts elsewhere in the tree.
    "dataset/train.jsonl",
    "dataset/validation.jsonl",
    "dataset/test.jsonl",
    # Secrets and training outputs.
    ".env",
    ".env.local",
    "outputs/run-1/adapter/adapter_model.safetensors",
    "checkpoints/checkpoint-100/optimizer.pt",
    "reports/leakage/leakage_report.json",
    "wandb/run-abc/logs",
    "credentials.json",
    "key.pem",
]

#: Paths that must stay tracked. These contain no real data: the schema
#: describes shape, the fixtures are synthetic and generated.
MUST_BE_TRACKED = [
    "data/README.md",
    "data/schema/training_example.schema.json",
    "data/schema/evaluation_example.schema.json",
    "data/schema/dataset_manifest.schema.json",
    "data/examples/README.md",
    "data/examples/manifest.json",
    "data/examples/synthetic_train.jsonl",
    "data/examples/synthetic_eval.jsonl",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "outputs/.gitkeep",
    ".env.example",
    "src/kleos_models/config.py",
    "configs/training/qlora_small.yaml",
    "scripts/train.py",
    "tests/conftest.py",
]


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Path:
    """A throwaway git repo carrying the real .gitignore and dummy files."""
    if shutil.which("git") is None:  # pragma: no cover - git is normally present
        pytest.skip("git is not installed")

    root = tmp_path_factory.mktemp("gitignore-sandbox")
    _git("init", "-q", ".", cwd=root)
    shutil.copy(REPO_ROOT / ".gitignore", root / ".gitignore")

    for relative in [*MUST_BE_IGNORED, *MUST_BE_TRACKED]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    return root


def is_ignored(sandbox: Path, relative: str) -> bool:
    return _git("check-ignore", "-q", relative, cwd=sandbox).returncode == 0


@pytest.mark.parametrize("relative", MUST_BE_IGNORED)
def test_private_paths_are_ignored(sandbox, relative):
    assert is_ignored(sandbox, relative), (
        f"{relative} would be COMMITTED to a public repository. "
        "The data/ rules are deny-by-default; check the allowlist in .gitignore."
    )


@pytest.mark.parametrize("relative", MUST_BE_TRACKED)
def test_safe_paths_are_tracked(sandbox, relative):
    assert not is_ignored(sandbox, relative), (
        f"{relative} is ignored but must be committed. "
        "Check the negation patterns in .gitignore — note that git does not "
        "support trailing comments on a pattern line."
    )


def test_staging_everything_admits_only_the_allowlist(sandbox):
    """The decisive check: what does `git add -A` actually stage under data/?"""
    _git("add", "-A", cwd=sandbox)
    staged = {
        line[3:].strip()
        for line in _git("status", "--porcelain", cwd=sandbox).stdout.splitlines()
        if line.strip()
    }
    staged_data = {p for p in staged if p.startswith("data/")}

    expected = {p for p in MUST_BE_TRACKED if p.startswith("data/")}
    assert staged_data == expected, (
        "the set of data/ files git would commit does not match the allowlist.\n"
        f"unexpected: {sorted(staged_data - expected)}\n"
        f"missing:    {sorted(expected - staged_data)}"
    )


def test_gitignore_has_no_trailing_comments_on_patterns():
    """Trailing comments are not supported and silently break a pattern.

    A line like `!data/schema/  # keep this` is a literal pattern containing
    spaces and a hash. It matches nothing, so the negation never takes effect and
    the files stay ignored without any error.
    """
    offenders = []
    for number, line in enumerate((REPO_ROOT / ".gitignore").read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            offenders.append(f"line {number}: {line}")

    assert not offenders, (
        "these .gitignore lines have trailing comments, which git treats as part "
        "of the pattern:\n" + "\n".join(offenders)
    )


def test_committed_data_files_contain_no_real_data():
    """Everything under data/ that is tracked must be synthetic or a schema."""
    from kleos_models.data.validation import scan_sensitive_content

    for path in (REPO_ROOT / "data" / "examples").glob("*.jsonl"):
        matches = scan_sensitive_content(path.read_text(encoding="utf-8"))
        assert not matches, f"{path.name} matched sensitive patterns: {matches}"
