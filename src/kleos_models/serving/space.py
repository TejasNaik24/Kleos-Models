# Getting Hermes onto Hugging Face: a private package repo (adapter, frozen tokenizer,
# manifest; no checkpoints or data) and a Space repo (app, README, requirements, serving
# record; no weights, data or secrets, which arrive as Space secrets and pinned downloads).
#
# Checks and pure transformations only; scripts/upload_deployment_package.py and
# scripts/stage_zerogpu_space.py do the I/O.

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kleos_models.data.loaders import file_sha256
from kleos_models.data.validation import scan_sensitive_content
from kleos_models.errors import ConfigError
from kleos_models.publishing import FORBIDDEN_PATTERNS
from kleos_models.serving.manifest import (
    MANIFEST_FILENAME,
    TOKENIZER_DIRNAME,
    DeploymentManifest,
    verify_package,
)
from kleos_models.serving.profile import profile_from_record

_PINNED = re.compile(r"^[0-9a-f]{40}$")


#: Human-readable files the build writes beside the hashed artifacts.
PACKAGE_DOC_FILES = frozenset({"deployment/README.md"})
#: Text files scanned before upload; weights are binary and tokenizer files are hash-pinned.
_SCANNED_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".md", ".txt"})


@dataclass(frozen=True)
class RemoteFile:
    """One file as the Hub reports it after an upload."""

    path: str
    size: int
    blob_id: str | None = None
    lfs_sha256: str | None = None


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file: what the Hub reports as ``blob_id``."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data, usedforsecurity=False).hexdigest()


def package_upload_files(package_dir: Path | str) -> tuple[DeploymentManifest, list[str]]:
    """Verify a package and list exactly the files that may be uploaded."""
    root = Path(package_dir)
    manifest = verify_package(root)

    recorded = {record.path for record in manifest.all_files()}
    allowed = recorded | {MANIFEST_FILENAME} | PACKAGE_DOC_FILES
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    problems: list[str] = []
    for relative in on_disk:
        if relative not in allowed:
            problems.append(f"{relative}: not part of the package manifest")
            continue
        name = relative.rsplit("/", 1)[-1]
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.match(name):
                problems.append(f"{relative}: forbidden name ({pattern.pattern})")
        if relative.startswith(f"{TOKENIZER_DIRNAME}/"):
            continue
        if Path(relative).suffix.lower() in _SCANNED_SUFFIXES:
            hits = scan_sensitive_content((root / relative).read_text(encoding="utf-8"))
            if hits:
                problems.append(f"{relative}: matched sensitive pattern(s) {', '.join(hits)}")

    if problems:
        raise ConfigError(
            "Refusing to upload this package.",
            details={"problems": problems},
            suggestions=[
                "Rebuild it with scripts/build_deployment_package.py into an empty directory.",
                "Only the adapter, the frozen tokenizer, the packaged model config and "
                "the manifest belong in the package repository.",
            ],
        )
    return manifest, on_disk


def compare_remote_files(
    package_dir: Path | str, files: list[str], remote: dict[str, RemoteFile]
) -> list[str]:
    """Check the Hub holds exactly the bytes that were verified locally."""
    root = Path(package_dir)
    problems: list[str] = []
    for relative in files:
        local = root / relative
        entry = remote.get(relative)
        if entry is None:
            problems.append(f"{relative}: missing from the repository")
            continue
        if entry.size != local.stat().st_size:
            problems.append(
                f"{relative}: {entry.size} bytes remotely, {local.stat().st_size} locally"
            )
            continue
        if entry.lfs_sha256 is not None:
            if entry.lfs_sha256 != file_sha256(local):
                problems.append(f"{relative}: remote sha256 differs")
        elif entry.blob_id != git_blob_sha1(local):
            problems.append(f"{relative}: remote git blob differs")
    for extra in sorted(set(remote) - set(files) - {".gitattributes"}):
        problems.append(f"{extra}: in the repository but not in the verified package")
    return problems


SPACE_SOURCE_DIR = Path("deploy") / "zerogpu-space"
REQUIREMENTS_TEMPLATE = "requirements.txt.template"
RECORD_NAME = "hermes_record.yaml"
COMMIT_PLACEHOLDER = "{{KLEOS_MODELS_COMMIT}}"
#: The Space repository holds these files and nothing else.
STAGED_FILES = ("README.md", "app.py", "requirements.txt", RECORD_NAME)

#: Base files baked into the image: the HF shards and configs. Not consolidated.safetensors
#: (the same 24.5 GB in Mistral's format), nor the tokenizer, which Hermes' package supplies.
BASE_PRELOAD_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        *(f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)),
    }
)
#: The Gradio version this was written against and the Python the torch wheel targets.
SPACE_SDK = {"sdk": "gradio", "sdk_version": "6.28.0", "python_version": "3.12"}


def parse_front_matter(text: str) -> dict[str, Any]:
    """The YAML block at the top of a Space README."""
    if not text.startswith("---\n"):
        raise ConfigError("The Space README has no YAML front matter.")
    block, separator, _ = text[4:].partition("\n---\n")
    if not separator:
        raise ConfigError("The Space README's front matter is not closed.")
    data = yaml.safe_load(block)
    if not isinstance(data, dict):
        raise ConfigError("The Space README's front matter is not a mapping.")
    return data


def check_space_readme(readme: str, record: dict[str, Any]) -> list[str]:
    """Problems with the Space configuration against the serving record."""
    config = parse_front_matter(readme)
    problems: list[str] = []
    for key, want in {**SPACE_SDK, "app_file": "app.py"}.items():
        if str(config.get(key)) != want:
            problems.append(f"{key} is {config.get(key)!r}, expected {want!r}")

    entries = config.get("preload_from_hub") or []
    if len(entries) != 1:
        problems.append(f"preload_from_hub must have exactly one entry, found {len(entries)}")
        return problems
    parts = str(entries[0]).split()
    if len(parts) != 3:
        problems.append("preload_from_hub entry must be '<repo> <files> <commit>'")
        return problems
    repo, files, commit = parts
    if repo != record["base_model"]:
        problems.append(f"preload repo is {repo!r}, the record serves {record['base_model']!r}")
    if commit != record["revision"]:
        problems.append(f"preload commit is {commit!r}, the record pins {record['revision']!r}")
    listed = set(files.split(","))
    expected = set(profile_from_record(record).base_files)
    if listed != expected:
        problems.append(
            "preload files differ from the base checkpoint's shards and configs: "
            f"extra {sorted(listed - expected)}, "
            f"missing {sorted(expected - listed)}"
        )
    return problems


def render_requirements(template: str, commit: str) -> str:
    """Fill in the kleos-models commit the Space installs."""
    if not _PINNED.match(commit):
        raise ConfigError(
            f"The kleos-models commit must be a 40-character sha, got {commit!r}.",
            suggestions=["A branch name would let the Space install code that was never tested."],
        )
    if template.count(COMMIT_PLACEHOLDER) != 1:
        raise ConfigError(f"The requirements template must contain {COMMIT_PLACEHOLDER} once.")
    rendered = template.replace(COMMIT_PLACEHOLDER, commit)
    if "{{" in rendered:
        raise ConfigError("The requirements template has an unfilled placeholder.")
    return rendered


def stage_space(
    source_dir: Path | str,
    record_path: Path | str,
    out_dir: Path | str,
    *,
    commit: str,
) -> list[Path]:
    """Render the Space repository into ``out_dir``, checked and allowlisted."""
    source = Path(source_dir)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ConfigError(
            f"{out} is not empty.",
            suggestions=["Stage into a new directory; the Space holds only the staged files."],
        )

    record_text = Path(record_path).read_text(encoding="utf-8")
    record = (yaml.safe_load(record_text) or {}).get("deployment") or {}
    record_name = profile_from_record(record).record_file
    readme = (source / "README.md").read_text(encoding="utf-8")
    problems = check_space_readme(readme, record)
    if problems:
        raise ConfigError(
            "The Space README disagrees with the serving record.",
            details={"problems": problems},
        )

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "README.md", out / "README.md")
    shutil.copyfile(source / "app.py", out / "app.py")
    template = (source / REQUIREMENTS_TEMPLATE).read_text(encoding="utf-8")
    (out / "requirements.txt").write_text(render_requirements(template, commit), encoding="utf-8")
    (out / record_name).write_text(record_text, encoding="utf-8")

    staged = sorted(out.iterdir())
    names = sorted(path.name for path in staged)
    if names != sorted((*STAGED_FILES[:-1], record_name)):
        raise ConfigError(f"Unexpected files staged: {names}")
    for path in staged:
        hits = scan_sensitive_content(path.read_text(encoding="utf-8"))
        if hits:
            raise ConfigError(f"{path.name} matched sensitive pattern(s): {', '.join(hits)}")
    return staged
