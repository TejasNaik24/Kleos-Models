#!/usr/bin/env python3
"""Scan the repository for secrets and private data (spec §2, §42).

This repository is public. This script is the automated part of keeping it that
way — a safety net, not a substitute for judgement.

It looks for credentials (API keys, tokens, JWTs, private keys), personal data
(emails, phone numbers), Supabase URLs and keys, and dataset files that should
never be committed.

Runs in CI and can be installed as a pre-commit hook:

    python scripts/check_no_private_data.py --install-hook

Usage::

    python scripts/check_no_private_data.py .
    python scripts/check_no_private_data.py --staged     # pre-commit mode
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: (name, pattern, severity) — "error" blocks a commit, "warn" reports only.
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "error"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}"), "error"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "error"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "error"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), "error"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "error"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "error"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "error",
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        "error",
    ),
    ("supabase_url", re.compile(r"https://[a-z0-9]{20}\.supabase\.co"), "error"),
    (
        "supabase_service_key",
        re.compile(r"\bservice_role\b.{0,40}\beyJ", re.DOTALL),
        "error",
    ),
    (
        "assigned_secret",
        re.compile(
            r"\b(?:api[_-]?key|secret|password|passwd|token|credential)\s*[:=]\s*"
            r"['\"][A-Za-z0-9_\-./+]{16,}['\"]",
            re.IGNORECASE,
        ),
        "error",
    ),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{24,}"), "error"),
    ("email_address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "warn"),
    (
        "us_phone",
        re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "warn",
    ),
]

#: Emails that are legitimately present in a public repository.
EMAIL_ALLOWLIST = re.compile(
    r"(?:noreply@|example\.com|example\.org|your[-_]?email|user@host|"
    r"\.png|\.jpg|@\{|@example|name@domain)",
    re.IGNORECASE,
)

#: Paths never scanned.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        "outputs",
        "checkpoints",
        "wandb",
        "runs",
        ".ipynb_checkpoints",
        "build",
        "dist",
        ".eggs",
        "htmlcov",
    }
)

SKIP_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ckpt",
        ".onnx",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".woff",
        ".woff2",
        ".ttf",
        ".ico",
        ".lock",
    }
)

#: Files whose presence is itself a problem in this public repository.
FORBIDDEN_PATHS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\.env$"), ".env must never be committed"),
    (re.compile(r"^\.env\.(?!example)"), "environment files must never be committed"),
    (re.compile(r"^data/raw/(?!\.gitkeep)"), "data/raw/ is for private data and must stay empty"),
    (
        re.compile(r"^data/processed/(?!\.gitkeep)"),
        "data/processed/ holds generated datasets and must not be committed",
    ),
    (re.compile(r".*\.pem$|.*\.key$|.*\.p12$"), "key material must never be committed"),
    (re.compile(r"^outputs/(?!\.gitkeep)"), "training outputs must not be committed"),
)

#: Files that must contain the patterns this scanner looks for, because they
#: define or test the detection itself. Everything listed here is reviewed on
#: the understanding that its "secrets" are documentation examples or test
#: fixtures — never real credentials.
#:
#: Keep this list short. Exempting a file turns the scanner off for it.
SELF_EXEMPT = frozenset(
    {
        "scripts/check_no_private_data.py",  # the patterns themselves
        "src/kleos_models/data/validation.py",  # dataset content scanner
        "src/kleos_models/publishing.py",  # upload safety scanner
        "tests/test_validation.py",  # asserts the scanner detects samples
        "tests/test_notebooks.py",  # asserts notebooks contain no secrets
        "docs/privacy.md",
        "SECURITY.md",
        ".env.example",
    }
)

MAX_FILE_BYTES = 2 * 1024 * 1024


@dataclass
class Finding:
    """One scanner hit."""

    path: Path
    line_number: int
    pattern: str
    severity: str
    excerpt: str

    def render(self, root: Path) -> str:
        try:
            location = self.path.relative_to(root)
        except ValueError:
            location = self.path
        icon = "✗" if self.severity == "error" else "!"
        return f"  {icon} {location}:{self.line_number}  [{self.pattern}]  {self.excerpt}"


def _iter_files(root: Path, paths: list[Path] | None = None):
    """Yield files to scan."""
    if paths:
        for path in paths:
            if path.is_file():
                yield path
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def _redact(line: str, match: re.Match[str]) -> str:
    """Show enough context to locate the hit without reprinting the secret."""
    text = match.group(0)
    shown = text[:6] + "…" if len(text) > 8 else "…"
    snippet = line.strip()[:100]
    return f"{shown!r} in {snippet[:60]!r}"


def scan_file(path: Path, root: Path) -> list[Finding]:
    """Scan one file for sensitive patterns."""
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    if relative in SELF_EXEMPT:
        return []

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    findings: list[Finding] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for name, pattern, severity in PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            if name == "email_address" and EMAIL_ALLOWLIST.search(match.group(0)):
                continue
            findings.append(
                Finding(
                    path=path,
                    line_number=line_number,
                    pattern=name,
                    severity=severity,
                    excerpt=_redact(line, match),
                )
            )
    return findings


def check_forbidden_paths(root: Path, paths: list[Path] | None = None) -> list[tuple[Path, str]]:
    """Find files that must not exist in this repository at all."""
    problems: list[tuple[Path, str]] = []
    for path in _iter_files(root, paths):
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            continue
        for pattern, reason in FORBIDDEN_PATHS:
            if pattern.match(relative):
                problems.append((path, reason))
                break
    return problems


def _staged_files(root: Path) -> list[Path]:
    """Files staged for commit."""
    try:
        output = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    return [root / name for name in output.splitlines() if (root / name).is_file()]


HOOK_SCRIPT = """#!/bin/sh
# KLEOS pre-commit hook: block secrets and private data.
exec python3 scripts/check_no_private_data.py --staged
"""


def install_hook(root: Path) -> int:
    """Install this scanner as a git pre-commit hook."""
    hooks_dir = root / ".git" / "hooks"
    if not hooks_dir.exists():
        print("✗ No .git/hooks directory. Run `git init` first.", file=sys.stderr)
        return 1
    hook_path = hooks_dir / "pre-commit"
    if hook_path.exists():
        print(f"! {hook_path} already exists; not overwriting.")
        print("  Add this line to it manually:")
        print("    python3 scripts/check_no_private_data.py --staged")
        return 0
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    hook_path.chmod(0o755)
    print(f"✓ Installed pre-commit hook at {hook_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", nargs="?", type=Path, default=REPO_ROOT, help="Directory to scan.")
    parser.add_argument("--staged", action="store_true", help="Scan only git-staged files.")
    parser.add_argument("--install-hook", action="store_true", help="Install the pre-commit hook.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures too.")
    args = parser.parse_args(argv)

    root = args.path.resolve()

    if args.install_hook:
        return install_hook(root)

    targets = _staged_files(root) if args.staged else None
    if args.staged and not targets:
        print("✓ No staged files to scan.")
        return 0

    print()
    print("=" * 72)
    print(f"Private-data scan — {root}")
    print("=" * 72)

    findings: list[Finding] = []
    scanned = 0
    for path in _iter_files(root, targets):
        scanned += 1
        findings.extend(scan_file(path, root))

    forbidden = check_forbidden_paths(root, targets)

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]

    print(f"\n  scanned  : {scanned} file(s)")
    print(f"  errors   : {len(errors)}")
    print(f"  warnings : {len(warnings)}")
    print(f"  forbidden paths : {len(forbidden)}")

    if forbidden:
        print("\nFiles that must not exist in this public repository:")
        for path, reason in forbidden:
            print(f"  ✗ {path.relative_to(root)}: {reason}")

    if errors:
        print("\nSecrets or credentials detected:")
        for finding in errors[:40]:
            print(finding.render(root))

    if warnings:
        print("\nPossible personal data (review these):")
        for finding in warnings[:20]:
            print(finding.render(root))
        if len(warnings) > 20:
            print(f"  … {len(warnings) - 20} more")

    failed = bool(errors) or bool(forbidden) or (args.strict and bool(warnings))

    print()
    if failed:
        print("✗ Private-data scan FAILED.", file=sys.stderr)
        print(
            "\n  This repository is PUBLIC. Do not commit until these are resolved.",
            file=sys.stderr,
        )
        print("  If a hit is a false positive, add the file to SELF_EXEMPT or", file=sys.stderr)
        print("  rewrite the line so it cannot be mistaken for a real secret.", file=sys.stderr)
        print("\n  If a secret was already committed, rotate it immediately —", file=sys.stderr)
        print(
            "  removing it from the working tree does not remove it from history.", file=sys.stderr
        )
        print("  See SECURITY.md.\n", file=sys.stderr)
        return 1

    print("✓ No secrets or private data detected.")
    if warnings:
        print(f"  ({len(warnings)} warning(s) above are worth a look.)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
