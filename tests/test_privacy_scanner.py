"""The private-data scanner must never reprint what it finds.

Its report is read in terminals and CI logs; echoing a credential there leaks
the very thing the scan exists to catch.
"""

from __future__ import annotations

import importlib.util
import re
import sys

from tests.conftest import REPO_ROOT


def load_scanner():
    name = "privacy_scanner_under_test"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "check_no_private_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_a_found_secret_is_masked_in_its_context():
    scanner = load_scanner()
    # Built at runtime so this file cannot trip the scan itself.
    token = "hf_" + "Q" * 34
    line = f"HF_TOKEN={token}"
    rendered = scanner._redact(line, re.search(r"hf_[A-Za-z0-9]{30,}", line))
    assert token not in rendered
    assert "hf_QQQ…" in rendered
    assert "HF_TOKEN=" in rendered  # still enough to locate the hit


def test_findings_in_a_file_never_carry_the_secret(tmp_path):
    scanner = load_scanner()
    token = "hf_" + "Z" * 34
    secret_file = tmp_path / "leaky.env"
    secret_file.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
    findings = scanner.scan_file(secret_file, tmp_path)
    assert findings
    assert all(token not in finding.render(tmp_path) for finding in findings)
