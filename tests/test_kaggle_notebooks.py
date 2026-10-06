"""The Kaggle notebooks that train and evaluate Logos v0.0.2 (H9).

Kaggle runs them unattended, so a mistake surfaces hours into a 12-hour session
or not at all. These tests hold what can be checked before then: the files are
what the builder writes, every code cell is valid Python, every script, flag
and config they name exists, the paths are the ones H9's config hash was
computed with, and nothing in them publishes or carries a secret.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Any

import pytest
from tests import test_notebooks
from tests.conftest import REPO_ROOT
from tests.test_logos_v002_config import (
    EXPERIMENT_ID,
    KAGGLE_DATASET,
    KAGGLE_OUTPUTS,
    LOGOS_V002_HASH,
)
from tests.test_notebooks import source_of

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_kaggle_notebooks as builder

KAGGLE_DIR = REPO_ROOT / "notebooks" / "kaggle"
EXPECTED = ("logos_v002_train.ipynb", "logos_v002_evaluate.ipynb")


@pytest.fixture(scope="module")
def loaded() -> dict[str, dict[str, Any]]:
    return {name: json.loads((KAGGLE_DIR / name).read_text(encoding="utf-8")) for name in EXPECTED}


def code_cells(notebook: dict[str, Any]) -> list[str]:
    return ["\n".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def run_calls(notebook: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """(script, flags) for every ``run(PY, "scripts/x.py", ...)`` in the notebook."""
    calls = []
    for cell in code_cells(notebook):
        for args in re.findall(r'run\(PY, "(scripts/[a-z_]+\.py)"(.*?)\)\n', cell + "\n", re.S):
            calls.append((args[0], re.findall(r'"(--[\w-]+)"', args[1])))
    return calls


class TestGenerated:
    def test_the_files_are_what_the_builder_writes(self):
        for name in EXPECTED:
            path = KAGGLE_DIR / name
            assert path.exists(), f"Run: python scripts/build_kaggle_notebooks.py ({name})"
            assert path.read_text(encoding="utf-8") == builder.render(builder.NOTEBOOKS[name])

    def test_they_are_valid_notebooks(self, loaded):
        nbformat = pytest.importorskip("nbformat")
        for name in loaded:
            nbformat.validate(nbformat.read(KAGGLE_DIR / name, as_version=4))

    def test_every_code_cell_is_python(self, loaded):
        for name, notebook in loaded.items():
            for index, cell in enumerate(code_cells(notebook)):
                ast.parse(cell, filename=f"{name} cell {index}")

    def test_no_stored_outputs(self, loaded):
        for notebook in loaded.values():
            assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")

    def test_no_shell_escapes(self, loaded):
        # A failing !command does not stop an unattended run; run() does.
        for notebook in loaded.values():
            for cell in code_cells(notebook):
                assert not re.search(r"^\s*[!%]", cell, re.M), cell[:80]


class TestPinnedToH9:
    def test_the_paths_are_the_pre_registered_ones(self, loaded):
        for notebook in loaded.values():
            text = source_of(notebook)
            assert f'DATASET = "{KAGGLE_DATASET}"' in text
            assert f'OUTPUTS = "{KAGGLE_OUTPUTS}"' in text
            assert f'EXPERIMENT_ID = "{EXPERIMENT_ID}"' in text

    def test_training_refuses_a_different_config_hash(self, loaded):
        text = source_of(loaded["logos_v002_train.ipynb"])
        assert f'CONFIG_HASH = "{LOGOS_V002_HASH}"' in text
        assert "!= CONFIG_HASH" in text

    def test_a_branch_name_is_refused(self, loaded):
        for notebook in loaded.values():
            assert 'COMMIT = "SET-ME"' in source_of(notebook)
            assert "len(COMMIT) != 40" in source_of(notebook)

    def test_the_sealed_files_are_checked(self, loaded):
        for notebook in loaded.values():
            assert "a4decaaf029b2273" in source_of(notebook)
        assert "a11ffad75f5147f9" in source_of(loaded["logos_v002_evaluate.ipynb"])

    def test_training_smoke_tests_and_gates_before_the_real_run(self, loaded):
        calls = run_calls(loaded["logos_v002_train.ipynb"])
        scripts = [script for script, _ in calls]
        assert scripts.index("scripts/check_smoke_gate.py") < max(
            i for i, s in enumerate(scripts) if s == "scripts/train.py"
        )
        gate = dict(calls)["scripts/check_smoke_gate.py"]
        assert {"--expect-modules", "--expect-trainable", "--min-spare-gb"} <= set(gate)

    def test_both_resume(self, loaded):
        train = source_of(loaded["logos_v002_train.ipynb"])
        assert '"--resume-from-checkpoint", "auto"' in train
        assert '"--resume"' in source_of(loaded["logos_v002_evaluate.ipynb"])

    def test_evaluation_runs_arm2_only(self, loaded):
        text = source_of(loaded["logos_v002_evaluate.ipynb"])
        assert '"--arm", "arm2_finetuned"' in text
        assert "arm1_base_orchestrated" not in text


class TestReferences:
    def test_scripts_and_configs_exist(self, loaded):
        for notebook in loaded.values():
            text = source_of(notebook)
            for script in re.findall(r"scripts/[a-z_]+\.py", text):
                assert (REPO_ROOT / script).exists(), script
            for config in re.findall(r"configs/[\w/]+\.yaml", text):
                assert (REPO_ROOT / config).exists(), config

    def test_every_flag_is_accepted(self, loaded):
        calls = [call for notebook in loaded.values() for call in run_calls(notebook)]
        assert calls
        for script, flags in calls:
            help_text = subprocess.run(
                [sys.executable, str(REPO_ROOT / script), "--help"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout
            assert help_text, f"{script} --help printed nothing"
            for flag in flags:
                assert flag in help_text, f"{script} does not accept {flag}"


class TestNothingLeaves:
    FORBIDDEN = ("git push", "push_to_hub", "huggingface-cli upload", "upload_folder", "HF_TOKEN")

    def test_nothing_is_published(self, loaded):
        for notebook in loaded.values():
            text = source_of(notebook)
            for phrase in self.FORBIDDEN:
                assert phrase not in text, phrase

    def test_no_secrets(self, loaded):
        for name, notebook in loaded.items():
            text = source_of(notebook)
            for pattern, label in test_notebooks.TestSecrets.PATTERNS:
                assert not pattern.search(text), f"{name} appears to contain a {label}"

    def test_weights_stay_out_of_the_saved_output(self, loaded):
        for notebook in loaded.values():
            text = source_of(notebook)
            assert 'os.environ["HF_HOME"] = "/tmp/hf_cache"' in text
            assert 'REPO = "/tmp/kleos-models"' in text


class TestResumeGuard:
    def test_an_incomplete_restored_checkpoint_is_set_aside_loudly(self, loaded):
        # Finding L-F5: a checkpoint without optimizer or scheduler state would
        # resume with a fresh optimizer and a restarted schedule, silently.
        text = source_of(loaded["logos_v002_train.ipynb"])
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
            assert name in text
        assert "incomplete-" in text
        assert "L-F5" in text


class TestUnattendedSessions:
    def test_the_private_dataset_never_reaches_the_saved_output(self, loaded):
        assert KAGGLE_DATASET.startswith("/tmp/")
        for notebook in loaded.values():
            assert "/kaggle/working/data" not in source_of(notebook)

    def test_long_runs_stop_at_the_session_deadline(self, loaded):
        long_runs = {"scripts/train.py", "scripts/evaluate.py"}
        for notebook in loaded.values():
            for cell in code_cells(notebook):
                for script, rest in re.findall(
                    r'run\(PY, "(scripts/[a-z_]+\.py)"(.*?)\)\n', cell + "\n", re.S
                ):
                    if script in long_runs:
                        assert "deadline=True" in rest, script

    def test_a_run_past_the_deadline_stops_cleanly_and_later_runs_are_skipped(
        self, loaded, tmp_path
    ):
        import time

        helpers = next(c for c in code_cells(loaded["logos_v002_train.ipynb"]) if "def run(" in c)
        namespace: dict[str, Any] = {
            "REPO": str(tmp_path),
            # The deadline passes one second into the command.
            "SESSION_START": time.time() - builder.SESSION_HOURS * 3600 + 1,
        }
        exec(compile(helpers, "helpers", "exec"), namespace)
        started = time.time()
        namespace["run"](sys.executable, "-c", "import time; time.sleep(60)", deadline=True)
        assert time.time() - started < 30
        assert namespace["STOPPED"] is True
        assert namespace["run"](sys.executable, "-c", "raise SystemExit(3)") == ""

    def test_a_failing_command_still_stops_the_notebook(self, loaded, tmp_path):
        import time

        helpers = next(c for c in code_cells(loaded["logos_v002_train.ipynb"]) if "def run(" in c)
        namespace: dict[str, Any] = {"REPO": str(tmp_path), "SESSION_START": time.time()}
        exec(compile(helpers, "helpers", "exec"), namespace)
        with pytest.raises(SystemExit, match="FAILED"):
            namespace["run"](sys.executable, "-c", "raise SystemExit(3)")

    def test_child_output_is_unbuffered(self, loaded):
        for notebook in loaded.values():
            text = source_of(notebook)
            assert 'os.environ["PYTHONUNBUFFERED"] = "1"' in text
            assert 'os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"' in text

    def test_an_unfinished_evaluation_says_how_to_resume(self, loaded):
        text = source_of(loaded["logos_v002_evaluate.ipynb"])
        assert "if not os.path.exists(RESULT)" in text


class TestOutputProbe:
    """Before relying on resume: does Kaggle keep a failed version's output?"""

    def test_the_probe_writes_a_marker_and_then_fails(self):
        probe = json.loads((KAGGLE_DIR / "output_probe.ipynb").read_text(encoding="utf-8"))
        cells = code_cells(probe)
        assert "/kaggle/working/probe/marker.txt" in cells[0]
        assert "raise SystemExit" in cells[-1]
        assert builder.render(builder.NOTEBOOKS["output_probe.ipynb"]) == (
            KAGGLE_DIR / "output_probe.ipynb"
        ).read_text(encoding="utf-8")
