"""Notebook tests.

Notebooks are the canonical entry point for training, so they get the same
scrutiny as code: valid JSON, no hard-coded secrets, and references that actually
resolve to files and flags that exist.
"""

from __future__ import annotations

import json
import re
import typing
from pathlib import Path

import pytest
from tests.conftest import REPO_ROOT

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
EXPECTED = [
    "00_environment_check.ipynb",
    "01_dataset_validation.ipynb",
    "02_train_qlora.ipynb",
    "03_evaluate.ipynb",
]


def notebooks() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def source_of(notebook: dict, kind: str | None = None) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if kind is None or cell["cell_type"] == kind
    )


@pytest.fixture(scope="module")
def loaded() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in notebooks()}


class TestPresence:
    def test_every_expected_notebook_exists(self):
        names = {p.name for p in notebooks()}
        missing = set(EXPECTED) - names
        assert not missing, (
            f"missing notebook(s): {sorted(missing)}. Run: python scripts/build_notebooks.py"
        )


class TestStructure:
    def test_notebooks_are_valid_nbformat(self, loaded):
        nbformat = pytest.importorskip("nbformat")
        for name in loaded:
            notebook = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
            nbformat.validate(notebook)

    def test_notebooks_have_code_and_prose(self, loaded):
        for name, notebook in loaded.items():
            kinds = [cell["cell_type"] for cell in notebook["cells"]]
            assert "code" in kinds, f"{name} has no code cells"
            assert "markdown" in kinds, f"{name} has no explanatory prose"

    def test_notebooks_request_a_gpu_runtime(self, loaded):
        for name, notebook in loaded.items():
            assert notebook["metadata"].get("accelerator") == "GPU", (
                f"{name} does not request a GPU runtime"
            )

    def test_no_stored_outputs(self, loaded):
        # Stored outputs bloat diffs and can leak data from a previous run.
        for name, notebook in loaded.items():
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    assert not cell.get("outputs"), f"{name} cell {index} has stored output"


class TestSecrets:
    """A notebook is the easiest place to accidentally commit a token."""

    PATTERNS: typing.ClassVar[list] = [
        (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
        (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "API key"),
        (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS key"),
        (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."), "JWT"),
        (
            re.compile(r"HF_TOKEN\s*=\s*[\"'][^\"'{}\s]{8,}[\"']"),
            "hard-coded HF_TOKEN assignment",
        ),
    ]

    def test_no_hard_coded_secrets(self, loaded):
        for name, notebook in loaded.items():
            text = source_of(notebook)
            for pattern, label in self.PATTERNS:
                assert not pattern.search(text), f"{name} appears to contain a {label}"

    def test_tokens_come_from_colab_secrets(self, loaded):
        for name in ("00_environment_check.ipynb", "02_train_qlora.ipynb"):
            text = source_of(loaded[name])
            if "HF_TOKEN" in text:
                assert "userdata" in text, f"{name} references HF_TOKEN without using Colab secrets"


class TestReferences:
    def test_referenced_scripts_exist(self, loaded):
        pattern = re.compile(r"scripts/([a-z_]+\.py)")
        for name, notebook in loaded.items():
            for script in set(pattern.findall(source_of(notebook))):
                assert (REPO_ROOT / "scripts" / script).exists(), (
                    f"{name} references scripts/{script}, which does not exist"
                )

    def test_referenced_configs_exist(self, loaded):
        pattern = re.compile(r"configs/[\w/]+\.yaml")
        for name, notebook in loaded.items():
            for config in set(pattern.findall(source_of(notebook))):
                assert (REPO_ROOT / config).exists(), (
                    f"{name} references {config}, which does not exist"
                )

    def test_referenced_cli_flags_exist(self, loaded):
        """Flags used in notebooks must be accepted by the scripts they call."""
        import subprocess

        invocations = re.findall(
            r"python (scripts/[a-z_]+\.py)((?:\s+\\?\s*--[\w-]+(?:\s+[^\s\\]+)?)*)",
            source_of(loaded["02_train_qlora.ipynb"]),
        )
        assert invocations, "no script invocations found in the training notebook"

        for script, flag_text in invocations:
            help_text = subprocess.run(
                [str(REPO_ROOT / ".venv" / "bin" / "python"), str(REPO_ROOT / script), "--help"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout
            if not help_text:
                continue
            for flag in re.findall(r"--[\w-]+", flag_text):
                assert flag in help_text, f"{script} does not accept {flag}"


class TestTrainingNotebookContent:
    """The training notebook is the canonical entry point; it must cover the basics."""

    @pytest.fixture
    def training(self, loaded):
        return loaded["02_train_qlora.ipynb"]

    def test_it_warns_against_reinstalling_torch(self, training):
        # Reinstalling torch on Colab breaks CUDA. This must be stated.
        text = source_of(training)
        assert "colab_setup" in text
        assert "torch" in text.lower()

    def test_it_checks_feasibility_before_training(self, training):
        text = source_of(training)
        assert "plan_run.py" in text

    def test_it_covers_resume(self, training):
        assert "--resume-from-checkpoint auto" in source_of(training)

    def test_it_covers_drive_persistence(self, training):
        text = source_of(training)
        assert "drive.mount" in text
        assert "MyDrive" in text

    def test_it_does_not_publish_automatically(self, training):
        """Publishing must require an explicit, deliberate action (spec §18)."""
        for cell in training["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if "publish_adapter.py" in source:
                active = [
                    line
                    for line in source.splitlines()
                    if "publish_adapter.py" in line and not line.strip().startswith("#")
                ]
                assert not active, (
                    "the training notebook has an uncommented publish command; "
                    "publishing must never happen as a side effect of training"
                )

    def test_it_states_that_loss_is_not_a_result(self, training):
        text = source_of(training, "markdown").lower()
        assert "loss curve is not a result" in text


class TestEvaluationNotebookContent:
    @pytest.fixture
    def evaluation(self, loaded):
        return loaded["03_evaluate.ipynb"]

    def test_it_runs_both_arms(self, evaluation):
        text = source_of(evaluation)
        assert "arm0_base" in text
        assert "arm2_finetuned" in text

    def test_it_explains_significance(self, evaluation):
        text = source_of(evaluation, "markdown").lower()
        assert "significant" in text

    def test_it_explains_the_ood_delta(self, evaluation):
        assert "ood" in source_of(evaluation, "markdown").lower()
