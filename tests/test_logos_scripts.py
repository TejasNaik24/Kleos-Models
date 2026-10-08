"""The build, upload and smoke scripts serve Logos under its own profile."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_deployment_package as build  # noqa: E402
import upload_deployment_package as upload  # noqa: E402
import zerogpu_smoke as smoke  # noqa: E402
from kleos_models.errors import ConfigError  # noqa: E402
from kleos_models.serving.manifest import load_expected_identity  # noqa: E402
from kleos_models.serving.profile import HERMES_PROFILE, load_profile  # noqa: E402

LOGOS_RECORD = ROOT / "configs" / "deployment" / "kleos_logos_v002.yaml"
HERMES_RECORD = ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"

FIELDS = {
    "model_name": "kleos-logos",
    "model_version": "v0.0.2",
    "experiment_id": "kleos-v007-ministral314breasoning-run1",
    "base_model": "mistralai/Ministral-3-14B-Reasoning-2512",
    "base_revision": "51f9210f3cd20f3452a80d5819d15dc61cc50630",
    "adapter_sha256": "a" * 64,
    "source_checkpoint": "checkpoint-175",
    "selection_metric": "eval_loss",
    "selection_value": 0.0293,
    "dataset_version": "kleos-policy-v0.0.7",
    "training_config_hash": "d" * 64,
    "fix_mistral_regex": True,
}


class TestPackageReadme:
    def test_hermes_readme_is_rendered_exactly_as_before(self):
        assert build.deployment_readme(FIELDS, HERMES_PROFILE) == build.DEPLOYMENT_README.format(
            **FIELDS
        )

    def test_the_logos_readme_points_at_its_space_and_its_report(self):
        text = build.deployment_readme(FIELDS, load_profile(LOGOS_RECORD))
        assert "HERMES_" not in text and "serve_hermes.py" not in text
        assert "Logos v0.0.2 on ZeroGPU" in text
        assert "kleos-v007-ministral314breasoning-run1-report.md" in text
        assert "kleos-v006-mistralnemo12b-run1-report.md" not in text

    def test_hermes_null_revision_wording_is_kept(self):
        text = build.deployment_readme(FIELDS, HERMES_PROFILE, original_revision=None)
        assert text == build.DEPLOYMENT_README.format(**FIELDS)

    def test_a_research_copy_that_already_pins_the_revision_is_not_called_null(self):
        text = build.deployment_readme(
            FIELDS, load_profile(LOGOS_RECORD), original_revision=FIELDS["base_revision"]
        )
        assert "revision: null" not in text
        assert "already pins the base\nrevision" in text
        assert "deliberate difference" not in text

    def test_a_different_research_revision_is_named(self):
        text = build.deployment_readme(
            FIELDS, load_profile(LOGOS_RECORD), original_revision="0" * 40
        )
        assert "revision: null" not in text
        assert "0" * 40 in text


def fake_hub(monkeypatch, stored: str | None) -> None:
    module = types.ModuleType("huggingface_hub")
    module.get_token = lambda: stored  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


class TestTokens:
    def test_hf_token_wins(self, monkeypatch):
        fake_hub(monkeypatch, "stored")
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert upload.resolve_token() == "from-env"

    def test_the_stored_login_is_used_without_hf_token(self, monkeypatch):
        fake_hub(monkeypatch, "stored")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert upload.resolve_token() == "stored"

    def test_no_token_at_all_says_how_to_log_in(self, monkeypatch):
        fake_hub(monkeypatch, None)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(ConfigError) as raised:
            upload.resolve_token()
        assert "hf auth login" in str(raised.value.suggestions)


def space_status(**overrides) -> dict:
    expected = load_expected_identity(HERMES_RECORD)
    status = {
        "model": {
            "adapter_sha256": expected["adapter_sha256"],
            "base_model": expected["base_model"],
            "base_revision": expected["base_revision"],
        },
        "runtime": expected["runtime"],
        "limits": {"max_new_tokens": 512},
    }
    status.update(overrides)
    return status


class TestSmokeIdentity:
    def test_hermes_identity_still_passes(self):
        expected = load_expected_identity(HERMES_RECORD)
        assert smoke.check_identity(space_status(), expected, HERMES_PROFILE) == []

    def test_a_different_served_budget_is_refused(self):
        expected = load_expected_identity(HERMES_RECORD)
        problems = smoke.check_identity(
            space_status(limits={"max_new_tokens": 256}), expected, HERMES_PROFILE
        )
        assert any("max_new_tokens" in p for p in problems)

    def test_a_reasoning_model_must_report_its_reasoning_reply(self):
        logos = load_profile(LOGOS_RECORD)
        expected = {**load_expected_identity(HERMES_RECORD), "generation": {"max_new_tokens": 512}}
        problems = smoke.check_identity(space_status(), expected, logos)
        assert any("reasoning" in p for p in problems)
        assert smoke.check_identity(space_status(reasoning=True), expected, logos) == []


def reference(max_new_tokens: int | None, traces: bool, count: int = 3) -> dict:
    """An evaluation results payload, shaped like arm2_finetuned.json."""
    records = []
    for index in range(count):
        record = {"example_id": f"e{index}", "response": "answer"}
        if traces:
            record["reasoning"] = "trace"
        records.append(record)
    payload: dict = {"results": records}
    if max_new_tokens is not None:
        payload["generation"] = {"max_new_tokens": max_new_tokens, "do_sample": False}
    return payload


LOGOS_EXPECTED = {"generation": {"max_new_tokens": 1024}}


class TestSmokeReference:
    """The reference is checked before any GPU call: the wrong one wastes a day's quota."""

    def test_logos_v002_results_are_accepted(self):
        logos = load_profile(LOGOS_RECORD)
        assert smoke.check_reference(reference(1024, traces=True), LOGOS_EXPECTED, logos) == []

    def test_logos_v001_results_are_refused(self):
        # Same file name, 512 tokens and no traces: Logos v0.0.1's evaluation.
        logos = load_profile(LOGOS_RECORD)
        problems = smoke.check_reference(reference(512, traces=False), LOGOS_EXPECTED, logos)
        assert any("max_new_tokens" in p for p in problems)
        assert any("trace" in p for p in problems)

    def test_a_thinking_model_needs_traces_to_compare(self):
        logos = load_profile(LOGOS_RECORD)
        problems = smoke.check_reference(reference(1024, traces=False), LOGOS_EXPECTED, logos)
        assert any("trace" in p for p in problems)

    def test_a_reference_without_results_is_refused(self):
        logos = load_profile(LOGOS_RECORD)
        assert smoke.check_reference({"results": []}, LOGOS_EXPECTED, logos)
        assert smoke.check_reference(["not", "a", "payload"], LOGOS_EXPECTED, logos)

    def test_hermes_references_pass_as_before(self):
        expected = load_expected_identity(HERMES_RECORD)
        assert smoke.check_reference(reference(None, traces=False), expected, HERMES_PROFILE) == []
        stated = expected["generation"]["max_new_tokens"]
        assert (
            smoke.check_reference(reference(stated, traces=False), expected, HERMES_PROFILE) == []
        )


def summary(compared: int, exact: int, traces_compared: int, traces_matching: int) -> dict:
    return {
        "compared": compared,
        "exact_matches": exact,
        "reasoning_compared": traces_compared,
        "reasoning_matches": traces_matching,
    }


class TestSmokeOutcome:
    def test_logos_passes_only_when_every_trace_was_compared(self):
        logos = load_profile(LOGOS_RECORD)
        assert smoke.smoke_outcome(summary(9, 9, 9, 9), 9, None, logos) == "pass"
        assert smoke.smoke_outcome(summary(9, 9, 0, 0), 9, None, logos) == "untraced"
        assert smoke.smoke_outcome(summary(9, 9, 8, 8), 9, None, logos) == "untraced"

    def test_a_different_trace_is_a_difference(self):
        logos = load_profile(LOGOS_RECORD)
        assert smoke.smoke_outcome(summary(9, 9, 9, 8), 9, None, logos) == "differs"

    def test_hermes_needs_no_traces(self):
        assert smoke.smoke_outcome(summary(9, 9, 0, 0), 9, None, HERMES_PROFILE) == "pass"
        assert smoke.smoke_outcome(summary(9, 8, 0, 0), 9, None, HERMES_PROFILE) == "differs"

    def test_a_stopped_or_short_run_is_incomplete(self):
        assert smoke.smoke_outcome(summary(5, 5, 0, 0), 9, None, HERMES_PROFILE) == "incomplete"
        assert (
            smoke.smoke_outcome(summary(9, 9, 0, 0), 9, "quota_exhausted", HERMES_PROFILE)
            == "incomplete"
        )
