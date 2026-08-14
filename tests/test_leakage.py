"""Leakage-detection tests (spec §11, §30).

Each detector is tested against a planted case, because a leakage checker that
silently detects nothing is worse than none at all — it produces false confidence
in an evaluation number.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import make_example

from kleos_models.data.leakage import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    LeakageKind,
    check_leakage,
    enforce_leakage_policy,
    jaccard,
    minhash_signature,
    normalize_text,
    shingles,
)
from kleos_models.data.schemas import TrainingExample
from kleos_models.errors import LeakageError


def example(example_id: str, *, user: str = "Rank alpha and beta.", **kwargs) -> TrainingExample:
    return TrainingExample.model_validate(make_example(example_id, user=user, **kwargs))


class TestNormalization:
    def test_case_and_punctuation_are_normalized(self):
        assert normalize_text("The Deadline, is Soon!") == normalize_text("the deadline is soon")

    def test_digits_collapse_to_a_placeholder(self):
        # Changing only a number does not make a scenario new.
        assert normalize_text("due in 3 days") == normalize_text("due in 7 days")

    def test_accents_are_stripped(self):
        assert normalize_text("café") == normalize_text("cafe")

    def test_whitespace_is_collapsed(self):
        assert normalize_text("a   b\n\nc") == "a b c"

    def test_distinct_text_stays_distinct(self):
        assert normalize_text("rank alpha first") != normalize_text("rank beta first")


class TestSimilarity:
    def test_identical_texts_have_jaccard_one(self):
        text = "the quick brown fox jumps"
        assert jaccard(shingles(text), shingles(text)) == 1.0

    def test_unrelated_texts_have_low_jaccard(self):
        left = shingles("prioritize the internship application deadline")
        right = shingles("zzzz qqqq wwww vvvv")
        assert jaccard(left, right) < 0.2

    def test_near_duplicates_score_high(self):
        left = shingles("Rank the internship application before the networking event.")
        right = shingles("Rank the internship application before the networking events.")
        assert jaccard(left, right) > 0.85

    def test_minhash_is_deterministic(self):
        text = shingles("some representative text for hashing")
        assert minhash_signature(text) == minhash_signature(text)

    def test_minhash_approximates_jaccard(self):
        left = shingles("the internship application closes in two days")
        right = shingles("the internship application closes in three days")
        exact = jaccard(left, right)
        left_signature = minhash_signature(left)
        right_signature = minhash_signature(right)
        estimate = sum(
            1 for a, b in zip(left_signature, right_signature, strict=True) if a == b
        ) / len(left_signature)
        assert abs(estimate - exact) < 0.25


class TestExactDuplicates:
    def test_cross_split_exact_duplicate_is_detected(self):
        train = [example("train-001")]
        test = [example("test-001")]  # identical content, different id
        report = check_leakage({"train": train, "test": test})
        findings = report.by_kind(LeakageKind.EXACT_DUPLICATE)
        assert findings
        assert findings[0].is_cross_split

    def test_distinct_examples_are_clean(self):
        train = [example("train-001", user="Rank alpha and beta.")]
        test = [example("test-001", user="Which tool should handle this lookup request?")]
        report = check_leakage({"train": train, "test": test})
        assert not report.by_kind(LeakageKind.EXACT_DUPLICATE)

    def test_within_split_duplicates_are_reported_but_not_cross_split(self):
        train = [example("train-001"), example("train-002")]
        report = check_leakage({"train": train})
        findings = report.by_kind(LeakageKind.EXACT_DUPLICATE)
        assert findings
        assert not findings[0].is_cross_split
        assert not report.fatal_findings

    def test_within_split_detection_can_be_disabled(self):
        train = [example("train-001"), example("train-002")]
        report = check_leakage({"train": train}, within_split=False)
        assert report.clean


class TestNormalizedDuplicates:
    def test_case_only_difference_is_detected(self):
        train = [example("train-001", user="Rank Alpha and Beta.")]
        test = [example("test-001", user="rank alpha and beta")]
        report = check_leakage({"train": train, "test": test})
        kinds = {f.kind for f in report.cross_split_findings}
        assert kinds & {LeakageKind.NORMALIZED_DUPLICATE, LeakageKind.EXACT_DUPLICATE}

    def test_number_only_difference_is_detected(self):
        train = [example("train-001", user="The deadline is in 3 days.")]
        test = [example("test-001", user="The deadline is in 9 days.")]
        report = check_leakage({"train": train, "test": test})
        assert report.cross_split_findings

    def test_exact_duplicates_are_not_double_reported(self):
        train = [example("train-001")]
        test = [example("test-001")]
        report = check_leakage({"train": train, "test": test})
        pairs = [
            (f.left_id, f.right_id)
            for f in report.findings
            if f.kind is LeakageKind.NORMALIZED_DUPLICATE
        ]
        assert not pairs, "a pair already reported as exact should not repeat as normalized"


class TestNearDuplicates:
    def test_near_duplicate_across_splits_is_detected(self):
        train = [
            example(
                "train-001",
                user="Rank the internship application, the networking event, and the resume review.",
            )
        ]
        test = [
            example(
                "test-001",
                user="Rank the internship applications, the networking event, and the resume reviews.",
            )
        ]
        report = check_leakage({"train": train, "test": test})
        assert report.cross_split_findings

    def test_near_duplicate_detection_can_be_disabled(self):
        train = [example("train-001", user="Rank the internship application and the event today.")]
        test = [example("test-001", user="Rank the internship applications and the events today.")]
        report = check_leakage({"train": train, "test": test}, detect_near_duplicates=False)
        assert not report.by_kind(LeakageKind.NEAR_DUPLICATE)

    def test_threshold_is_respected(self):
        train = [example("train-001", user="Prioritize the application deadline this week.")]
        test = [example("test-001", user="Choose which tool answers a scheduling question.")]
        strict = check_leakage({"train": train, "test": test}, threshold=0.99)
        assert not strict.by_kind(LeakageKind.NEAR_DUPLICATE)


class TestIdCollisions:
    def test_same_id_across_splits_is_detected(self):
        train = [example("shared-id", user="Train content here.")]
        test = [example("shared-id", user="Completely different test content.")]
        report = check_leakage({"train": train, "test": test})
        findings = report.by_kind(LeakageKind.ID_COLLISION)
        assert findings
        assert findings[0].is_cross_split


class TestScenarioRepeats:
    def test_scenario_family_spanning_splits_is_detected(self):
        train = [example("train-001", user="Variant A", scenario_family="family-1")]
        test = [
            example(
                "test-001",
                user="Variant B entirely different wording here",
                scenario_family="family-1",
            )
        ]
        report = check_leakage({"train": train, "test": test})
        assert report.by_kind(LeakageKind.SCENARIO_REPEAT)

    def test_distinct_families_are_clean(self):
        train = [example("train-001", user="Variant A", scenario_family="family-1")]
        test = [example("test-001", user="Something quite different", scenario_family="family-2")]
        report = check_leakage({"train": train, "test": test})
        assert not report.by_kind(LeakageKind.SCENARIO_REPEAT)


class TestEntityLeakage:
    def test_entity_group_in_both_splits_is_detected(self):
        train = [example("train-001", user="Train text about set A.", entities="set_a")]
        test = [example("test-001", user="Different test text entirely.", entities="set_a")]
        report = check_leakage({"train": train, "test": test})
        assert report.by_kind(LeakageKind.ENTITY_LEAK)

    def test_disjoint_entity_groups_are_clean(self):
        train = [example("train-001", user="Train text about set A.", entities="set_a")]
        test = [example("test-001", user="Different test text entirely.", entities="set_b")]
        report = check_leakage({"train": train, "test": test})
        assert not report.by_kind(LeakageKind.ENTITY_LEAK)


class TestReportOutput:
    def test_report_serializes_to_json(self, tmp_path):
        report = check_leakage({"train": [example("ex-a")], "test": [example("ex-b")]})
        paths = report.write(tmp_path)
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert "counts" in payload
        assert "findings" in payload
        assert payload["near_duplicate_threshold"] == DEFAULT_NEAR_DUPLICATE_THRESHOLD

    def test_markdown_summary_is_written(self, tmp_path):
        report = check_leakage({"train": [example("ex-a")], "test": [example("ex-b")]})
        paths = report.write(tmp_path)
        markdown = paths["markdown"].read_text(encoding="utf-8")
        assert "# Leakage report" in markdown

    def test_clean_report_says_so(self, tmp_path):
        report = check_leakage({"train": [example("ex-a", user="Unique training text here.")]})
        assert report.clean
        assert "No leakage detected" in report.render_markdown()


class TestPolicyEnforcement:
    def test_fatal_policy_raises_on_cross_split_duplicate(self):
        report = check_leakage({"train": [example("ex-a")], "test": [example("ex-b")]})
        with pytest.raises(LeakageError, match="cross-split leakage"):
            enforce_leakage_policy(report, fail_on="fatal")

    def test_none_policy_never_raises(self):
        report = check_leakage({"train": [example("ex-a")], "test": [example("ex-b")]})
        enforce_leakage_policy(report, fail_on="none")

    def test_any_policy_raises_on_a_scenario_repeat(self):
        train = [example("train-001", user="Variant A", scenario_family="family-1")]
        test = [
            example(
                "test-001", user="Wholly different wording appears here", scenario_family="family-1"
            )
        ]
        report = check_leakage({"train": train, "test": test})
        with pytest.raises(LeakageError):
            enforce_leakage_policy(report, fail_on="any")

    def test_fatal_policy_tolerates_a_scenario_repeat_alone(self):
        # A scenario repeat warrants review; it is not automatically fatal.
        train = [example("train-001", user="Variant A", scenario_family="family-1")]
        test = [
            example(
                "test-001", user="Wholly different wording appears here", scenario_family="family-1"
            )
        ]
        report = check_leakage({"train": train, "test": test})
        enforce_leakage_policy(report, fail_on="fatal")

    def test_clean_report_passes_every_policy(self):
        # Disjoint entity groups as well as disjoint text: sharing an entity
        # group across splits is itself a (correctly detected) leak.
        report = check_leakage(
            {
                "train": [example("ex-a", user="Unique content one.", entities="set_a")],
                "test": [example("ex-b", user="Entirely separate content two.", entities="set_b")],
            }
        )
        assert report.clean
        for policy in ("none", "fatal", "any"):
            enforce_leakage_policy(report, fail_on=policy)


class TestCommittedFixtures:
    def test_fixtures_have_no_cross_split_leakage(self, fixture_examples, fixture_eval_examples):
        report = check_leakage({"train": fixture_examples, "evaluation": fixture_eval_examples})
        assert not report.cross_split_findings, (
            "the committed fixtures leak between train and eval:\n"
            + "\n".join(f.detail or f.kind.value for f in report.cross_split_findings)
        )
