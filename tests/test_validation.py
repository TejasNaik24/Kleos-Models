"""Dataset validation and coverage tests (spec §25, §30).

Validation exists to catch problems that make an experiment meaningless:
duplicates, coverage gaps, unreviewed data, placeholder text, private content
that must never reach a public repository, and examples that teach a private fact
rather than a policy.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_example

from kleos_models.data.coverage import build_coverage_report, render_coverage_report
from kleos_models.data.schemas import TrainingExample
from kleos_models.data.validation import (
    approximate_token_count,
    build_quality_report,
    render_quality_report,
    scan_sensitive_content,
    validate_examples,
)


def build(example_id="test-000001", **kwargs) -> TrainingExample:
    return TrainingExample.model_validate(make_example(example_id, **kwargs))


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


class TestBasicValidation:
    def test_valid_examples_produce_no_errors(self, training_examples):
        report = validate_examples(training_examples)
        assert report.ok
        assert not report.errors

    def test_empty_dataset_is_an_error(self):
        report = validate_examples([])
        assert not report.ok
        assert "empty_dataset" in codes(report)

    def test_duplicate_ids_are_an_error(self):
        examples = [build("dup-000001"), build("dup-000001", assistant="Different answer here.")]
        report = validate_examples(examples)
        assert "duplicate_ids" in codes(report)
        assert not report.ok

    def test_report_renders_readably(self, training_examples):
        assert "validated" in validate_examples(training_examples).render()

    def test_report_serializes(self, training_examples):
        payload = validate_examples(training_examples).to_dict()
        assert payload["ok"] is True
        assert "findings" in payload


class TestQualityGate:
    def test_unreviewed_examples_warn_by_default(self):
        report = validate_examples([build(quality_status="draft")])
        assert "unreviewed_examples" in codes(report)
        assert report.ok  # a warning, not an error

    def test_unreviewed_examples_can_be_fatal(self):
        report = validate_examples([build(quality_status="draft")], require_reviewed=True)
        assert not report.ok

    def test_rejected_examples_are_always_an_error(self):
        report = validate_examples([build(quality_status="rejected")])
        assert "rejected_examples_present" in codes(report)
        assert not report.ok


class TestCoverageWarnings:
    def test_a_constant_axis_is_flagged(self):
        # Eight examples all at urgency=high cannot support a claim about urgency.
        examples = [build(f"const-{i:06d}", urgency="high") for i in range(8)]
        report = validate_examples(examples)
        assert "constant_variation_axis" in codes(report)

    def test_a_varied_axis_is_not_flagged(self):
        examples = [build(f"varied-{i:06d}", urgency=["high", "low"][i % 2]) for i in range(8)]
        findings = [
            f
            for f in validate_examples(examples).findings
            if f.code == "constant_variation_axis" and f.context.get("axis") == "urgency"
        ]
        assert not findings

    def test_unregistered_axes_are_surfaced(self):
        report = validate_examples([build(made_up_axis="value")])
        assert "unregistered_variation_axes" in codes(report)


class TestContentScanning:
    def test_placeholder_text_is_an_error(self):
        report = validate_examples([build(assistant="TODO: write the real answer")])
        assert "placeholder_content" in codes(report)
        assert not report.ok

    def test_lorem_ipsum_is_caught(self):
        report = validate_examples([build(assistant="Lorem ipsum dolor sit amet.")])
        assert "placeholder_content" in codes(report)

    def test_an_email_address_is_an_error(self):
        # This repository is public; an email in the data is a leak.
        report = validate_examples([build(user="Contact person@company.com about it.")])
        assert "possible_sensitive_content" in codes(report)
        assert not report.ok

    def test_an_api_key_is_an_error(self):
        report = validate_examples(
            [build(user="Use key sk-abcdefghijklmnopqrstuvwxyz012345 to authenticate.")]
        )
        assert "possible_sensitive_content" in codes(report)

    def test_content_scanning_can_be_disabled(self):
        report = validate_examples([build(user="Contact person@company.com")], scan_content=False)
        assert "possible_sensitive_content" not in codes(report)

    def test_scanner_recognizes_several_secret_shapes(self):
        assert "email_address" in scan_sensitive_content("a@b.com")
        assert "aws_access_key" in scan_sensitive_content("AKIAIOSFODNN7EXAMPLE")
        assert "hf_token" in scan_sensitive_content("hf_" + "a" * 34)
        assert not scan_sensitive_content("perfectly ordinary text")


class TestPolicyNotFacts:
    def test_fact_teaching_phrasing_is_flagged(self):
        # Spec §9: examples must teach a decision policy, not a private fact.
        report = validate_examples(
            [build(assistant="Prioritize it because he works at that company.")]
        )
        assert "possible_fact_memorization" in codes(report)

    def test_policy_phrasing_is_not_flagged(self):
        report = validate_examples(
            [
                build(
                    assistant="Prioritize the item with the nearest deadline and strongest evidence."
                )
            ]
        )
        assert "possible_fact_memorization" not in codes(report)

    def test_the_finding_explains_how_to_fix_it(self):
        report = validate_examples([build(assistant="Do it because she works at the lab.")])
        finding = next(f for f in report.findings if f.code == "possible_fact_memorization")
        assert "evidence" in finding.context["guidance"]


class TestLengthChecks:
    def test_overlong_examples_warn(self):
        report = validate_examples([build(user="word " * 400)], max_tokens=50)
        assert "examples_exceed_max_tokens" in codes(report)

    def test_short_examples_do_not_warn(self):
        report = validate_examples([build()], max_tokens=4096)
        assert "examples_exceed_max_tokens" not in codes(report)

    def test_token_approximation_scales_with_length(self):
        assert approximate_token_count("one two three") > approximate_token_count("one")

    def test_a_real_tokenizer_can_be_supplied(self):
        report = validate_examples(
            [build(user="word " * 100)],
            max_tokens=10,
            token_counter=lambda text: len(text.split()),
        )
        assert "examples_exceed_max_tokens" in codes(report)


class TestQualityReport:
    def test_distributions_are_computed(self, training_examples):
        report = build_quality_report(training_examples, version="v1")
        assert report.total_examples == len(training_examples)
        assert report.task_distribution
        assert report.domain_distribution
        assert sum(report.domain_distribution.values()) == len(training_examples)

    def test_token_statistics_are_computed(self, training_examples):
        report = build_quality_report(training_examples)
        assert report.token_stats["max"] >= report.token_stats["median"]
        assert report.token_stats["mean"] > 0

    def test_missing_metadata_is_reported(self):
        report = build_quality_report([build()])
        assert "urgency" in report.missing_metadata

    def test_report_renders(self, training_examples):
        rendered = render_quality_report(build_quality_report(training_examples, version="v1"))
        assert "Dataset quality report" in rendered
        assert "Variation-axis coverage" in rendered

    def test_report_serializes(self, training_examples):
        payload = build_quality_report(training_examples).to_dict()
        assert "task_distribution" in payload
        assert "token_stats" in payload


class TestCoverageReport:
    def test_marginal_coverage_counts_values(self, training_examples):
        report = build_coverage_report(training_examples)
        assert "domain" in report.axes
        assert report.axes["domain"].distinct_values == 4

    def test_joint_coverage_identifies_situation_types(self, training_examples):
        report = build_coverage_report(training_examples, joint_axes=["domain", "urgency"])
        assert report.observed_cells > 0
        assert report.possible_cells >= report.observed_cells

    def test_empty_cells_are_reported(self, training_examples):
        report = build_coverage_report(training_examples, joint_axes=["domain", "urgency"])
        # 4 domains x 3 urgencies = 12 possible, only 8 examples: gaps exist.
        assert report.empty_cells
        assert report.cell_fill_rate < 1.0

    def test_thin_cells_are_reported(self, training_examples):
        report = build_coverage_report(training_examples, joint_axes=["domain"], min_cell_count=5)
        assert report.thin_cells

    def test_constant_axis_is_detected(self):
        examples = [build(f"c-{i:06d}", urgency="high") for i in range(5)]
        report = build_coverage_report(examples)
        assert report.axes["urgency"].is_constant

    def test_imbalance_is_measured(self):
        examples = [build(f"i-{i:06d}", urgency="high" if i < 9 else "low") for i in range(10)]
        assert report_imbalance(examples) == pytest.approx(9.0)

    def test_report_renders_with_the_diversity_reminder(self, training_examples):
        rendered = render_coverage_report(build_coverage_report(training_examples))
        assert "Variation-axis coverage" in rendered
        assert "not diverse because it is large" in rendered

    def test_report_serializes(self, training_examples):
        payload = build_coverage_report(training_examples).to_dict()
        assert "cell_fill_rate" in payload
        assert "joint_counts" in payload


def report_imbalance(examples) -> float:
    return build_coverage_report(examples).axes["urgency"].imbalance_ratio


class TestCommittedFixtures:
    def test_fixtures_validate_cleanly(self, fixture_examples):
        report = validate_examples(fixture_examples, require_reviewed=True)
        assert report.ok, report.render()

    def test_fixtures_contain_no_sensitive_content(self, fixture_examples):
        for example in fixture_examples:
            assert not scan_sensitive_content(example.conversation_text()), (
                f"fixture {example.id} matched a sensitive-content pattern"
            )

    def test_fixtures_cover_several_tasks(self, fixture_examples):
        tasks = {e.task for e in fixture_examples}
        assert len(tasks) >= 3, "fixtures should exercise more than one task"

    def test_fixtures_cover_several_domains(self, fixture_examples):
        domains = {e.variation_axes.domain for e in fixture_examples}
        assert len(domains) >= 3
