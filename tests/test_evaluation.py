"""Evaluation tests (spec §21-§24, §30, §54).

Metric calculations, grader behaviour, consistency, OOD separation, capability
deltas and report generation. Several tests exist specifically to pin
research-integrity properties: OOD must not be reported when unmeasurable, and a
non-significant improvement must not be described as a win.
"""

from __future__ import annotations

import math

import pytest

from kleos_models.config import EvaluationConfig, GenerationConfig
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.capability import build_capability_delta
from kleos_models.evaluation.consistency import build_consistency_groups
from kleos_models.evaluation.faithfulness import assess_faithfulness, extract_claims
from kleos_models.evaluation.graders import (
    ClassificationGrader,
    ExactMatchGrader,
    HeuristicRubricGrader,
    LLMJudgeGrader,
    RankingGrader,
    SetMatchGrader,
    extract_json,
    extract_label,
    extract_ranking,
    get_grader,
)
from kleos_models.evaluation.metrics import (
    bootstrap_difference,
    classification_scores,
    exact_match,
    kendall_tau,
    ndcg,
    normalize_answer,
    set_precision_recall_f1,
    spearman_footrule,
    summarize,
    token_f1,
    top_1_accuracy,
)
from kleos_models.evaluation.ood import build_ood_report, compare_ood_reports
from kleos_models.evaluation.reports import compare_results, render_markdown_report
from kleos_models.evaluation.runner import run_evaluation
from kleos_models.inference.backends import EchoBackend


class TestNormalization:
    def test_case_articles_and_punctuation_are_normalized(self):
        assert normalize_answer("The Q3 Report.") == normalize_answer("q3 report")

    def test_distinct_answers_stay_distinct(self):
        assert normalize_answer("alpha") != normalize_answer("beta")


class TestTextMetrics:
    def test_exact_match_is_one_for_equal_answers(self):
        assert exact_match("alpha", "The alpha.") == 1.0

    def test_exact_match_is_zero_for_different_answers(self):
        assert exact_match("alpha", "beta") == 0.0

    def test_token_f1_is_one_for_identical_text(self):
        assert token_f1("the quick fox", "quick the fox") == 1.0

    def test_token_f1_is_zero_for_disjoint_text(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_token_f1_is_partial_for_overlap(self):
        score = token_f1("alpha beta gamma", "alpha beta delta")
        assert 0.0 < score < 1.0


class TestClassificationScores:
    def test_perfect_predictions(self):
        scores = classification_scores(["a", "b", "a"], ["a", "b", "a"])
        assert scores.accuracy == 1.0
        assert scores.macro_f1 == 1.0

    def test_all_wrong_predictions(self):
        scores = classification_scores(["a", "a"], ["b", "b"])
        assert scores.accuracy == 0.0

    def test_macro_f1_penalizes_a_majority_class_predictor(self):
        # 9 of 10 are "no"; always predicting "no" scores 0.9 accuracy but
        # should not look competent on macro F1.
        references = ["no"] * 9 + ["yes"]
        predictions = ["no"] * 10
        scores = classification_scores(predictions, references)
        assert scores.accuracy == pytest.approx(0.9)
        assert scores.macro_f1 < 0.5

    def test_per_class_scores_are_reported(self):
        scores = classification_scores(["a", "b"], ["a", "a"])
        assert "a" in scores.per_class
        assert scores.support["a"] == 2

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            classification_scores(["a"], ["a", "b"])

    def test_empty_input_is_handled(self):
        assert classification_scores([], []).accuracy == 0.0


class TestSetMetrics:
    def test_perfect_set_match(self):
        assert set_precision_recall_f1(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)

    def test_partial_set_match(self):
        precision, recall, f1 = set_precision_recall_f1(["a", "c"], ["a", "b"])
        assert precision == 0.5
        assert recall == 0.5
        assert f1 == 0.5

    def test_empty_prediction_scores_zero(self):
        assert set_precision_recall_f1([], ["a"]) == (0.0, 0.0, 0.0)

    def test_empty_reference_and_prediction_is_perfect(self):
        assert set_precision_recall_f1([], []) == (1.0, 1.0, 1.0)


class TestRankingMetrics:
    def test_ndcg_is_one_for_a_perfect_ranking(self):
        assert ndcg(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_ndcg_is_lower_for_a_reversed_ranking(self):
        assert ndcg(["c", "b", "a"], ["a", "b", "c"]) < 1.0

    def test_ndcg_rewards_the_correct_top_item(self):
        good = ndcg(["a", "c", "b"], ["a", "b", "c"])
        bad = ndcg(["c", "b", "a"], ["a", "b", "c"])
        assert good > bad

    def test_ndcg_handles_an_empty_reference(self):
        assert ndcg(["a"], []) == 0.0

    def test_kendall_tau_is_one_for_identical_orderings(self):
        assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_kendall_tau_is_zero_for_reversed_orderings(self):
        assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == pytest.approx(0.0)

    def test_kendall_tau_is_neutral_when_uncomparable(self):
        assert kendall_tau(["a"], ["a"]) == 0.5

    def test_footrule_is_one_for_identical_orderings(self):
        assert spearman_footrule(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_footrule_drops_when_items_move(self):
        assert spearman_footrule(["c", "b", "a"], ["a", "b", "c"]) < 1.0

    def test_top_1_accuracy(self):
        assert top_1_accuracy(["a", "z"], ["a", "b"]) == 1.0
        assert top_1_accuracy(["z", "a"], ["a", "b"]) == 0.0
        assert top_1_accuracy([], ["a"]) == 0.0


class TestSummaries:
    def test_mean_and_dispersion(self):
        summary = summarize("m", [0.0, 1.0, 0.5])
        assert summary.mean == pytest.approx(0.5)
        assert summary.count == 3
        assert summary.std > 0

    def test_single_value_has_no_standard_error(self):
        assert summarize("m", [0.7]).stderr == 0.0

    def test_empty_values_are_handled(self):
        summary = summarize("m", [])
        assert summary.mean == 0.0
        assert summary.count == 0

    def test_confidence_interval_brackets_the_mean(self):
        summary = summarize("m", [0.2, 0.4, 0.6, 0.8])
        low, high = summary.confidence_interval()
        assert low < summary.mean < high


class TestBootstrap:
    def test_a_clear_improvement_is_significant(self):
        base = [0.0] * 30
        finetuned = [1.0] * 30
        result = bootstrap_difference(base, finetuned, iterations=500, seed=1)
        assert result["difference"] == pytest.approx(1.0)
        assert result["significant_at_05"] is True

    def test_identical_scores_are_not_significant(self):
        scores = [0.5] * 30
        result = bootstrap_difference(scores, scores, iterations=500, seed=1)
        assert result["difference"] == 0.0
        assert result["significant_at_05"] is False

    def test_a_tiny_noisy_difference_is_not_significant(self):
        # This is the case the reporting layer must not describe as a win.
        base = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5]
        finetuned = [0.52, 0.58, 0.44, 0.5, 0.47, 0.53]
        result = bootstrap_difference(base, finetuned, iterations=800, seed=3)
        assert result["significant_at_05"] is False

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="equal-length"):
            bootstrap_difference([0.1], [0.1, 0.2])

    def test_empty_input_is_handled(self):
        assert bootstrap_difference([], [])["n"] == 0


class TestResponseParsing:
    def test_json_in_a_fenced_block_is_extracted(self):
        assert extract_json('Here:\n```json\n{"label": "a"}\n```') == {"label": "a"}

    def test_bare_json_is_extracted(self):
        assert extract_json('The answer is {"label": "a"} okay') == {"label": "a"}

    def test_unparseable_text_returns_none(self):
        assert extract_json("no json here") is None

    def test_numbered_list_becomes_a_ranking(self):
        assert extract_ranking("1. alpha\n2. beta\n3. gamma") == ["alpha", "beta", "gamma"]

    def test_bulleted_list_becomes_a_ranking(self):
        assert extract_ranking("- alpha\n- beta") == ["alpha", "beta"]

    def test_json_array_becomes_a_ranking(self):
        assert extract_ranking('["alpha", "beta"]') == ["alpha", "beta"]

    def test_ranking_falls_back_to_mention_order(self):
        result = extract_ranking("I would do beta then alpha.", candidates=["alpha", "beta"])
        assert result == ["beta", "alpha"]

    def test_label_is_extracted_from_json(self):
        assert extract_label('{"decision": "memory_search"}') == "memory_search"

    def test_label_is_extracted_from_prose(self):
        label = extract_label(
            "I would use memory_search for this.", allowed=["memory_search", "web_search"]
        )
        assert label == "memory_search"

    def test_earliest_mentioned_option_wins(self):
        label = extract_label(
            "Use web_search. Not memory_search.", allowed=["memory_search", "web_search"]
        )
        assert label == "web_search"


class TestGraders:
    def test_exact_match_grader(self):
        result = ExactMatchGrader().grade("alpha", {"text": "alpha"})
        assert result.score == 1.0

    def test_classification_grader_correct(self):
        result = ClassificationGrader().grade(
            "memory_search", {"label": "memory_search", "options": ["memory_search", "none"]}
        )
        assert result.score == 1.0
        assert result.details["predicted_label"] == "memory_search"

    def test_classification_grader_incorrect(self):
        result = ClassificationGrader().grade(
            "none", {"label": "memory_search", "options": ["memory_search", "none"]}
        )
        assert result.score == 0.0

    def test_ranking_grader_scores_a_perfect_ranking_highly(self):
        result = RankingGrader().grade("1. a\n2. b\n3. c", {"ranking": ["a", "b", "c"]})
        assert result.score == pytest.approx(1.0)
        assert result.sub_scores["top_1_accuracy"] == 1.0

    def test_ranking_grader_reports_subscores_separately(self):
        # A good ordering with the wrong top item is still the wrong answer.
        result = RankingGrader().grade("1. b\n2. a\n3. c", {"ranking": ["a", "b", "c"]})
        assert result.sub_scores["top_1_accuracy"] == 0.0
        assert result.sub_scores["ndcg"] > 0.5

    def test_ranking_grader_flags_a_parse_failure(self):
        result = RankingGrader().grade("no ranking here", {"ranking": ["a", "b"]})
        assert result.parse_failed
        assert result.score == 0.0

    def test_ranking_grader_requires_a_reference(self):
        with pytest.raises(EvaluationError, match="non-empty"):
            RankingGrader().grade("1. a", {})

    def test_set_match_grader(self):
        result = SetMatchGrader().grade('["a", "b"]', {"items": ["a", "b"]})
        assert result.score == 1.0

    def test_unknown_grader_is_rejected(self):
        with pytest.raises(EvaluationError, match="Unknown grader"):
            get_grader("nonexistent_grader")


class TestHeuristicRubric:
    def test_covering_required_points_scores_well(self):
        grader = HeuristicRubricGrader()
        result = grader.grade(
            "Recommendation: complete evidence-1 first, since evidence-2 blocks it. "
            "You should prioritize that.",
            {
                "required_points": ["evidence-1", "evidence-2"],
                "evidence_ids": ["evidence-1", "evidence-2"],
                "expected_decision": "complete evidence-1",
            },
        )
        assert result.score > 0.7
        assert result.sub_scores["correctness"] == 1.0

    def test_missing_required_points_is_penalized(self):
        grader = HeuristicRubricGrader()
        result = grader.grade(
            "I recommend something unrelated entirely.",
            {"required_points": ["evidence-1", "evidence-2"], "expected_decision": "x"},
        )
        assert result.sub_scores["critical_omission"] == 1.0
        assert result.score < 0.5

    def test_unsupported_claim_phrasing_is_penalized(self):
        grader = HeuristicRubricGrader()
        with_hedge = grader.grade(
            "Obviously you should prioritize evidence-1. Everyone knows that.",
            {"required_points": ["evidence-1"], "expected_decision": "evidence-1"},
        )
        assert with_hedge.sub_scores["unsupported_claims"] > 0

    def test_forbidden_content_is_penalized(self):
        grader = HeuristicRubricGrader()
        result = grader.grade(
            "You should apply; guaranteed acceptance awaits.",
            {"required_points": [], "forbidden_points": ["guaranteed acceptance"]},
        )
        assert result.sub_scores["unsupported_claims"] > 0

    def test_actionability_requires_a_recommendation(self):
        grader = HeuristicRubricGrader()
        vague = grader.grade("There are several considerations here.", {"required_points": []})
        assert vague.sub_scores["actionability"] == 0.0

    def test_unknown_dimension_is_rejected(self):
        with pytest.raises(EvaluationError, match="Unknown rubric dimension"):
            HeuristicRubricGrader(dimensions=["vibes"])


class TestLLMJudge:
    def test_judge_without_a_callable_raises(self):
        with pytest.raises(EvaluationError, match="no judge function"):
            LLMJudgeGrader().grade("response", {"label": "a"})

    def test_judge_scores_are_normalized(self):
        grader = LLMJudgeGrader(
            judge_fn=lambda prompt: '{"correctness": 4, "evidence_usage": 2}',
            judge_model="test-judge",
            dimensions=["correctness", "evidence_usage"],
        )
        result = grader.grade("response", {"label": "a"})
        assert result.sub_scores["correctness"] == 1.0
        assert result.sub_scores["evidence_usage"] == 0.5
        # The judge identity must be recorded: scores from different judges are
        # not comparable.
        assert result.details["judge_model"] == "test-judge"

    def test_unparseable_judge_output_is_flagged(self):
        grader = LLMJudgeGrader(judge_fn=lambda prompt: "I cannot grade this")
        result = grader.grade("response", {"label": "a"})
        assert result.parse_failed


class TestConsistency:
    def test_agreeing_group_is_consistent(self):
        report = build_consistency_groups(
            example_ids=["a", "b", "c"],
            group_ids=["g1", "g1", "g1"],
            decisions=["alpha", "alpha", "alpha"],
        )
        assert report.agreement_rate == 1.0
        assert report.groups[0].is_consistent

    def test_disagreeing_group_is_inconsistent(self):
        report = build_consistency_groups(
            example_ids=["a", "b"],
            group_ids=["g1", "g1"],
            decisions=["alpha", "beta"],
        )
        assert report.agreement_rate == 0.0
        assert report.inconsistent_groups

    def test_consistent_but_wrong_is_distinguished_from_correct(self):
        # A model that is consistently wrong must not look good.
        report = build_consistency_groups(
            example_ids=["a", "b"],
            group_ids=["g1", "g1"],
            decisions=["beta", "beta"],
            reference_decisions=["alpha", "alpha"],
        )
        assert report.agreement_rate == 1.0
        assert report.correct_agreement_rate == 0.0

    def test_singleton_groups_are_skipped_not_scored(self):
        report = build_consistency_groups(example_ids=["a"], group_ids=["g1"], decisions=["alpha"])
        assert report.evaluated_groups == 0
        assert report.skipped_singletons == 1

    def test_majority_share_is_graded(self):
        report = build_consistency_groups(
            example_ids=["a", "b", "c", "d"],
            group_ids=["g1"] * 4,
            decisions=["alpha", "alpha", "alpha", "beta"],
        )
        assert report.groups[0].majority_share == 0.75
        assert not report.groups[0].is_consistent

    def test_flips_are_attributed_to_a_perturbation(self):
        report = build_consistency_groups(
            example_ids=["a", "b", "c"],
            group_ids=["g1"] * 3,
            decisions=["alpha", "alpha", "beta"],
            perturbation_kinds=["none", "paraphrase", "evidence_order"],
        )
        assert report.flips_by_perturbation().get("evidence_order") == 1

    def test_mismatched_input_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            build_consistency_groups(example_ids=["a"], group_ids=["g"], decisions=[])


class TestOOD:
    def test_scores_are_partitioned_by_tag(self):
        report = build_ood_report(
            scores=[1.0, 1.0, 0.0, 0.0],
            split_tags=["in_distribution", "in_distribution", "ood", "ood"],
        )
        assert report.in_distribution_score == 1.0
        assert report.ood_score == 0.0
        assert report.generalization_gap == 1.0

    def test_report_is_unmeasurable_without_ood_examples(self):
        # Spec §36: no OOD evaluation means no generalization claim.
        report = build_ood_report(scores=[1.0], split_tags=["in_distribution"])
        assert not report.measurable
        assert "NOT MEASURABLE" in report.render()

    def test_per_shift_breakdown_is_produced(self):
        report = build_ood_report(
            scores=[1.0, 0.5, 0.0],
            split_tags=["in_distribution", "ood", "ood"],
            shift_kinds=[None, "unseen_domains", "conflicting_evidence"],
        )
        shifts = {s.shift for s in report.per_shift}
        assert shifts == {"unseen_domains", "conflicting_evidence"}

    def test_worst_shift_is_identified(self):
        report = build_ood_report(
            scores=[1.0, 0.9, 0.1],
            split_tags=["in_distribution", "ood", "ood"],
            shift_kinds=[None, "unseen_domains", "conflicting_evidence"],
        )
        worst = report.worst_shift()
        assert worst is not None
        assert worst.shift == "conflicting_evidence"

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            build_ood_report(scores=[1.0], split_tags=[])

    def test_comparison_flags_in_distribution_gain_with_ood_loss(self):
        base = build_ood_report(scores=[0.5, 0.5], split_tags=["in_distribution", "ood"])
        finetuned = build_ood_report(scores=[0.9, 0.2], split_tags=["in_distribution", "ood"])
        comparison = compare_ood_reports(base, finetuned)
        assert "DEGRADED out-of-distribution" in comparison["verdict"]


class TestCapability:
    def test_no_change_is_not_a_regression(self):
        delta = build_capability_delta(base_scores=[0.8] * 10, fine_tuned_scores=[0.8] * 10)
        assert delta.delta == 0.0
        assert not delta.regressed

    def test_a_drop_is_reported_as_a_regression(self):
        delta = build_capability_delta(base_scores=[0.8] * 10, fine_tuned_scores=[0.6] * 10)
        assert delta.delta < 0
        assert delta.regressed
        assert "REGRESSION" in delta.render()

    def test_relative_delta_is_computed(self):
        delta = build_capability_delta(base_scores=[0.5] * 10, fine_tuned_scores=[0.4] * 10)
        assert delta.relative_delta == pytest.approx(-0.2)

    def test_per_category_breakdown(self):
        delta = build_capability_delta(
            base_scores=[1.0, 1.0, 0.0, 0.0],
            fine_tuned_scores=[1.0, 1.0, 1.0, 1.0],
            categories=["math", "math", "reading", "reading"],
        )
        assert delta.per_category["reading"]["delta"] == 1.0
        assert delta.per_category["math"]["delta"] == 0.0

    def test_unpaired_scores_are_rejected(self):
        # Both arms must run the identical fixed suite.
        with pytest.raises(ValueError, match="paired scores"):
            build_capability_delta(base_scores=[0.5], fine_tuned_scores=[0.5, 0.6])


class TestFaithfulness:
    def test_claims_present_in_context_are_supported(self):
        result = assess_faithfulness(
            "The deadline is March 3 for evidence-1.",
            context_text="evidence-1: the deadline is March 3",
            provided_evidence_ids=["evidence-1"],
            decisive_evidence_ids=["evidence-1"],
        )
        assert result.unsupported_claim_rate == 0.0
        assert result.evidence_coverage == 1.0

    def test_claims_absent_from_context_are_unsupported(self):
        result = assess_faithfulness(
            "The deadline is December 25.",
            context_text="evidence-1: the deadline is March 3",
        )
        assert result.unsupported_claim_rate > 0

    def test_fabricated_citations_are_detected(self):
        result = assess_faithfulness(
            "According to evidence-9, act now.",
            context_text="evidence-1: something",
            provided_evidence_ids=["evidence-1"],
        )
        assert result.fabricated_ids
        assert result.citation_precision < 1.0

    def test_claim_extraction_finds_numbers_and_dates(self):
        claims = extract_claims("Submit 3 forms by Mar 15 for $200.")
        assert any("3" in c for c in claims)
        assert any("200" in c for c in claims)

    def test_sentence_starters_are_not_treated_as_claims(self):
        claims = extract_claims("The answer is clear. This is fine.")
        assert "The" not in claims
        assert "This" not in claims


class TestRunner:
    def test_evaluation_scores_every_example(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="alpha-task"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"]),
            arm="arm0_base",
            benchmark_path="test",
        )
        assert result.count == len(fixture_eval_examples)
        assert result.arm == "arm0_base"

    def test_per_task_metrics_are_produced(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="alpha-task"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"]),
            benchmark_path="test",
        )
        assert result.per_task
        assert all("mean" in scores for scores in result.per_task.values())

    def test_ood_is_measurable_with_tagged_fixtures(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="alpha-task"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"]),
            benchmark_path="test",
        )
        assert result.ood is not None
        assert result.ood.measurable

    def test_consistency_is_computed_from_scenario_families(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="alpha-task"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"]),
            benchmark_path="test",
        )
        assert result.consistency is not None
        assert result.consistency.evaluated_groups > 0

    def test_empty_benchmark_is_rejected(self):
        with pytest.raises(EvaluationError, match="no examples"):
            run_evaluation(EchoBackend(), [], EvaluationConfig(arms=["arm0_base"]))

    def test_multiple_seeds_with_greedy_decoding_warns(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="x"),
            fixture_eval_examples,
            EvaluationConfig(
                arms=["arm0_base"], seeds=[1, 2], generation=GenerationConfig(do_sample=False)
            ),
            benchmark_path="test",
        )
        assert any("identical output" in w for w in result.warnings)

    def test_truncation_is_reported_as_a_warning(self, fixture_eval_examples):
        result = run_evaluation(
            EchoBackend(default="x"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"], max_examples=2),
            benchmark_path="test",
        )
        assert result.count == 2
        assert any("truncated benchmark" in w for w in result.warnings)

    def test_results_serialize_and_can_omit_responses(self, fixture_eval_examples, tmp_path):
        result = run_evaluation(
            EchoBackend(default="x"),
            fixture_eval_examples,
            EvaluationConfig(arms=["arm0_base"]),
            benchmark_path="test",
        )
        path = result.save(tmp_path / "results.json", include_responses=False)
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["results"]
        assert "response" not in payload["results"][0]


class TestComparison:
    def _payload(self, arm: str, scores: dict[str, float], **extra) -> dict:
        return {
            "arm": arm,
            "benchmark_path": "bench",
            "generation": {"temperature": 0.0},
            "model": {"base_model": "m", **extra},
            "results": [
                {"example_id": k, "task": "notification_prioritization", "score": v}
                for k, v in scores.items()
            ],
        }

    def test_improvement_is_reported(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 0.0, "b": 0.0, "c": 0.0}),
            self._payload("arm2_finetuned", {"a": 1.0, "b": 1.0, "c": 1.0}, adapter_path="ad"),
        )
        assert comparison.per_task[0].absolute_delta == 1.0
        assert comparison.improved_tasks

    def test_regression_is_reported(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 1.0, "b": 1.0}),
            self._payload("arm2_finetuned", {"a": 0.0, "b": 0.0}, adapter_path="ad"),
        )
        assert comparison.regressed_tasks
        assert "REGRESSED" in comparison._conclusion()

    def test_no_change_is_reported_as_no_change(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 0.5, "b": 0.5}),
            self._payload("arm2_finetuned", {"a": 0.5, "b": 0.5}, adapter_path="ad"),
        )
        assert "no meaningful difference" in comparison._conclusion()

    def test_different_benchmarks_produce_a_comparability_warning(self):
        base = self._payload("arm0_base", {"a": 0.5})
        finetuned = self._payload("arm2_finetuned", {"a": 0.9}, adapter_path="ad")
        finetuned["benchmark_path"] = "other_bench"
        comparison = compare_results(base, finetuned)
        assert any("Different benchmarks" in w for w in comparison.comparability_warnings)

    def test_missing_adapter_produces_a_warning(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 0.5}),
            self._payload("arm2_finetuned", {"a": 0.9}),
        )
        assert any("no adapter path" in w for w in comparison.comparability_warnings)

    def test_different_decoding_produces_a_warning(self):
        base = self._payload("arm0_base", {"a": 0.5})
        finetuned = self._payload("arm2_finetuned", {"a": 0.9}, adapter_path="ad")
        finetuned["generation"] = {"temperature": 0.9}
        comparison = compare_results(base, finetuned)
        assert any("Decoding settings differ" in w for w in comparison.comparability_warnings)

    def test_scores_are_paired_by_example_id(self):
        base = self._payload("arm0_base", {"a": 0.0, "b": 1.0})
        finetuned = self._payload("arm2_finetuned", {"b": 1.0, "a": 1.0}, adapter_path="ad")
        comparison = compare_results(base, finetuned)
        assert comparison.per_task[0].count == 2
        assert comparison.per_task[0].absolute_delta == pytest.approx(0.5)

    def test_markdown_report_states_when_ood_was_not_measured(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 0.5}),
            self._payload("arm2_finetuned", {"a": 0.9}, adapter_path="ad"),
        )
        markdown = render_markdown_report(comparison)
        assert "No generalization claim can be made" in markdown

    def test_markdown_report_includes_a_where_it_failed_section(self):
        comparison = compare_results(
            self._payload("arm0_base", {"a": 0.9}),
            self._payload("arm2_finetuned", {"a": 0.1}, adapter_path="ad"),
        )
        markdown = render_markdown_report(comparison)
        assert "## Where did it fail?" in markdown


class TestMathHelpers:
    def test_ndcg_matches_a_hand_computed_value(self):
        # Reference relevance: a=3, b=2, c=1. Predicted order c, b, a.
        # DCG = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1 + 1.2619 + 1.5 = 3.7619
        # IDCG = 3/1 + 2/1.585 + 1/2 = 3 + 1.2619 + 0.5 = 4.7619
        expected = (1 + 2 / math.log2(3) + 3 / 2) / (3 + 2 / math.log2(3) + 0.5)
        assert ndcg(["c", "b", "a"], ["a", "b", "c"]) == pytest.approx(expected, rel=1e-6)
