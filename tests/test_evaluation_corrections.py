"""Corrected evaluation measures (findings H-F5, H-F7, H-F11 to H-F14).

The corrections are computed *beside* the original numbers: every version-1 field
keeps its definition and value. What is pinned here:

* the cluster bootstrap is deterministic, refuses to estimate from too few
  groups, and renders a zero p-value as a bound;
* consistency by ``group_id`` and by family, each with an oracle ceiling, so a
  metric that even a perfect model fails is visible as such;
* the answerable and should-decline subsets;
* the runner's version-2 record: group, subset and grader details per example,
  the benchmark's identity, generation statistics, atomic writes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from tests.conftest import make_eval_example

from kleos_models.config import EvaluationConfig
from kleos_models.data.schemas import EvaluationExample
from kleos_models.evaluation.corrections import (
    ANSWERABLE,
    SHOULD_DECLINE,
    annotate_records,
    benchmark_index,
    build_corrected,
    consistency_summary,
    generation_stats,
    render_corrected,
    subset_of,
    targets_fingerprint,
)
from kleos_models.evaluation.metrics import (
    cluster_bootstrap_mean,
    paired_cluster_bootstrap_difference,
    render_p_value,
)
from kleos_models.evaluation.runner import (
    RESULTS_SCHEMA_VERSION,
    run_evaluation,
    write_json_atomic,
)
from kleos_models.inference.backends import EchoBackend, GenerationOutput

# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


class TestClusterBootstrap:
    VALUES = (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0)
    CLUSTERS = ("a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f")

    def test_it_is_deterministic(self):
        first = cluster_bootstrap_mean(self.VALUES, self.CLUSTERS)
        assert first == cluster_bootstrap_mean(self.VALUES, self.CLUSTERS)
        assert first["estimable"] and first["clusters"] == 6

    def test_the_interval_contains_the_mean(self):
        result = cluster_bootstrap_mean(self.VALUES, self.CLUSTERS)
        assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]

    def test_clusters_widen_the_interval(self):
        # Perfectly correlated pairs: one effective observation per group.
        values = [1.0, 1.0, 0.0, 0.0] * 5
        paired = [f"g{i // 2}" for i in range(20)]
        singletons = [f"s{i}" for i in range(20)]
        grouped = cluster_bootstrap_mean(values, paired)
        independent = cluster_bootstrap_mean(values, singletons)
        assert (grouped["ci95_high"] - grouped["ci95_low"]) > (
            independent["ci95_high"] - independent["ci95_low"]
        )

    def test_too_few_clusters_are_not_estimable(self):
        result = cluster_bootstrap_mean([1.0, 0.0, 1.0], ["a", "b", "c"])
        assert result["estimable"] is False
        assert "ci95_low" not in result

    def test_lengths_must_match(self):
        with pytest.raises(ValueError):
            cluster_bootstrap_mean([1.0], ["a", "b"])

    def test_paired_difference_of_identical_arms_is_zero(self):
        result = paired_cluster_bootstrap_difference(self.VALUES, self.VALUES, self.CLUSTERS)
        assert result["difference"] == 0.0
        assert result["ci95_low"] == result["ci95_high"] == 0.0
        assert result["significant_at_05"] is False

    def test_a_uniform_improvement_is_significant(self):
        left = [0.0] * 12
        right = [1.0] * 12
        result = paired_cluster_bootstrap_difference(left, right, self.CLUSTERS)
        assert result["difference"] == 1.0
        assert result["p_value"] == 0.0
        assert result["significant_at_05"] is True
        assert render_p_value(result["p_value"], result["iterations"]) == "p<0.0005"

    def test_paired_too_few_clusters(self):
        result = paired_cluster_bootstrap_difference([0.0, 1.0], [1.0, 1.0], ["a", "b"])
        assert result["estimable"] is False and result["p_value"] is None

    def test_p_value_rendering(self):
        assert render_p_value(0.0123) == "p=0.0123"
        assert render_p_value(None) == "p n/a"


# ---------------------------------------------------------------------------
# Corrections over records
# ---------------------------------------------------------------------------


def record(
    example_id: str,
    *,
    group: str | None,
    family: str | None,
    decision: str,
    reference: str,
    score: float = 1.0,
    subset: str | None = ANSWERABLE,
    task: str = "tool_routing",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "task": task,
        "score": score,
        "sub_scores": {"judgment": score},
        "group_id": group,
        "scenario_family": family,
        "subset": subset,
        "decision": decision,
        "reference_decision": reference,
        "perturbation_kind": "paraphrase",
        **extra,
    }


def family_mixing_records(model_answers: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One family, two groups with different correct labels (the v0.0.6 shape)."""
    answers = model_answers or {}
    rows = []
    for group, label in (("g1", "evidence"), ("g2", "scope")):
        for i in range(3):
            example_id = f"{group}-{i}"
            rows.append(
                record(
                    example_id,
                    group=group,
                    family="fam",
                    decision=answers.get(example_id, label),
                    reference=label,
                )
            )
    return rows


class TestConsistencyUnits:
    def test_a_family_that_mixes_labels_defeats_even_an_oracle(self):
        by_family = consistency_summary(family_mixing_records(), "scenario_family")
        assert by_family["oracle_agreement_rate"] == 0.0
        assert by_family["oracle_flips"] > 0

    def test_group_id_is_a_sound_unit(self):
        by_group = consistency_summary(family_mixing_records(), "group_id")
        assert by_group["evaluated_groups"] == 2
        assert by_group["agreement_rate"] == by_group["oracle_agreement_rate"] == 1.0
        assert by_group["flips"] == by_group["oracle_flips"] == 0

    def test_a_real_flip_is_counted(self):
        rows = family_mixing_records({"g1-2": "deadline"})
        by_group = consistency_summary(rows, "group_id")
        assert by_group["agreement_rate"] == 0.5
        assert by_group["flips"] == 1

    def test_consistency_can_be_restricted_to_a_subset(self):
        rows = family_mixing_records()
        for row in rows[3:]:
            row["subset"] = SHOULD_DECLINE
        answerable = consistency_summary(rows, "group_id", subset=ANSWERABLE)
        assert answerable["evaluated_groups"] == 1

    def test_without_the_key_nothing_is_measured(self):
        rows = [record("x", group=None, family=None, decision="a", reference="a")]
        assert consistency_summary(rows, "group_id")["evaluated_groups"] == 0


class TestSubsetsAndIdentity:
    def test_subset_comes_from_the_reference(self):
        assert subset_of({"confident": True}) == ANSWERABLE
        assert subset_of({"confident": False}) == SHOULD_DECLINE
        assert subset_of({"label": "x"}) is None

    def benchmark_rows(self) -> list[dict[str, Any]]:
        rows = []
        for i, confident in enumerate([True, True, False]):
            row = make_eval_example(f"ex-{i}", reference={"label": "a", "confident": confident})
            row["metadata"]["group_id"] = f"grp-{i}"
            rows.append(row)
        return rows

    def test_the_index_reads_raw_rows_and_validated_examples_alike(self):
        rows = self.benchmark_rows()
        from_dicts = benchmark_index(rows)
        from_objects = benchmark_index(EvaluationExample.model_validate(r) for r in rows)
        assert from_dicts == from_objects
        assert from_dicts["ex-2"]["subset"] == SHOULD_DECLINE
        assert from_dicts["ex-0"]["has_evidence_ids"] is False

    def test_annotation_joins_group_and_subset(self):
        index = benchmark_index(self.benchmark_rows())
        stored = [{"example_id": "ex-0", "task": "tool_routing", "score": 1.0}]
        (annotated,) = annotate_records(stored, index)
        assert annotated["group_id"] == "grp-0" and annotated["subset"] == ANSWERABLE
        assert "group_id" not in stored[0]  # the input is not mutated

    @pytest.mark.parametrize(
        "stored",
        [
            {"example_id": "ex-9", "task": "tool_routing"},
            {"example_id": "ex-0", "task": "context_prioritization"},
            {"example_id": "ex-0", "task": "tool_routing", "group_id": "grp-7"},
        ],
        ids=["unknown-example", "task-mismatch", "group-mismatch"],
    )
    def test_annotation_refuses_a_different_benchmark(self, stored):
        with pytest.raises(ValueError):
            annotate_records([stored], benchmark_index(self.benchmark_rows()))

    def test_the_targets_fingerprint_ignores_order_and_seeds(self):
        rows = family_mixing_records()
        assert targets_fingerprint(rows) == targets_fingerprint(list(reversed(rows)))
        assert targets_fingerprint(rows) == targets_fingerprint(rows + rows[:2])
        changed = [dict(rows[0], reference_decision="other"), *rows[1:]]
        assert targets_fingerprint(changed) != targets_fingerprint(rows)


class TestGenerationStats:
    def test_answers_that_used_the_whole_budget_are_counted(self):
        rows = [
            {
                "completion_tokens": 512,
                "prompt_tokens": 300,
                "latency_seconds": 30.0,
                "response": "x",
            },
            {
                "completion_tokens": 90,
                "prompt_tokens": 280,
                "latency_seconds": 10.0,
                "response": "y",
            },
            {
                "completion_tokens": 100,
                "prompt_tokens": 290,
                "latency_seconds": 12.0,
                "response": " ",
            },
        ]
        stats = generation_stats(rows, max_new_tokens=512)
        assert stats["hit_max_new_tokens"] == 1
        assert stats["empty_responses"] == 1
        assert stats["completion_tokens"]["max"] == 512
        assert stats["latency_seconds"]["total"] == 52.0

    def test_without_a_budget_nothing_is_claimed(self):
        assert generation_stats([{"completion_tokens": 5}])["hit_max_new_tokens"] is None


class TestCorrectedBlock:
    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for g in range(6):
            for i in range(2):
                subset = ANSWERABLE if g < 4 else SHOULD_DECLINE
                rows.append(
                    record(
                        f"g{g}-{i}",
                        group=f"g{g}",
                        family=f"f{g % 2}",
                        decision="a",
                        reference="a",
                        score=1.0 if subset == ANSWERABLE else 0.0,
                        subset=subset,
                    )
                )
        return rows

    def test_subsets_are_reported_apart(self):
        corrected = build_corrected(self.rows())
        assert corrected["subsets"][ANSWERABLE]["mean"] == 1.0
        assert corrected["subsets"][SHOULD_DECLINE]["mean"] == 0.0
        assert corrected["subsets"][SHOULD_DECLINE]["estimable"] is False  # 2 groups
        assert corrected["overall"]["clusters"] == 6

    def test_missing_group_ids_are_reported_not_papered_over(self):
        rows = self.rows()
        rows[0]["group_id"] = None
        overall = build_corrected(rows)["overall"]
        assert overall["estimable"] is False
        assert "group_id" in overall["reason"]

    def test_vacuous_evidence_coverage_is_named(self):
        index = {"x": {"has_evidence_ids": False}}
        corrected = build_corrected(self.rows(), index=index)
        assert corrected["faithfulness"]["evidence_coverage"].startswith("vacuous")
        assert "UNCALIBRATED" in corrected["faithfulness"]["citation_heuristic"]

    def test_the_block_renders(self):
        text = render_corrected(build_corrected(self.rows()))
        assert "answerable" in text and "oracle" in text


# ---------------------------------------------------------------------------
# The runner's version-2 record
# ---------------------------------------------------------------------------


def benchmark() -> list[EvaluationExample]:
    rows = []
    for i in range(6):
        confident = i < 4
        row = make_eval_example(
            f"ex-{i}",
            reference={
                "label": "memory_search" if confident else "none",
                "options": ["memory_search", "none"],
                "confident": confident,
            },
            scenario_family="fam",
        )
        row["metadata"]["group_id"] = f"grp-{i // 2}"
        row["metadata"]["perturbation_kind"] = "paraphrase" if i % 2 else None
        rows.append(EvaluationExample.model_validate(row))
    return rows


class TestRunnerRecords:
    def run(self, **config: Any):
        backend = EchoBackend(default="memory_search")
        return run_evaluation(
            backend,
            benchmark(),
            EvaluationConfig(**config),
            arm="arm2_finetuned",
            benchmark_path="bench.jsonl",
            benchmark_sha256="f" * 64,
            default_grader="classification",
        )

    def test_records_carry_group_subset_and_details(self):
        payload = self.run().to_dict()
        first = payload["results"][0]
        assert first["group_id"] == "grp-0"
        assert first["subset"] == ANSWERABLE
        assert payload["results"][5]["subset"] == SHOULD_DECLINE
        # Finding H-F7: what the grader extracted is saved, not only the score.
        assert first["details"] == {
            "predicted_label": "memory_search",
            "expected_label": "memory_search",
        }

    def test_version_two_fields_sit_beside_the_originals(self):
        payload = self.run().to_dict()
        assert payload["schema_version"] == RESULTS_SCHEMA_VERSION == 2
        assert payload["benchmark_sha256"] == "f" * 64
        assert payload["benchmark_fingerprint"] == targets_fingerprint(payload["results"])
        assert payload["generation_stats"]["responses"] == 6
        assert payload["corrected"]["subsets"][ANSWERABLE]["n"] == 4
        # Version-1 fields are all still there.
        for key in ("arm", "overall", "metrics", "per_task", "consistency", "results"):
            assert key in payload

    def test_the_configured_grouping_key_is_honoured(self):
        by_family = self.run().consistency
        assert by_family is not None and by_family.group_key == "scenario_family"
        assert by_family.evaluated_groups == 1
        by_group = self.run(consistency={"group_key": "group_id"}).consistency
        assert by_group is not None and by_group.group_key == "group_id"
        assert by_group.evaluated_groups == 3

    def test_replayed_generations_report_their_recorded_latency(self):
        class Replaying(EchoBackend):
            def generate(self, messages, config, **kwargs):
                output = super().generate(messages, config, **kwargs)
                return GenerationOutput(
                    text=output.text,
                    metadata={"recorded_latency_seconds": 12.5},
                )

        result = run_evaluation(
            Replaying(default="memory_search"),
            benchmark(),
            EvaluationConfig(),
            default_grader="classification",
        )
        assert {r.latency_seconds for r in result.results} == {12.5}


class TestAtomicWrites:
    def test_a_write_replaces_the_file_whole(self, tmp_path):
        target = tmp_path / "result.json"
        write_json_atomic(target, {"a": 1})
        write_json_atomic(target, {"a": 2})
        assert json.loads(target.read_text()) == {"a": 2}
        assert [p.name for p in tmp_path.iterdir()] == ["result.json"]

    def test_a_failed_write_leaves_the_old_file(self, tmp_path):
        target = tmp_path / "result.json"
        write_json_atomic(target, {"a": 1})
        with pytest.raises(TypeError):
            write_json_atomic(target, {1j: "a complex key cannot be JSON"})
        assert json.loads(target.read_text()) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["result.json"]
