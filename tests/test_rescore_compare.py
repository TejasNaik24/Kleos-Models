"""Re-reporting stored results, and comparing two models.

``rescore.py --mode annotate`` is how Hermes' stored results get the corrected
measures without being re-run or modified. ``compare.py --cross-model`` is how
H8b is decided. Pinned here:

* annotation never changes an original field, and refuses results whose scores
  no longer reproduce or whose benchmark differs;
* the citation heuristic's gold floor is counted from the gold answers;
* the primary verdict follows the pre-registered rule, on the cluster interval;
* compare's exit codes let a driver script branch on the outcome.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.conftest import REPO_ROOT, make_eval_example

from kleos_models.config import EvaluationConfig
from kleos_models.data.schemas import EvaluationExample
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.reports import (
    compare_results,
    primary_verdict,
    render_markdown_report,
)
from kleos_models.evaluation.runner import run_evaluation
from kleos_models.inference.backends import EchoBackend

#: Keys a version-1 results file does not have.
V2_KEYS = (
    "schema_version",
    "benchmark_sha256",
    "benchmark_fingerprint",
    "generation_stats",
    "corrected",
    "resource_usage",
    "resume",
)
V2_RECORD_KEYS = ("group_id", "subset", "details")


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_rows(count: int = 16) -> list[dict[str, Any]]:
    """16 items in 8 groups: 6 answerable groups, 2 should-decline ones."""
    rows = []
    for i in range(count):
        confident = i < 12
        row = make_eval_example(
            f"ex-{i:02d}",
            reference={
                "label": "memory_search" if confident else "none",
                "options": ["memory_search", "none"],
                "confident": confident,
            },
            scenario_family=f"fam-{i % 3}",
        )
        row["metadata"]["group_id"] = f"grp-{i // 2}"
        row["metadata"]["perturbation_kind"] = "paraphrase" if i % 2 else None
        # The citation heuristic only judges citations when the prompt supplies ids.
        row["messages"][0]["content"] = "Which tool handles this? See item A1."
        rows.append(row)
    return rows


def write_benchmark(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def evaluate(answers: dict[str, str], *, arm: str = "arm2_finetuned") -> dict[str, Any]:
    examples = [EvaluationExample.model_validate(r) for r in benchmark_rows()]
    result = run_evaluation(
        EchoBackend(responses=answers, default="none"),
        examples,
        EvaluationConfig(),
        arm=arm,
        default_grader="classification",
    )
    return result.to_dict()


def as_version_1(payload: dict[str, Any]) -> dict[str, Any]:
    """What a results file written before schema version 2 looks like."""
    old = copy.deepcopy(payload)
    for key in V2_KEYS:
        old.pop(key, None)
    for row in old["results"]:
        for key in V2_RECORD_KEYS:
            row.pop(key, None)
    return old


def all_correct() -> dict[str, str]:
    return {r["id"]: r["reference"]["label"] for r in benchmark_rows()}


# ---------------------------------------------------------------------------
# rescore.py --mode annotate
# ---------------------------------------------------------------------------


@pytest.fixture
def stored(tmp_path) -> tuple[Path, Path]:
    bench = write_benchmark(tmp_path / "benchmark.jsonl", benchmark_rows())
    results = tmp_path / "arm2_finetuned.json"
    results.write_text(json.dumps(as_version_1(evaluate(all_correct()))), encoding="utf-8")
    return results, bench


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestAnnotate:
    def run(self, *argv: str) -> int:
        return load_script("rescore").main(list(argv))

    def test_every_original_field_is_kept_verbatim(self, stored, tmp_path):
        results, bench = stored
        before = sha(results)
        out = tmp_path / "annotated.json"
        assert (
            self.run(
                "--mode",
                "annotate",
                "--results",
                str(results),
                "--benchmark",
                str(bench),
                "--output",
                str(out),
            )
            == 0
        )
        assert sha(results) == before  # the source is only read

        original = json.loads(results.read_text())
        annotated = json.loads(out.read_text())
        for key, value in original.items():
            if key != "results":
                assert annotated[key] == value, key
        for old_row, new_row in zip(original["results"], annotated["results"], strict=True):
            assert {k: new_row[k] for k in old_row} == old_row
            assert new_row["group_id"] and new_row["subset"] in ("answerable", "should_decline")
            assert new_row["details"]["expected_label"]

    def test_the_corrected_measures_are_added(self, stored, tmp_path):
        results, bench = stored
        out = tmp_path / "annotated.json"
        self.run(
            "--mode",
            "annotate",
            "--results",
            str(results),
            "--benchmark",
            str(bench),
            "--output",
            str(out),
        )
        annotated = json.loads(out.read_text())
        assert annotated["schema_version"] == 1  # annotated, not re-run
        assert annotated["benchmark_sha256"] == sha(bench)
        assert annotated["annotated"]["scores_reproduced"] is True
        assert annotated["corrected"]["subsets"]["answerable"]["n"] == 12
        assert annotated["corrected"]["consistency"]["group_id"]["evaluated_groups"] == 8
        assert annotated["generation_stats"]["responses"] == 16

    def test_scores_that_do_not_reproduce_stop_the_annotation(self, stored, tmp_path):
        results, bench = stored
        payload = json.loads(results.read_text())
        payload["results"][0]["score"] = 0.5  # as if graded by another grader version
        results.write_text(json.dumps(payload), encoding="utf-8")
        args = ["--mode", "annotate", "--results", str(results), "--benchmark", str(bench)]
        with pytest.raises(EvaluationError, match="do not reproduce"):
            self.run(*args, "--output", str(tmp_path / "a.json"))
        assert self.run(*args, "--output", str(tmp_path / "b.json"), "--allow-score-drift") == 0
        annotated = json.loads((tmp_path / "b.json").read_text())
        assert annotated["annotated"]["score_drift"] == [
            {"example_id": "ex-00", "stored": 0.5, "regraded": 1.0}
        ]
        assert annotated["results"][0]["score"] == 0.5  # the original value stays

    def test_a_different_benchmark_is_refused(self, stored, tmp_path):
        results, _ = stored
        rows = benchmark_rows()
        rows[0]["reference"]["label"] = "none"
        other = write_benchmark(tmp_path / "other.jsonl", rows)
        with pytest.raises(EvaluationError, match="differ from this benchmark"):
            self.run(
                "--mode",
                "annotate",
                "--results",
                str(results),
                "--benchmark",
                str(other),
                "--output",
                str(tmp_path / "x.json"),
            )

    def test_the_gold_floor_is_counted_from_the_gold_answers(self, stored, tmp_path):
        results, bench = stored
        gold = tmp_path / "test.jsonl"
        lines = []
        for i, row in enumerate(benchmark_rows()):
            answer = "Use memory_search per evidence E7." if i < 3 else "Use memory_search."
            lines.append(
                json.dumps(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
            )
        gold.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out, summary = tmp_path / "annotated.json", tmp_path / "summary.md"
        assert (
            self.run(
                "--mode",
                "annotate",
                "--results",
                str(results),
                "--benchmark",
                str(bench),
                "--output",
                str(out),
                "--gold-targets",
                str(gold),
                "--summary",
                str(summary),
            )
            == 0
        )
        floor = json.loads(out.read_text())["corrected"]["faithfulness"]["gold_floor"]
        assert (floor["gold_answers_found"], floor["gold_answers_flagged"]) == (16, 3)
        text = summary.read_text()
        assert "Original vs corrected" in text and "3/16" in text

    def test_inputs_are_never_overwritten(self, stored):
        results, bench = stored
        assert (
            self.run(
                "--mode",
                "annotate",
                "--results",
                str(results),
                "--benchmark",
                str(bench),
                "--output",
                str(results),
            )
            == 1
        )

    def test_annotate_only_flags_are_refused_in_regrade_mode(self, stored, tmp_path):
        results, bench = stored
        with pytest.raises(SystemExit):
            self.run(
                "--results",
                str(results),
                "--benchmark",
                str(bench),
                "--output",
                str(tmp_path / "r.json"),
                "--summary",
                str(tmp_path / "s.md"),
            )


class TestRegrade:
    def test_decisions_and_consistency_follow_the_new_grade(self, stored, tmp_path):
        results, bench = stored
        payload = json.loads(results.read_text())
        for row in payload["results"]:
            row["decision"] = "stale"
        payload["consistency"]["agreement_rate"] = 0.999  # a block computed from them
        payload["consistency"]["groups"] = []
        results.write_text(json.dumps(payload), encoding="utf-8")
        out = tmp_path / "regraded.json"
        assert (
            load_script("rescore").main(
                ["--results", str(results), "--benchmark", str(bench), "--output", str(out)]
            )
            == 0
        )
        regraded = json.loads(out.read_text())
        assert {row["decision"] for row in regraded["results"]} == {"memory_search", "none"}
        # Recomputed from the new decisions: each family mixes both labels.
        assert regraded["consistency"]["agreement_rate"] == 0.0
        assert len(regraded["consistency"]["groups"]) == 3
        assert "details" in regraded["results"][0]


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestPrimaryVerdict:
    @pytest.mark.parametrize(
        ("low", "high", "verdict"),
        [
            (0.03, 0.09, "better"),
            (-0.09, -0.03, "worse"),
            (-0.015, 0.018, "equivalent"),
            (-0.05, 0.04, "inconclusive"),
            (0.005, 0.015, "better"),  # significant, but inside the margin
        ],
    )
    def test_the_pre_registered_rule(self, low, high, verdict):
        significance = {"estimable": True, "ci95_low": low, "ci95_high": high}
        assert primary_verdict(significance, 0.02)[0] == verdict

    def test_an_inestimable_interval_is_inconclusive(self):
        verdict, why = primary_verdict({"estimable": False, "reason": "2 cluster(s)"}, 0.02)
        assert verdict == "inconclusive" and "2 cluster(s)" in why


def model_block(base_model: str) -> dict[str, Any]:
    return {"base_model": base_model, "adapter_path": f"/runs/{base_model}/adapter"}


class TestCrossModel:
    def payloads(self, right_answers: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        left = evaluate(all_correct())
        right = evaluate(right_answers)
        left["model"] = model_block("hermes-base")
        right["model"] = model_block("logos-base")
        return left, right

    def test_equal_models_are_not_called_different(self):
        left, right = self.payloads(all_correct())
        report = compare_results(
            left, right, cross_model=True, primary_subset="answerable", equivalence_margin=0.02
        )
        assert not report.incomparable
        assert not any("Different base models" in w for w in report.comparability_warnings)
        assert report.primary["verdict"] == "equivalent"
        assert {s.task for s in report.subsets} == {"answerable", "should_decline"}
        assert report.subsets[0].cluster_significance["clusters"] == 6

    def test_a_worse_model_is_reported_worse(self):
        answers = all_correct()
        for i in range(12):
            answers[f"ex-{i:02d}"] = "none"  # every answerable item wrong
        left, right = self.payloads(answers)
        report = compare_results(
            left, right, cross_model=True, primary_subset="answerable", equivalence_margin=0.02
        )
        assert report.primary["verdict"] == "worse"
        assert "WORSE" in report._conclusion()
        markdown = render_markdown_report(report)
        assert "Primary comparison (pre-registered)" in markdown and "By subset" in markdown

    def test_stored_results_without_group_ids_say_so(self):
        left, right = self.payloads(all_correct())
        left = as_version_1(left)
        report = compare_results(
            left, right, cross_model=True, primary_subset="answerable", equivalence_margin=0.02
        )
        assert report.per_task[0].cluster_verdict == "n/a"
        assert "annotate" in report.per_task[0].cluster_significance["reason"]


class TestCompareScript:
    def run(self, tmp_path: Path, left: dict[str, Any], right: dict[str, Any], *extra: str) -> int:
        base, finetuned = tmp_path / "left.json", tmp_path / "right.json"
        base.write_text(json.dumps(left), encoding="utf-8")
        finetuned.write_text(json.dumps(right), encoding="utf-8")
        return load_script("compare").main(
            ["--base", str(base), "--finetuned", str(finetuned), *extra]
        )

    def test_a_comparison_exits_0(self, tmp_path):
        base = evaluate({}, arm="arm1_base_orchestrated")
        finetuned = evaluate(all_correct())
        assert self.run(tmp_path, base, finetuned) == 0

    def test_different_benchmarks_exit_2(self, tmp_path):
        base, finetuned = evaluate({}), evaluate(all_correct())
        base["benchmark_sha256"], finetuned["benchmark_sha256"] = "a" * 64, "b" * 64
        base["results"][0]["reference_decision"] = "something else"
        assert self.run(tmp_path, base, finetuned) == 2

    def test_a_regression_exits_1_only_when_asked(self, tmp_path):
        base = evaluate(all_correct(), arm="arm1_base_orchestrated")
        worse = evaluate({})
        assert self.run(tmp_path, base, worse) == 0
        assert self.run(tmp_path, base, worse, "--fail-on-regression") == 1

    def test_a_worse_primary_verdict_exits_1_when_asked(self, tmp_path):
        left = evaluate(all_correct())
        right = evaluate({r["id"]: "none" for r in benchmark_rows()})
        left["model"], right["model"] = model_block("hermes"), model_block("logos")
        args = ("--cross-model", "--primary-subset", "answerable", "--fail-on-regression")
        assert self.run(tmp_path, left, right, *args) == 1
