"""The evaluation runner (spec sections 19, 20, 21).

Runs one research arm over a benchmark under fixed conditions and produces a
result object carrying per-example records, aggregate metrics, consistency, OOD
and faithfulness.

Two invariants make the results comparable:

1. **Conditions are held identical across arms.** Same benchmark, same decoding
   settings, same seeds, same graders. The runner takes the arm as a parameter and
   changes nothing else, so the only difference between ``arm0_base`` and
   ``arm2_finetuned`` is the adapter.
2. **Per-example records are always kept.** Aggregates alone would make it
   possible to report a favourable subset without deciding to. The raw records let
   any aggregate be recomputed and checked.

This module imports no torch; the backend supplies generation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.config import EvaluationConfig, GenerationConfig
from kleos_models.data.schemas import EvaluationExample
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.capability import CapabilityDelta
from kleos_models.evaluation.consistency import ConsistencyReport, build_consistency_groups
from kleos_models.evaluation.faithfulness import (
    FaithfulnessReport,
    assess_faithfulness,
)
from kleos_models.evaluation.graders import Grader, GradeResult, get_grader
from kleos_models.evaluation.metrics import MetricSummary, summarize
from kleos_models.evaluation.ood import OODReport, build_ood_report
from kleos_models.inference.backends import Backend
from kleos_models.inference.generate import (
    OrchestrationConfig,
    build_prompt,
    collect_evidence_ids,
    extract_context_text,
    summarize_arm,
)
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExampleResult:
    """The full record for one evaluated example."""

    example_id: str
    task: str
    arm: str
    seed: int
    score: float
    grader: str
    response: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    split_tag: str = "in_distribution"
    ood_shift: str | None = None
    scenario_family: str | None = None
    perturbation_kind: str | None = None
    decision: str = ""
    reference_decision: str | None = None
    faithfulness: dict[str, Any] | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_failed: bool = False
    had_reasoning: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_response: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "example_id": self.example_id,
            "task": self.task,
            "arm": self.arm,
            "seed": self.seed,
            "score": round(self.score, 4),
            "grader": self.grader,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "split_tag": self.split_tag,
            "ood_shift": self.ood_shift,
            "scenario_family": self.scenario_family,
            "perturbation_kind": self.perturbation_kind,
            "decision": self.decision,
            "reference_decision": self.reference_decision,
            "latency_seconds": round(self.latency_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "parse_failed": self.parse_failed,
            "had_reasoning": self.had_reasoning,
        }
        if self.faithfulness:
            payload["faithfulness"] = self.faithfulness
        if include_response:
            payload["response"] = self.response
        return payload


@dataclass
class EvaluationResult:
    """The complete outcome of evaluating one arm."""

    arm: str
    model_description: dict[str, Any]
    benchmark_path: str
    generation: dict[str, Any]
    results: list[ExampleResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    consistency: ConsistencyReport | None = None
    ood: OODReport | None = None
    faithfulness: FaithfulnessReport | None = None
    capability: CapabilityDelta | None = None
    arm_config: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def overall(self) -> MetricSummary:
        """Mean score across every evaluated example."""
        return summarize("overall", [r.score for r in self.results])

    def scores_for(self, *, split_tag: str | None = None, task: str | None = None) -> list[float]:
        """Per-example scores, optionally filtered."""
        return [
            r.score
            for r in self.results
            if (split_tag is None or r.split_tag == split_tag) and (task is None or r.task == task)
        ]

    def to_dict(self, *, include_responses: bool = True) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "arm_config": self.arm_config,
            "model": self.model_description,
            "benchmark_path": self.benchmark_path,
            "generation": self.generation,
            "seeds": self.seeds,
            "example_count": self.count,
            "duration_seconds": round(self.duration_seconds, 2),
            "overall": self.overall.to_dict(),
            "metrics": self.metrics,
            "per_task": self.per_task,
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "ood": self.ood.to_dict() if self.ood else None,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "capability": self.capability.to_dict() if self.capability else None,
            "warnings": self.warnings,
            "results": [r.to_dict(include_response=include_responses) for r in self.results],
        }

    def save(self, path: Path | str, *, include_responses: bool = True) -> Path:
        """Write results to JSON.

        Args:
            include_responses: Keep raw model text. Turn this off when the
                benchmark contains sanitized-but-sensitive context and the results
                file will be shared.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(include_responses=include_responses), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote evaluation results to %s", target)
        return target

    def render(self) -> str:
        lines = [
            "=" * 72,
            f"Evaluation — arm {self.arm}",
            "=" * 72,
            f"  model            : {self.model_description.get('base_model', '-')}",
            f"  adapter          : {self.model_description.get('adapter_path') or 'none (base)'}",
            f"  benchmark        : {self.benchmark_path}",
            f"  examples         : {self.count}",
            f"  seeds            : {self.seeds}",
            f"  duration         : {self.duration_seconds:.1f}s",
            "",
            f"  overall score    : {self.overall.render()}",
        ]
        parse_failures = sum(1 for r in self.results if r.parse_failed)
        if parse_failures:
            lines.append(
                f"  parse failures   : {parse_failures} "
                f"({parse_failures / max(self.count, 1):.1%}) — the grader could not "
                "extract an answer"
            )
        if self.per_task:
            lines.extend(["", "  Per task:"])
            for task, scores in sorted(self.per_task.items()):
                lines.append(f"    {task:<34} {scores['mean']:.4f}  (n={scores['count']})")
        if self.ood:
            lines.extend(["", *(f"  {line}" for line in self.ood.render().splitlines())])
        if self.consistency:
            lines.extend(["", *(f"  {line}" for line in self.consistency.render().splitlines())])
        if self.faithfulness and self.faithfulness.count:
            lines.extend(["", *(f"  {line}" for line in self.faithfulness.render().splitlines())])
        if self.warnings:
            lines.extend(["", "  Warnings:"])
            lines.extend(f"    - {w}" for w in self.warnings)
        lines.append("")
        return "\n".join(lines)


def _resolve_grader(name: str, cache: dict[str, Grader]) -> Grader:
    if name not in cache:
        cache[name] = get_grader(name)
    return cache[name]


def _extract_decision(result: GradeResult, response: str) -> str:
    """Best available representation of the decision, for consistency grouping."""
    for key in ("predicted_label", "predicted_ranking", "predicted_items"):
        value = result.details.get(key)
        if value:
            return json.dumps(value, sort_keys=True) if isinstance(value, list) else str(value)
    return response.strip()[:200]


def _reference_decision(example: EvaluationExample) -> str | None:
    """The reference decision used for correct-agreement scoring."""
    reference = example.reference
    for key in ("label", "expected_decision", "text"):
        if key in reference:
            return str(reference[key])
    if "ranking" in reference and isinstance(reference["ranking"], list):
        return json.dumps([str(i) for i in reference["ranking"]], sort_keys=True)
    return None


def run_evaluation(
    backend: Backend,
    examples: Sequence[EvaluationExample],
    config: EvaluationConfig,
    *,
    arm: str = "arm0_base",
    orchestration: OrchestrationConfig | None = None,
    benchmark_path: str = "unknown",
    default_grader: str | None = None,
    include_faithfulness: bool = True,
) -> EvaluationResult:
    """Evaluate one arm over a benchmark.

    Args:
        backend: Generation backend for this arm.
        examples: Benchmark items.
        config: Evaluation settings — decoding, seeds, consistency, OOD.
        arm: Research arm label, recorded on every example result.
        orchestration: Scaffolding config. Defaults to the arm's own.
        benchmark_path: Recorded for provenance.
        default_grader: Grader for examples that do not name one.
        include_faithfulness: Run faithfulness assessment.

    Returns:
        An :class:`EvaluationResult`.
    """
    if not examples:
        raise EvaluationError(
            "The benchmark contains no examples.",
            details={"benchmark_path": benchmark_path},
            suggestions=[
                "Check evaluation.benchmark_path points at a non-empty JSONL file.",
                "Validate it: python scripts/validate_dataset.py --eval <path>",
            ],
        )

    orchestration = orchestration or OrchestrationConfig.for_arm(
        arm, prompt_path=config.orchestration_prompt_path
    )
    generation: GenerationConfig = config.generation
    grader_cache: dict[str, Grader] = {}
    warnings: list[str] = []

    selected = list(examples)
    if config.max_examples is not None and len(selected) > config.max_examples:
        # Truncate deterministically from the front so repeated runs see the same
        # subset — a random subset per run would make comparisons noisy.
        selected = selected[: config.max_examples]
        warnings.append(
            f"Evaluated only the first {config.max_examples} of {len(examples)} examples "
            "(evaluation.max_examples). This is a truncated benchmark, not the full one."
        )

    if len(config.seeds) > 1 and not generation.do_sample:
        warnings.append(
            f"{len(config.seeds)} seeds configured with greedy decoding "
            "(do_sample=false); every seed will produce identical output. Set "
            "generation.do_sample=true for seed variance to mean anything."
        )

    logger.info(
        "Evaluating arm %s on %d example(s) with %d seed(s)",
        arm,
        len(selected),
        len(config.seeds),
    )

    results: list[ExampleResult] = []
    faithfulness_report = FaithfulnessReport()
    started = time.perf_counter()

    for seed in config.seeds:
        for example in selected:
            messages = build_prompt(example, orchestration)
            grader_name = example.grader or default_grader or "exact_match"
            grader = _resolve_grader(grader_name, grader_cache)

            call_started = time.perf_counter()
            output = backend.generate(messages, generation, example_id=example.id, seed=seed)
            latency = time.perf_counter() - call_started

            grade = grader.grade(output.text, example.reference, example=example)

            faithfulness_payload: dict[str, Any] | None = None
            if include_faithfulness:
                assessment = assess_faithfulness(
                    output.text,
                    context_text=extract_context_text(example),
                    provided_evidence_ids=collect_evidence_ids(example),
                    decisive_evidence_ids=[
                        str(e) for e in example.reference.get("evidence_ids", [])
                    ],
                )
                faithfulness_report.results.append(assessment)
                faithfulness_payload = assessment.to_dict()

            results.append(
                ExampleResult(
                    example_id=example.id,
                    task=example.task,
                    arm=arm,
                    seed=seed,
                    score=grade.score,
                    grader=grader_name,
                    response=output.text,
                    sub_scores=grade.sub_scores,
                    split_tag=example.split_tag or "in_distribution",
                    ood_shift=example.ood_shift,
                    scenario_family=example.metadata.scenario_family,
                    perturbation_kind=example.metadata.perturbation_kind,
                    decision=_extract_decision(grade, output.text),
                    reference_decision=_reference_decision(example),
                    faithfulness=faithfulness_payload,
                    latency_seconds=latency,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    parse_failed=grade.parse_failed,
                    had_reasoning=output.reasoning is not None,
                    details=grade.details,
                )
            )

    duration = time.perf_counter() - started

    # --- aggregates ---------------------------------------------------------
    overall = summarize("overall", [r.score for r in results])
    metrics: dict[str, Any] = {"overall": overall.to_dict()}

    sub_score_names = {name for r in results for name in r.sub_scores}
    for name in sorted(sub_score_names):
        values = [r.sub_scores[name] for r in results if name in r.sub_scores]
        metrics[name] = summarize(name, values).to_dict()

    per_task: dict[str, dict[str, Any]] = {}
    for task in sorted({r.task for r in results}):
        task_scores = [r.score for r in results if r.task == task]
        per_task[task] = summarize(task, task_scores).to_dict()

    parse_failures = sum(1 for r in results if r.parse_failed)
    if parse_failures > len(results) * 0.2:
        warnings.append(
            f"{parse_failures} of {len(results)} responses could not be parsed into an "
            "answer. The score may be measuring output-format compliance rather than "
            "judgment; inspect the raw responses before drawing conclusions."
        )

    # --- OOD ----------------------------------------------------------------
    ood_report: OODReport | None = None
    if config.ood.enabled:
        ood_report = build_ood_report(
            scores=[r.score for r in results],
            split_tags=[r.split_tag for r in results],
            shift_kinds=[r.ood_shift for r in results],
            metric_name="score",
        )
        if not ood_report.measurable:
            warnings.append(
                "OOD evaluation was requested but the benchmark has no examples tagged "
                "split_tag='ood'. No generalization claim can be made from this run."
            )

    # --- consistency --------------------------------------------------------
    consistency_report: ConsistencyReport | None = None
    if config.consistency.enabled:
        grouped = [r for r in results if r.scenario_family]
        if grouped:
            consistency_report = build_consistency_groups(
                example_ids=[r.example_id for r in grouped],
                group_ids=[r.scenario_family or "" for r in grouped],
                decisions=[r.decision for r in grouped],
                scores=[r.score for r in grouped],
                perturbation_kinds=[r.perturbation_kind or "" for r in grouped],
                reference_decisions=[r.reference_decision for r in grouped],
                min_group_size=config.consistency.min_group_size,
                group_key=config.consistency.group_key,
            )
        else:
            warnings.append(
                "Consistency testing was requested but no example carries "
                "metadata.scenario_family, so equivalent scenarios cannot be grouped."
            )

    result = EvaluationResult(
        arm=arm,
        model_description=backend.describe(),
        benchmark_path=benchmark_path,
        generation=generation.model_dump(mode="json"),
        results=results,
        metrics=metrics,
        per_task=per_task,
        consistency=consistency_report,
        ood=ood_report,
        faithfulness=faithfulness_report if include_faithfulness else None,
        arm_config=summarize_arm(arm, orchestration),
        seeds=list(config.seeds),
        duration_seconds=duration,
        warnings=warnings,
    )

    logger.info(
        "Arm %s: overall %.4f over %d example(s) in %.1fs",
        arm,
        overall.mean,
        len(results),
        duration,
    )
    for warning in warnings:
        logger.warning("%s", warning)
    return result


def load_evaluation_result(path: Path | str) -> dict[str, Any]:
    """Load a saved results file.

    Returns the raw dict rather than a reconstructed object: ``compare.py`` needs
    to read results produced by other versions of this code, and a permissive
    reader is the right tool for that.
    """
    source = Path(path)
    if not source.exists():
        raise EvaluationError(
            f"Evaluation results file not found: {source}",
            suggestions=[
                "Run scripts/evaluate.py first to produce it.",
                "Check the path — results are written to outputs/<run>/eval_<arm>.json.",
            ],
        )
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Results file {source} is not valid JSON.",
            details={"error": str(exc)},
        ) from exc
