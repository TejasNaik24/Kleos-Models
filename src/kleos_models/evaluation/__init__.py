"""Evaluation layer: metrics, graders, consistency, OOD, faithfulness, reporting.

The scoring layer imports no torch. Metrics, graders, consistency, OOD analysis and
report generation can all be tested and re-run offline against saved results, which
is what keeps re-analysis cheap and CI meaningful.
"""

from kleos_models.evaluation.capability import (
    CAPABILITY_SUITE_VERSION,
    CapabilityDelta,
    build_capability_delta,
)
from kleos_models.evaluation.consistency import (
    ConsistencyGroup,
    ConsistencyReport,
    build_consistency_groups,
)
from kleos_models.evaluation.faithfulness import (
    FaithfulnessReport,
    FaithfulnessResult,
    assess_faithfulness,
)
from kleos_models.evaluation.graders import (
    GRADER_REGISTRY,
    Grader,
    GradeResult,
    HeuristicRubricGrader,
    get_grader,
    register_grader,
)
from kleos_models.evaluation.metrics import (
    MetricSummary,
    bootstrap_difference,
    classification_scores,
    exact_match,
    kendall_tau,
    ndcg,
    normalize_answer,
    summarize,
    token_f1,
)
from kleos_models.evaluation.ood import OODReport, build_ood_report, compare_ood_reports
from kleos_models.evaluation.reports import (
    ComparisonReport,
    TaskComparison,
    compare_results,
    render_markdown_report,
    write_experiment_report,
)
from kleos_models.evaluation.runner import (
    EvaluationResult,
    ExampleResult,
    load_evaluation_result,
    run_evaluation,
)

__all__ = [
    "CAPABILITY_SUITE_VERSION",
    "GRADER_REGISTRY",
    "CapabilityDelta",
    "ComparisonReport",
    "ConsistencyGroup",
    "ConsistencyReport",
    "EvaluationResult",
    "ExampleResult",
    "FaithfulnessReport",
    "FaithfulnessResult",
    "GradeResult",
    "Grader",
    "HeuristicRubricGrader",
    "MetricSummary",
    "OODReport",
    "TaskComparison",
    "assess_faithfulness",
    "bootstrap_difference",
    "build_capability_delta",
    "build_consistency_groups",
    "build_ood_report",
    "classification_scores",
    "compare_ood_reports",
    "compare_results",
    "exact_match",
    "get_grader",
    "kendall_tau",
    "load_evaluation_result",
    "ndcg",
    "normalize_answer",
    "register_grader",
    "render_markdown_report",
    "run_evaluation",
    "summarize",
    "token_f1",
    "write_experiment_report",
]
