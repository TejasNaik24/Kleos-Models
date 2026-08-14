"""Data layer: schemas, loading, validation, formatting, splitting, leakage.

Nothing in this package imports torch. Dataset preparation, validation and
auditing therefore run on any machine, in CI, and in a notebook without a GPU.
"""

from kleos_models.data.coverage import (
    CoverageReport,
    build_coverage_report,
    render_coverage_report,
)
from kleos_models.data.formatting import (
    ConversationFormatter,
    FormattedExample,
    FormattingStats,
)
from kleos_models.data.leakage import (
    LeakageKind,
    LeakageReport,
    check_leakage,
    enforce_leakage_policy,
)
from kleos_models.data.loaders import (
    DatasetBundle,
    load_dataset_bundle,
    load_evaluation_examples,
    load_examples,
    load_manifest,
    write_jsonl,
)
from kleos_models.data.schemas import (
    DatasetManifest,
    EvaluationExample,
    ExampleMetadata,
    Message,
    TrainingExample,
    VariationAxes,
)
from kleos_models.data.splitting import SplitResult, split_examples, verify_split
from kleos_models.data.validation import (
    QualityReport,
    Severity,
    ValidationReport,
    build_quality_report,
    render_quality_report,
    validate_examples,
)

__all__ = [
    "ConversationFormatter",
    "CoverageReport",
    "DatasetBundle",
    "DatasetManifest",
    "EvaluationExample",
    "ExampleMetadata",
    "FormattedExample",
    "FormattingStats",
    "LeakageKind",
    "LeakageReport",
    "Message",
    "QualityReport",
    "Severity",
    "SplitResult",
    "TrainingExample",
    "ValidationReport",
    "VariationAxes",
    "build_coverage_report",
    "build_quality_report",
    "check_leakage",
    "enforce_leakage_policy",
    "load_dataset_bundle",
    "load_evaluation_examples",
    "load_examples",
    "load_manifest",
    "render_coverage_report",
    "render_quality_report",
    "split_examples",
    "validate_examples",
    "verify_split",
    "write_jsonl",
]
