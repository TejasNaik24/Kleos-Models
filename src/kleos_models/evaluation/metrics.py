"""Evaluation metrics (spec section 21).

The research question is about *judgment and correctness*, not tone, so the
metrics here measure decisions: did the model pick the right item, rank things in
the right order, cite the evidence it claims to.

Implemented in pure Python with no numpy dependency in the hot paths, so the
scoring layer imports without torch and runs anywhere.

Conventions
-----------
* Every metric returns a float where **higher is better**.
* Metrics degrade explicitly: an undefined metric (no positives, empty ranking)
  returns 0.0 and records why, rather than raising or returning NaN.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalize a free-text answer for comparison.

    Lower-cases, strips accents, removes articles and punctuation, collapses
    whitespace. Standard practice for extractive QA scoring, and it stops
    "The Q3 report." and "q3 report" being counted as a disagreement.

    Article stripping is skipped when it would empty the string. A label that is
    literally ``"a"`` or ``"the"`` is unusual but legal, and normalizing it to
    the empty string would silently collapse it together with every other such
    label.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.casefold()

    without_articles = _ARTICLES.sub(" ", lowered)
    if not _SPACE.sub(" ", _PUNCT.sub(" ", without_articles)).strip():
        without_articles = lowered

    without_punct = _PUNCT.sub(" ", without_articles)
    return _SPACE.sub(" ", without_punct).strip()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def exact_match(prediction: str, reference: str) -> float:
    """1.0 when the normalized strings are identical."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between two texts."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class ClassificationScores:
    """Per-class and aggregate classification metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "per_class": {
                label: {k: round(v, 4) for k, v in scores.items()}
                for label, scores in self.per_class.items()
            },
            "support": self.support,
        }


def classification_scores(
    predictions: Sequence[str], references: Sequence[str]
) -> ClassificationScores:
    """Accuracy plus macro/micro precision, recall and F1.

    Macro-averaging is reported alongside micro because KLEOS label distributions
    are imbalanced (most notifications are not urgent), and accuracy alone would
    let a majority-class predictor look competent.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references differ in length: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return ClassificationScores(0.0, 0.0, 0.0, 0.0, 0.0)

    normalized_predictions = [normalize_answer(p) for p in predictions]
    normalized_references = [normalize_answer(r) for r in references]
    labels = sorted(set(normalized_references) | set(normalized_predictions))

    correct = sum(
        1 for p, r in zip(normalized_predictions, normalized_references, strict=True) if p == r
    )
    accuracy = correct / len(predictions)

    per_class: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    true_positives = false_positives = false_negatives = 0

    for label in labels:
        tp = sum(
            1
            for p, r in zip(normalized_predictions, normalized_references, strict=True)
            if p == label and r == label
        )
        fp = sum(
            1
            for p, r in zip(normalized_predictions, normalized_references, strict=True)
            if p == label and r != label
        )
        fn = sum(
            1
            for p, r in zip(normalized_predictions, normalized_references, strict=True)
            if p != label and r == label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        support[label] = tp + fn
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    # Macro-average over classes that actually occur in the references.
    present = [label for label in labels if support.get(label, 0) > 0] or labels
    macro_precision = sum(per_class[c]["precision"] for c in present) / len(present)
    macro_recall = sum(per_class[c]["recall"] for c in present) / len(present)
    macro_f1 = sum(per_class[c]["f1"] for c in present) / len(present)

    micro_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    micro_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    confusion: dict[str, dict[str, int]] = {}
    for predicted, reference in zip(normalized_predictions, normalized_references, strict=True):
        confusion.setdefault(reference, {})
        confusion[reference][predicted] = confusion[reference].get(predicted, 0) + 1

    return ClassificationScores(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_class=per_class,
        support=support,
        confusion=confusion,
    )


# ---------------------------------------------------------------------------
# Set metrics
# ---------------------------------------------------------------------------


def set_precision_recall_f1(
    predicted: Sequence[str], reference: Sequence[str]
) -> tuple[float, float, float]:
    """Precision, recall and F1 over two sets of items.

    Used for "which context items are relevant" style tasks, where the answer is
    a set rather than a single label.
    """
    predicted_set = {normalize_answer(p) for p in predicted}
    reference_set = {normalize_answer(r) for r in reference}
    if not reference_set:
        return (1.0, 1.0, 1.0) if not predicted_set else (0.0, 1.0, 0.0)
    if not predicted_set:
        return 0.0, 0.0, 0.0

    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(reference_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain."""
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg(
    predicted_order: Sequence[str], ideal_order: Sequence[str], *, k: int | None = None
) -> float:
    """Normalized DCG of a predicted ranking against an ideal one.

    Relevance is assigned by position in the ideal ranking: the top item scores
    ``n``, the next ``n-1``, and so on. This rewards getting the most important
    item first, which is what prioritization tasks care about.
    """
    if not ideal_order:
        return 0.0
    relevance = {normalize_answer(item): len(ideal_order) - i for i, item in enumerate(ideal_order)}
    cutoff = k or len(ideal_order)

    predicted_relevance = [
        relevance.get(normalize_answer(item), 0.0) for item in predicted_order[:cutoff]
    ]
    ideal_relevance = sorted(relevance.values(), reverse=True)[:cutoff]

    ideal_dcg = dcg(ideal_relevance)
    return dcg(predicted_relevance) / ideal_dcg if ideal_dcg else 0.0


def kendall_tau(predicted_order: Sequence[str], reference_order: Sequence[str]) -> float:
    """Kendall's tau-a rank correlation, rescaled to [0, 1].

    0.5 means "no better than random ordering", 1.0 is a perfect match, 0.0 is
    exactly reversed. Rescaled so the metric obeys the higher-is-better rule.
    """
    common = [
        item
        for item in (normalize_answer(i) for i in reference_order)
        if item in {normalize_answer(p) for p in predicted_order}
    ]
    if len(common) < 2:
        return 0.5

    predicted_rank = {normalize_answer(item): index for index, item in enumerate(predicted_order)}
    reference_rank = {normalize_answer(item): index for index, item in enumerate(reference_order)}

    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            left, right = common[i], common[j]
            predicted_delta = predicted_rank[left] - predicted_rank[right]
            reference_delta = reference_rank[left] - reference_rank[right]
            if predicted_delta * reference_delta > 0:
                concordant += 1
            elif predicted_delta * reference_delta < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.5
    tau = (concordant - discordant) / total
    return (tau + 1.0) / 2.0


def spearman_footrule(predicted_order: Sequence[str], reference_order: Sequence[str]) -> float:
    """Normalized Spearman footrule similarity in [0, 1].

    Sums absolute rank displacement, normalized so 1.0 is identical ordering.
    Complements Kendall's tau: tau counts pairwise inversions, footrule measures
    how far items moved.
    """
    reference_rank = {normalize_answer(item): index for index, item in enumerate(reference_order)}
    predicted_rank = {normalize_answer(item): index for index, item in enumerate(predicted_order)}
    common = set(reference_rank) & set(predicted_rank)
    if not common:
        return 0.0

    n = len(reference_order)
    displacement = sum(abs(reference_rank[item] - predicted_rank[item]) for item in common)
    max_displacement = (n * n) // 2 if n > 1 else 1
    return max(0.0, 1.0 - displacement / max_displacement)


def top_1_accuracy(predicted_order: Sequence[str], reference_order: Sequence[str]) -> float:
    """Whether the top-ranked item matches.

    Reported separately because for prioritization the first item usually is the
    decision; a good tau with the wrong top item is still the wrong answer.
    """
    if not predicted_order or not reference_order:
        return 0.0
    return float(normalize_answer(predicted_order[0]) == normalize_answer(reference_order[0]))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    """An aggregated metric with dispersion, so single numbers are not oversold."""

    name: str
    mean: float
    count: int
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0

    @property
    def stderr(self) -> float:
        return self.std / math.sqrt(self.count) if self.count > 1 else 0.0

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Normal-approximation confidence interval around the mean."""
        margin = z * self.stderr
        return self.mean - margin, self.mean + margin

    def to_dict(self) -> dict[str, Any]:
        low, high = self.confidence_interval()
        return {
            "name": self.name,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "stderr": round(self.stderr, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "count": self.count,
        }

    def render(self) -> str:
        low, high = self.confidence_interval()
        return f"{self.mean:.4f} ± {self.stderr:.4f} (95% CI {low:.4f}–{high:.4f}, n={self.count})"


def summarize(name: str, values: Sequence[float]) -> MetricSummary:
    """Aggregate per-example scores into a summary with dispersion."""
    if not values:
        return MetricSummary(name=name, mean=0.0, count=0)
    count = len(values)
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / (count - 1) if count > 1 else 0.0
    return MetricSummary(
        name=name,
        mean=mean,
        count=count,
        std=math.sqrt(variance),
        minimum=min(values),
        maximum=max(values),
    )


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test on the difference between two score vectors.

    Reports a confidence interval and a two-sided p-value for
    ``mean(right) - mean(left)``.

    Why this and not a t-test: evaluation scores are usually bounded and far from
    normal (many exact 0.0 and 1.0), and a paired bootstrap makes no distributional
    assumption. It is also the honest way to answer "did fine-tuning actually
    help", which is the entire point of the experiment.
    """
    import random

    if len(left) != len(right):
        raise ValueError(
            f"paired bootstrap needs equal-length vectors: {len(left)} vs {len(right)}"
        )
    if not left:
        return {"difference": 0.0, "p_value": 1.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}

    observed = sum(right) / len(right) - sum(left) / len(left)
    rng = random.Random(seed)
    indices = range(len(left))

    differences: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(indices) for _ in indices]
        left_mean = sum(left[i] for i in sample) / len(sample)
        right_mean = sum(right[i] for i in sample) / len(sample)
        differences.append(right_mean - left_mean)

    differences.sort()
    low = differences[int(0.025 * iterations)]
    high = differences[min(int(0.975 * iterations), iterations - 1)]

    # Two-sided p-value: how often the resampled difference crosses zero.
    if observed >= 0:
        p_value = 2 * sum(1 for d in differences if d <= 0) / iterations
    else:
        p_value = 2 * sum(1 for d in differences if d >= 0) / iterations

    return {
        "difference": round(observed, 4),
        "p_value": round(min(1.0, p_value), 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "n": len(left),
        "iterations": iterations,
        "significant_at_05": bool(min(1.0, p_value) < 0.05),
    }


#: Registered metric functions taking ``(prediction, reference)``.
TEXT_METRICS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
}

#: Registered ranking metrics taking ``(predicted_order, reference_order)``.
RANKING_METRICS = {
    "ndcg": ndcg,
    "kendall_tau": kendall_tau,
    "spearman_footrule": spearman_footrule,
    "top_1_accuracy": top_1_accuracy,
}
