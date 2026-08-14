"""Graders: turn a model response into a score against a reference (spec section 21).

A grader is a small strategy object with one job — score one response — so that
new task types are added by registering a grader rather than by editing the
evaluation runner.

Three families:

* **Deterministic** — exact match, classification, set overlap, ranking. These are
  reproducible and free, and they cover the tasks where a right answer exists.
* **Heuristic rubric** — structural checks (does the answer cite evidence, cover
  the required points, avoid unsupported claims). Offline and reproducible; a
  coarse approximation of human judgement, and labelled as such.
* **LLM judge** — an interface only. It is opt-in, never the default, and it
  records the judge model in the result so a score can never be silently
  attributed to a different judge.

Every grader returns a :class:`GradeResult` with a score in [0, 1] where higher is
better, plus the sub-scores that produced it. Aggregate numbers alone hide the
failure modes that matter.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kleos_models.constants import NEGATIVE_RUBRIC_DIMENSIONS, RUBRIC_DIMENSIONS
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.metrics import (
    classification_scores,
    exact_match,
    kendall_tau,
    ndcg,
    normalize_answer,
    set_precision_recall_f1,
    token_f1,
    top_1_accuracy,
)
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GradeResult:
    """The outcome of grading one response."""

    score: float
    grader: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "grader": self.grader,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "details": self.details,
            "parse_failed": self.parse_failed,
        }


class Grader(ABC):
    """Base grader."""

    name: str = "grader"

    @abstractmethod
    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        """Score a response against a reference."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", re.MULTILINE)


def extract_json(text: str) -> Any | None:
    """Pull a JSON object or array out of a model response.

    Models wrap JSON in prose or fences far more often than they emit it cleanly,
    and treating that as a wrong answer would measure formatting compliance rather
    than judgment.
    """
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    bare = _BARE_JSON.search(text)
    if bare:
        try:
            return json.loads(bare.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def extract_ranking(text: str, *, candidates: list[str] | None = None) -> list[str]:
    """Extract an ordered list from a response.

    Tries JSON first, then a numbered or bulleted list, then — if candidates are
    known — the order in which they are first mentioned.
    """
    parsed = extract_json(text)
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    if isinstance(parsed, dict):
        for key in ("ranking", "order", "priorities", "items"):
            value = parsed.get(key)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return value

    lines = _NUMBERED_LINE.findall(text)
    if len(lines) >= 2:
        return [line.strip() for line in lines]

    if candidates:
        # Word-boundary matching, not a raw substring search: looking for the
        # candidate "a" inside "no ranking here" would otherwise match the "a"
        # in "ranking" and manufacture a ranking out of prose.
        positions: list[tuple[int, str]] = []
        for candidate in candidates:
            match = re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text, re.IGNORECASE)
            if match:
                positions.append((match.start(), candidate))
        if positions:
            positions.sort()
            return [candidate for _, candidate in positions]

    return []


def extract_label(text: str, *, allowed: list[str] | None = None) -> str:
    """Extract a single classification label from a response."""
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        for key in ("label", "decision", "answer", "category", "priority"):
            if key in parsed and isinstance(parsed[key], (str, int, float)):
                return str(parsed[key])

    if allowed:
        normalized_text = normalize_answer(text)
        # Prefer the label appearing earliest, so a trailing restatement of the
        # options does not override the actual answer.
        hits = [
            (normalized_text.find(normalize_answer(label)), label)
            for label in allowed
            if normalize_answer(label) in normalized_text
        ]
        if hits:
            hits.sort()
            return hits[0][1]

    return text.strip().split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Deterministic graders
# ---------------------------------------------------------------------------


class ExactMatchGrader(Grader):
    """String equality after normalization. Reference key: ``text``."""

    name = "exact_match"

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        expected = str(reference.get("text", reference.get("label", "")))
        score = exact_match(response, expected)
        return GradeResult(
            score=score,
            grader=self.name,
            sub_scores={"exact_match": score, "token_f1": token_f1(response, expected)},
        )


class ClassificationGrader(Grader):
    """Single-label classification. Reference keys: ``label``, optional ``options``."""

    name = "classification"

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        expected = str(reference.get("label", ""))
        options = reference.get("options")
        predicted = extract_label(response, allowed=list(options) if options else None)
        score = exact_match(predicted, expected)
        return GradeResult(
            score=score,
            grader=self.name,
            sub_scores={"accuracy": score},
            details={"predicted_label": predicted, "expected_label": expected},
            parse_failed=not predicted.strip(),
        )


class SetMatchGrader(Grader):
    """Set selection scored by F1. Reference key: ``items``."""

    name = "set_match"

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        expected = [str(item) for item in reference.get("items", [])]
        parsed = extract_json(response)
        predicted: list[str] = []
        if isinstance(parsed, list):
            predicted = [str(item) for item in parsed]
        elif isinstance(parsed, dict):
            for key in ("items", "selected", "relevant"):
                if isinstance(parsed.get(key), list):
                    predicted = [str(item) for item in parsed[key]]
                    break
        if not predicted:
            predicted = extract_ranking(response, candidates=expected)

        precision, recall, f1 = set_precision_recall_f1(predicted, expected)
        return GradeResult(
            score=f1,
            grader=self.name,
            sub_scores={"precision": precision, "recall": recall, "f1": f1},
            details={"predicted_items": predicted, "expected_items": expected},
            parse_failed=not predicted,
        )


class RankingGrader(Grader):
    """Ordering quality. Reference key: ``ranking``.

    The headline score blends nDCG with top-1 accuracy, because for prioritization
    the first item usually *is* the decision. Sub-scores are reported separately so
    a good tau with the wrong top item is still visible.
    """

    name = "ranking"

    def __init__(self, *, ndcg_weight: float = 0.6) -> None:
        self.ndcg_weight = ndcg_weight

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        expected = [str(item) for item in reference.get("ranking", [])]
        if not expected:
            raise EvaluationError(
                "RankingGrader requires a non-empty reference['ranking'].",
                details={"reference_keys": sorted(reference)},
            )

        predicted = extract_ranking(response, candidates=expected)
        if not predicted:
            return GradeResult(
                score=0.0,
                grader=self.name,
                sub_scores={"ndcg": 0.0, "kendall_tau": 0.0, "top_1_accuracy": 0.0},
                details={"expected_ranking": expected},
                parse_failed=True,
            )

        ndcg_score = ndcg(predicted, expected)
        tau = kendall_tau(predicted, expected)
        top1 = top_1_accuracy(predicted, expected)
        blended = self.ndcg_weight * ndcg_score + (1 - self.ndcg_weight) * top1

        return GradeResult(
            score=blended,
            grader=self.name,
            sub_scores={"ndcg": ndcg_score, "kendall_tau": tau, "top_1_accuracy": top1},
            details={"predicted_ranking": predicted, "expected_ranking": expected},
        )


# ---------------------------------------------------------------------------
# Heuristic rubric grader
# ---------------------------------------------------------------------------

_HEDGE_PATTERNS = (
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bi (?:cannot|can't|am unable to)\b", re.IGNORECASE),
)

#: Phrasings asserting a fact with no evidential grounding.
_UNSUPPORTED_PATTERNS = (
    re.compile(r"\bobviously\b", re.IGNORECASE),
    re.compile(r"\beveryone knows\b", re.IGNORECASE),
    re.compile(r"\bit is well known\b", re.IGNORECASE),
    re.compile(r"\bcertainly the (?:best|worst)\b", re.IGNORECASE),
)

_ACTION_PATTERNS = (
    re.compile(r"\b(?:should|recommend|suggest|next step|action|prioriti[sz]e)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE),
)


class HeuristicRubricGrader(Grader):
    """Structural rubric scoring with no API calls (spec section 21).

    Deliberately *coarse*. It checks properties that can be verified from the text
    itself — required points covered, cited evidence ids present, forbidden content
    absent, an actionable recommendation given — and reports each dimension
    separately.

    It is not a substitute for human judgement or an LLM judge on open-ended
    quality. It exists so that rubric-graded tasks have a reproducible,
    zero-cost, deterministic baseline that runs in CI, and so a rubric score can
    never quietly become "whatever the judge model felt that day".

    Reference keys:
        ``required_points``  — substrings/phrases that must be covered
        ``forbidden_points`` — content that must not appear
        ``evidence_ids``     — evidence identifiers the answer should cite
        ``expected_decision``— the decision the answer should reach
    """

    name = "heuristic_rubric"

    def __init__(self, *, dimensions: list[str] | None = None) -> None:
        self.dimensions = dimensions or [
            "correctness",
            "evidence_usage",
            "actionability",
            "critical_omission",
            "unsupported_claims",
        ]
        unknown = set(self.dimensions) - set(RUBRIC_DIMENSIONS)
        if unknown:
            raise EvaluationError(
                f"Unknown rubric dimension(s): {sorted(unknown)}",
                details={"valid_dimensions": list(RUBRIC_DIMENSIONS)},
            )

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        normalized = normalize_answer(response)
        sub_scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        required = [str(p) for p in reference.get("required_points", [])]
        forbidden = [str(p) for p in reference.get("forbidden_points", [])]
        evidence_ids = [str(e) for e in reference.get("evidence_ids", [])]
        expected_decision = reference.get("expected_decision")

        # correctness — does it reach the expected decision?
        if "correctness" in self.dimensions:
            if expected_decision is None:
                correctness = 1.0 if not required else self._coverage(normalized, required)[0]
            else:
                correctness = 1.0 if normalize_answer(str(expected_decision)) in normalized else 0.0
            sub_scores["correctness"] = correctness

        # critical_omission — required points NOT covered (inverted below).
        if "critical_omission" in self.dimensions or "correctness" in self.dimensions:
            coverage, missing = self._coverage(normalized, required)
            details["required_point_coverage"] = round(coverage, 4)
            details["missing_points"] = missing
            if "critical_omission" in self.dimensions:
                sub_scores["critical_omission"] = 1.0 - coverage

        # evidence_usage — are cited evidence ids present?
        if "evidence_usage" in self.dimensions:
            if evidence_ids:
                cited = [e for e in evidence_ids if normalize_answer(e) in normalized]
                usage = len(cited) / len(evidence_ids)
                details["cited_evidence"] = cited
            else:
                usage = 1.0 if not required else min(1.0, len(normalized.split()) / 40)
            sub_scores["evidence_usage"] = usage

        # actionability — does it actually recommend something?
        if "actionability" in self.dimensions:
            actionable = any(pattern.search(response) for pattern in _ACTION_PATTERNS)
            hedged = any(pattern.search(response) for pattern in _HEDGE_PATTERNS)
            sub_scores["actionability"] = float(actionable and not hedged)

        # unsupported_claims — asserted-without-grounding phrasings, plus anything
        # the reference explicitly forbids.
        if "unsupported_claims" in self.dimensions:
            hits = [p.pattern for p in _UNSUPPORTED_PATTERNS if p.search(response)]
            forbidden_hits = [f for f in forbidden if normalize_answer(f) in normalized]
            details["unsupported_phrases"] = hits
            details["forbidden_hits"] = forbidden_hits
            penalty = min(1.0, 0.34 * (len(hits) + 2 * len(forbidden_hits)))
            sub_scores["unsupported_claims"] = penalty

        if "relevance" in self.dimensions:
            sub_scores["relevance"] = self._coverage(normalized, required)[0] if required else 1.0
        if "prioritization_quality" in self.dimensions:
            ranking = reference.get("ranking", [])
            predicted = extract_ranking(response, candidates=[str(r) for r in ranking])
            sub_scores["prioritization_quality"] = (
                ndcg(predicted, [str(r) for r in ranking]) if ranking else 0.0
            )

        score = self._aggregate(sub_scores)
        return GradeResult(
            score=score,
            grader=self.name,
            sub_scores=sub_scores,
            details=details,
        )

    @staticmethod
    def _coverage(normalized_response: str, points: list[str]) -> tuple[float, list[str]]:
        """Fraction of required points present, plus the missing ones."""
        if not points:
            return 1.0, []
        missing = [p for p in points if normalize_answer(p) not in normalized_response]
        return (len(points) - len(missing)) / len(points), missing

    @staticmethod
    def _aggregate(sub_scores: dict[str, float]) -> float:
        """Average the dimensions, inverting the ones where higher is worse."""
        if not sub_scores:
            return 0.0
        adjusted = [
            (1.0 - value) if name in NEGATIVE_RUBRIC_DIMENSIONS else value
            for name, value in sub_scores.items()
        ]
        return sum(adjusted) / len(adjusted)


# ---------------------------------------------------------------------------
# LLM judge (interface only)
# ---------------------------------------------------------------------------


class LLMJudgeGrader(Grader):
    """Rubric grading by a judge model (spec section 21).

    Deliberately **not implemented against any provider**. Spec section 20 says not
    to implement proprietary APIs unless requested, while still designing the
    interface so they can be added later.

    Supply a ``judge_fn(prompt) -> str`` to activate it. The judge model id is
    recorded in every result, so scores from different judges can never be pooled
    by accident — that would be exactly the kind of silent change to the
    measurement instrument that spec section 36 forbids.
    """

    name = "llm_judge"

    def __init__(
        self,
        judge_fn: Any = None,
        *,
        judge_model: str = "unspecified",
        dimensions: list[str] | None = None,
    ) -> None:
        self.judge_fn = judge_fn
        self.judge_model = judge_model
        self.dimensions = dimensions or list(RUBRIC_DIMENSIONS)

    def build_prompt(self, response: str, reference: dict[str, Any]) -> str:
        """Build the judging prompt. Public so it can be reviewed and versioned."""
        rubric = "\n".join(f"- {d}: rate 0-4" for d in self.dimensions)
        return (
            "You are grading an assistant response against a rubric.\n"
            "Rate each dimension from 0 (worst) to 4 (best).\n"
            "Respond ONLY with a JSON object mapping dimension names to integers.\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Reference:\n{json.dumps(reference, indent=2)}\n\n"
            f"Response to grade:\n{response}\n"
        )

    def grade(self, response: str, reference: dict[str, Any], **context: Any) -> GradeResult:
        if self.judge_fn is None:
            raise EvaluationError(
                "LLMJudgeGrader was selected but no judge function was supplied.",
                details={"judge_model": self.judge_model},
                suggestions=[
                    "LLM-judge grading is opt-in and needs an explicit judge callable.",
                    "Use 'heuristic_rubric' for offline, reproducible rubric scoring.",
                    "If you add a judge, record its model id: judge scores from "
                    "different judges are not comparable.",
                ],
            )

        raw = self.judge_fn(self.build_prompt(response, reference))
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            return GradeResult(
                score=0.0,
                grader=self.name,
                details={"judge_model": self.judge_model, "raw_response": str(raw)[:500]},
                parse_failed=True,
            )

        sub_scores = {
            dimension: max(0.0, min(1.0, float(parsed[dimension]) / 4.0))
            for dimension in self.dimensions
            if dimension in parsed and isinstance(parsed[dimension], (int, float))
        }
        if not sub_scores:
            return GradeResult(
                score=0.0,
                grader=self.name,
                details={"judge_model": self.judge_model},
                parse_failed=True,
            )

        adjusted = [
            (1.0 - value) if name in NEGATIVE_RUBRIC_DIMENSIONS else value
            for name, value in sub_scores.items()
        ]
        return GradeResult(
            score=sum(adjusted) / len(adjusted),
            grader=self.name,
            sub_scores=sub_scores,
            details={"judge_model": self.judge_model},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY: dict[str, type[Grader]] = {
    "exact_match": ExactMatchGrader,
    "classification": ClassificationGrader,
    "set_match": SetMatchGrader,
    "ranking": RankingGrader,
    "heuristic_rubric": HeuristicRubricGrader,
    "llm_judge": LLMJudgeGrader,
}


def get_grader(name: str, **kwargs: Any) -> Grader:
    """Instantiate a grader by name.

    Raises:
        EvaluationError: when the name is not registered.
    """
    grader_class = GRADER_REGISTRY.get(name)
    if grader_class is None:
        raise EvaluationError(
            f"Unknown grader {name!r}.",
            details={"registered_graders": sorted(GRADER_REGISTRY)},
            suggestions=[
                "Set evaluation.graders to a registered grader.",
                "Add a new one by subclassing Grader and registering it in GRADER_REGISTRY.",
            ],
        )
    return grader_class(**kwargs)


def register_grader(name: str, grader_class: type[Grader]) -> None:
    """Register a custom grader."""
    GRADER_REGISTRY[name] = grader_class


__all__ = [
    "GRADER_REGISTRY",
    "ClassificationGrader",
    "ExactMatchGrader",
    "GradeResult",
    "Grader",
    "HeuristicRubricGrader",
    "LLMJudgeGrader",
    "RankingGrader",
    "SetMatchGrader",
    "classification_scores",
    "extract_json",
    "extract_label",
    "extract_ranking",
    "get_grader",
    "register_grader",
]
