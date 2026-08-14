"""Faithfulness: is the answer grounded in the supplied evidence? (spec section 21)

KLEOS assembles context from retrieval and memory, so the failure mode that
matters is not "the model said something false about the world" but "the model
asserted something the supplied context does not support".

Three measurable properties:

``evidence_coverage``
    Of the evidence the reference marks as decisive, how much did the answer use?
``citation_precision``
    Of the evidence ids the answer cites, how many were actually provided?
    Citing a non-existent source is a fabrication, and it is checkable.
``unsupported_claim_rate``
    Fraction of specific claims (numbers, dates, named entities) appearing in the
    answer but nowhere in the supplied context.

These are heuristics over text, not entailment checks. They are precise about what
they measure and are not presented as a general hallucination metric.

This module imports no torch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.evaluation.metrics import normalize_answer
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Claim-bearing tokens: numbers, dates, percentages, currency, capitalized names.
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.IGNORECASE,
)
_CURRENCY = re.compile(r"[$€£]\s?\d[\d,.]*")
_PROPER_NOUN = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
_CITATION = re.compile(
    r"\b(?:evidence|source|doc|item|ref)[-_ ]?([A-Za-z0-9_-]{1,32})\b", re.IGNORECASE
)

#: Words that start a sentence and are capitalized for that reason alone.
_SENTENCE_STARTERS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Here",
        "It",
        "They",
        "You",
        "Your",
        "We",
        "Our",
        "If",
        "When",
        "While",
        "Since",
        "Because",
        "However",
        "Therefore",
        "Given",
        "Based",
        "Prioritize",
        "Recommend",
        "Consider",
        "Note",
        "First",
        "Second",
        "Third",
        "Finally",
        "Both",
        "All",
    }
)


@dataclass
class FaithfulnessResult:
    """Faithfulness scores for one response."""

    evidence_coverage: float
    citation_precision: float
    unsupported_claim_rate: float
    cited_ids: list[str] = field(default_factory=list)
    fabricated_ids: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Blended faithfulness in [0, 1], higher is better."""
        return (
            self.evidence_coverage + self.citation_precision + (1.0 - self.unsupported_claim_rate)
        ) / 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "citation_precision": round(self.citation_precision, 4),
            "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
            "cited_ids": self.cited_ids,
            "fabricated_ids": self.fabricated_ids,
            "unsupported_claims": self.unsupported_claims[:20],
            "supported_claim_count": len(self.supported_claims),
        }


def extract_claims(text: str) -> list[str]:
    """Extract specific, checkable claim tokens from a response.

    Deliberately limited to tokens whose presence in the context can be verified
    directly. Vague assertions are not extracted, because scoring them without
    entailment would produce a number that looks rigorous and is not.
    """
    claims: set[str] = set()
    claims.update(match.group(0) for match in _NUMBER.finditer(text))
    claims.update(match.group(0) for match in _DATE.finditer(text))
    claims.update(match.group(0) for match in _CURRENCY.finditer(text))

    for match in _PROPER_NOUN.finditer(text):
        candidate = match.group(0)
        first_word = candidate.split()[0]
        if first_word in _SENTENCE_STARTERS and len(candidate.split()) == 1:
            continue
        claims.add(candidate)

    return sorted(claims)


def extract_citations(text: str) -> list[str]:
    """Extract evidence identifiers the response claims to cite."""
    return sorted({match.group(1) for match in _CITATION.finditer(text)})


def assess_faithfulness(
    response: str,
    *,
    context_text: str,
    provided_evidence_ids: Sequence[str] = (),
    decisive_evidence_ids: Sequence[str] = (),
) -> FaithfulnessResult:
    """Score how well a response is grounded in its supplied context.

    Args:
        response: The model's answer.
        context_text: All context the model was given, concatenated.
        provided_evidence_ids: Evidence ids actually supplied.
        decisive_evidence_ids: Ids the reference marks as necessary for the answer.

    Returns:
        A :class:`FaithfulnessResult`.
    """
    normalized_context = normalize_answer(context_text)

    # --- coverage of decisive evidence -------------------------------------
    if decisive_evidence_ids:
        normalized_response = normalize_answer(response)
        used = [
            evidence_id
            for evidence_id in decisive_evidence_ids
            if normalize_answer(str(evidence_id)) in normalized_response
        ]
        coverage = len(used) / len(decisive_evidence_ids)
    else:
        coverage = 1.0

    # --- citation precision -------------------------------------------------
    cited = extract_citations(response)
    provided_normalized = {normalize_answer(str(e)) for e in provided_evidence_ids}
    if cited and provided_normalized:
        fabricated = [c for c in cited if normalize_answer(c) not in provided_normalized]
        precision = (len(cited) - len(fabricated)) / len(cited)
    else:
        fabricated = []
        precision = 1.0

    # --- unsupported claims -------------------------------------------------
    claims = extract_claims(response)
    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims:
        if normalize_answer(claim) in normalized_context:
            supported.append(claim)
        else:
            unsupported.append(claim)
    rate = len(unsupported) / len(claims) if claims else 0.0

    return FaithfulnessResult(
        evidence_coverage=coverage,
        citation_precision=precision,
        unsupported_claim_rate=rate,
        cited_ids=cited,
        fabricated_ids=fabricated,
        unsupported_claims=unsupported,
        supported_claims=supported,
    )


@dataclass
class FaithfulnessReport:
    """Aggregate faithfulness across a benchmark."""

    results: list[FaithfulnessResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    def _mean(self, attribute: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attribute) for r in self.results) / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def responses_with_fabricated_citations(self) -> int:
        return sum(1 for r in self.results if r.fabricated_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_score": round(self.mean_score, 4),
            "mean_evidence_coverage": round(self._mean("evidence_coverage"), 4),
            "mean_citation_precision": round(self._mean("citation_precision"), 4),
            "mean_unsupported_claim_rate": round(self._mean("unsupported_claim_rate"), 4),
            "responses_with_fabricated_citations": self.responses_with_fabricated_citations,
        }

    def render(self) -> str:
        if not self.results:
            return "Faithfulness: not evaluated (no responses)."
        return "\n".join(
            [
                "Faithfulness",
                f"  responses evaluated       : {self.count}",
                f"  mean score                : {self.mean_score:.4f}",
                f"  evidence coverage         : {self._mean('evidence_coverage'):.4f}",
                f"  citation precision        : {self._mean('citation_precision'):.4f}",
                f"  unsupported claim rate    : {self._mean('unsupported_claim_rate'):.4f}",
                f"  fabricated citations in   : {self.responses_with_fabricated_citations} "
                "response(s)",
                "",
                "  These are text-level heuristics, not entailment checks. They detect",
                "  claims and citations absent from the supplied context; they do not",
                "  verify semantic support.",
            ]
        )
