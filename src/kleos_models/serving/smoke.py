"""The frozen-output smoke suite, shared by every serving path.

`scripts/hermes_smoke.py` (in-process, any GPU host or the Docker image) and
`scripts/zerogpu_smoke.py` (over the network, against a ZeroGPU Space) must run
the *same* examples and judge them the *same* way, or their results cannot be
compared. Both import from here.

This produces reproducibility evidence, never a score: nine examples say nothing
about quality that the 349-example benchmark does not already say better.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from kleos_models.errors import KleosError

#: Task families the suite must cover. Named rather than derived so a benchmark
#: that silently loses a family is caught instead of quietly skipped.
REQUIRED_TASKS = (
    "workspace_reasoning",
    "memory_conflict_resolution",
    "tool_routing",
    "recommendation_generation",
    "mission_control_briefing",
    "notification_prioritization",
    "context_prioritization",
)


def select_suite(examples: list[Any], *, per_task: int = 1, abstention: int = 2) -> list[Any]:
    """Pick a small, deterministic, representative set.

    Sorted by example id, so the same benchmark always yields the same suite and
    a passing run today is comparable with one next month. Abstention cases are
    added explicitly because they are the behaviour most likely to reveal a
    tokenizer or prompt-assembly difference.
    """
    by_task: dict[str, list[Any]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: str(e.id)):
        by_task[str(example.task)].append(example)

    selected: dict[str, Any] = {}
    for task in sorted(by_task):
        for example in by_task[task][:per_task]:
            selected[str(example.id)] = example

    declines = [
        e
        for e in sorted(examples, key=lambda e: str(e.id))
        if (e.reference or {}).get("confident") is False
    ]
    for example in declines[:abstention]:
        selected[str(example.id)] = example

    missing = [task for task in REQUIRED_TASKS if task not in by_task]
    if missing:
        raise KleosError(
            f"The benchmark is missing required task families: {missing}",
            suggestions=["Rebuild it with scripts/build_benchmark.py from the sealed release."],
        )
    return [selected[key] for key in sorted(selected)]


def compare_output(expected: str, actual: str) -> dict[str, Any]:
    """Exact comparison, plus where the two first diverge when they differ.

    The divergence position helps tell a numeric drift late in generation
    (identical opening, one token flips) from a setup error (different from the
    first word).
    """
    want = (expected or "").strip()
    have = (actual or "").strip()
    if want == have:
        return {"match": True, "first_difference": None}
    first = next(
        (i for i, (a, b) in enumerate(zip(want, have, strict=False)) if a != b),
        min(len(want), len(have)),
    )
    return {
        "match": False,
        "first_difference": first,
        "expected_chars": len(want),
        "actual_chars": len(have),
        "shared_prefix_fraction": round(first / max(len(want), 1), 3),
    }


def load_reference(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Frozen per-example records from an evaluation results JSON, keyed by id."""
    records = payload.get("results") or payload.get("examples") or []
    return {str(r["example_id"]): r for r in records if "example_id" in r}


#: Roles the serving API accepts. A benchmark turn outside these cannot be sent
#: as-is, and is reported rather than silently rewritten.
API_ROLES = frozenset({"system", "user", "assistant"})


def api_messages(example: Any) -> list[dict[str, str]] | None:
    """The example's messages as the API takes them, or None if not expressible."""
    messages = []
    for message in example.messages:
        if message.role not in API_ROLES or getattr(message, "name", None):
            return None
        messages.append({"role": message.role, "content": message.content})
    return messages


def judge_response(
    example: Any, reference: dict[str, Any] | None, response: dict[str, Any]
) -> dict[str, Any]:
    """One example's reproducibility evidence from a status-contract response.

    Exact text first. Then the two diagnostics that localise a difference:
    prompt token count (equal means tokenization and chat template agree, so a
    difference is in the arithmetic) and the extracted decision (equal means a
    difference did not change what Hermes decided).
    """
    record: dict[str, Any] = {
        "example_id": str(example.id),
        "task": str(example.task),
        "status": response.get("status"),
        "ok": bool(response.get("ok")),
    }
    if not record["ok"]:
        record["retry_after_seconds"] = response.get("retry_after_seconds")
        return record

    text = response.get("text") or ""
    usage = response.get("usage") or {}
    record.update(
        {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": response.get("finish_reason"),
            "empty": not text.strip(),
            "timings": response.get("timings") or {},
            "diagnostics": response.get("diagnostics") or {},
            "response": text,
        }
    )
    if reference is None:
        return record

    from kleos_models.evaluation.graders import get_grader

    # The evaluation's own decision rule, so "agrees" means what it meant there.
    from kleos_models.evaluation.runner import _extract_decision

    grade = get_grader(example.grader or "exact_match").grade(
        text, example.reference, example=example
    )
    record.update(
        {
            "exact": compare_output(reference.get("response", ""), text),
            "prompt_tokens_match": usage.get("prompt_tokens") == reference.get("prompt_tokens"),
            "completion_tokens_match": (
                usage.get("completion_tokens") == reference.get("completion_tokens")
            ),
            "decision_matches": _extract_decision(grade, text) == reference.get("decision"),
        }
    )
    return record


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and observed measurements for a smoke run. Never a quality score."""
    answered = [r for r in records if r.get("ok")]
    compared = [r for r in answered if "exact" in r]
    cold = [r for r in answered if r["diagnostics"].get("cold_start")]
    warm = [r for r in answered if r["diagnostics"].get("cold_start") is False]

    def throughput(rows: list[dict[str, Any]]) -> list[float]:
        return [
            r["completion_tokens"] / r["timings"]["gpu_generate_s"]
            for r in rows
            if r["timings"].get("gpu_generate_s")
        ]

    def phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "calls": len(rows),
            "median_gpu_acquire_s": _median([r["timings"]["gpu_acquire_s"] for r in rows]),
            "median_gpu_generate_s": _median([r["timings"]["gpu_generate_s"] for r in rows]),
            "median_total_s": _median([r["timings"]["total_s"] for r in rows]),
            "median_client_wall_s": _median(
                [r["client_wall_s"] for r in rows if "client_wall_s" in r]
            ),
            "median_tokens_per_second": _median(throughput(rows)),
        }

    statuses: dict[str, int] = defaultdict(int)
    for r in records:
        statuses[str(r.get("status"))] += 1
    diagnostics = answered[0]["diagnostics"] if answered else {}
    return {
        "requests": len(records),
        "answered": len(answered),
        "statuses": dict(statuses),
        "empty": sum(1 for r in answered if r.get("empty")),
        "compared": len(compared),
        "exact_matches": sum(1 for r in compared if r["exact"]["match"]),
        "prompt_token_matches": sum(1 for r in compared if r["prompt_tokens_match"]),
        "completion_token_matches": sum(1 for r in compared if r["completion_tokens_match"]),
        "decision_matches": sum(1 for r in compared if r["decision_matches"]),
        "measurements": {
            "label": "Observed ZeroGPU smoke-test measurements",
            "device": diagnostics.get("device"),
            "compute_capability": diagnostics.get("compute_capability"),
            "cuda_runtime": diagnostics.get("cuda_runtime"),
            "peak_vram_gib": max(
                (r["diagnostics"].get("peak_vram_gib") or 0 for r in answered), default=None
            ),
            "cold": phase(cold),
            "warm": phase(warm),
            # Upper bound on quota used: includes scheduling and, on a cold
            # call, moving the weights onto the GPU.
            "gpu_call_seconds_total": round(sum(r["timings"]["gpu_call_s"] for r in answered), 1),
            # Lower bound: time spent generating.
            "gpu_generate_seconds_total": round(
                sum(r["timings"]["gpu_generate_s"] for r in answered), 1
            ),
        },
    }
