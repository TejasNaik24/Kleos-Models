#!/usr/bin/env python3
"""Prove the exact frozen artifact loads and answers outside the training notebook.

This is a deployment check, not an experiment. It does **not** produce a score,
and its output must never be reported as one — it runs a handful of examples, so
any number it computes would be noise next to the 349-example benchmark.

What it proves:

1. The package verifies (hashes, pinned revision, adapter config).
2. The exact base revision, frozen tokenizer and adapter load together.
3. Representative KLEOS prompts produce non-empty responses.
4. Those responses match what the frozen v0.0.6 evaluation recorded for the
   same example ids.

Point 4 is the real test. Decoding is greedy and seeded, so the same weights on
the same prompts should reproduce the same text. A mismatch means the deployment
path differs from the evaluation path somewhere — tokenizer, revision,
quantization, or hardware. It is a reproducibility problem to investigate, and
never a reason to retrain.

Usage::

    python scripts/hermes_smoke.py \\
        --package /path/to/hermes-v0.0.6 \\
        --benchmark /path/to/benchmark.jsonl \\
        --reference /path/to/arm2_finetuned.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.data.loaders import load_evaluation_examples
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


def select_suite(examples: list[Any], *, per_task: int, abstention: int) -> list[Any]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True, help="Deployment package root.")
    parser.add_argument("--benchmark", type=Path, required=True, help="benchmark.jsonl.")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Frozen arm2 evaluation JSON, to compare responses against.",
    )
    parser.add_argument("--per-task", type=int, default=1, help="Examples per task family.")
    parser.add_argument("--abstention", type=int, default=2, help="Should-decline examples to add.")
    parser.add_argument("--output", type=Path, help="Write the smoke results here.")
    parser.add_argument(
        "--show-text", action="store_true", help="Print responses (they contain prompt content)."
    )
    parser.add_argument(
        "--require-remote-revision",
        action="store_true",
        help="Confirm the base commit with the model registry before loading.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Hermes exact-artifact smoke test")

    from kleos_models.serving.loader import load_deployment

    deployment = load_deployment(
        args.package, verify=True, require_remote_revision=args.require_remote_revision
    )
    identity = deployment.describe()
    print(f"  model     : {identity['model_name']} {identity['model_version']}")
    print(f"  base      : {identity['base_model']}@{identity['base_revision'][:12]}…")
    print(f"  adapter   : {identity['adapter_sha256'][:16]}…")
    print(f"  tokenizer : fix_mistral_regex={identity['tokenizer_fix_mistral_regex']}")

    examples = load_evaluation_examples(args.benchmark, strict=True)
    suite = select_suite(examples, per_task=args.per_task, abstention=args.abstention)
    print(f"\n  benchmark : {args.benchmark} ({len(examples)} example(s))")
    print(f"  suite     : {len(suite)} example(s)\n")

    reference: dict[str, str] = {}
    if args.reference:
        import json

        payload = json.loads(args.reference.read_text(encoding="utf-8"))
        for record in payload.get("results") or payload.get("examples") or []:
            if "response" in record:
                reference[str(record["example_id"])] = record["response"]
        if not reference:
            print("  ! the reference file carries no responses; comparison is skipped\n")

    results: list[dict[str, Any]] = []
    empty = 0
    compared = 0
    matched = 0

    for example in suite:
        output = deployment.generate(list(example.messages))
        text = (output.text or "").strip()
        if not text:
            empty += 1

        status = "—"
        expected = reference.get(str(example.id))
        if expected is not None:
            compared += 1
            if text == expected.strip():
                matched += 1
                status = "match"
            else:
                status = "DIFFERS"

        print(
            f"    {example.id!s:<28} {example.task!s:<28} "
            f"{output.completion_tokens:>4} tok  {status}"
        )
        if args.show_text:
            print(f"        {text[:300]}")

        results.append(
            {
                "example_id": str(example.id),
                "task": str(example.task),
                "completion_tokens": output.completion_tokens,
                "prompt_tokens": output.prompt_tokens,
                "empty": not text,
                "reference_available": expected is not None,
                "matches_reference": (expected is not None and text == expected.strip()),
                "response": text,
            }
        )

    print(f"\n  responses : {len(results)}, empty: {empty}")
    if compared:
        print(f"  reference : {matched}/{compared} reproduce the frozen v0.0.6 output exactly")

    if args.output:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "results": results,
                    "matched": matched,
                    "compared": compared,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  written   : {args.output}")

    if empty:
        print(f"\n✗ {empty} example(s) produced no text. The artifact is not serving correctly.\n")
        return 1

    if compared and matched < compared:
        # Not a failure of the artifact, and explicitly not a reason to retrain.
        print(
            f"\n! {compared - matched} of {compared} responses differ from the frozen "
            "evaluation.\n"
            "  The adapter hashes matched, so the weights are right. Investigate the "
            "serving path:\n"
            "    · a different GPU (fp16 arithmetic is not bit-identical across devices)\n"
            "    · quantization settings differing from evaluation\n"
            "    · tokenizer behaviour (check fix_mistral_regex and the packaged files)\n"
            "    · prompt assembly (the system-prompt merge, deviation D5)\n"
            "  Do not retrain. Record what you find.\n"
        )
        return 2

    print("\n✓ The frozen Hermes artifact loads and serves outside the training notebook.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
