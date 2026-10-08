#!/usr/bin/env python3
"""Check the ZeroGPU Space serves the frozen Hermes v0.0.6 outputs, and measure it.

The same nine examples, judged the same way, as scripts/hermes_smoke.py — but
over the network, through the reference KLEOS client, against the Space. The
benchmark and the frozen reference stay on this machine; the Space only ever
sees the prompts.

What it establishes, in order:

1. The Space is the frozen artifact: /status identity and runtime contract equal
   the serving record (adapter hash, base revision, NF4, float16 compute).
2. Each response equals the frozen v0.0.6 evaluation output, exactly.
3. Where one differs: whether the prompt token count matches (tokenization and
   chat template agree, so the difference is arithmetic) and whether the
   extracted decision still matches.
4. Observed cold and warm timings, throughput, peak VRAM and GPU seconds.

Outputs that differ are a STOP: record them, do not retrain, do not change the
model or its decoding to make them match. See docs/deployment.md.

Quota: every call spends the caller's daily ZeroGPU quota (HF_TOKEN's account).
Nine examples plus one warm repeat is sized to fit a free account's daily
allowance with margin. The run stops at the first quota refusal and keeps what
it measured; it never loops against a spent quota.

Environment: the model's API key (``HERMES_API_KEY`` for Hermes, ``LOGOS_API_KEY``
for Logos: the Space's key) and a Hugging Face token with read access, required
for a private Space and charged for the GPU time: HF_TOKEN if set, otherwise the
token stored by ``hf auth login``. ``--record`` selects the model (Hermes' by
default). For a thinking model the trace is compared as well as the answer.

Usage::

    python scripts/zerogpu_smoke.py --space YOUR_USERNAME/kleos-hermes \\
        --benchmark /path/to/benchmark.jsonl \\
        --reference /path/to/arm2_finetuned.json \\
        --output /path/to/zerogpu_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import REPO_ROOT, add_common_arguments, print_header, run, setup_logging
from kleos_models.data.loaders import load_evaluation_examples
from kleos_models.errors import ConfigError
from kleos_models.serving.client import HermesClient, HermesClientSettings
from kleos_models.serving.manifest import load_expected_identity
from kleos_models.serving.profile import HERMES_PROFILE, ServingProfile, load_profile
from kleos_models.serving.smoke import (
    api_messages,
    judge_response,
    load_reference,
    select_suite,
    summarize_run,
)

DEFAULT_RECORD = REPO_ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"


def _hf_token() -> str | None:
    """The operator's token for a private Space: HF_TOKEN, else the stored login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def check_identity(
    status: dict[str, Any],
    expected: dict[str, Any],
    profile: ServingProfile = HERMES_PROFILE,
) -> list[str]:
    """Differences between what the Space reports and the serving record."""
    model = status.get("model") or {}
    problems = []
    for key in ("adapter_sha256", "base_model", "base_revision"):
        if model.get(key) != expected.get(key):
            problems.append(f"{key}: Space {model.get(key)!r}, record {expected.get(key)!r}")
    if (status.get("runtime") or {}) != expected.get("runtime"):
        problems.append(f"runtime: Space {status.get('runtime')!r}, record {expected['runtime']!r}")
    served = (status.get("limits") or {}).get("max_new_tokens")
    wanted = (expected.get("generation") or {}).get("max_new_tokens")
    if wanted is not None and served != wanted:
        problems.append(f"max_new_tokens: Space serves {served!r}, record evaluated {wanted!r}")
    if bool(status.get("reasoning")) != profile.reasoning:
        problems.append(
            f"reasoning: Space reports {bool(status.get('reasoning'))}, record expects "
            f"{profile.reasoning}"
        )
    return problems


def check_reference(
    payload: Any, expected: dict[str, Any], profile: ServingProfile = HERMES_PROFILE
) -> list[str]:
    """Reasons the reference is not the evaluation the record describes."""
    records = load_reference(payload) if isinstance(payload, dict) else {}
    if not records:
        return ["the reference holds no per-example results"]
    problems = []
    used = (payload.get("generation") or {}).get("max_new_tokens")
    wanted = (expected.get("generation") or {}).get("max_new_tokens")
    if used is not None and wanted is not None and used != wanted:
        problems.append(
            f"max_new_tokens: the reference was generated with {used!r}, the record "
            f"evaluated {wanted!r}"
        )
    if profile.reasoning:
        untraced = sum("reasoning" not in record for record in records.values())
        if untraced:
            problems.append(
                f"{untraced} of {len(records)} reference answers carry no thinking trace"
            )
    return problems


def smoke_outcome(
    summary: dict[str, Any], suite_size: int, stopped: str | None, profile: ServingProfile
) -> str:
    """The run's verdict: pass, incomplete, untraced or differs."""
    if summary["compared"] != suite_size or stopped is not None:
        return "incomplete"
    if profile.reasoning and summary["reasoning_compared"] != summary["compared"]:
        return "untraced"
    traces_match = summary["reasoning_matches"] == summary["reasoning_compared"]
    if summary["exact_matches"] == summary["compared"] and traces_match:
        return "pass"
    return "differs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--space", required=True, help="Space id, owner/name.")
    parser.add_argument("--benchmark", type=Path, required=True, help="benchmark.jsonl.")
    parser.add_argument("--reference", type=Path, required=True, help="Frozen arm2 results JSON.")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="Serving record.")
    parser.add_argument("--output", type=Path, help="Write the full results here (JSON).")
    parser.add_argument("--per-task", type=int, default=1, help="Examples per task family.")
    parser.add_argument("--abstention", type=int, default=2, help="Should-decline examples.")
    parser.add_argument("--only", help="Comma-separated example ids (e.g. to finish a run).")
    parser.add_argument(
        "--warm-repeats", type=int, default=1, help="Re-run the first example this many times."
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds per call.")
    parser.add_argument(
        "--connect-wait",
        type=float,
        default=900.0,
        help="Seconds to wait for a sleeping Space to wake and load.",
    )
    parser.add_argument(
        "--show-text", action="store_true", help="Print responses (they contain prompt content)."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    profile = load_profile(args.record)
    expected = load_expected_identity(args.record)
    reference_payload = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_problems = check_reference(reference_payload, expected, profile)
    if reference_problems:
        print(f"\n✗ {args.reference} is not the {profile.display_name} evaluation to compare with:")
        for problem in reference_problems:
            print(f"    {problem}")
        print("  Nothing was sent to the Space.\n")
        return 1
    key_name = profile.env("API_KEY")
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ConfigError(
            f"{key_name} is not set.",
            suggestions=["Use the Space's key; export it in this shell with `read -s`."],
        )
    client = HermesClient(
        HermesClientSettings(
            enabled=True,
            provider="zerogpu",
            space=args.space,
            api_key=api_key,
            hf_token=_hf_token(),
            timeout=args.timeout,
            connect_wait=args.connect_wait,
            key_header=profile.key_header,
            env_prefix=profile.env_prefix,
            contract_version=profile.contract_version,
        )
    )

    print_header(f"{profile.short_name} on ZeroGPU — frozen-output smoke test")

    # Identity first, before spending any GPU time.
    started = time.perf_counter()
    status = client.status()
    status_wall = time.perf_counter() - started
    if not status.get("ok"):
        print(f"\n✗ The Space is not ready: {status.get('status')} — {status.get('message')}\n")
        return 1
    problems = check_identity(status, expected, profile)
    versions = status.get("versions") or {}
    print(f"  space     : {args.space} (status answered in {status_wall:.1f}s)")
    print(f"  adapter   : {status['model'].get('adapter_sha256', '')[:16]}…")
    print(
        f"  base      : {status['model'].get('base_model')}@{str(status['model'].get('base_revision'))[:12]}…"
    )
    print(f"  runtime   : {status.get('runtime')}")
    print(f"  versions  : {versions}")
    if problems:
        print("\n✗ The Space is not the frozen artifact the record describes:")
        for problem in problems:
            print(f"    {problem}")
        print()
        return 1

    examples = load_evaluation_examples(args.benchmark, strict=True)
    suite = select_suite(examples, per_task=args.per_task, abstention=args.abstention)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        suite = [e for e in suite if str(e.id) in wanted]
    reference = load_reference(reference_payload)
    print(f"\n  suite     : {len(suite)} example(s); reference records: {len(reference)}\n")

    records: list[dict[str, Any]] = []
    stopped: str | None = None
    plan = [(e, False) for e in suite] + [(suite[0], True)] * max(args.warm_repeats, 0)
    for example, repeat in plan:
        messages = api_messages(example)
        if messages is None:
            records.append(
                {
                    "example_id": str(example.id),
                    "task": str(example.task),
                    "status": "unsupported",
                    "ok": False,
                }
            )
            print(f"    {example.id!s:<28} has a turn the API cannot carry (tool role); not sent")
            continue
        call_started = time.perf_counter()
        response = client.generate(messages, request_id=f"smoke-{example.id}")
        wall = time.perf_counter() - call_started
        record = judge_response(example, reference.get(str(example.id)), response)
        record["client_wall_s"] = round(wall, 3)
        record["warm_repeat"] = repeat
        records.append(record)

        if not record["ok"]:
            print(f"    {example.id!s:<28} {record['status']}")
            if record["status"] == "quota_exhausted":
                stopped = "quota_exhausted"
                retry = record.get("retry_after_seconds")
                print(
                    "      quota spent; stopping. "
                    + (f"Hugging Face says retry in {retry}s." if retry else "")
                )
                break
            continue

        exact = record.get("exact", {})
        verdict = "match" if exact.get("match") else f"DIFFERS@{exact.get('first_difference')}"
        trace = record.get("reasoning_exact")
        if trace is not None and exact.get("match") and not trace.get("match"):
            verdict = f"TRACE@{trace.get('first_difference')}"
        tokens = "=" if record.get("prompt_tokens_match") else "≠"
        cold = "cold" if record["diagnostics"].get("cold_start") else "warm"
        print(
            f"    {example.id!s:<28} {verdict:<14} prompt{tokens}{record['prompt_tokens']:<5} "
            f"out {record['completion_tokens']:<4} gen {record['timings'].get('gpu_generate_s')}s "
            f"wall {wall:.1f}s {cold}{' (repeat)' if repeat else ''}"
        )
        if args.show_text:
            print(f"        {record['response'][:300]}")

    first_pass = [r for r in records if not r.get("warm_repeat")]
    summary = summarize_run(first_pass)
    summary["measurements"] = summarize_run(records)["measurements"]
    repeats = [r for r in records if r.get("warm_repeat") and r.get("ok")]
    first = next((r for r in first_pass if r.get("ok")), None)
    summary["warm_repeat_identical"] = (
        all(r["response"] == first["response"] for r in repeats) if repeats and first else None
    )

    m = summary["measurements"]
    version = "v0.0.6" if not profile.reasoning else profile.display_name
    print(
        f"\n  exact     : {summary['exact_matches']}/{summary['compared']} reproduce the frozen "
        f"{version} output"
    )
    if summary["reasoning_compared"]:
        print(
            f"  trace     : {summary['reasoning_matches']}/{summary['reasoning_compared']} "
            "reproduce the frozen thinking trace"
        )
    print(
        f"  prompt tok: {summary['prompt_token_matches']}/{summary['compared']} equal the evaluation's"
    )
    print(f"  decisions : {summary['decision_matches']}/{summary['compared']} agree")
    print(f"  statuses  : {summary['statuses']}")
    print(f"\n  {m['label']}:")
    print(
        f"    device       : {m['device']} (sm {m['compute_capability']}, CUDA {m['cuda_runtime']})"
    )
    print(f"    peak VRAM    : {m['peak_vram_gib']} GiB")
    print(f"    cold         : {m['cold']}")
    print(f"    warm         : {m['warm']}")
    print(f"    GPU seconds  : {m['gpu_generate_seconds_total']}–{m['gpu_call_seconds_total']} s")
    if summary["warm_repeat_identical"] is not None:
        print(f"    repeat same  : {summary['warm_repeat_identical']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "space": args.space,
                    "status": status,
                    "summary": summary,
                    "stopped": stopped,
                    "results": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  written   : {args.output}")

    if summary["empty"] and not profile.reasoning:
        print("\n✗ Some responses were empty. The Space is not serving correctly.\n")
        return 1
    if summary["empty"]:
        # Unclosed thinking leaves the answer empty: model behaviour, graded as usual.
        print(f"\n  ! {summary['empty']} answer(s) empty: the thinking never closed.")
    outcome = smoke_outcome(summary, len(suite), stopped, profile)
    if outcome == "pass":
        print(
            f"\n✓ The ZeroGPU Space reproduces the frozen {profile.display_name} outputs exactly.\n"
        )
        return 0
    if outcome == "incomplete":
        print("\n! Incomplete run. Finish it later with --only <remaining ids>.\n")
        return 2
    if outcome == "untraced":
        print(
            f"\n! Traces were compared for {summary['reasoning_compared']} of "
            f"{summary['compared']} answers, so this run does not verify the Space.\n"
        )
        return 2
    print(
        "\n! STOP: responses differ from the frozen evaluation. Do not retrain and do not\n"
        "  change the model, tokenizer or decoding to make them match. Record the\n"
        "  differences above (prompt-token equality localises them) and investigate.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(run(main))
