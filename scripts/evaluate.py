#!/usr/bin/env python3
"""Evaluate a research arm on a KLEOS benchmark (spec §19-§23).

Conditions are held identical across arms — same benchmark, decoding, graders and
seeds — so a difference in score is attributable to the arm and not the harness.

Usage::

    python scripts/evaluate.py --config configs/training/qlora_small.yaml \
                               --arm arm0_base --output outputs/base_results.json

    python scripts/evaluate.py --config configs/training/qlora_small.yaml \
                               --arm arm2_finetuned \
                               --adapter outputs/<experiment-id>/adapter \
                               --output outputs/finetuned_results.json

    # Exercise the harness with no model and no GPU:
    python scripts/evaluate.py --config configs/training/qlora_small.yaml --echo

    # After a disconnect, re-run the same command with --resume:
    python scripts/evaluate.py ... --output outputs/finetuned_results.json --resume

Every finished generation goes to ``<output>.partial.jsonl`` as it happens, so an
interrupted arm continues where it stopped. The partial file records the run's
identity (config, arm, benchmark bytes, adapter weights, code, libraries, GPU);
a resume under a different identity is refused. A finished result is never
overwritten without --overwrite.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import (
    add_common_arguments,
    add_config_arguments,
    print_header,
    run,
    setup_logging,
)
from kleos_models.config import load_config
from kleos_models.constants import RESEARCH_ARMS
from kleos_models.data.loaders import load_evaluation_examples
from kleos_models.evaluation.resume import (
    PartialWriter,
    ResumingBackend,
    adapter_identity,
    build_identity,
    identity_sha256,
    load_partial,
    partial_path,
    sha256_file,
    start_partial,
)
from kleos_models.evaluation.runner import load_evaluation_result, run_evaluation
from kleos_models.experiments.environment import set_global_seed
from kleos_models.experiments.manifest import build_manifest
from kleos_models.inference.generate import OrchestrationConfig
from kleos_models.logging_utils import get_logger

logger = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--arm", choices=list(RESEARCH_ARMS), default="arm0_base", help="Research arm to evaluate."
    )
    parser.add_argument(
        "--adapter", type=Path, help="LoRA adapter directory (required for fine-tuned arms)."
    )
    parser.add_argument("--benchmark", type=Path, help="Benchmark JSONL, overriding the config.")
    parser.add_argument("--output", type=Path, help="Where to write the results JSON.")
    parser.add_argument("--max-examples", type=int, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--no-responses", action="store_true", help="Omit raw model text from the results file."
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Use the stub EchoBackend to exercise the harness without a model.",
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted evaluation from <output>.partial.jsonl. Refused "
            "if anything that could change a generation differs. Harmless on a "
            "first run; a finished result with the same identity is left as it is."
        ),
    )
    restart.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing result or partial evaluation at --output and start again.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header(f"KLEOS evaluation — {args.arm}")
    config = load_config(
        args.config,
        overrides=args.overrides,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )

    benchmark_path = args.benchmark or config.evaluation.benchmark_path
    if benchmark_path is None:
        print(
            "\n✗ No benchmark configured. Pass --benchmark or set evaluation.benchmark_path.\n",
            file=sys.stderr,
        )
        return 1

    if args.max_examples:
        config.evaluation.max_examples = args.max_examples

    set_global_seed(config.seed)

    examples = load_evaluation_examples(benchmark_path, strict=True)
    benchmark_sha256 = sha256_file(Path(benchmark_path))
    print(f"  benchmark : {benchmark_path} ({len(examples)} example(s))")
    print(f"  sha256    : {benchmark_sha256}")
    print(f"  model     : {config.model.base_model}")
    print(f"  arm       : {args.arm}")

    adapter_path = args.adapter or config.evaluation.adapter_path
    output = args.output or Path(config.training.output_dir) / f"eval_{args.arm}.json"
    partial = partial_path(output)

    # --- identity: everything that could change a generation ----------------
    from kleos_models.compat import library_versions
    from kleos_models.experiments.environment import capture_git_info
    from kleos_models.models.feasibility import probe_gpu

    orchestration = OrchestrationConfig.for_arm(
        args.arm, prompt_path=config.evaluation.orchestration_prompt_path
    )
    selected_ids = [e.id for e in examples]
    if config.evaluation.max_examples is not None:
        selected_ids = selected_ids[: config.evaluation.max_examples]
    git = capture_git_info()
    gpu = probe_gpu()
    identity = build_identity(
        arm=args.arm,
        seeds=config.evaluation.seeds,
        example_ids=selected_ids,
        benchmark_sha256=benchmark_sha256,
        generation=config.evaluation.generation.model_dump(mode="json"),
        orchestration_prompt=orchestration.system_prompt,
        model={**config.model.model_dump(mode="json"), "echo_backend": args.echo},
        adapter=adapter_identity(adapter_path) if not args.echo else None,
        git_commit=f"{git.commit}{' (dirty)' if git.dirty else ''}",
        libraries=library_versions(),
        compute_capability=gpu.capability_string,
    )
    digest = identity_sha256(identity)

    # --- never overwrite a finished result by accident ------------------------
    if args.overwrite:
        partial.unlink(missing_ok=True)
    elif output.exists():
        finished = load_evaluation_result(output)
        recorded = (finished.get("resume") or {}).get("identity_sha256")
        if args.resume and recorded == digest:
            print(f"\n✓ {output} is already complete for this identity; nothing to do.\n")
            return 0
        reason = (
            "it was produced under a different identity"
            if args.resume and recorded
            else "it records no identity to check against"
            if args.resume
            else "a finished result is never replaced silently"
        )
        print(
            f"\n✗ {output} already exists and {reason}.\n"
            "  Pass --overwrite to replace it, or choose another --output.\n",
            file=sys.stderr,
        )
        return 1
    elif partial.exists() and not args.resume:
        print(
            f"\n✗ An interrupted evaluation exists at {partial}.\n"
            "  Pass --resume to continue it, or --overwrite to start again.\n",
            file=sys.stderr,
        )
        return 1

    recorded_generations = {}
    dropped_torn_line = False
    if partial.exists():
        state = load_partial(partial, identity)
        recorded_generations = state.generations
        dropped_torn_line = state.dropped_torn_line
        print(f"  resume    : {len(recorded_generations)} generation(s) recorded in {partial.name}")
    else:
        start_partial(partial, identity)

    load_started = time.perf_counter()
    if args.echo:
        from kleos_models.inference.backends import EchoBackend

        print("\n  ! Using EchoBackend: canned responses, no model. Results from this")
        print("    run exercise the harness and mean nothing about any model.\n")
        backend = EchoBackend(default="alpha-task")
    else:
        from kleos_models.inference.backends import build_backend

        backend = build_backend(
            args.arm,
            config.model,
            adapter_path=adapter_path,
            reasoning_mode=config.model.reasoning.default_mode,
        )
    load_seconds = time.perf_counter() - load_started

    writer = PartialWriter(partial)
    resuming = ResumingBackend(backend, recorded=recorded_generations, writer=writer)
    run_started = time.perf_counter()
    try:
        result = run_evaluation(
            resuming,
            examples,
            config.evaluation,
            arm=args.arm,
            orchestration=orchestration,
            benchmark_path=str(benchmark_path),
            benchmark_sha256=benchmark_sha256,
        )
    finally:
        writer.close()

    result.resume = {
        "identity_sha256": digest,
        "replayed_generations": resuming.replayed,
        "new_generations": resuming.generated,
        "dropped_torn_line": dropped_torn_line,
    }
    result.resource_usage = {
        "model_load_seconds": round(load_seconds, 1),
        "evaluation_wall_seconds": round(time.perf_counter() - run_started, 1),
        "gpu": gpu.name,
        "compute_capability": gpu.capability_string,
        "peak_vram_gb": _peak_vram_gb(),
    }

    print("\n" + result.render())

    result.save(output, include_responses=not args.no_responses)
    # The finished result holds every generation; the partial has done its job.
    partial.unlink(missing_ok=True)

    manifest = build_manifest(
        config,
        kind="evaluation",
        dataset_version=str(benchmark_path),
        dataset_hash=benchmark_sha256,
    )
    manifest.metrics = {"overall": result.overall.mean, **result.metrics}
    manifest.mark_completed()
    manifest.add_artifact("results", str(output))
    manifest.save(output.parent, filename=f"manifest_eval_{args.arm}.json")

    print(f"✓ Results written to {output}")
    print(f"\nNext: python scripts/compare.py --base <base.json> --finetuned {output}\n")
    return 0


def _peak_vram_gb() -> float | None:
    """Peak allocated VRAM of this process, or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


if __name__ == "__main__":
    raise SystemExit(run(main))
