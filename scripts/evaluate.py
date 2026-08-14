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
"""

from __future__ import annotations

import argparse
import sys
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
from kleos_models.evaluation.runner import run_evaluation
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
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header(f"KLEOS evaluation — {args.arm}")
    config = load_config(args.config, overrides=args.overrides, dataset_path=args.dataset)

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
    print(f"  benchmark : {benchmark_path} ({len(examples)} example(s))")
    print(f"  model     : {config.model.base_model}")
    print(f"  arm       : {args.arm}")

    adapter_path = args.adapter or config.evaluation.adapter_path
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

    result = run_evaluation(
        backend,
        examples,
        config.evaluation,
        arm=args.arm,
        orchestration=OrchestrationConfig.for_arm(
            args.arm, prompt_path=config.evaluation.orchestration_prompt_path
        ),
        benchmark_path=str(benchmark_path),
    )

    print("\n" + result.render())

    output = args.output or Path(config.training.output_dir) / f"eval_{args.arm}.json"
    result.save(output, include_responses=not args.no_responses)

    manifest = build_manifest(
        config,
        kind="evaluation",
        dataset_version=str(benchmark_path),
    )
    manifest.metrics = {"overall": result.overall.mean, **result.metrics}
    manifest.mark_completed()
    manifest.add_artifact("results", str(output))
    manifest.save(output.parent, filename=f"manifest_eval_{args.arm}.json")

    print(f"✓ Results written to {output}")
    print(f"\nNext: python scripts/compare.py --base <base.json> --finetuned {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
