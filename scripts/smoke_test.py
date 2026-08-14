#!/usr/bin/env python3
"""Pre-flight smoke test (spec §29).

Run this before spending a GPU session. It checks, in increasing order of cost:

1. Python version and package imports
2. Configuration loading and validation
3. Data schemas and the bundled fixtures
4. The data pipeline: validation, splitting, leakage, coverage
5. The evaluation harness, end to end, with a stub backend
6. Optional deps: torch, transformers, peft, bitsandbytes
7. GPU detection and feasibility
8. Tokenizer loading and formatting (needs --model and network)
9. Output directory creation

Steps that cannot run are reported as SKIP with the reason, never as a pass.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --model configs/models/qwen3_8b.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, run, setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_PYTHON = (3, 11)


@dataclass
class CheckResult:
    """Outcome of one smoke check."""

    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""
    error: str = ""


@dataclass
class SmokeReport:
    """All smoke-test results."""

    results: list[CheckResult] = field(default_factory=list)

    def record(self, name: str, status: str, detail: str = "", error: str = "") -> None:
        icon = {"pass": "✓", "fail": "✗", "skip": "·"}[status]
        line = f"  {icon} {name:<44} {detail}"
        print(line, file=sys.stderr if status == "fail" else sys.stdout)
        if error:
            for entry in error.strip().splitlines()[:6]:
                print(f"      {entry}", file=sys.stderr)
        self.results.append(CheckResult(name, status, detail, error))

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def skipped(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "skip"]

    @property
    def passed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "pass"]


def check(report: SmokeReport, name: str):
    """Decorator running a check and recording pass/fail without aborting."""

    def wrapper(fn):
        try:
            detail = fn()
            report.record(name, "pass", detail or "")
        except _Skip as skip:
            report.record(name, "skip", str(skip))
        except Exception as exc:
            report.record(name, "fail", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        return fn

    return wrapper


class _Skip(Exception):
    """Raised by a check that cannot run in this environment."""


def run_checks(model_config_path: Path | None = None) -> SmokeReport:
    """Run every smoke check."""
    report = SmokeReport()

    print("\n── environment " + "─" * 48)

    @check(report, "Python version")
    def _python() -> str:
        if sys.version_info < MIN_PYTHON:
            raise RuntimeError(
                f"Python {'.'.join(map(str, MIN_PYTHON))}+ required, "
                f"found {sys.version_info.major}.{sys.version_info.minor}"
            )
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @check(report, "Core imports (no torch required)")
    def _imports() -> str:
        import kleos_models
        import kleos_models.data
        import kleos_models.evaluation
        import kleos_models.experiments

        return f"kleos_models {kleos_models.__version__}"

    @check(report, "Import isolation (light layer stays torch-free)")
    def _isolation() -> str:
        import kleos_models.data
        import kleos_models.evaluation  # noqa: F401

        leaked = [m for m in ("torch", "transformers") if m in sys.modules]
        if leaked and not _torch_installed():
            raise RuntimeError(f"light layer imported {leaked}")
        return "clean" if not leaked else "torch present (installed)"

    print("\n── configuration " + "─" * 46)

    @check(report, "Load configs/training/qlora_small.yaml")
    def _config() -> str:
        from kleos_models.config import load_config

        config = load_config(REPO_ROOT / "configs" / "training" / "qlora_small.yaml")
        return f"{config.name}, hash {config.short_hash()}"

    @check(report, "Load every model config")
    def _model_configs() -> str:
        from kleos_models.config import load_model_config
        from kleos_models.models.adapters import get_adapter

        paths = sorted((REPO_ROOT / "configs" / "models").glob("*.yaml"))
        if not paths:
            raise RuntimeError("no model configs found")
        for path in paths:
            config = load_model_config(path)
            adapter = get_adapter(config)
            adapter.resolve_reasoning_mode()
        return f"{len(paths)} config(s), all adapters resolved"

    @check(report, "Invalid config is rejected")
    def _config_rejects() -> str:
        from kleos_models.config import TrainingConfig

        try:
            TrainingConfig(learning_rate=-1.0)
        except Exception:
            return "negative learning rate rejected"
        raise RuntimeError("an invalid learning rate was accepted")

    print("\n── data pipeline " + "─" * 46)

    @check(report, "JSON schemas match the pydantic models")
    def _schemas() -> str:
        import json

        from kleos_models.data.schemas import DatasetManifest, EvaluationExample, TrainingExample

        pairs = {
            "training_example.schema.json": TrainingExample,
            "evaluation_example.schema.json": EvaluationExample,
            "dataset_manifest.schema.json": DatasetManifest,
        }
        for filename, model in pairs.items():
            path = REPO_ROOT / "data" / "schema" / filename
            if not path.exists():
                raise RuntimeError(f"{filename} missing; run prepare_dataset.py --emit-schemas")
            stored = json.loads(path.read_text(encoding="utf-8"))
            current = model.model_json_schema()
            if stored.get("properties") != current.get("properties"):
                raise RuntimeError(f"{filename} is out of date")
        return f"{len(pairs)} schema(s) current"

    @check(report, "Load and validate the bundled fixtures")
    def _fixtures() -> str:
        from kleos_models.data.loaders import load_examples
        from kleos_models.data.schemas import TrainingExample
        from kleos_models.data.validation import validate_examples

        path = REPO_ROOT / "data" / "examples" / "synthetic_train.jsonl"
        examples, _ = load_examples(path, model=TrainingExample, strict=True)
        result = validate_examples(examples, split_name="fixtures")
        if not result.ok:
            raise RuntimeError(f"{len(result.errors)} validation error(s)")
        return f"{len(examples)} example(s) valid"

    @check(report, "Invalid example is rejected")
    def _rejects() -> str:
        from kleos_models.data.schemas import TrainingExample

        try:
            TrainingExample.model_validate(
                {
                    "id": "bad",
                    "task": "notification_prioritization",
                    "messages": [{"role": "user", "content": "hi"}],
                    "variation_axes": {"domain": "career"},
                }
            )
        except Exception:
            return "example with no assistant turn rejected"
        raise RuntimeError("a malformed example was accepted")

    @check(report, "Splitting is deterministic and non-overlapping")
    def _splitting() -> str:
        from kleos_models.config import SplitConfig
        from kleos_models.data.loaders import load_examples
        from kleos_models.data.schemas import TrainingExample
        from kleos_models.data.splitting import split_examples

        examples, _ = load_examples(
            REPO_ROOT / "data" / "examples" / "synthetic_train.jsonl",
            model=TrainingExample,
            strict=True,
        )
        config = SplitConfig(
            strategy="group",
            seed=42,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        first = split_examples(examples, config)
        second = split_examples(examples, config)
        if [e.id for e in first.train] != [e.id for e in second.train]:
            raise RuntimeError("split is not deterministic")
        return f"{first.counts}, verified"

    @check(report, "Leakage detection finds a planted duplicate")
    def _leakage() -> str:
        from kleos_models.data.leakage import check_leakage
        from kleos_models.data.loaders import load_examples
        from kleos_models.data.schemas import TrainingExample

        examples, _ = load_examples(
            REPO_ROOT / "data" / "examples" / "synthetic_train.jsonl",
            model=TrainingExample,
            strict=True,
        )
        planted = examples[0].model_copy(update={"id": "planted-duplicate"})
        result = check_leakage({"train": examples, "test": [planted]})
        if not result.cross_split_findings:
            raise RuntimeError("a planted cross-split duplicate was not detected")
        return f"{len(result.cross_split_findings)} cross-split finding(s) detected"

    @check(report, "Coverage report")
    def _coverage() -> str:
        from kleos_models.data.coverage import build_coverage_report
        from kleos_models.data.loaders import load_examples
        from kleos_models.data.schemas import TrainingExample

        examples, _ = load_examples(
            REPO_ROOT / "data" / "examples" / "synthetic_train.jsonl",
            model=TrainingExample,
            strict=True,
        )
        coverage = build_coverage_report(examples)
        return f"{coverage.observed_cells} situation type(s) observed"

    print("\n── evaluation harness " + "─" * 41)

    @check(report, "Metrics")
    def _metrics() -> str:
        from kleos_models.evaluation.metrics import exact_match, kendall_tau, ndcg

        assert exact_match("The Answer.", "answer") == 1.0
        assert ndcg(["a", "b", "c"], ["a", "b", "c"]) == 1.0
        assert kendall_tau(["a", "b"], ["b", "a"]) < 0.5
        return "exact match, nDCG, Kendall tau"

    @check(report, "Graders")
    def _graders() -> str:
        from kleos_models.evaluation.graders import GRADER_REGISTRY, get_grader

        grader = get_grader("ranking")
        result = grader.grade("1. alpha\n2. beta", {"ranking": ["alpha", "beta"]})
        if result.score <= 0:
            raise RuntimeError("ranking grader scored a correct answer at zero")
        return f"{len(GRADER_REGISTRY)} grader(s) registered"

    @check(report, "End-to-end evaluation with the stub backend")
    def _evaluation() -> str:
        from kleos_models.config import EvaluationConfig
        from kleos_models.data.loaders import load_evaluation_examples
        from kleos_models.evaluation.runner import run_evaluation
        from kleos_models.inference.backends import EchoBackend

        examples = load_evaluation_examples(
            REPO_ROOT / "data" / "examples" / "synthetic_eval.jsonl", strict=True
        )
        result = run_evaluation(
            EchoBackend(default="alpha-task"),
            examples,
            EvaluationConfig(arms=["arm0_base"]),
            arm="arm0_base",
            benchmark_path="smoke",
        )
        if result.count != len(examples):
            raise RuntimeError("not every example was evaluated")
        return f"{result.count} example(s) scored, OOD measurable={bool(result.ood and result.ood.measurable)}"

    @check(report, "Comparison and report generation")
    def _comparison() -> str:
        from kleos_models.evaluation.reports import compare_results

        base = {
            "arm": "arm0_base",
            "benchmark_path": "b",
            "generation": {},
            "model": {"base_model": "m"},
            "results": [{"example_id": "1", "task": "t", "score": 0.5}],
        }
        finetuned = {
            "arm": "arm2_finetuned",
            "benchmark_path": "b",
            "generation": {},
            "model": {"base_model": "m", "adapter_path": "a"},
            "results": [{"example_id": "1", "task": "t", "score": 0.9}],
        }
        comparison = compare_results(base, finetuned)
        if not comparison.per_task:
            raise RuntimeError("comparison produced no per-task rows")
        return "per-task deltas computed"

    print("\n── experiment infrastructure " + "─" * 34)

    @check(report, "Manifest round-trip")
    def _manifest() -> str:
        from kleos_models.config import load_config
        from kleos_models.experiments.manifest import ExperimentManifest, build_manifest

        config = load_config(REPO_ROOT / "configs" / "training" / "debug.yaml")
        manifest = build_manifest(config, kind="training")
        with tempfile.TemporaryDirectory() as tmp:
            path = manifest.save(tmp)
            restored = ExperimentManifest.load(path)
        if restored.config_hash != manifest.config_hash:
            raise RuntimeError("manifest did not round-trip")
        return f"git commit: {manifest.git.get('commit', 'unavailable')[:12]}"

    @check(report, "Output directory is writable")
    def _outputs() -> str:
        target = REPO_ROOT / "outputs"
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".smoke_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return str(target.relative_to(REPO_ROOT))

    print("\n── optional training dependencies " + "─" * 29)

    @check(report, "torch")
    def _torch() -> str:
        try:
            import torch
        except ImportError as exc:
            raise _Skip('not installed — pip install -e ".[train]"') from exc
        return f"{torch.__version__} (cuda={torch.cuda.is_available()})"

    @check(report, "transformers")
    def _transformers() -> str:
        try:
            import transformers
        except ImportError as exc:
            raise _Skip('not installed — pip install -e ".[train]"') from exc
        from kleos_models.compat import check_transformers_version

        warning = check_transformers_version()
        return f"{transformers.__version__}" + (f" — {warning}" if warning else "")

    @check(report, "peft")
    def _peft() -> str:
        try:
            import peft
        except ImportError as exc:
            raise _Skip('not installed — pip install -e ".[train]"') from exc
        return peft.__version__

    @check(report, "bitsandbytes (4-bit quantization)")
    def _bnb() -> str:
        from kleos_models.models.quantization import quantization_support

        support = quantization_support()
        if not support.bitsandbytes_installed:
            raise _Skip('not installed — pip install -e ".[train,quant]" (CUDA only)')
        if not support.cuda_available:
            raise _Skip(f"installed but no CUDA device ({support.reason})")
        return f"{support.bitsandbytes_version}, bf16={support.bf16_supported}"

    @check(report, "GPU detection and feasibility")
    def _gpu() -> str:
        from kleos_models.config import load_config
        from kleos_models.models.adapters import get_adapter
        from kleos_models.models.feasibility import assess_feasibility, probe_gpu

        gpu = probe_gpu()
        config = load_config(REPO_ROOT / "configs" / "training" / "qlora_small.yaml")
        adapter = get_adapter(config.model)
        assessment = assess_feasibility(
            config.model,
            config.training,
            gpu=gpu,
            target_modules=adapter.default_target_modules,
        )
        if not gpu.available:
            raise _Skip(f"no CUDA GPU ({gpu.source}); training must run on Colab or a CUDA host")
        return f"{gpu.name}, {assessment.tier.value}"

    if model_config_path:
        print("\n── model access " + "─" * 47)

        @check(report, f"Tokenizer + formatting ({model_config_path.name})")
        def _tokenizer() -> str:
            if not _torch_installed():
                raise _Skip('transformers not installed — pip install -e ".[train]"')
            from kleos_models.config import load_model_config
            from kleos_models.data.formatting import ConversationFormatter
            from kleos_models.data.loaders import load_examples
            from kleos_models.data.schemas import TrainingExample
            from kleos_models.models.adapters import get_adapter
            from kleos_models.models.loading import load_tokenizer

            config = load_model_config(model_config_path)
            adapter = get_adapter(config)
            tokenizer = load_tokenizer(config, adapter)
            formatter = ConversationFormatter(
                tokenizer,
                max_seq_length=config.max_seq_length,
                template_kwargs=adapter.chat_template_kwargs(adapter.resolve_reasoning_mode()),
            )
            examples, _ = load_examples(
                REPO_ROOT / "data" / "examples" / "synthetic_train.jsonl",
                model=TrainingExample,
                strict=True,
            )
            formatted = formatter.format_example(examples[0])
            if formatted is None or formatted.target_token_count == 0:
                raise RuntimeError("formatting produced no supervised tokens")
            return f"{formatted.target_token_count}/{len(formatted.input_ids)} tokens supervised"

    return report


def _torch_installed() -> bool:
    from kleos_models.compat import is_available

    return is_available("transformers")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model config to check tokenizer loading against (downloads the tokenizer).",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args if args.verbose else argparse.Namespace(verbose=False, quiet=True))

    print()
    print("=" * 72)
    print("KLEOS smoke test")
    print("=" * 72)

    report = run_checks(args.model)

    print()
    print("=" * 72)
    print(
        f"  {len(report.passed)} passed, {len(report.failures)} failed, "
        f"{len(report.skipped)} skipped"
    )
    print("=" * 72)

    if report.skipped:
        print("\nSkipped (not failures — these need optional dependencies or hardware):")
        for result in report.skipped:
            print(f"  · {result.name}: {result.detail}")

    if report.failures:
        print("\n✗ Smoke test FAILED:", file=sys.stderr)
        for result in report.failures:
            print(f"    {result.name}: {result.detail}", file=sys.stderr)
        print("\n  See docs/troubleshooting.md.\n", file=sys.stderr)
        return 1

    print("\n✓ Smoke test passed.")
    if not _torch_installed():
        print("\n  Note: torch/transformers are not installed, so model loading and")
        print('  training were not exercised. Install with: pip install -e ".[train]"')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
