"""Hugging Face publishing helpers and model-card generation (spec §18, §35).

Two responsibilities, both about restraint:

1. **Decide what may leave the machine.** Uploading is an allowlist, not a
   blocklist. Only named artifacts are eligible, and each one is still scanned
   for private-looking content before it is offered. Raw training data, `.env`
   and checkpoints are never eligible.

2. **Write a model card that does not overclaim.** The card states what the model
   is better than *only when an evaluation result shows it*. With no evaluation
   attached, it says so plainly rather than implying a benefit.

This module imports no torch.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_models.data.validation import scan_sensitive_content
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Filenames eligible for upload. An allowlist: anything not named here is
#: refused, so a new artifact type cannot leak by default.
ALLOWED_UPLOAD_NAMES: frozenset[str] = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
        "config.yaml",
        "manifest.json",
        "metrics.json",
    }
)

#: Tokenizer artifacts are deliberately NOT published with an adapter.
#:
#: A PEFT adapter is not self-contained — inference loads the base model, so the
#: base repository is always a dependency and its tokenizer is always available.
#: KLEOS never adds tokens or resizes embeddings (LoRA excludes ``embed_tokens``
#: and ``lm_head``), so the tokenizer is byte-identical to the base model's and
#: duplicating it buys nothing.
#:
#: The rule is all-or-nothing on purpose. Publishing ``tokenizer_config.json``
#: without ``tokenizer.json`` yields a repository that looks like it carries a
#: tokenizer but cannot build one — no vocabulary source — which is worse than
#: shipping none. That partial bundle is exactly what the 5MB scan cap used to
#: produce, since the vocabulary file is ~17MB and the config files are tiny.
#:
#: This is a packaging decision, not a security refusal, which is why these names
#: live here rather than in FORBIDDEN_PATTERNS.
TOKENIZER_ARTIFACTS: frozenset[str] = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
)

#: Files that must never be uploaded, whatever else happens.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\.env"),
    re.compile(r".*\.jsonl$"),  # datasets
    re.compile(r"^checkpoint-"),
    re.compile(r".*\.log$"),
    re.compile(r"^events\.jsonl$"),
    re.compile(r".*\.key$|.*\.pem$"),
)

#: Files scanned for sensitive content before upload.
_SCANNED_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".md", ".txt"})

#: Maximum size scanned inline. Larger text files are unusual here and are
#: refused rather than skipped.
_MAX_SCAN_BYTES = 5 * 1024 * 1024


def _is_forbidden(name: str) -> str | None:
    """Return a rejection reason if the filename is forbidden."""
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.match(name):
            return f"matches the forbidden pattern {pattern.pattern!r}"
    return None


def _scan_file(path: Path) -> str | None:
    """Return a rejection reason if the file looks like it contains secrets."""
    if path.suffix.lower() not in _SCANNED_SUFFIXES:
        return None
    try:
        size = path.stat().st_size
    except OSError:  # pragma: no cover
        return "could not stat the file"
    if size > _MAX_SCAN_BYTES:
        return f"text file is unexpectedly large ({size / 1e6:.1f} MB) and was not scanned"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return "could not read the file"
    matches = scan_sensitive_content(content)
    if matches:
        return f"matched sensitive pattern(s): {', '.join(matches)}"
    return None


def collect_upload_files(
    adapter_dir: Path, run_dir: Path | None = None
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Decide which files may be uploaded.

    Args:
        adapter_dir: Directory holding the adapter weights.
        run_dir: The run directory, whose config/manifest/metrics are also eligible.

    Returns:
        ``(allowed, rejected)`` where each rejection carries a reason.
    """
    allowed: list[Path] = []
    rejected: list[tuple[Path, str]] = []

    candidates: list[Path] = sorted(p for p in adapter_dir.iterdir() if p.is_file())
    if run_dir is not None and run_dir != adapter_dir:
        for name in ("config.yaml", "manifest.json", "metrics.json"):
            path = run_dir / name
            if path.exists():
                candidates.append(path)
        # The run's tokenizer/ directory is deliberately not enumerated. See
        # TOKENIZER_ARTIFACTS: the tokenizer comes from the pinned base model,
        # and a partial copy here would be worse than none.

    for path in candidates:
        forbidden = _is_forbidden(path.name)
        if forbidden:
            rejected.append((path, forbidden))
            continue
        if path.name in TOKENIZER_ARTIFACTS:
            # Rejected with its own reason rather than a bare allowlist miss, so
            # the log says "by design" instead of looking like a failure.
            rejected.append(
                (
                    path,
                    "tokenizer artifact — not duplicated; the tokenizer is loaded "
                    "from the pinned base model (see docs/publishing.md)",
                )
            )
            continue
        if path.name not in ALLOWED_UPLOAD_NAMES:
            rejected.append((path, "not in the upload allowlist"))
            continue
        sensitive = _scan_file(path)
        if sensitive:
            rejected.append((path, sensitive))
            logger.warning("Refusing to upload %s: %s", path.name, sensitive)
            continue
        allowed.append(path)

    return allowed, rejected


#: A 40-character hex string is a resolved git/Hub commit. Anything else
#: (``main``, a tag, a branch) is a moving pointer.
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


def is_pinned_revision(revision: str | None) -> bool:
    """Whether a revision identifies one immutable commit."""
    return bool(revision) and bool(_PINNED_REVISION.match(str(revision)))


def _revision_note(revision: str) -> str:
    """Say plainly whether the base weights this adapter needs are pinned.

    A LoRA adapter is deltas against specific base weights. If the recorded
    revision is a moving pointer, a reader cannot know which weights it was
    trained against, and the card must not imply otherwise.
    """
    if is_pinned_revision(revision):
        return (
            f"> The base revision is pinned to `{revision}`. Loading any other "
            "revision pairs this adapter with weights it was not trained "
            "against; the failure is silent."
        )
    return (
        f"> **The base revision is `{revision}`, which is a moving pointer, not a "
        "pinned commit.** The exact base weights this adapter was trained "
        "against are therefore not recoverable from this release, and results "
        "may not reproduce if the base repository has since changed."
    )


def _describe_results(results: dict[str, Any] | None) -> str:
    """Render evaluation results, or say clearly that there are none."""
    if not results:
        return (
            "**No evaluation results were attached to this release.**\n\n"
            "Nothing is therefore claimed about this adapter's performance. Do not\n"
            "assume it is better than the base model — that has not been measured\n"
            "here.\n"
        )

    lines = [
        f"Benchmark: `{results.get('benchmark_path', 'unknown')}`  ",
        f"Arm: `{results.get('arm', 'unknown')}`  ",
        f"Examples: {results.get('example_count', 'unknown')}",
        "",
    ]
    overall = results.get("overall") or {}
    if overall:
        lines.extend(
            [
                f"Overall score: **{overall.get('mean', 0):.4f}** "
                f"(95% CI {overall.get('ci95_low', 0):.4f}–{overall.get('ci95_high', 0):.4f}, "
                f"n={overall.get('count', 0)})",
                "",
            ]
        )
    per_task = results.get("per_task") or {}
    if per_task:
        lines.extend(["| Task | Score | n |", "| --- | ---: | ---: |"])
        for task, scores in sorted(per_task.items()):
            lines.append(f"| `{task}` | {scores.get('mean', 0):.4f} | {scores.get('count', 0)} |")
        lines.append("")

    ood = results.get("ood")
    if ood and ood.get("measurable"):
        lines.extend(
            [
                "### Out-of-distribution",
                "",
                f"- in-distribution: {ood.get('in_distribution_score', 0):.4f}",
                f"- out-of-distribution: {ood.get('ood_score', 0):.4f}",
                f"- generalization gap: {ood.get('generalization_gap', 0):+.4f}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Out-of-distribution",
                "",
                "Not measured. **No generalization claim is made.**",
                "",
            ]
        )
    return "\n".join(lines)


def _describe_comparison(comparison: dict[str, Any] | None) -> str:
    """Render a base-vs-fine-tuned comparison without overclaiming."""
    if not comparison:
        return (
            "No base-vs-fine-tuned comparison was attached, so **this adapter is not\n"
            "claimed to improve on the base model**.\n"
        )

    lines = [
        "| Task | Base | Fine-tuned | Δ | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task in comparison.get("per_task", []):
        lines.append(
            f"| `{task['task']}` | {task['base']:.4f} | {task['finetuned']:.4f} | "
            f"{task['absolute_delta']:+.4f} | {task['verdict']} |"
        )
    lines.append("")

    regressed = comparison.get("regressed_task_count", 0)
    improved = comparison.get("improved_task_count", 0)
    if regressed:
        lines.extend(
            [
                f"**{regressed} task(s) regressed** relative to the base model. Any",
                "improvement above is a trade-off, not a uniform gain.",
                "",
            ]
        )
    elif not improved:
        lines.extend(["No task improved beyond the noise threshold.", ""])

    capability = comparison.get("capability_delta") or {}
    if capability.get("regressed"):
        lines.extend(
            [
                f"**General capability regressed** by "
                f"{abs(capability.get('relative_delta', 0)):.1%} on the fixed regression",
                "suite. Weigh this against any task-specific gain.",
                "",
            ]
        )
    return "\n".join(lines)


def build_model_card(
    *,
    repo_id: str,
    manifest: ExperimentManifest | None = None,
    results: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Generate a model card (spec §35).

    Includes model name, base model, method, dataset description and privacy
    statement, tasks, hyperparameters, hardware, evaluation methodology, results,
    limitations, intended and prohibited uses, licence and reproducibility info.

    Deliberately does not claim superiority unless ``comparison`` demonstrates it.
    """
    model = manifest.model if manifest else {}
    lora = manifest.lora.get("lora_config", manifest.lora) if manifest else {}
    training = (manifest.effective_config.get("training", {}) if manifest else {}) or {}
    base_model = model.get("base_model", "unknown")
    revision = str(model.get("revision") or "main")

    front_matter = [
        "---",
        "library_name: peft",
        "tags:",
        "  - lora",
        "  - qlora",
        "  - kleos",
        "  - peft",
    ]
    if base_model != "unknown":
        front_matter.append(f"base_model: {base_model}")
    front_matter.extend(["license: other", "---", ""])

    lines = [
        *front_matter,
        f"# {repo_id}",
        "",
        f"A KLEOS behavioural LoRA adapter for `{base_model}`.",
        "",
        "> This adapter was produced by a research pipeline whose purpose is to",
        "> determine **whether** KLEOS-specific fine-tuning improves task performance.",
        "> A published adapter is not evidence that it does. See the Results section.",
        "",
        "## Model details",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Base model | `{base_model}` |",
        f"| Revision | `{model.get('revision', 'unknown')}` |",
        f"| Model family | {model.get('family', 'unknown')} |",
        f"| Architecture | `{model.get('architecture', 'unknown')}` |",
        f"| Parameters | {model.get('parameter_count', 'unknown')} |",
    ]
    if model.get("active_parameter_count"):
        lines.append(f"| Active parameters (MoE) | {model['active_parameter_count']} |")
    lines.extend(
        [
            "| Fine-tuning method | LoRA / QLoRA (PEFT) |",
            f"| Reasoning mode | {manifest.reasoning_mode if manifest else 'unknown'} |",
            f"| Quantization | {(manifest.quantization or {}).get('mode', 'unknown') if manifest else 'unknown'} |",
            "",
            "### LoRA configuration",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| r | {lora.get('r', 'unknown')} |",
            f"| alpha | {lora.get('alpha', 'unknown')} |",
            f"| dropout | {lora.get('dropout', 'unknown')} |",
            f"| target modules | `{lora.get('target_modules', 'unknown')}` |",
            "",
            "### Training hyperparameters",
            "",
            "> These are **initial engineering defaults**, not tuned values.",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| learning rate | {training.get('learning_rate', 'unknown')} |",
            f"| epochs | {training.get('num_train_epochs', 'unknown')} |",
            f"| batch size (device) | {training.get('per_device_train_batch_size', 'unknown')} |",
            f"| gradient accumulation | {training.get('gradient_accumulation_steps', 'unknown')} |",
            f"| optimizer | {training.get('optim', 'unknown')} |",
            f"| seed | {manifest.seed if manifest else 'unknown'} |",
            "",
            "### Training hardware",
            "",
            f"`{(manifest.hardware or {}).get('name', 'unknown') if manifest else 'unknown'}`"
            + (
                f" ({manifest.hardware.get('total_memory_gb')} GB)"
                if manifest and manifest.hardware.get("total_memory_gb")
                else ""
            ),
            "",
            "## Training data",
            "",
            f"Dataset version: `{manifest.dataset_version if manifest else 'unknown'}`  ",
            f"Dataset content hash: `{(manifest.dataset_hash or 'unknown')[:16] if manifest else 'unknown'}`",
            "",
            "### Data privacy statement",
            "",
            "This adapter was trained on **behavioural policy examples**, not on private",
            "user facts. The training data teaches decision procedures — how to weigh a",
            "deadline against evidence quality, when to route to a tool, how to resolve",
            "conflicting records — and deliberately does **not** teach facts about any",
            "individual.",
            "",
            "Private user data (conversations, memories, resumes, documents) is handled",
            "by KLEOS's retrieval and memory layers and is **not** encoded in these",
            "weights. No raw user data, credentials or personal identifiers were included",
            "in training, and none are published with this adapter.",
            "",
            "## Tasks",
            "",
            f"Primary task: `{manifest.task if manifest and manifest.task else 'not restricted to a single task'}`",
            "",
            "## Evaluation methodology",
            "",
            "Evaluation compares research arms under identical conditions — same",
            "benchmark, same decoding settings, same graders, same seeds — so that any",
            "difference is attributable to the adapter rather than the harness. Metrics",
            "target judgment and correctness (classification, ranking, rubric,",
            "faithfulness) rather than tone. In-distribution, out-of-distribution and",
            "consistency results are reported separately and never blended.",
            "",
            "## Results",
            "",
            _describe_results(results),
            "",
            "### Comparison against the base model",
            "",
            _describe_comparison(comparison),
            "",
            "## Intended use",
            "",
            "Research into whether behavioural fine-tuning improves task-specific judgment",
            "for an AI operating system. It is intended for evaluation and further",
            "research, not for production deployment.",
            "",
            "## Limitations and known failure modes",
            "",
            "- Trained on a narrow behavioural distribution; behaviour outside KLEOS-like",
            "  decision tasks is unmodified at best and degraded at worst.",
            "- Fine-tuning can damage general capability. Check the capability figures",
            "  above; if none are shown, that regression was **not measured**.",
            "- Hyperparameters are engineering defaults, not tuned values.",
            "- The adapter inherits every limitation and bias of the base model.",
            "- No safety alignment work was performed on top of the base model.",
            "",
            "## Prohibited uses",
            "",
            "- Do not use to infer or generate claims about real individuals.",
            "- Do not use in a setting where an incorrect prioritization or routing",
            "  decision carries safety, legal, medical or financial consequences.",
            "- Do not present its output as a factual record without verifying it against",
            "  the underlying source.",
            "",
            "## Licence",
            "",
            f"This adapter derives from `{base_model}` and is subject to that model's",
            "licence terms. The KLEOS training and evaluation code is MIT-licensed.",
            "Check the base model's licence before redistributing.",
            "",
            "## Reproducibility",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Experiment id | `{manifest.experiment_id if manifest else 'unknown'}` |",
            f"| Config hash | `{manifest.config_hash if manifest else 'unknown'}` |",
            f"| Seed | {manifest.seed if manifest else 'unknown'} |",
            f"| Code commit | `{(manifest.git or {}).get('commit', 'unavailable') if manifest else 'unavailable'}` |",
            f"| transformers | {(manifest.software or {}).get('transformers', 'unknown') if manifest else 'unknown'} |",
            f"| peft | {(manifest.software or {}).get('peft', 'unknown') if manifest else 'unknown'} |",
            f"| torch | {(manifest.software or {}).get('torch', 'unknown') if manifest else 'unknown'} |",
            "",
        ]
    )

    if manifest and manifest.adjustments:
        lines.extend(
            [
                "### Automatic configuration adjustments",
                "",
                "The requested configuration was changed automatically during this run:",
                "",
                "| Field | From | To | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for adjustment in manifest.adjustments:
            lines.append(
                f"| `{adjustment['field']}` | {adjustment['original']} | "
                f"{adjustment['adjusted']} | {adjustment['reason']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Usage",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'BASE = "{base_model}"',
            f'REVISION = "{revision}"',
            "",
            "base = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)",
            f'model = PeftModel.from_pretrained(base, "{repo_id}")',
            "tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)",
            "```",
            "",
            _revision_note(revision),
            "",
            "### Tokenizer",
            "",
            f"This repository does **not** ship a tokenizer. Load it from `{base_model}` "
            f"at revision `{revision}`, exactly as in the snippet above.",
            "",
            "That is not an omission. A LoRA adapter is not self-contained — inference "
            "loads the base model regardless — and this adapter never modified the "
            "vocabulary: LoRA excludes `embed_tokens` and `lm_head`, and no tokens were "
            "added. The base tokenizer is therefore the correct one, and duplicating "
            "~17MB of identical vocabulary here would add nothing.",
            "",
            "> If the base model is a vision-language checkpoint (for example",
            "> Mistral Small 3.2), use `AutoModelForImageTextToText` instead —",
            "> `AutoModelForCausalLM` cannot load it.",
            "",
            "---",
            "",
            f"_Card generated by `scripts/publish_adapter.py` on "
            f"{datetime.now(UTC).date().isoformat()}._",
            "",
        ]
    )
    return "\n".join(lines)
