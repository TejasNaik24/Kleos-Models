"""Resumable evaluation: a disconnect costs the example in flight, not the arm.

An arm of the KLEOS benchmark takes two to three hours on a free T4, longer than
a Colab session reliably lasts. Every finished generation is therefore appended
to ``<output>.partial.jsonl`` and flushed to disk before the next one starts. A
resumed run replays those generations and generates only the rest.

Greedy decoding makes a replayed generation the one an uninterrupted run would
have produced, so the finished result is the result of an uninterrupted run.
Grading is never replayed: it always reruns over every response, with the code
that finishes the run.

The partial file's first line is an identity: everything that could change a
generation (arm, seeds, the examples and the benchmark's bytes, decoding, the
orchestration prompt, the model config, the adapter's weights, the code commit,
library versions and the GPU's compute capability). A resume under a different
identity is refused, because it would splice two different evaluations into one
result. A torn last line, from a runtime killed mid-write, is dropped and its
example generated again.

This module imports no torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_models.errors import EvaluationError
from kleos_models.inference.backends import GenerationOutput
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

PARTIAL_SUFFIX = ".partial.jsonl"
PARTIAL_KIND = "kleos-evaluation-partial"
PARTIAL_VERSION = 1


def partial_path(output: Path) -> Path:
    """Where the in-progress generations for ``output`` are kept."""
    return output.with_name(output.name + PARTIAL_SUFFIX)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_identity(adapter_path: Path | str | None) -> dict[str, Any] | None:
    """sha256 of an adapter's weights and config: its identity, whatever its path."""
    if adapter_path is None:
        return None
    directory = Path(adapter_path)
    files = sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and (p.name.startswith("adapter_model.") or p.name == "adapter_config.json")
    )
    if not any(name.startswith("adapter_model.") for name in files):
        raise EvaluationError(
            f"No adapter weights found in {directory}.",
            suggestions=["Point --adapter at the directory holding adapter_model.safetensors."],
        )
    return {name: sha256_file(directory / name) for name in files}


def build_identity(
    *,
    arm: str,
    seeds: Sequence[int],
    example_ids: Sequence[str],
    benchmark_sha256: str,
    generation: Mapping[str, Any],
    orchestration_prompt: str | None,
    model: Mapping[str, Any],
    adapter: Mapping[str, Any] | None,
    git_commit: str | None,
    libraries: Mapping[str, str | None],
    compute_capability: str,
) -> dict[str, Any]:
    """Everything that could change a generation, as JSON-ready data."""
    return {
        "arm": arm,
        "seeds": list(seeds),
        "example_count": len(example_ids),
        "example_ids_sha256": sha256_bytes("\n".join(example_ids).encode("utf-8")),
        "benchmark_sha256": benchmark_sha256,
        "generation": dict(generation),
        "orchestration_prompt_sha256": (
            sha256_bytes(orchestration_prompt.encode("utf-8")) if orchestration_prompt else None
        ),
        "model": json.loads(json.dumps(model, default=str)),
        "adapter": dict(adapter) if adapter else None,
        "git_commit": git_commit,
        "libraries": dict(libraries),
        "compute_capability": compute_capability,
    }


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(identity, sort_keys=True, default=str).encode("utf-8"))


def _differences(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(recorded) | set(current))
    return [key for key in keys if recorded.get(key) != current.get(key)]


@dataclass
class PartialState:
    """What an earlier, interrupted run of the same evaluation finished."""

    path: Path
    generations: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    dropped_torn_line: bool = False


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(descriptor)


def start_partial(path: Path, identity: Mapping[str, Any]) -> None:
    """Create the partial file with its identity header, replacing nothing."""
    header = {
        "kind": PARTIAL_KIND,
        "version": PARTIAL_VERSION,
        "identity_sha256": identity_sha256(identity),
        "identity": identity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def load_partial(path: Path, identity: Mapping[str, Any]) -> PartialState:
    """Read an interrupted run's generations, refusing a different evaluation.

    Raises:
        EvaluationError: when the header is missing or names a different
            identity, or a line other than the last is unreadable.
    """
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    # A file ending in "\n" splits into a trailing empty element; anything else
    # there is a line that was being written when the runtime died.
    tail = lines.pop()
    state = PartialState(path=path)

    if not lines:
        raise EvaluationError(
            f"{path} has no complete header line; it cannot be resumed.",
            suggestions=["Delete it, or pass --overwrite to start the evaluation again."],
        )
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path} has an unreadable header: {exc}") from exc
    if header.get("kind") != PARTIAL_KIND or header.get("version") != PARTIAL_VERSION:
        raise EvaluationError(
            f"{path} is not a KLEOS partial evaluation (version {PARTIAL_VERSION})."
        )

    recorded = header.get("identity") or {}
    if header.get("identity_sha256") != identity_sha256(identity):
        changed = _differences(recorded, identity)
        raise EvaluationError(
            "The interrupted evaluation was run under a different identity, so its "
            "generations cannot be reused.",
            details={"partial": str(path), "differs_in": ", ".join(changed) or "(header digest)"},
            suggestions=[
                "Resume with exactly the same config, arm, benchmark, adapter, code "
                "and libraries as the interrupted run.",
                "Or pass --overwrite to discard it and evaluate from the start.",
            ],
        )

    good_bytes = len(lines[0]) + 1
    for number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"{path} line {number} is unreadable, and it is not the last line: "
                "the file is damaged, not merely interrupted.",
                details={"error": str(exc)},
                suggestions=["Pass --overwrite to evaluate from the start."],
            ) from exc
        state.generations[(int(record["seed"]), str(record["example_id"]))] = record
        good_bytes += len(line) + 1

    if tail:
        # The runtime died while this line was being written.
        state.dropped_torn_line = True
        with path.open("r+b") as handle:
            handle.truncate(good_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        logger.warning("Dropped a torn last line from %s; that example is generated again.", path)
    return state


class PartialWriter:
    """Append-only, fsynced record of finished generations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


class ResumingBackend:
    """Wrap a backend: replay recorded generations, record new ones.

    The runner sees an ordinary backend. Replayed outputs carry their original
    latency in ``metadata['recorded_latency_seconds']`` so generation statistics
    describe the real run, not the replay.
    """

    def __init__(
        self,
        inner: Any,
        *,
        recorded: Mapping[tuple[int, str], Mapping[str, Any]],
        writer: PartialWriter,
    ) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "backend")
        self._recorded = dict(recorded)
        self._writer = writer
        self.replayed = 0
        self.generated = 0

    def generate(self, messages: Sequence[Any], config: Any, **kwargs: Any) -> GenerationOutput:
        key = (int(kwargs.get("seed", 0)), str(kwargs.get("example_id")))
        record = self._recorded.get(key)
        if record is not None:
            self.replayed += 1
            return GenerationOutput(
                text=record["text"],
                reasoning=record.get("reasoning"),
                prompt_tokens=int(record.get("prompt_tokens", 0)),
                completion_tokens=int(record.get("completion_tokens", 0)),
                finish_reason=record.get("finish_reason", "stop"),
                metadata={
                    **(record.get("metadata") or {}),
                    "recorded_latency_seconds": record.get("latency_seconds", 0.0),
                },
            )

        started = time.perf_counter()
        output: GenerationOutput = self.inner.generate(messages, config, **kwargs)
        latency = time.perf_counter() - started
        self._writer.append(
            {
                "seed": key[0],
                "example_id": key[1],
                "text": output.text,
                "reasoning": output.reasoning,
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "finish_reason": output.finish_reason,
                "metadata": json.loads(json.dumps(output.metadata, default=str)),
                "latency_seconds": round(latency, 3),
            }
        )
        self.generated += 1
        return output

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = self.inner.describe()
        return description
