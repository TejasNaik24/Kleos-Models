"""The data contract reads kleos-policy-v0.0.7's assistant ``reasoning`` field.

Schema 1.1 adds one optional field to assistant messages: a policy-derived
reasoning trace, which a reasoning model is trained to emit inside its thinking
span. Examples without it must serialize and hash exactly as before, so every
earlier release keeps its identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.conftest import V007_RELEASE, make_example

from kleos_models.config import DatasetConfig
from kleos_models.constants import DATASET_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from kleos_models.data.loaders import load_dataset_bundle, write_jsonl
from kleos_models.data.schemas import Message, TrainingExample
from kleos_models.data.validation import validate_examples

V006_TEST_SHA256 = "a4decaaf029b227366846d44018b0b1b669d21f068800a9f3175e15ec4783980"


def with_reasoning(text: str = "Score each item.") -> dict[str, Any]:
    payload = make_example()
    payload["version"] = "1.1"
    payload["messages"][-1]["reasoning"] = text
    return payload


class TestReasoningField:
    def test_a_v007_style_example_loads(self) -> None:
        example = TrainingExample.model_validate(with_reasoning())
        assert example.messages[-1].reasoning == "Score each item."

    def test_reasoning_is_assistant_only(self) -> None:
        with pytest.raises(ValidationError, match="assistant"):
            Message(role="user", content="x", reasoning="y")

    def test_reasoning_is_never_blank(self) -> None:
        with pytest.raises(ValidationError, match="reasoning"):
            Message(role="assistant", content="x", reasoning="  ")

    def test_an_example_without_reasoning_writes_exactly_as_before(self, tmp_path: Path) -> None:
        example = TrainingExample.model_validate(make_example())
        write_jsonl([example], tmp_path / "a.jsonl")
        line = json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))
        assert all(set(m) == {"role", "content", "name"} for m in line["messages"])

    def test_reasoning_is_written_when_present(self, tmp_path: Path) -> None:
        write_jsonl([TrainingExample.model_validate(with_reasoning())], tmp_path / "a.jsonl")
        line = json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))
        assert line["messages"][-1]["reasoning"] == "Score each item."
        assert line["messages"][-1]["name"] is None

    def test_a_written_example_with_reasoning_reads_back_identically(self, tmp_path: Path) -> None:
        example = TrainingExample.model_validate(with_reasoning())
        write_jsonl([example], tmp_path / "a.jsonl")
        again = TrainingExample.model_validate_json((tmp_path / "a.jsonl").read_text())
        assert again == example

    def test_the_conversation_text_without_reasoning_is_unchanged(self) -> None:
        plain = TrainingExample.model_validate(make_example())
        expected = "\n".join(f"{m.role}: {m.content}" for m in plain.messages)
        assert plain.conversation_text() == expected

    def test_reasoning_enters_the_content_hash(self) -> None:
        a = TrainingExample.model_validate(with_reasoning("a."))
        b = TrainingExample.model_validate(with_reasoning("b."))
        assert a.content_hash() != b.content_hash()

    def test_reasoning_stays_out_of_the_prompt_only_hash(self) -> None:
        a = TrainingExample.model_validate(with_reasoning("a."))
        b = TrainingExample.model_validate(with_reasoning("b."))
        assert a.content_hash(include_assistant=False) == b.content_hash(include_assistant=False)

    def test_stripping_drops_the_reasoning_field(self) -> None:
        stripped = TrainingExample.model_validate(with_reasoning()).strip_reasoning_spans()
        assert stripped.messages[-1].reasoning is None
        assert stripped.messages[-1].content == make_example()["messages"][-1]["content"]


class TestSchemaVersions:
    def test_the_written_schema_version_is_still_1_0(self) -> None:
        assert DATASET_SCHEMA_VERSION == "1.0"
        assert SUPPORTED_SCHEMA_VERSIONS == ("1.0", "1.1")

    def test_schema_1_1_does_not_warn(self) -> None:
        report = validate_examples([TrainingExample.model_validate(with_reasoning())])
        assert "unexpected_schema_version" not in {f.code for f in report.warnings}

    def test_an_unknown_schema_version_still_warns(self) -> None:
        payload = make_example()
        payload["version"] = "9.9"
        report = validate_examples([TrainingExample.model_validate(payload)])
        assert "unexpected_schema_version" in {f.code for f in report.warnings}


@pytest.fixture(scope="module")
def bundle() -> Any:
    config = DatasetConfig.model_validate({"path": str(V007_RELEASE), "require_manifest": True})
    return load_dataset_bundle(config, require_splits=("train", "validation", "test"))


@pytest.mark.requires_v007
class TestSealedV007Release:
    def test_every_split_loads(self, bundle: Any) -> None:
        assert len(bundle.train) and len(bundle.validation) and len(bundle.test)

    def test_train_and_validation_answers_all_carry_reasoning(self, bundle: Any) -> None:
        for example in [*bundle.train, *bundle.validation]:
            assert example.messages[-1].reasoning, example.id
            assert example.version == "1.1"

    def test_every_example_starts_with_a_system_message(self, bundle: Any) -> None:
        # Review Focus 3: the Reasoning template adds its own default "how you
        # should think" system prompt to any conversation without one.
        for example in [*bundle.train, *bundle.validation, *bundle.test]:
            assert example.messages[0].role == "system", example.id

    def test_the_test_split_carries_none(self, bundle: Any) -> None:
        assert all(m.reasoning is None for e in bundle.test for m in e.messages)
        assert {e.version for e in bundle.test} == {"1.0"}

    def test_the_test_split_is_byte_identical_to_v006(self) -> None:
        digest = hashlib.sha256((V007_RELEASE / "test.jsonl").read_bytes()).hexdigest()
        assert digest == V006_TEST_SHA256
