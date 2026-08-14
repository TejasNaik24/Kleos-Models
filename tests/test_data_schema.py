"""Data-contract tests (spec §30).

Valid examples are accepted; malformed ones are rejected with a useful message.
The published JSON Schema is checked against the pydantic models so the two
cannot drift apart.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from tests.conftest import REPO_ROOT, make_eval_example, make_example

from kleos_models.constants import DATASET_SCHEMA_VERSION
from kleos_models.data.schemas import (
    DatasetManifest,
    EvaluationExample,
    Message,
    TrainingExample,
    contains_reasoning,
    strip_reasoning,
)


class TestValidExamples:
    def test_minimal_example_is_accepted(self):
        example = TrainingExample.model_validate(make_example())
        assert example.id == "test-000001"
        assert example.task == "notification_prioritization"
        assert len(example.messages) == 3

    def test_domain_mirrors_variation_axis(self):
        example = TrainingExample.model_validate(make_example(domain="research"))
        assert example.domain == "research"
        assert example.variation_axes.domain == "research"

    def test_system_prompt_is_exposed(self):
        example = TrainingExample.model_validate(make_example())
        assert example.system_prompt == "You are the KLEOS reasoning layer."

    def test_example_without_system_message_is_valid(self):
        example = TrainingExample.model_validate(make_example(system=None))
        assert example.system_prompt is None
        assert example.messages[0].role == "user"

    def test_multi_turn_conversation_is_accepted(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second answer"},
        ]
        example = TrainingExample.model_validate(payload)
        assert len(example.assistant_targets) == 2

    def test_extra_variation_axes_are_allowed_and_flagged(self):
        example = TrainingExample.model_validate(make_example(experimental_axis="pilot_value"))
        assert example.variation_axes.unknown_axes() == {"experimental_axis": "pilot_value"}
        assert "domain" in example.variation_axes.known_axes()


class TestRejectedExamples:
    def test_missing_assistant_message_is_rejected(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        with pytest.raises(ValidationError, match="assistant"):
            TrainingExample.model_validate(payload)

    def test_conversation_not_ending_with_assistant_is_rejected(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "follow-up with no answer"},
        ]
        with pytest.raises(ValidationError, match="final message"):
            TrainingExample.model_validate(payload)

    def test_consecutive_assistant_messages_are_rejected(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        with pytest.raises(ValidationError, match=r"[Cc]onsecutive assistant"):
            TrainingExample.model_validate(payload)

    def test_assistant_before_any_user_is_rejected(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "unprompted"},
        ]
        with pytest.raises(ValidationError, match="preceded by a user"):
            TrainingExample.model_validate(payload)

    def test_system_message_not_first_is_rejected(self):
        payload = make_example()
        payload["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "late system prompt"},
            {"role": "assistant", "content": "a"},
        ]
        with pytest.raises(ValidationError, match="system message must be first"):
            TrainingExample.model_validate(payload)

    def test_invalid_role_is_rejected(self):
        payload = make_example()
        payload["messages"][1]["role"] = "moderator"
        with pytest.raises(ValidationError):
            TrainingExample.model_validate(payload)

    def test_empty_message_content_is_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            Message(role="user", content="   ")

    def test_unknown_task_is_rejected(self):
        payload = make_example()
        payload["task"] = "not_a_registered_task"
        with pytest.raises(ValidationError, match="unknown task"):
            TrainingExample.model_validate(payload)

    def test_malformed_id_is_rejected(self):
        payload = make_example()
        payload["id"] = "!!"
        with pytest.raises(ValidationError, match="id"):
            TrainingExample.model_validate(payload)

    def test_missing_required_field_is_rejected(self):
        payload = make_example()
        del payload["variation_axes"]
        with pytest.raises(ValidationError, match="variation_axes"):
            TrainingExample.model_validate(payload)

    def test_unknown_top_level_field_is_rejected(self):
        payload = make_example()
        payload["unexpected_field"] = "value"
        with pytest.raises(ValidationError):
            TrainingExample.model_validate(payload)

    def test_contradictory_domain_is_rejected(self):
        payload = make_example(domain="career")
        payload["domain"] = "research"
        with pytest.raises(ValidationError, match="contradicts"):
            TrainingExample.model_validate(payload)

    def test_unknown_source_is_rejected(self):
        payload = make_example()
        payload["metadata"]["source"] = "scraped_from_production"
        with pytest.raises(ValidationError, match="unknown source"):
            TrainingExample.model_validate(payload)

    def test_tool_message_without_name_is_rejected(self):
        with pytest.raises(ValidationError, match="require a 'name'"):
            Message(role="tool", content="result")


class TestEvaluationExample:
    def test_valid_example_is_accepted(self):
        example = EvaluationExample.model_validate(make_eval_example())
        assert example.reference["label"] == "memory_search"
        assert example.split_tag == "in_distribution"

    def test_reference_is_required(self):
        payload = make_eval_example()
        payload["reference"] = {}
        with pytest.raises(ValidationError, match="reference"):
            EvaluationExample.model_validate(payload)

    def test_prompt_messages_stop_at_the_first_assistant_turn(self):
        payload = make_eval_example()
        payload["messages"] = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "leaked reference answer"},
        ]
        example = EvaluationExample.model_validate(payload)
        assert len(example.prompt_messages) == 1
        assert "leaked" not in example.conversation_text()

    def test_ood_tagging_round_trips(self):
        example = EvaluationExample.model_validate(
            make_eval_example(split_tag="ood", ood_shift="unseen_domains")
        )
        assert example.split_tag == "ood"
        assert example.ood_shift == "unseen_domains"


class TestReasoningStripping:
    def test_well_formed_think_block_is_removed(self):
        text = "<think>internal deliberation</think>The answer is alpha."
        assert strip_reasoning(text) == "The answer is alpha."

    def test_dangling_close_tag_is_removed(self):
        # Qwen Thinking templates pre-open <think>, so generated text often has
        # only the closing tag.
        text = "reasoning that was never opened</think>The answer is beta."
        assert strip_reasoning(text) == "The answer is beta."

    def test_text_without_reasoning_is_unchanged(self):
        assert strip_reasoning("Just an answer.") == "Just an answer."

    def test_contains_reasoning_detects_both_forms(self):
        assert contains_reasoning("<think>x</think>y")
        assert contains_reasoning("x</think>y")
        assert not contains_reasoning("plain text")

    def test_stripping_applies_to_assistant_turns_only(self):
        payload = make_example(
            user="<think>this stays</think> user text",
            assistant="<think>this goes</think>Final answer.",
        )
        example = TrainingExample.model_validate(payload).strip_reasoning_spans()
        assert example.assistant_targets == ["Final answer."]
        assert "<think>" in example.messages[1].content


class TestContentHashing:
    def test_identical_content_hashes_match(self):
        left = TrainingExample.model_validate(make_example("id-a"))
        right = TrainingExample.model_validate(make_example("id-b"))
        assert left.content_hash() == right.content_hash()

    def test_different_content_hashes_differ(self):
        left = TrainingExample.model_validate(make_example("id-a"))
        right = TrainingExample.model_validate(make_example("id-a", assistant="Different."))
        assert left.content_hash() != right.content_hash()

    def test_excluding_assistant_changes_the_hash(self):
        example = TrainingExample.model_validate(make_example())
        assert example.content_hash() != example.content_hash(include_assistant=False)


class TestGroupKey:
    def test_group_id_takes_precedence(self):
        payload = make_example(scenario_family="family-1")
        payload["metadata"]["group_id"] = "group-9"
        example = TrainingExample.model_validate(payload)
        assert example.group_key() == "group-9"

    def test_scenario_family_is_the_fallback(self):
        example = TrainingExample.model_validate(make_example(scenario_family="family-1"))
        assert example.group_key() == "family-1"

    def test_id_is_the_last_resort(self):
        example = TrainingExample.model_validate(make_example("solo-001"))
        assert example.group_key() == "solo-001"

    def test_explicit_key_reads_a_variation_axis(self):
        example = TrainingExample.model_validate(make_example(domain="research"))
        assert example.group_key("domain") == "research"


class TestDatasetManifest:
    def test_content_hash_is_deterministic(self):
        manifest = DatasetManifest(
            version="v1", file_hashes={"train.jsonl": "abc", "test.jsonl": "def"}
        ).finalize()
        other = DatasetManifest(
            version="v1", file_hashes={"test.jsonl": "def", "train.jsonl": "abc"}
        ).finalize()
        assert manifest.content_hash == other.content_hash

    def test_different_files_produce_different_hashes(self):
        left = DatasetManifest(version="v1", file_hashes={"train.jsonl": "abc"}).finalize()
        right = DatasetManifest(version="v1", file_hashes={"train.jsonl": "xyz"}).finalize()
        assert left.content_hash != right.content_hash

    def test_private_flag_defaults_to_false(self):
        assert DatasetManifest(version="v1").contains_private_data is False


class TestPublishedJsonSchema:
    """The committed JSON Schema must match the pydantic models."""

    @pytest.mark.parametrize(
        ("filename", "model"),
        [
            ("training_example.schema.json", TrainingExample),
            ("evaluation_example.schema.json", EvaluationExample),
            ("dataset_manifest.schema.json", DatasetManifest),
        ],
    )
    def test_schema_file_is_current(self, filename, model):
        path = REPO_ROOT / "data" / "schema" / filename
        assert path.exists(), (
            f"{filename} is missing. Run: python scripts/prepare_dataset.py --emit-schemas"
        )
        stored = json.loads(path.read_text(encoding="utf-8"))
        current = model.model_json_schema()
        assert stored.get("properties") == current.get("properties"), (
            f"{filename} has drifted from the pydantic model. "
            "Run: python scripts/prepare_dataset.py --emit-schemas"
        )

    def test_schema_version_constant_matches_default(self):
        example = TrainingExample.model_validate(make_example())
        assert example.version == DATASET_SCHEMA_VERSION


class TestCommittedFixtures:
    def test_fixtures_are_valid(self, fixture_examples):
        assert len(fixture_examples) >= 10
        assert all(e.metadata.source == "development_fixture" for e in fixture_examples)

    def test_fixtures_are_labelled_synthetic(self, fixture_examples):
        for example in fixture_examples:
            assert "SYNTHETIC DEVELOPMENT FIXTURE" in (example.metadata.notes or ""), (
                "every fixture must be unmistakably labelled as a development fixture"
            )

    def test_fixture_ids_are_unique(self, fixture_examples):
        ids = [e.id for e in fixture_examples]
        assert len(ids) == len(set(ids))

    def test_eval_fixtures_include_ood_examples(self, fixture_eval_examples):
        ood = [e for e in fixture_eval_examples if e.split_tag == "ood"]
        assert ood, "eval fixtures must include OOD examples or OOD reporting is untestable"
        assert all(e.ood_shift for e in ood)

    def test_eval_fixtures_include_a_consistency_group(self, fixture_eval_examples):
        families: dict[str, int] = {}
        for example in fixture_eval_examples:
            if example.metadata.scenario_family:
                families[example.metadata.scenario_family] = (
                    families.get(example.metadata.scenario_family, 0) + 1
                )
        assert any(count >= 2 for count in families.values()), (
            "eval fixtures need at least one multi-example scenario_family "
            "or consistency testing is untestable"
        )
