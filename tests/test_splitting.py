"""Splitting tests (spec §12, §30).

The properties that matter: splits are deterministic, they never overlap, groups
stay together, and held-out strategies genuinely hold values out.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_example

from kleos_models.config import SplitConfig
from kleos_models.data.schemas import TrainingExample
from kleos_models.data.splitting import split_examples, verify_split
from kleos_models.errors import DatasetIntegrityError


def build(count: int, **kwargs) -> list[TrainingExample]:
    """Build a list of distinct examples."""
    return [
        TrainingExample.model_validate(
            make_example(
                f"ex-{i:06d}",
                user=f"Scenario {i}: rank item-{i}a and item-{i}b.",
                assistant=f"1. item-{i}a\n2. item-{i}b\n\nReasoning: nearer deadline.",
                **kwargs,
            )
        )
        for i in range(count)
    ]


class TestRandomSplit:
    def test_all_examples_are_assigned(self, training_examples):
        config = SplitConfig(strategy="random", seed=42)
        result = split_examples(training_examples, config)
        assert result.total == len(training_examples)

    def test_split_is_deterministic(self, training_examples):
        config = SplitConfig(strategy="random", seed=7)
        first = split_examples(training_examples, config)
        second = split_examples(training_examples, config)
        assert [e.id for e in first.train] == [e.id for e in second.train]
        assert [e.id for e in first.test] == [e.id for e in second.test]

    def test_different_seeds_give_different_splits(self):
        examples = build(60)
        left = split_examples(examples, SplitConfig(strategy="random", seed=1))
        right = split_examples(examples, SplitConfig(strategy="random", seed=2))
        assert [e.id for e in left.train] != [e.id for e in right.train]

    def test_no_overlap_between_splits(self, training_examples):
        result = split_examples(training_examples, SplitConfig(strategy="random", seed=42))
        train_ids = {e.id for e in result.train}
        val_ids = {e.id for e in result.validation}
        test_ids = {e.id for e in result.test}
        assert not train_ids & val_ids
        assert not train_ids & test_ids
        assert not val_ids & test_ids

    def test_random_split_carries_a_development_only_warning(self, training_examples):
        result = split_examples(training_examples, SplitConfig(strategy="random", seed=42))
        assert any("development" in note.lower() for note in result.notes)

    def test_fractions_are_approximately_respected(self):
        examples = build(100)
        result = split_examples(
            examples,
            SplitConfig(
                strategy="random",
                seed=42,
                train_fraction=0.6,
                validation_fraction=0.2,
                test_fraction=0.2,
            ),
        )
        assert 55 <= len(result.train) <= 65
        assert 15 <= len(result.validation) <= 25


class TestGroupSplit:
    def test_grouped_examples_stay_together(self):
        examples = [
            TrainingExample.model_validate(
                make_example(
                    f"ex-{i:04d}",
                    scenario_family=f"family-{i % 5:02d}",
                    user=f"Scenario {i}",
                    assistant=f"Answer {i}",
                )
            )
            for i in range(40)
        ]
        result = split_examples(examples, SplitConfig(strategy="group", seed=42))

        placement: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for example in result.split(split_name):
                family = example.metadata.scenario_family
                assert family is not None
                if family in placement:
                    assert placement[family] == split_name, (
                        f"family {family} was split across {placement[family]} and {split_name}"
                    )
                placement[family] = split_name

    def test_singleton_groups_produce_a_warning(self):
        examples = build(20)  # no scenario_family → every group is a singleton
        result = split_examples(examples, SplitConfig(strategy="group", seed=42))
        assert any("size 1" in note or "singleton" in note.lower() for note in result.notes)

    def test_explicit_group_key_is_honoured(self):
        examples = [
            TrainingExample.model_validate(
                make_example(
                    f"ex-{i:04d}",
                    domain=["career", "research"][i % 2],
                    user=f"S{i}",
                    assistant=f"A{i}",
                )
            )
            for i in range(30)
        ]
        result = split_examples(
            examples, SplitConfig(strategy="group", seed=42, group_key="domain")
        )
        placement: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for example in result.split(split_name):
                domain = example.variation_axes.domain
                if domain in placement:
                    assert placement[domain] == split_name
                placement[domain] = split_name


class TestHoldoutSplits:
    def _mixed_entities(self, count: int = 60) -> list[TrainingExample]:
        sets = ["set_a", "set_b", "set_c", "set_d"]
        return [
            TrainingExample.model_validate(
                make_example(
                    f"ex-{i:04d}",
                    entities=sets[i % len(sets)],
                    user=f"Scenario {i}",
                    assistant=f"Answer {i}",
                )
            )
            for i in range(count)
        ]

    def test_entity_holdout_keeps_held_out_entities_out_of_training(self):
        examples = self._mixed_entities()
        result = split_examples(
            examples,
            SplitConfig(strategy="entity_holdout", seed=42, holdout_values=["set_d"]),
        )
        assert result.holdout_values == ["set_d"]
        train_entities = {e.variation_axes.entities for e in result.train}
        assert "set_d" not in train_entities
        test_entities = {e.variation_axes.entities for e in result.test}
        assert test_entities == {"set_d"}

    def test_validation_stays_in_distribution(self):
        # Validation must come from SEEN values: early stopping on OOD data would
        # leak exactly the signal the test split is meant to measure.
        examples = self._mixed_entities()
        result = split_examples(
            examples,
            SplitConfig(strategy="entity_holdout", seed=42, holdout_values=["set_d"]),
        )
        validation_entities = {e.variation_axes.entities for e in result.validation}
        assert "set_d" not in validation_entities

    def test_domain_holdout_holds_out_a_domain(self):
        domains = ["career", "research", "coursework", "projects"]
        examples = [
            TrainingExample.model_validate(
                make_example(f"ex-{i:04d}", domain=domains[i % 4], user=f"S{i}", assistant=f"A{i}")
            )
            for i in range(40)
        ]
        result = split_examples(
            examples,
            SplitConfig(strategy="domain_holdout", seed=42, holdout_values=["projects"]),
        )
        assert "projects" not in {e.variation_axes.domain for e in result.train}
        assert {e.variation_axes.domain for e in result.test} == {"projects"}

    def test_format_holdout_holds_out_a_format(self):
        formats = ["prose", "bullets", "json"]
        examples = [
            TrainingExample.model_validate(
                make_example(f"ex-{i:04d}", format=formats[i % 3], user=f"S{i}", assistant=f"A{i}")
            )
            for i in range(30)
        ]
        result = split_examples(
            examples, SplitConfig(strategy="format_holdout", seed=42, holdout_values=["json"])
        )
        assert "json" not in {e.variation_axes.format for e in result.train}

    def test_holdout_is_chosen_deterministically_without_explicit_values(self):
        examples = self._mixed_entities()
        config = SplitConfig(strategy="entity_holdout", seed=42)
        first = split_examples(examples, config)
        second = split_examples(examples, config)
        assert first.holdout_values == second.holdout_values
        assert first.holdout_values, "a holdout value should have been chosen"

    def test_unknown_holdout_value_is_rejected(self):
        examples = self._mixed_entities()
        with pytest.raises(DatasetIntegrityError, match="do not occur"):
            split_examples(
                examples,
                SplitConfig(strategy="entity_holdout", seed=42, holdout_values=["set_zzz"]),
            )

    def test_single_valued_attribute_cannot_be_held_out(self):
        # Every example shares one entity set, so there is nothing to hold out.
        examples = build(20, entities="set_a")
        with pytest.raises(DatasetIntegrityError, match=r"only 1 distinct value"):
            split_examples(examples, SplitConfig(strategy="entity_holdout", seed=42))

    def test_holding_out_everything_is_rejected(self):
        examples = build(10, entities="set_a") + build(10, entities="set_b")
        for i, example in enumerate(examples):
            object.__setattr__(example, "id", f"unique-{i:04d}")
        with pytest.raises(DatasetIntegrityError, match="no training data"):
            split_examples(
                examples,
                SplitConfig(strategy="entity_holdout", seed=42, holdout_values=["set_a", "set_b"]),
            )

    def test_scenario_family_holdout_keeps_families_intact(self):
        examples = [
            TrainingExample.model_validate(
                make_example(
                    f"ex-{i:04d}",
                    scenario_family=f"family-{i % 6:02d}",
                    user=f"S{i}",
                    assistant=f"A{i}",
                )
            )
            for i in range(36)
        ]
        result = split_examples(examples, SplitConfig(strategy="scenario_family_holdout", seed=42))
        placement: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for example in result.split(split_name):
                family = example.metadata.scenario_family
                assert family is not None
                if family in placement:
                    assert placement[family] == split_name
                placement[family] = split_name


class TestVerification:
    def test_verify_rejects_overlap(self, training_examples):
        result = split_examples(training_examples, SplitConfig(strategy="random", seed=42))
        # Plant an overlap that a correct split would never produce.
        result.test.append(result.train[0])
        with pytest.raises(DatasetIntegrityError, match=r"[Ss]plit overlap"):
            verify_split(result)

    def test_verify_rejects_lost_examples(self, training_examples):
        result = split_examples(training_examples, SplitConfig(strategy="random", seed=42))
        result.train.pop()
        with pytest.raises(DatasetIntegrityError, match="lost or duplicated"):
            verify_split(result, expected_total=len(training_examples))

    def test_verify_rejects_an_empty_training_set(self, training_examples):
        result = split_examples(training_examples, SplitConfig(strategy="random", seed=42))
        result.train.clear()
        with pytest.raises(DatasetIntegrityError, match="empty training set"):
            verify_split(result)


class TestSplitConfigValidation:
    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            SplitConfig(train_fraction=0.5, validation_fraction=0.2, test_fraction=0.2)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown split strategy"):
            SplitConfig(strategy="magic")

    def test_empty_dataset_is_rejected(self):
        with pytest.raises(DatasetIntegrityError, match="empty dataset"):
            split_examples([], SplitConfig())
