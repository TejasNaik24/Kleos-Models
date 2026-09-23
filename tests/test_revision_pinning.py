"""Base-model revision pinning (audit finding F3).

A LoRA adapter is deltas against *specific* base weights. If the base revision
is a moving pointer, an upstream re-upload silently pairs the adapter with
weights it never saw — no error is raised, the judgment just degrades. These
tests exist so that cannot happen by accident, and so the training and serving
pins cannot drift apart.

They also pin the honesty requirement: a model card generated from an unpinned
run must say so rather than imply reproducibility it cannot offer.
"""

from __future__ import annotations

import yaml
from tests.conftest import CONFIGS_DIR

from kleos_models.config import load_model_config
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.publishing import build_model_card, is_pinned_revision

#: The base revision KLEOS trains and serves against, verified 2026-09-15.
PINNED_SHA = "2f494a194c5b980dfb9772cb92d26cbb671fce5a"

MINISTRAL_CONFIG = CONFIGS_DIR / "models" / "ministral_8b.yaml"
DEPLOYMENT_CONFIG = CONFIGS_DIR / "deployment" / "kleos_v006_ministral8b.yaml"

#: The base revision KLEOS Hermes trains and serves against, verified 2026-09-15.
HERMES_PINNED_SHA = "04d8a90549d23fc6bd7f642064003592df51e9b3"
#: The frozen Hermes adapter. Byte-identical to checkpoint-200 of the run.
HERMES_ADAPTER_SHA = "dc121fa36ce1409ca142e8da61a809f9386e32d0ef09c4214efa271d5b857b32"

NEMO_CONFIG = CONFIGS_DIR / "models" / "mistral_nemo_12b.yaml"
HERMES_DEPLOYMENT_CONFIG = CONFIGS_DIR / "deployment" / "kleos_hermes_v006.yaml"


def _deployment() -> dict:
    return yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))["deployment"]


def _hermes() -> dict:
    return yaml.safe_load(HERMES_DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))["deployment"]


class TestIsPinnedRevision:
    def test_a_full_sha_is_pinned(self):
        assert is_pinned_revision(PINNED_SHA)

    def test_moving_pointers_are_not_pinned(self):
        for revision in ("main", "master", "v1.0", "refs/heads/main", "", None):
            assert not is_pinned_revision(revision), revision

    def test_a_short_sha_is_not_accepted(self):
        # Short hashes are ambiguous and can collide as a repository grows.
        assert not is_pinned_revision(PINNED_SHA[:12])

    def test_uppercase_is_not_accepted(self):
        # The Hub emits lowercase; accepting both would let two spellings of the
        # same commit compare unequal in the drift check below.
        assert not is_pinned_revision(PINNED_SHA.upper())


class TestMinistralIsPinned:
    """The config KLEOS actually trains with must never revert to a pointer."""

    def test_ministral_revision_is_a_pinned_sha(self):
        config = load_model_config(MINISTRAL_CONFIG)
        assert is_pinned_revision(config.revision), (
            f"ministral_8b.yaml has revision={config.revision!r}. It must be a "
            "40-character commit sha; 'main' lets upstream move under a trained "
            "adapter without any error being raised."
        )

    def test_ministral_revision_is_the_verified_sha(self):
        assert load_model_config(MINISTRAL_CONFIG).revision == PINNED_SHA

    def test_ministral_base_model_is_unchanged(self):
        assert load_model_config(MINISTRAL_CONFIG).base_model == (
            "mistralai/Ministral-8B-Instruct-2410"
        )


class TestDeploymentMatchesTraining:
    """Serving a different revision than was trained is a silent failure."""

    def test_deployment_config_exists(self):
        assert DEPLOYMENT_CONFIG.exists()

    def test_deployment_revision_matches_the_training_config(self):
        training = load_model_config(MINISTRAL_CONFIG)
        assert _deployment()["revision"] == training.revision, (
            "The serving pin and the training pin have drifted apart. An adapter "
            "served against different base weights than it was trained on "
            "degrades silently."
        )

    def test_deployment_base_model_matches_the_training_config(self):
        training = load_model_config(MINISTRAL_CONFIG)
        assert _deployment()["base_model"] == training.base_model

    def test_deployment_revision_is_pinned(self):
        assert is_pinned_revision(_deployment()["revision"])

    def test_deployment_ships_no_adapter_weights_path_by_default(self):
        # The adapter lives on private storage; a committed path would leak where.
        assert _deployment()["adapter_path"].startswith("${env:")


class TestHermesDeploymentMatchesTraining:
    """KLEOS Hermes v0.0.6, the first KLEOS model prepared for serving.

    Its research artifact records ``revision: null`` in adapter_config.json
    (finding H-F1) and is frozen that way as historical evidence. The deployment
    path must therefore carry the pin itself, and it must be the same pin the
    model was trained against.
    """

    def test_hermes_deployment_config_exists(self):
        assert HERMES_DEPLOYMENT_CONFIG.exists()

    def test_hermes_revision_matches_the_training_config(self):
        training = load_model_config(NEMO_CONFIG)
        assert _hermes()["revision"] == training.revision, (
            "The Hermes serving pin and training pin have drifted apart. An "
            "adapter served against different base weights than it was trained "
            "on degrades silently."
        )

    def test_hermes_base_model_matches_the_training_config(self):
        assert _hermes()["base_model"] == load_model_config(NEMO_CONFIG).base_model

    def test_hermes_revision_is_the_verified_sha(self):
        assert _hermes()["revision"] == HERMES_PINNED_SHA
        assert is_pinned_revision(_hermes()["revision"])

    def test_hermes_ships_no_adapter_weights_path_by_default(self):
        assert _hermes()["adapter_path"].startswith("${env:")

    def test_hermes_records_the_frozen_adapter_identity(self):
        deployment = _hermes()
        assert deployment["adapter_sha256"] == HERMES_ADAPTER_SHA
        assert deployment["source_checkpoint"] == "checkpoint-200"
        assert deployment["trainable_parameters"] == 57_016_320

    def test_hermes_records_the_sealed_dataset(self):
        deployment = _hermes()
        assert deployment["dataset_version"] == "kleos-policy-v0.0.6"
        assert len(deployment["dataset_sha256"]) == 64

    def test_hermes_decoding_matches_the_evaluated_conditions(self):
        generation = _hermes()["generation"]
        assert generation["do_sample"] is False
        assert generation["temperature"] == 0.0

    def test_hermes_states_the_tokenizer_contract_explicitly(self):
        tokenizer = _hermes()["tokenizer"]
        # Not absent, not inherited from a library default: stated. v0.0.6
        # trained and evaluated with the Mistral regex unpatched.
        assert tokenizer["fix_mistral_regex"] is False
        assert tokenizer["source"] == "frozen_package"

    def test_hermes_pins_the_compute_dtype_it_was_evaluated_in(self):
        # v0.0.6 trained and evaluated on a T4, where `auto` meant float16. On any
        # newer GPU (ZeroGPU's Blackwell included) `auto` means bfloat16, so the
        # serving record must say float16 outright.
        runtime = _hermes()["runtime"]
        assert runtime["compute_dtype"] == "float16"
        assert runtime["quantization_mode"] == "nf4"
        assert runtime["double_quant"] is True

    def test_hermes_carries_its_measured_limitations(self):
        notes = " ".join(_hermes()["notes"]).lower()
        for expected in ("json", "abstain", "rephrasing"):
            assert expected in notes, (
                "A serving record must carry the limitations that were measured, "
                "so a consumer cannot discover them in production."
            )


class TestOtherConfigsDidNotInheritTheSha:
    """The verified sha belongs to ONE checkpoint and must not be copy-pasted."""

    def test_no_model_config_mixes_up_the_two_shas(self):
        for path, own in (
            (MINISTRAL_CONFIG, PINNED_SHA),
            (NEMO_CONFIG, HERMES_PINNED_SHA),
        ):
            assert load_model_config(path).revision == own

    def test_no_other_model_config_uses_the_hermes_sha(self):
        for path in sorted((CONFIGS_DIR / "models").glob("*.yaml")):
            if path.name == "mistral_nemo_12b.yaml":
                continue
            assert load_model_config(path).revision != HERMES_PINNED_SHA, (
                f"{path.name} carries Mistral-Nemo's commit sha. That sha "
                "identifies a different checkpoint and would load the wrong "
                "weights or fail outright."
            )

    def test_no_other_model_config_uses_the_ministral_sha(self):
        for path in sorted((CONFIGS_DIR / "models").glob("*.yaml")):
            if path.name == "ministral_8b.yaml":
                continue
            revision = load_model_config(path).revision
            assert revision != PINNED_SHA, (
                f"{path.name} carries Ministral-8B's commit sha. That sha "
                "identifies a different checkpoint and would load the wrong "
                "weights or fail outright."
            )


class TestModelCardStatesRevisionHonestly:
    """A card must not imply reproducibility the release cannot deliver."""

    @staticmethod
    def _card(revision: str) -> str:
        manifest = ExperimentManifest(experiment_id="test-run")
        manifest.model = {
            "base_model": "mistralai/Ministral-8B-Instruct-2410",
            "revision": revision,
        }
        return build_model_card(repo_id="user/kleos-test", manifest=manifest)

    def test_a_pinned_card_records_the_sha_in_the_load_snippet(self):
        card = self._card(PINNED_SHA)
        assert "revision=REVISION" in card
        assert PINNED_SHA in card
        assert "moving pointer" not in card

    def test_an_unpinned_card_warns_that_weights_are_unidentifiable(self):
        # This is the case the v0.0.6 run falls into: it was trained with 'main'
        # and the resolved commit is not recoverable from the artifacts.
        card = self._card("main")
        assert "moving pointer" in card
        assert "not recoverable" in card

    def test_the_tokenizer_is_loaded_from_the_base_at_the_same_revision(self):
        card = self._card(PINNED_SHA)
        assert "AutoTokenizer.from_pretrained(BASE, revision=REVISION)" in card
