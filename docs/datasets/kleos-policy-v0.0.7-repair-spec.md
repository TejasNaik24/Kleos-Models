# `kleos-policy-v0.0.7` — repair specification

**For the private `kleos-training-data` repository, which generates KLEOS
releases.** This repository only consumes sealed releases and never generates or
edits one, so nothing here is implemented here. Written 2026-09-24 from an audit
of the sealed v0.0.6 release (aggregates only; no example content was read into
this document).

**Why a v0.0.7.** Hermes (Mistral-Nemo 12B) and Ministral-8B, trained on v0.0.6,
failed in the same ways, with the same consistency and the same abstention table
case for case. A larger base did not move either. The limits are in the data. KLEOS
Logos v0.0.1 still trains on v0.0.6 unchanged, so its comparison with Hermes
isolates the base model. Logos v0.0.2 would train on v0.0.7.

---

## What v0.0.6 gets wrong (measured)

| Problem | Measurement |
| --- | --- |
| Four decline labels never taught | `request_ambiguous` (30 test cases), `missing_input` (25), `stale_explicit_conflict` (20), `insufficient_separation` (3): 0 occurrences in train or validation, as a label or anywhere in answer text. They are exactly the 78 `confident: false` test cases. Ceilings for any model trained on v0.0.6: `deciding_factor` 0.7765, judgment 0.9255 |
| Deciding factor mostly unstated | 492 of 820 training answers (60%) have no "What decided it:" line. 0 of the 124 training declines carry a label. Prose answers always have one; bullets 290/390 and Slack 38/51 lack it |
| Family–label gaps | `brief.what_needs_a_decision` and `notif.deadline_vs_evidence` are labelled `impact` in test and only `evidence` in train. With the decline labels, 94/349 test cases (26.9%) need a family–label pairing training never shows |
| Conditional abstention is confounded | In `rec.verify_before_recommending` training, all 25 declines have evidence_quality weak, ambiguity high, presentation as given, and domain career or coursework; all 30 commits have mixed, medium, reversed, and projects or research. Test breaks the link. Both trained models declined 0/25 there. The family has 6 base scenarios in train |
| Axis hygiene | `conflicting_evidence` uses both JSON null (806 rows) and the string "none" (90); `workspace` mixes capitalisation and is null in 67.6% of rows; `entities`, `deadlines` and `source_type` are null in every row; several values fall outside the data contract's registry (`urgency: critical`, `evidence_quality: conflicting`, `conflicting_evidence: moderate/strong`, `format: slack_thread`); the `schema` perturbation kind is not a registered kind |
| No evidence ids | No prompt contains an id-like token, so citation-based measures have nothing to check against |
| Provenance gaps | `gate_report_hash: null`; `pipeline_version` names no generator commit |

---

## Invariants

A release that breaks any of these is not v0.0.7.

1. **`test.jsonl` is byte-identical to v0.0.6**: sha256
   `a4decaaf029b227366846d44018b0b1b669d21f068800a9f3175e15ec4783980`. The benchmark
   then rebuilds to `a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266`,
   and every v0.0.6 result stays directly comparable with every v0.0.7 one.
2. **JSON stays out of train and validation.** No training or validation prompt
   embeds a JSON array and no target is a JSON object, as in v0.0.6. The format
   holdout is what the benchmark measures.
3. **No test contamination.** No test `group_id`, scenario fingerprint or near-duplicate
   (Jaccard ≥ 0.85 over the conversation) in train or validation. v0.0.6: 0
   train–test near-duplicates; keep it at 0. New training scenarios must not be
   rewrites of test scenarios.
4. **Groups do not cross splits**, and every row carries `metadata.group_id`.
5. **Sealed and validated**: `RELEASE.lock` covers every file, the schema passes this
   repository's `scripts/validate_dataset.py`, and `contains_private_data` is false.

---

## Requirements

### R1 — Teach every label the benchmark asks for

- The four decline labels appear in train and validation as the deciding factor of
  declined answers, each in **at least 20 training and 5 validation examples**.
- **Every** training and validation answer, committed or declined, states its
  deciding factor on a "What decided it: <label>" line in the form the grader
  reads (`evaluation/graders.py`, `_DECIDED_BY`).
- `brief.what_needs_a_decision` and `notif.deadline_vs_evidence` include `impact`
  examples in training.

**Acceptance:** `extract_deciding_factor` recovers a label from 100% of training and
validation answers; each of the 9 benchmark options occurs at least 20 times in
training.

### R2 — De-confound conditional abstention

For every family where the right answer is sometimes to decline and sometimes to
commit (`rec.verify_before_recommending`, `rec.abstain_without_evidence`,
`notif.tied_signals`, `mem.unresolvable_conflict`):

- **Contrastive pairs.** For each base scenario, a decline and a commit version
  that share every nuisance axis (domain, presentation order, context length,
  ambiguity wording) and differ only in whether the evidence is sufficient.
- **At least 20 base scenarios per family** in training (v0.0.6: 6 in
  `rec.verify_before_recommending`, 3 in `rec.abstain_without_evidence`).

**Acceptance:** within each family's training split, no value of any nuisance axis
occurs with only one decision; and a depth-2 decision tree over the nuisance axes
alone predicts the family's decline decision no better than the majority class
plus 0.10.

### R3 — Axis and group hygiene

- One representation of "none" for `conflicting_evidence`; consistent `workspace`
  casing, null only where no workspace applies.
- Every axis value registered in the data contract (`docs/data-contract.md`), or the
  contract extended in this repository first.
- Perturbation kinds from the registered set (`constants.PERTURBATION_KINDS`).
- `entities` populated where a task has named entities, so entity leakage and
  entity generalisation become checkable.
- Every group label-identical: the same label, `confident`, top-1 and candidate set
  across its perturbations, as all 78 v0.0.6 test groups are.

**Acceptance:** `scripts/validate_dataset.py --dataset <release>` passes with no axis
warnings; a per-group check finds 0 groups with mixed labels.

### R4 — A separate hard set, never the frozen test

Conditional abstention with unseen nuisance combinations, entity shift and new
formats need a held-out set of their own: a separately named and hashed file (for
example `hard.jsonl`), in its own benchmark. It must never be merged into
`test.jsonl`, which invariant 1 freezes.

**Acceptance:** its own `RELEASE.lock` entry; `group_id`s and scenario fingerprints
disjoint from all three splits; 0 near-duplicates against `test.jsonl`.

### R5 — Provenance you can check on receipt

- `provenance.json` records the generator's commit, the quality-gate report and its
  non-null `gate_report_hash`, and the pinned data-contract commit.
- A composition report per split: counts by task, family, label, decision and
  every axis, so the checks above can be re-run without reading examples.

---

## What this repository checks on receipt

Before any training on v0.0.7:

```bash
python scripts/validate_dataset.py --dataset <v0.0.7 dir>
shasum -a 256 <v0.0.7 dir>/test.jsonl        # must equal a4decaaf…3980
python scripts/build_benchmark.py --dataset <v0.0.7 dir> --output /tmp/bench
shasum -a 256 /tmp/bench/benchmark.jsonl     # must equal a11ffad7…b266
```

Then the R1–R3 acceptance checks, as aggregates only, and the leakage report from
training's pre-flight (fatal findings 0, train–test near-duplicates 0).

**What stays comparable.** With the test split frozen, v0.0.7 moves only the
training side. Every model, whichever release trained it, is scored on the same
349 items, so a v0.0.7 gain is a data effect. Since the benchmark's labels are now
all learnable, the ceilings above lift to 1.0.
