# Logos findings L-F1 to L-F8

> This is a frozen research record, written while the work was done and
> preserved as written. Editorial changes for public release are limited to
> replacing private storage paths and account identifiers with placeholders
> (marked as editor's notes) and retargeting links to documents that moved. No
> number, table, finding or decision was changed. Conventions:
> [docs/experiments/README.md](README.md).

Findings L-F1 to L-F5 were recorded in `docs/logos.md`, section 10 ("Findings
from this phase"), between 2026-09-24 and 2026-10-01, during the Logos v0.0.1
work, and are reproduced below as written. In them, "this page" is that version
of `docs/logos.md`, "this phase" is the Logos v0.0.1 work, and "the runbook" with
its numbered cells is now [the Logos v0.0.1 Colab runbook](../runbooks/logos-v001-colab.md).
Findings L-F6 to L-F8 are recorded with the Logos v0.0.2 run and indexed at the
end of this page.

## L-F1 to L-F5 (Logos v0.0.1)

**L-F1 — `Mistral3VLMAdapter`'s LoRA scoping does not work on transformers 5.**
VERIFIED with transformers 5.16.1 and peft 0.20.0 on a tiny
`Mistral3ForConditionalGeneration`. Two separate misses:

- Target validation keeps only modules whose names *start with* `language_model`.
  transformers 5 names them `model.language_model.*`, so nothing matches and
  attaching LoRA raises (loudly, at least).
- The vision tower is kept out of LoRA by listing `"vision_tower"` in PEFT's
  `exclude_modules`. PEFT matches list entries by exact name or `.suffix`, which no
  projection inside the tower has. With the adapter's own LoRA config, 7 of 14 LoRA
  modules landed in the vision tower.

Not fixed: the adapter serves only `configs/models/mistral_small_3_2.yaml`, which
no KLEOS run uses, and Logos does not touch it (its text view has no vision
modules). `tests/test_vlm_targeting_finding.py` holds both as strict xfails, so a
fix makes them fail until this entry is closed. A fix would match on a
`language_model.` path segment rather than a prefix, and pass PEFT a regex (or
explicit module names) instead of suffix lists.

**L-F2 — The tokenizer class depends on `config.json` being present.** VERIFIED
with transformers 5.16.1. From the Hub repository (config.json included),
`AutoTokenizer` builds `TokenizersBackend`, which gives the longest training example
as 442 tokens. From a folder of the tokenizer files alone it builds
`LlamaTokenizer`, which splits the same text into about 12% more tokens with
different ids: 492 for the same example. The first measurements for this page were
made that way and have been corrected. Training and evaluation load from the Hub,
and a tokenizer saved by training reloads as `TokenizersBackend` with identical
ids, so runs are consistent. **Any serving package for Logos must load its
tokenizer as `TokenizersBackend`**, never rebuild it from bare files; otherwise
serving tokenizes differently from training.

**L-F3 — Evaluation resume lost a killed session's work on Colab's Drive mount.**
OBSERVED on Colab, 2026-09-28 and 29; the cause is INFERRED. Fixed in
`src/kleos_models/evaluation/resume.py`.

- **What happened:** `PartialWriter` kept `<output>.partial.jsonl` open for the whole
  arm, appending and fsyncing each generation.
  - A Logos arm2 session answered more than 100 questions before Colab killed the
    runtime.
  - The next session printed `resume : 0 generation(s) recorded`. Only the identity
    header, which `start_partial` writes and closes on its own, had reached Drive.
- **Why (inferred):** the evidence fits a mount that uploads a file only once it is
  closed, whatever `fsync` does. Files that were written and then closed all
  survived the same kills: checkpoints, headers and `events.jsonl`, which is opened
  per event.
- **The fix:** the writer now opens and closes the file for every record.
  `tests/test_evaluation_resume.py` checks that no handle stays open between
  records. File handling is the only change; generation is untouched.
- **Confirmed on Colab, 2026-10-01:** with the fix (`ed5a987`), Logos' arm1
  session was killed partway through. The next session found its 172
  generations on Drive and resumed from them.

**L-F4 — A fresh start into a folder that already holds checkpoints would delete
the best one.** VERIFIED against transformers 5.16.1's `rotate_checkpoints`, on
2026-09-25. Not fixed.

- **The gap:** `scripts/train.py` has no guard against starting from step 0 in a
  run folder that already holds checkpoints, for example by running the training
  cell without `--resume-from-checkpoint`.
- **What would happen:** transformers sorts checkpoints by step number. It
  protects the newest and the *new* run's best, so the old run's best is not
  protected. At the new run's first save, with Logos' folder as it stood
  (checkpoint-25 beside 175, 225 and 250), rotation would have deleted
  `checkpoint-175`, the selected adapter.
- **What prevented it:** only the runbook's instruction to use cell 12.
- **The fix:** refuse a fresh start where checkpoints exist unless the operator
  says so explicitly.

**L-F5 — `validate_checkpoint` does not check optimizer or scheduler state.**
VERIFIED with transformers 5.16.1. Not fixed.

- **What it requires:** `trainer_state.json` and a weights file. That is enough to
  reject a checkpoint without weights, such as Logos' session-1 `checkpoint-125`
  (H-F4 again).
- **What it misses:** a checkpoint missing `optimizer.pt` or `scheduler.pt` passes.
  transformers then skips loading both without a warning, and the run continues
  with a fresh optimizer and a learning-rate schedule restarted from warmup.
- **This run:** both resume points were checked by hand first (D9).
- **The fix:** require the optimizer and scheduler files whenever a checkpoint is
  used to resume.
- **Also affected:** `training.log` is written through a `logging.FileHandler` held
  open the same way. The log of a training session that was killed may be missing
  from Drive. Its events, checkpoints and manifest are not affected.

The other findings of this phase, H-F11 to H-F14, are about how the evaluation
measures. They are recorded with the run they were found in:
[kleos-v006-mistralnemo12b-run1-report.md](kleos-v006-mistralnemo12b-run1-report.md#11-findings).

## L-F6 to L-F8 (Logos v0.0.2)

Recorded on 2026-10-06 in
[the Logos v0.0.2 run report, section 10](kleos-v007-ministral314breasoning-run1-report.md#10-findings),
which holds the evidence for each.

| Finding | What it records | Status, 2026-10-07 |
| --- | --- | --- |
| L-F6 | `device_map: auto` did not balance the layers across two T4s: GPU 0 took the embedding and 9 layers, GPU 1 took 31 layers, the norm and `lm_head` (10.16 GB measured against an estimated 8.53 GB) | Open. No effect on the run, which kept 4.0 GB spare |
| L-F7 | `finish_reason` misses an answer cut by the token budget after a closed trace: 6 of 349 answers read "stop" | Resolved in serving: a served reply says "length" whenever the budget filled or the thinking never closed. The evaluation code and its stored results are unchanged |
| L-F8 | The evaluation reports no progress while it runs | Open |
