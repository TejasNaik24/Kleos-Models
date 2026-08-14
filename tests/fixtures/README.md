# Test fixtures

Most tests build their data in-memory through the factories in `conftest.py`
(`make_example`, `make_eval_example`), which keeps each test's inputs visible
beside its assertions.

This directory exists for fixtures that must be files on disk — malformed JSONL
for loader error paths, for instance.

Fixtures shared with the pipeline itself live in `data/examples/`.
