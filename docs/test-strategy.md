# Test Strategy

ODEM uses fast contract tests for most behavior and a single real smoke run for
runtime proof.

## Test Layers

- Unit tests cover parsing, slicing, array loading, schema constants, numeric
  helpers, and model lookup.
- Contract tests cover manifests, checksums, static PDF requirements, report
  fields, docs links, CLI options, release evidence, and script interfaces.
- Visual tests check nonblank PNGs, compact sparse plots, and GIF creation.
- Smoke tests run `configs/smoke.yaml` end to end without expanding the full
  sweep.

## Fixtures

Shared fixtures live in `tests/fixtures/`. Prefer fixture bundles for pure
contract tests and reserve real subprocess runs for CLI, release-script, or
smoke behavior that cannot be proven in process.

## Slow Gates

Full 3024-combo release runs are slow gates and must stay outside `pytest`.
Use `scripts/run_release_sweep.py` for release evidence, and keep pytest focused
on dry-run chunk coverage, release-verifier contracts, and one smoke execution.

## When to Add Contract Tests

Add contract tests whenever a change affects public CLI commands, run-bundle
artifact names, manifest semantics, report fields, docs examples, CI scripts,
or generated visualization guarantees.
