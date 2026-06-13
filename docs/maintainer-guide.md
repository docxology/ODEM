# Maintainer Guide

This guide is for engineers changing ODEM internals while keeping run bundles,
reports, and validation reproducible.

## Source Boundaries

- `odem.config` owns YAML parsing, typed sweep expansion, boolean validation,
  model-name validation, and positive-noise checks.
- `odem.experiment` owns run orchestration, combo slicing, result persistence,
  requested-output metadata, and strict failure behavior.
- `odem.artifacts` and `odem.validation` own manifest records, byte counts,
  SHA-256 checksums, required raw arrays, and requested static PDF checks.
- `odem.analysis`, `odem.reporting`, and `odem.visualization` must read from
  saved bundles rather than live tensors.
- `algorithms/` and `functions/` remain numerical internals. Add safe
  extension points through `odem` modules before exposing new workflow behavior.

## Safe Extension Rules

Use these safe extension rules before adding new workflow behavior:

- Add new bundle artifacts through `RunBundle.record_file` or `RunBundle`
  save helpers so checksums stay current.
- Add new summary/report fields through `odem.summary_schema` before changing
  Markdown, CSV, or tests.
- Add new array loading or shape logic to `odem.arrays`; do not duplicate
  ad hoc `.npy` loading in analysis or visualization.
- Add new model names through `odem.model_registry` so config errors stay
  explicit.
- Keep full-sweep outputs outside git. Use `/tmp/odem-full` for local release
  evidence unless a release manager chooses another external storage target.

## Review Checklist

- Does the change preserve manifest-backed raw data as the source of truth?
- Does every generated PNG, GIF, PDF, JSON, CSV, or HTML artifact either stay
  unrecorded or have an updated manifest record?
- Do config and CLI docs use live option names from `python -m odem.cli --help`?
- Does the test layer match the risk: unit for helpers, contract for artifacts,
  smoke for one real run, release script for full-sweep evidence?
