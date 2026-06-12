# CI and Benchmarks

The CI workflow runs on Python 3.11, Python 3.12, and Python 3.13.

## CI Gates

The matrix workflow runs:

- `pytest -q`
- `python -m compileall -q .`
- `git diff --check`
- config validation for `parameters.yaml` and `configs/smoke.yaml`
- smoke run, summarize, report, benchmark, release archive, release verification
- wheel build

The smoke workflow keeps runtime bounded and uses `configs/smoke.yaml`.

## Visual Benchmarks

Use `scripts/benchmark_visuals.py` to measure dashboard and GIF generation for
one completed bundle:

```bash
python scripts/benchmark_visuals.py --run-dir results/RUN_ID --output-json /tmp/odem-visual-benchmark.json --max-dashboard-seconds 20 --max-animation-seconds 30
```

The script rewrites the dashboard and GIF, then refreshes manifest checksums so
the bundle remains valid.

## Release Script

Use `scripts/run_release_sweep.py` for full release evidence. Use `--dry-run`
first to verify chunks and output directories, then run with `--force` only
when replacing known local output directories. The runner invokes
`scripts/verify_release_evidence.py` after archiving; use that verifier directly
when checking an existing release directory.
