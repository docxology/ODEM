# ODEM Agent Instructions

These instructions apply to this repository. They sit below user-level
instructions and above general assumptions about the project.

## Project Surface

ODEM is a Python/PyTorch research codebase for Online Dynamic Expectation
Maximisation and Online Generalised Predictive Coding. The stable orchestration
surface is the `odem` package:

- `odem.config` validates YAML sweeps and expands `ExperimentCombo` objects.
- `odem.model_registry` resolves generative process, dynamics, and likelihood
  callables with clear validation errors.
- `odem.slicing` owns explicit combo slices and SLURM array normalization.
- `odem.experiment` runs selected combinations and writes run bundles.
- `odem.run_outputs` defines the ODEM output tuple schema, raw-array
  names, derived precision schedules, and snapshot-safe scalars.
- `odem.artifacts` writes raw arrays, metadata, manifests, and checksums.
- `odem.validation` validates completed bundle status, required files, bytes,
  SHA-256 hashes, array-shape contracts, recorded visual artifacts, and numeric
  artifact warnings.
- `odem.arrays` owns safe `.npy` loading, finite checks, and state-estimate
  shape contracts used by analysis and visualization.
- `odem.numerics` owns shared quadratic forms, finite tensor checks, and safe
  covariance-to-precision inversion.
- `odem.analysis` summarizes completed bundles and excludes invalid/nonfinite
  rows from ranking.
- `odem.plot_style` and `odem.plot_data` provide shared static-figure style and
  finite plotting data helpers.
- `odem.visual_checks` validates recorded PNG, GIF, and PDF artifact integrity.
- `odem.visualization` renders dashboards, sweep figures, and GIF animations.
- `odem.summary_schema` defines summary, CSV, and sweep-report fields shared by
  analysis, reporting, and tests.
- `odem.reporting` writes Markdown/HTML/JSON/CSV reports and static artifact
  gallery pages from validated data.
- `odem.release_evidence` archives and verifies local release manifests,
  reports, tarballs, visual assets, and gallery links.
- `odem.cli` backs both `python main.py` and `python -m odem.cli`.

The `algorithms/`, `functions/`, and `data_utils/` modules are load-bearing
numerical code. Prefer wrapping or testing them from the `odem` package instead
of rewriting their numerical contracts casually. Report and visualization
workflows should route through `odem.reporting` and `odem.visualization`.

## Commands

Use Python 3.11-3.13. The repo’s expected verification commands are:

```bash
pytest -q
python -m compileall -q .
git diff --check
python main.py validate --config parameters.yaml
python main.py validate --config configs/smoke.yaml
python main.py run --config configs/smoke.yaml --results-dir /tmp/odem-smoke/results --logs-dir /tmp/odem-smoke/logs --max-combos 1 --no-static-plots --dashboard --animations --no-progress
python main.py summarize /tmp/odem-smoke/results --output-json /tmp/odem-smoke/summary.json --output-csv /tmp/odem-smoke/summary.csv
python main.py report /tmp/odem-smoke/results --output-dir /tmp/odem-smoke/reports
python scripts/benchmark_visuals.py --run-dir /tmp/odem-smoke/results/RUN_ID --output-json /tmp/odem-smoke/visual_benchmark.json --max-dashboard-seconds 20 --max-animation-seconds 30
python scripts/archive_release_evidence.py --results-dir /tmp/odem-smoke/results --output-dir /tmp/odem-smoke/release --expected-combos 1
python scripts/verify_release_evidence.py --release-dir /tmp/odem-smoke/release --expected-combos 1
python -m pip wheel . --no-deps -w /tmp/odem-wheel
```

`parameters.yaml` currently expands to 3024 combinations.
`configs/smoke.yaml` must remain a single-combination fast integration config.
The full local release gate is:

```bash
python scripts/run_release_sweep.py --config parameters.yaml --results-dir /tmp/odem-full/results --logs-dir /tmp/odem-full/logs --release-dir /tmp/odem-full/release --expected-combos 3024 --chunk-size 84 --workers 2 --best-run-animation --force --no-progress
python scripts/verify_release_evidence.py --release-dir /tmp/odem-full/release --expected-combos 3024
```

## Development Rules

- Keep source-of-truth data in run-bundle raw `.npy` arrays and JSON metadata;
  treat plots, reports, dashboards, and GIFs as derived artifacts.
- Do not commit local run outputs under `results/`, `logs/`, `/tmp/odem-smoke`,
  wheel output directories, caches, or build artifacts unless explicitly asked.
- Preserve strict validation behavior: incomplete manifests, missing required
  arrays, checksum mismatches, stale byte counts, invalid slices, empty axes,
  string booleans, and nonpositive white-noise sigmas should fail loudly.
- Runner strict mode is the default. Use `--allow-partial` only when partial
  sweep outputs are intentionally acceptable.
- When documenting commands, prefer `--no-progress`, `--output-json`, and
  `--output-csv`.
- Add tests before behavior changes. For docs updates, extend
  `tests/test_docs_contract.py` when the claim can be checked cheaply.

## Documentation Contract

Keep `README.md`, this file, and `docs/*.md` aligned with live code. Before
claiming documentation is accurate, check:

- documented local links resolve,
- documented CLI subcommands and options appear in `python -m odem.cli --help`
  or the relevant subcommand help,
- Python snippets in `docs/api-reference.md` execute in an isolated workspace,
- count-bearing claims are rederived from `python main.py validate`, not copied
  from memory.
