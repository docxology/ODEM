# Architecture

ODEM has three layers.

## Numerical Core

The core numerical loop lives in `algorithms/ODEM.py` and calls:

- `D_step.step` for fast hidden-state inference,
- `M_step.step` for precision updates,
- `E_step.step` for parameter updates,
- `functions/vfe_calculation/*` for variational free-energy terms,
- `functions/noise_generation/*` for process and observation noise schedules.

The core returns raw arrays and tensors. It does not decide how a research run
should be organized on disk.

## Public Orchestration Layer

The `odem` package is the stable orchestration layer:

- `ParameterSweep` converts YAML into validated `ExperimentCombo` objects.
- `odem.model_registry` validates and resolves configured model callables.
- `resolve_combo_slice` in `odem.slicing` applies local bounds, `--max-combos`,
  and zero- or one-based SLURM array task normalization.
- `ExperimentRunner` runs a selected combo range.
- `odem.run_outputs` maps the algorithm return tuple into named raw arrays,
  derived precision schedules, and scalar metadata.
- `RunBundle` writes raw arrays, JSON metadata, and checksums.
- `validate_run_bundle` verifies completed bundle manifests, required files,
  byte counts, hashes, and nonfinite numeric artifacts.
- `odem.arrays` gives analysis and visualization one shared implementation for
  safe array loading, finite checks, and `x_noisy`/`gen_x_estimates[:, 0, :]`
  shape validation.
- `odem.numerics` provides shared finite checks and stable linear-algebra
  helpers used by VFE, posterior covariance, and the ODEM loop.
- `odem.summary_schema` keeps Markdown, CSV, JSON, and tests on the same field
  names.
- `odem.plot_style` and `odem.plot_data` keep dashboard, sweep, and gallery
  visual outputs compact and deterministic.
- `odem.visual_checks` keeps PNG, GIF, and PDF artifact integrity checks shared
  between bundle validation and release verification.
- Analysis, reporting, and visualization functions read from run bundles rather
  than from in-memory state.
- `odem.release_evidence` packages and verifies local release evidence from
  completed sweep outputs, reports, visual assets, and archive metadata.

This separation keeps numerical research code composable while making runs
repeatable and auditable.

## Artifact Layer

Each experiment combination gets one run directory under `results/`. A run
directory is self-describing because it contains:

- `combo.json`,
- `snapshot.json`,
- raw `.npy` outputs,
- `manifest.json`,
- optional `plots/`,
- optional `animations/`.

The manifest records file size and SHA-256 for each artifact created through
the bundle API.

Report and plot workflows should use `odem.reporting` and
`odem.visualization`. Release archive workflows should use
`odem.release_evidence`.
