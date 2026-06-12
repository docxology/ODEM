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
- `ExperimentRunner` runs a selected combo range.
- `RunBundle` writes raw arrays, JSON metadata, and checksums.
- Analysis and visualization functions read from run bundles rather than from
  in-memory state.

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
