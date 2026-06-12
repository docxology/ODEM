# ODEM Documentation

This documentation is organized around the workflow of an experiment and the
maintainer contracts that keep the workflow reproducible:

1. Define a validated parameter sweep in YAML.
2. Run one or more combinations through the ODEM numerical loop.
3. Save every raw output into a manifest-backed run bundle.
4. Render static figures, dashboards, and optional animations.
5. Summarize runs and sweeps into reproducible analysis tables.
6. Publish Markdown/HTML reports for single runs or whole sweeps.

## Workflow Docs

- [Configuration](configuration.md)
- [Running Experiments](running-experiments.md)
- [Artifacts and Data](artifacts-and-data.md)
- [Visualization and Animation](visualization-and-animation.md)
- [Analysis](analysis.md)
- [Reports](reports.md)
- [Reproducibility and Parallelism](reproducibility-and-parallelism.md)
- [API Reference](api-reference.md)

## Maintainer Docs

- [Architecture](architecture.md)
- [Maintainer Guide](maintainer-guide.md)
- [Validation Contract](validation-contract.md)
- [Release Evidence](release-evidence.md)
- [CI and Benchmarks](ci-and-benchmarks.md)
- [Test Strategy](test-strategy.md)
- [Testing](testing.md)

## Package Boundaries

The public package is intentionally small:

- `odem.config` loads and validates sweeps.
- `odem.model_registry` resolves model function names to callables.
- `odem.slicing` resolves local and SLURM combo index ranges.
- `odem.experiment` runs combos and writes bundles.
- `odem.run_outputs` defines the ODEM output tuple, raw arrays, derived
  precision schedules, and snapshot scalars.
- `odem.arrays` centralizes safe `.npy` loading, finite checks, and
  state-estimate shape validation.
- `odem.numerics` centralizes quadratic forms, finite tensor checks, and safe
  precision inversions.
- `odem.artifacts` manages raw data, metadata, checksums, and manifests.
- `odem.validation` verifies completed bundle status, required artifact files,
  byte counts, SHA-256 hashes, and nonfinite numeric warnings.
- `odem.visualization` renders plot outputs from saved arrays.
- `odem.analysis` summarizes completed run directories.
- `odem.summary_schema` defines the shared summary/report/CSV field contract.
- `odem.plot_style` and `odem.plot_data` provide shared static-visual helpers.
- `odem.reporting` creates Markdown/HTML reports, sweep figures, and static
  artifact galleries.
- `odem.release_evidence` archives and verifies local release evidence.
- `odem.cli` exposes those capabilities through `python main.py` or `odem`.

The `algorithms/` and `functions/` modules remain numerical internals for
direct research work, while workflow code should prefer the `odem` package
boundaries. Reports and plots are generated through `odem.reporting` and
`odem.visualization`.

The root [TODO](../TODO.md) points to the local release-evidence workflow and
does not track completed implementation work.
