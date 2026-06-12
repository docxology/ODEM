# ODEM Documentation

This documentation is organized around the workflow of an experiment:

1. Define a validated parameter sweep in YAML.
2. Run one or more combinations through the ODEM numerical loop.
3. Save every raw output into a manifest-backed run bundle.
4. Render static figures, dashboards, and optional animations.
5. Summarize runs and sweeps into reproducible analysis tables.
6. Publish Markdown/HTML reports for single runs or whole sweeps.

The public package is intentionally small:

- `odem.config` loads and validates sweeps.
- `odem.experiment` runs combos and writes bundles.
- `odem.artifacts` manages raw data, metadata, checksums, and manifests.
- `odem.visualization` renders plot outputs from saved arrays.
- `odem.analysis` summarizes completed run directories.
- `odem.reporting` creates Markdown/HTML reports and sweep figures.
- `odem.cli` exposes those capabilities through `python main.py` or `odem`.

The legacy `algorithms/` and `functions/` modules remain available for direct
research hacking, but new code should prefer the `odem` package boundaries.
