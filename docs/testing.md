# Testing

Run the default suite:

```bash
pytest -q
```

Selected shell examples are executed by the docs contract:

```bash
# docs-contract: run
python main.py validate --config configs/smoke.yaml
```

```bash
# docs-contract: run
python scripts/benchmark_visuals.py --help
```

```bash
# docs-contract: run
python scripts/archive_release_evidence.py --help
```

```bash
# docs-contract: run
python scripts/verify_release_evidence.py --help
```

```bash
# docs-contract: run
python scripts/run_release_sweep.py --help
```

The tests are intentionally fast and deterministic. They cover:

- YAML sweep validation and expansion,
- strict boolean, non-empty-axis, and positive-noise validation,
- run-bundle manifests, required raw arrays, checksums, and tamper detection,
- run-output schema mapping, model registry validation, and numerical finite
  guards,
- run and sweep summaries with finite-ranking and shape-contract checks,
- dashboard, sweep-summary, gallery, and animation creation from validated raw
  arrays,
- release-evidence archive and verification contracts,
- CLI help and documented option behavior without running full sweeps,
- documentation link, snippet, Python, and shell command contracts,
- import-safe `main.py`,
- precision/log-precision round trips,
- VFE quadratic-form and covariance-posterior guards,
- colored-noise kernel degeneracy,
- initializer/config-loader parity,
- SLURM slice calculation,
- real smoke-run output contracts for raw data, reports, dashboards, and GIFs.

Heavy numerical sweeps should remain opt-in. Use `configs/smoke.yaml` or
`--max-combos` for smoke tests and reserve full sweeps for release or
publication gates.
