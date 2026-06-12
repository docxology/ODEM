# Testing

Run the default suite:

```bash
pytest -q
```

The tests are intentionally fast and deterministic. They cover:

- YAML sweep validation and expansion,
- run-bundle manifests and checksums,
- run and sweep summaries,
- dashboard and animation creation from raw arrays,
- CLI help behavior without running experiments,
- import-safe `main.py`,
- precision/log-precision round trips,
- colored-noise kernel degeneracy,
- legacy initializer compatibility,
- SLURM slice calculation.

Heavy numerical sweeps should remain opt-in. Use `configs/smoke.yaml` or
`--max-combos` for smoke tests and reserve full sweeps for release or
publication gates.
