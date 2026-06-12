# Reproducibility and Parallelism

## Reproducibility Defaults

The runner seeds Python, NumPy, and PyTorch through
`functions.initialisation.seed.generate`. Raw arrays are saved before plots, so
figures can be regenerated from the bundle.

Important reproducibility surfaces:

- `parameters.yaml`,
- `combo.json`,
- `snapshot.json`,
- raw `.npy` arrays,
- `manifest.json` checksums,
- test output.

## Parallel Execution

For local slicing:

```bash
python main.py run --start-index 0 --end-index 100
python main.py run --start-index 100 --end-index 200
```

For SLURM arrays, set `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT`. The
runner splits combo indices with `numpy.array_split`, matching the legacy
behavior while avoiding reliance on positional `sys.argv`.

Use `--no-slurm` to ignore SLURM environment variables during local debugging.

## Failure Handling

Per-combo exceptions are captured in `logs/*/failed_combos.json` with tracebacks.
Completed combos remain in their own run bundles and can be analyzed
independently.
