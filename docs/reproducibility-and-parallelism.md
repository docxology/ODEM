# Reproducibility and Parallelism

## Reproducibility Defaults

The runner seeds Python, NumPy, and PyTorch through
`functions.initialisation.seed.generate`. Raw arrays are saved before plots, so
figures can be regenerated from the bundle.

Important reproducibility surfaces:

- `parameters.yaml`,
- `combo.json`,
- `snapshot.json`,
- the named output schema in `odem.run_outputs`,
- raw `.npy` arrays,
- `manifest.json` checksums,
- finite numerical guards in `odem.numerics`,
- test output.

## Parallel Execution

For local slicing:

```bash
python main.py run --start-index 0 --end-index 100
python main.py run --start-index 100 --end-index 200
```

For SLURM arrays, set `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT`.
`SLURM_ARRAY_TASK_MIN` is honored when present, so one-based arrays such as
`MIN=1, ID=1` select the first split. The runner splits combo indices with
`numpy.array_split` through `odem.slicing` while avoiding reliance on
positional `sys.argv`.

Use `--no-slurm` to ignore SLURM environment variables during local debugging.

## Release Evidence

Run and archive the full release sweep with the orchestrator:

```bash
python scripts/run_release_sweep.py --config parameters.yaml --results-dir /tmp/odem-full/results --logs-dir /tmp/odem-full/logs --release-dir /tmp/odem-full/release --expected-combos 3024 --chunk-size 84 --workers 2 --best-run-animation --force --no-progress
```

The runner chunks the sweep, writes reports, writes `release_evidence.json`,
creates `odem_release_evidence.tar.gz`, writes `release_progress.jsonl`, and
then verifies the archive. It exits nonzero when the completed valid run count
does not match `--expected-combos` or when release evidence fails verification.

Visual runtime checks can be run against any completed bundle:

```bash
python scripts/benchmark_visuals.py --run-dir results/RUN_ID --output-json /tmp/odem-visual-benchmark.json --max-dashboard-seconds 20 --max-animation-seconds 30
```

## Failure Handling

Per-combo exceptions are captured in `logs/*/failed_combos.json` with
tracebacks. Strict mode is the default: failures make the API raise after logs
are written and make the CLI exit nonzero. Completed combos remain in their own
run bundles and can be analyzed independently. Use `--allow-partial` only when
that partial-output behavior is intentional.
