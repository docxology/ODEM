# Reports

Reports are derived artifacts built from run bundles and sweep summaries.

## Run Reports

```bash
python main.py report results/RUN_ID
```

Outputs:

- `reports/run_report.md`
- `reports/run_report.html`
- `reports/run_summary.json`

Run reports include free action, MSE, accuracy/complexity totals, finite-output
status, final posterior values, model metadata, and manifest artifact entries.

## Sweep Reports

```bash
python main.py report results
```

Outputs:

- `reports/sweep_report.md`
- `reports/sweep_report.html`
- `reports/sweep_summary.json`
- `reports/sweep_summary.csv`
- `reports/sweep_summary.png`

Sweep reports rank runs by free action and keep MSE visible as a second
diagnostic. The PNG is intended as a quick scan of the frontier rather than a
replacement for raw arrays or full statistical analysis.

## Programmatic API

```python
from odem.reporting import create_run_report, create_sweep_report

run_paths = create_run_report("results/RUN_ID")
sweep_paths = create_sweep_report("results")
```

Both functions return paths to the generated artifacts.
