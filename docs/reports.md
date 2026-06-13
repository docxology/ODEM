# Reports

Reports are derived artifacts built from run bundles and sweep summaries.
`odem.summary_schema` defines the shared JSON/CSV/Markdown field names used by
the reporting code and docs-contract tests.

## Run Reports

```bash
python main.py report results/RUN_ID
```

Outputs:

- `reports/run_report.md`
- `reports/run_report.html`
- `reports/run_summary.json`
- `reports/index.html`

Run reports validate the source bundle before writing report files. They include
free action, MSE, accuracy/complexity totals, finite-output status, bundle
validation status, ranking eligibility, final posterior values, model metadata,
and manifest artifact entries.

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
- `reports/index.html`

Sweep reports rank only runs with finite free-action values, valid manifests,
and compatible state-estimate shapes. Invalid or diagnostic-only rows remain in
the JSON/CSV/Markdown outputs with their validation issues, but they cannot be
selected as the best run. The PNG is always written; when no rows are rankable,
it contains a deterministic empty-state panel.
Each report directory also includes a static no-dependency artifact gallery at
`index.html` that links Markdown/HTML/JSON/CSV files and embeds available
PNG/GIF visual artifacts.

## Programmatic API

```python
from odem.reporting import create_run_report, create_sweep_report

run_paths = create_run_report("results/RUN_ID")
sweep_paths = create_sweep_report("results")
```

Both functions return paths to the generated artifacts.
`ReportPaths.gallery_path` points to `index.html`, and `ReportPaths.assets`
lists visual assets linked by the gallery.

Report generation should use `odem.reporting`.
