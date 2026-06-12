# API Reference

## `odem.config`

```python
from odem.config import load_sweep

sweep = load_sweep("parameters.yaml")
sweep.combination_count
sweep.combos[0].to_legacy_tuple()
```

Use this for validated configuration loading.

## `odem.experiment`

```python
from odem.experiment import ExperimentRunner, RunnerOptions

result = ExperimentRunner(RunnerOptions(max_combos=1)).run()
```

Use this for running selected combinations and writing bundles.

## `odem.artifacts`

```python
from odem.artifacts import RunBundle

bundle = RunBundle.create("results", combo_index=0)
bundle.save_array("vfe", values)
bundle.write_manifest(status="completed")
```

Use this for reproducible raw-data and metadata persistence.

## `odem.analysis`

```python
from odem.analysis import summarize_run, summarize_sweep

run_summary = summarize_run("results/RUN_ID")
sweep_rows = summarize_sweep("results")
```

Use this for machine-readable post-run summaries.

## `odem.visualization`

```python
from odem.visualization import render_diagnostic_dashboard, create_state_animation

render_diagnostic_dashboard("results/RUN_ID")
create_state_animation("results/RUN_ID")
```

Use this to render plots from saved arrays.

## `odem.reporting`

```python
from odem.reporting import create_run_report, create_sweep_report

create_run_report("results/RUN_ID")
create_sweep_report("results")
```

Use this to generate Markdown/HTML reports and sweep-summary assets.
