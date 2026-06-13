# API Reference

## Stable Imports

The package root exports the common workflow API:

```python
from odem import (
    OUTPUT_NAMES,
    SUMMARY_CSV_FIELDNAMES,
    RunBundle,
    RunOutputs,
    archive_release_evidence,
    extract_state_estimate,
    load_sweep,
    resolve_model_functions,
    resolve_combo_slice,
    summarize_run,
    validate_run_bundle,
    verify_release_evidence,
)

assert "vfe" in OUTPUT_NAMES
assert callable(load_sweep)
assert callable(summarize_run)
assert callable(validate_run_bundle)
assert callable(archive_release_evidence)
assert callable(verify_release_evidence)
assert callable(extract_state_estimate)
assert callable(resolve_combo_slice)
assert callable(resolve_model_functions)
assert RunOutputs is not None
assert RunBundle is not None
assert "free_action" in SUMMARY_CSV_FIELDNAMES
```

Submodules such as `odem.config`, `odem.slicing`, `odem.experiment`,
`odem.arrays`, `odem.artifacts`, `odem.validation`, `odem.analysis`,
`odem.summary_schema`, `odem.visualization`, `odem.visual_checks`,
`odem.reporting`, and `odem.release_evidence` are also supported for callers
that need narrower imports.

## `odem.config`

```python
from pathlib import Path
from odem.config import load_sweep

repo_root = Path.cwd() if (Path.cwd() / "parameters.yaml").exists() else Path(globals().get("repo_root", Path.cwd()))
sweep = load_sweep(repo_root / "configs" / "smoke.yaml")
sweep.combination_count
sweep.combos[0].to_parameter_tuple()
```

Use this for validated configuration loading.

## `odem.slicing`

```python
from odem.slicing import resolve_combo_slice

assert resolve_combo_slice(10, start_index=2, end_index=5) == (2, 5)
assert resolve_combo_slice(10, max_combos=3, use_slurm_env=False) == (0, 3)
```

Use this for reproducible local and SLURM-compatible combo selection.

## `odem.model_registry`

```python
from odem.model_registry import resolve_model_functions

resolved = resolve_model_functions(generative_process="glv", dynamics="lorenz", likelihood="identity")
assert callable(resolved.process_build)
assert callable(resolved.dynamics)
assert callable(resolved.likelihood)
```

Use this for validated model-name lookup.

## `odem.experiment`

```python
from pathlib import Path
from odem.experiment import ExperimentRunner, RunnerOptions

repo_root = Path.cwd() if (Path.cwd() / "configs" / "smoke.yaml").exists() else Path(globals().get("repo_root", Path.cwd()))
options = RunnerOptions(
    config_path=repo_root / "configs" / "smoke.yaml",
    results_dir=Path("results"),
    logs_dir=Path("logs"),
    max_combos=1,
    static_plots=False,
    dashboard=False,
    animations=False,
    tqdm_disable=True,
)
runner = ExperimentRunner(options)
```

Use this for running selected combinations and writing bundles.

## `odem.run_outputs`

```python
import numpy as np
from odem.run_outputs import OUTPUT_NAMES, RunOutputs

raw_outputs = tuple(np.ones(2) if name not in {"free_action", "mse"} else 1.0 for name in OUTPUT_NAMES)
outputs = RunOutputs.from_tuple(raw_outputs)
assert "vfe" in outputs.raw_arrays
```

Use this at the boundary between the numerical return tuple and named bundle
artifacts.

## `odem.artifacts`

```python
import numpy as np
from odem.artifacts import RunBundle
from odem.validation import validate_run_bundle

bundle = RunBundle.create("results", combo_index=0, run_id="example-run")
values = np.array([1.0, 2.0])
bundle.save_array("vfe", values)
bundle.write_manifest(status="created")
validation = validate_run_bundle(bundle.path, require_completed=False)
assert validation.valid
```

Use this for reproducible raw-data and metadata persistence.

## `odem.arrays`

```python
import numpy as np
from odem.arrays import all_numeric_finite, extract_state_estimate, state_estimate_issue

x = np.array([[0.0, 1.0], [1.0, 2.0]])
estimates = np.array([[[0.1, 0.9]], [[1.2, 1.8]]])
truth, estimate = extract_state_estimate(x, estimates)
assert truth.shape == estimate.shape
assert state_estimate_issue(x, estimates) is None
assert all_numeric_finite({"x": truth, "estimate": estimate})
```

Use this when loading raw arrays or validating state-estimate shapes outside the
CLI.

## `odem.numerics`

```python
import torch
from odem.numerics import ensure_finite_tensor, quadratic_form

vector = torch.tensor([1.0, 2.0], dtype=torch.float64)
matrix = torch.eye(2, dtype=torch.float64)
assert quadratic_form(vector, matrix).item() == 5.0
ensure_finite_tensor(vector, name="vector", phase="docs")
```

Use this for shared numerical guardrails.

## `odem.analysis`

```python
from odem.analysis import summarize_run, summarize_sweep

assert callable(summarize_run)
assert callable(summarize_sweep)
```

Use this for machine-readable post-run summaries.

## `odem.summary_schema`

```python
from odem.summary_schema import SUMMARY_CSV_FIELDNAMES, SWEEP_REPORT_COLUMNS

assert "valid_bundle" in SUMMARY_CSV_FIELDNAMES
assert "Free action" in SWEEP_REPORT_COLUMNS
```

Use this when tests, reports, or external tooling need the stable summary field
contract.

## `odem.plot_style` and `odem.plot_data`

```python
import numpy as np
from odem.plot_data import finite_series, short_label
from odem.plot_style import THEME

assert finite_series(np.array([1.0]), name="series").shape == (1,)
assert short_label("abcdef", max_len=4) == "abc..."
assert THEME.blue.startswith("#")
```

Use these for deterministic static visual outputs.

## `odem.visualization`

```python
from odem.visualization import render_diagnostic_dashboard, create_state_animation

assert callable(render_diagnostic_dashboard)
assert callable(create_state_animation)
```

Use this to render plots from saved arrays.

## `odem.reporting`

```python
from odem.reporting import create_run_report, create_sweep_report

assert callable(create_run_report)
assert callable(create_sweep_report)
```

Use this to generate Markdown/HTML reports and sweep-summary assets.

## `odem.release_evidence`

```python
from odem.release_evidence import ReleaseVerification, archive_release_evidence, verify_release_evidence

assert ReleaseVerification is not None
assert callable(archive_release_evidence)
assert callable(verify_release_evidence)
```

Use this to archive and verify local release evidence from completed sweep
outputs.
