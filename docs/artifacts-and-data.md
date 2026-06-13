# Artifacts and Data

Every run bundle is designed to be auditable and machine-readable.

## Core Metadata

- `combo.json`: the exact expanded combo in parameter tuple order.
- `snapshot.json`: structured model, optimizer, prior, noise, free-action, and
  MSE metadata.
- `manifest.json`: run ID, combo index, status, metadata, artifact paths, byte
  counts, and SHA-256 checksums.

## Raw Arrays

The runner saves raw arrays for:

- `vfe`, `accuracy`, `complexity`,
- `gen_sensations`, `gen_predictions`, `gen_x_estimates`,
- `x_clean`, `x_noisy`, `y`,
- `theta`, `lambda_x`, `lambda_y`,
- posterior covariance traces,
- white and colored noise,
- sigma and precision schedules.

These `.npy` files are the source of truth for downstream figures and analyses.

## Requested Visual Artifacts

When static plotting is requested, completed-run validation requires the core
PDF set in `plots/` to be recorded in the manifest. Runs created with
`--no-static-plots` are valid without those PDFs. Dashboards, sweep figures,
gallery pages, and GIFs are checksum-verified whenever they are recorded.
Plots are derived artifacts. Use `odem.arrays` for safe loading, numeric
finite checks, and shared state-estimate shape validation.

## Bundle API

```python
import numpy as np
from odem.artifacts import RunBundle
from odem.validation import validate_run_bundle

bundle = RunBundle.create("results", combo_index=0)
bundle.save_array("vfe", np.array([1.0, 2.0]))
bundle.save_json("snapshot", {"fa": 3.0})
bundle.write_manifest(status="created")

validation = validate_run_bundle(bundle.path, require_completed=False)
assert validation.valid
```

Completed runner-produced bundles are stricter than ad hoc working bundles:
`validate_run_bundle(path)` requires `status == "completed"`, `combo.json`,
`snapshot.json`, every raw array emitted by the runner, finite checksum/byte
records, and readable `.npy` files. Nonfinite numeric values are reported as
validation warnings and make summaries ineligible for ranking.
