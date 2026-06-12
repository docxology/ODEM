# Artifacts and Data

Every run bundle is designed to be auditable and machine-readable.

## Core Metadata

- `combo.json`: the exact expanded combo in legacy tuple order.
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
Plots are derived artifacts.

## Bundle API

```python
import numpy as np
from odem.artifacts import RunBundle

bundle = RunBundle.create("results", combo_index=0)
bundle.save_array("vfe", np.array([1.0, 2.0]))
bundle.save_json("snapshot", {"fa": 3.0})
bundle.write_manifest(status="completed")
```
