# Configuration

`parameters.yaml` defines a sweep. Scalar fields run once. List-valued sweep
fields expand by Cartesian product.

Validate before running:

```bash
python main.py validate --config parameters.yaml
```

The current default config expands to 3,024 combinations.

## Required Sections

- `priors.theta`: parameter prior means and standard deviations.
- `priors.lambda`: expected precisions and log-precision uncertainty.
- `optimizer`: D/E/M update schedules and covariance behavior.
- `gp`: generative process, time step, duration, and state-noise schedule.
- `gm`: generative model dynamics, likelihood, generalised-coordinate order,
  and observation-noise schedule.

## Validation Rules

The loader rejects configurations that violate important runtime assumptions:

- `kx` must equal `ky + 1`.
- `gp.dt` and `gp.T` must be positive.
- colored-noise kernels must be positive odd integers.
- noise schedule modes must be one of `linear`, `exp`, `sigmoid`, `log`, or
  `gaussian`.
- precision expectations and prior standard deviations must be positive.
- named generative process, dynamics, and likelihood functions must exist.

## Programmatic Loading

```python
from odem.config import load_sweep

sweep = load_sweep("parameters.yaml")
print(sweep.combination_count)
first_combo = sweep.combos[0]
```
