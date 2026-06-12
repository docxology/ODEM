# Visualization and Animation

ODEM supports three visualization levels.

## Static Plot Dispatcher

The legacy `PlotDispatcher` renders detailed PDFs from a run bundle:

```bash
python main.py plot results/RUN_ID --static
```

It reads saved raw arrays and writes into `plots/`.

## Diagnostic Dashboard

The compact dashboard renders a PNG containing:

- VFE over time,
- cumulative free action,
- accuracy/complexity balance,
- first-dimension state tracking.

```bash
python main.py plot results/RUN_ID --dashboard
```

## State Animation

Animations compare true/noisy state trajectories with inferred state
trajectories:

```bash
python main.py plot results/RUN_ID --animations
```

The default output is `animations/state_estimation.gif`. The animation helper
uses the saved `x_noisy.npy` and `gen_x_estimates.npy` arrays, so it can be run
after the numerical experiment completes.
