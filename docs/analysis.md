# Analysis

Analysis reads run bundles rather than live Python variables.

Summarize one run:

```bash
python main.py summarize results/RUN_ID
```

Summarize a sweep:

```bash
python main.py summarize results --json results/summary.json --csv results/summary.csv
```

The summary includes:

- run ID and combo index,
- number of time steps,
- free action,
- MSE recomputed from raw state arrays when available,
- total accuracy and complexity,
- final theta and lambda values,
- whether all numeric `.npy` outputs are finite.

Rows are sorted by free action so the best candidate appears first.

Reports build on the same summaries:

```bash
python main.py report results/RUN_ID
python main.py report results
```

Run reports include model metadata, free action, MSE, final posterior values,
finite-output status, and manifest artifacts. Sweep reports include ranked runs,
JSON/CSV summaries, and a sweep summary PNG.
