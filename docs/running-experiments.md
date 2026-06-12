# Running Experiments

The default entrypoint is still:

```bash
python main.py run --config parameters.yaml
```

For a quick smoke run:

```bash
python main.py run --config configs/smoke.yaml --quiet-progress --no-static-plots
```

For a local slice:

```bash
python main.py run --start-index 10 --end-index 20
```

For a run without static PDFs:

```bash
python main.py run --max-combos 1 --no-static-plots
```

For animations:

```bash
python main.py run --max-combos 1 --animations
```

For reports after a run:

```bash
python main.py report results/RUN_ID
python main.py report results
```

## Python API

```python
from pathlib import Path
from odem.experiment import ExperimentRunner, RunnerOptions

options = RunnerOptions(
    config_path=Path("parameters.yaml"),
    results_dir=Path("results"),
    max_combos=1,
    tqdm_disable=True,
)
result = ExperimentRunner(options).run()
```

The runner catches per-combo failures, logs tracebacks, and continues the sweep.
This is important for large sweeps where one unstable parameter setting should
not destroy all completed work.
