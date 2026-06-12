# Online Generalised Predictive Coding through ODEM

A Python/PyTorch implementation of **Online Dynamic Expectation Maximisation
(ODEM)** for **Online Generalised Predictive Coding** under the **Free Energy
Principle (FEP)**.

ODEM performs online data assimilation through separated temporal scales,
jointly estimating:

- hidden dynamic states,
- unknown model parameters,
- state and observation uncertainty through precision learning.

The repository now exposes both the original `python main.py` entrypoint and a
modular `odem` package for validated configuration, reproducible run bundles,
analytics, static visualizations, and optional animations.

<p align="center">
  <img src="example/lorenz-GM-kx=3.png" alt="Lorenz GM kx=3" width="700"/>
  <br>
  <em>
  Figure 1: State estimation using a Lorenz generative model against a
  Generalised Lotka-Volterra generative process under model mismatch.
  </em>
</p>

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python main.py validate --config parameters.yaml
python main.py run --config configs/smoke.yaml --quiet-progress --no-static-plots
python main.py summarize results
pytest -q
```

The default `parameters.yaml` expands to a larger sweep. Use
`configs/smoke.yaml` for quick local verification, and use `--max-combos`,
`--start-index`, and `--end-index` for segmented execution.

## Features

- Validated YAML parameter sweeps via `odem.config`.
- Backward-compatible `python main.py` CLI.
- Manifest-backed run bundles with raw `.npy` arrays, JSON metadata, checksums,
  static plots, compact dashboards, and optional GIF animations.
- Sweep summarization and Markdown/HTML reports ordered by free action with
  finite-output checks.
- SLURM array compatibility plus explicit local index slicing.
- Focused pytest suite covering configuration, artifacts, analysis,
  visualization, numerical transforms, and legacy compatibility.

## Repository Structure

```text
ODEM/
├── algorithms/          # Core D/E/M and ODEM numerical loop
├── functions/           # Generative models, VFE, noise, plotting utilities
├── odem/                # Public package: config, runner, artifacts, analysis, CLI
├── docs/                # Modular reference and workflow documentation
├── tests/               # Deterministic unit and smoke tests
├── example/             # Example rendered figure
├── parameters.yaml      # Main experiment sweep
├── main.py              # Backward-compatible CLI entrypoint
└── pyproject.toml       # Package and test metadata
```

## Documentation

Start with [docs/index.md](docs/index.md), then use the module that matches the
task:

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Running Experiments](docs/running-experiments.md)
- [Artifacts and Data](docs/artifacts-and-data.md)
- [Visualization and Animation](docs/visualization-and-animation.md)
- [Analysis](docs/analysis.md)
- [Reports](docs/reports.md)
- [Testing](docs/testing.md)
- [Reproducibility and Parallelism](docs/reproducibility-and-parallelism.md)
- [API Reference](docs/api-reference.md)

## Citation

If you use this codebase in research, cite:

```bibtex
@article{bazargani2026online,
  title={Online Generalised Predictive Coding},
  author={Bazargani, Mehran HZ and Urbas, Szymon and Razi, Adeel and Murphy, Thomas Brendan and Friston, Karl},
  journal={arXiv preprint arXiv:2605.02675},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
