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

Use Python 3.11-3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python main.py validate --config parameters.yaml
python main.py run --config configs/smoke.yaml --no-progress --no-static-plots
python main.py summarize results --output-json results/summary.json
pytest -q
```

The default `parameters.yaml` expands to a larger sweep. Use
`configs/smoke.yaml` for quick local verification, and use `--max-combos`,
`--start-index`, and `--end-index` for segmented execution.

## Features

- Validated YAML parameter sweeps via `odem.config`.
- `python main.py` CLI entrypoint.
- Shared raw-array loading and state-estimate shape contracts via
  `odem.arrays`.
- Central run-output schema and precision-schedule derivation via
  `odem.run_outputs`.
- Validated model lookup via `odem.model_registry` and shared numerical guards
  via `odem.numerics`.
- Validated manifest-backed run bundles with raw `.npy` arrays, JSON metadata,
  checksums, static plots, compact dashboards, and optional GIF animations.
- Dedicated `odem.validation` checks for completed bundle status, required
  artifacts, byte counts, SHA-256 hashes, array-shape contracts, recorded
  visual artifact integrity, and nonfinite numeric warnings.
- Shared visualization helpers for multi-panel diagnostic dashboards, sweep
  summary figures, state-estimation GIFs, and PNG/GIF/PDF integrity checks.
- Sweep summarization and Markdown/HTML reports ordered by free action with
  finite-output and bundle-validation checks, using the shared
  `odem.summary_schema` field contract.
- Static HTML artifact galleries that link report data, PNG/PDF assets, and GIF
  animations without adding runtime dependencies.
- Local release evidence archiving and verification via `odem.release_evidence`.
- SLURM array compatibility plus explicit local index slicing through
  `odem.slicing`.
- CI workflow coverage for Python 3.11, 3.12, and 3.13.
- Scripted visual runtime benchmarks plus local release-evidence archive and
  verification workflows.
- Focused pytest suite covering configuration, artifacts, analysis,
  visualization, numerical transforms, and source/module contracts.

## Repository Structure

```text
ODEM/
├── algorithms/          # Core D/E/M and ODEM numerical loop
├── functions/           # Generative models, VFE, noise, plotting utilities
├── odem/                # Public package: config, outputs, slicing, artifacts, analysis, reports, CLI
├── docs/                # Modular reference and workflow documentation
├── scripts/             # Benchmark and release-evidence utilities
├── tests/               # Deterministic unit and smoke tests
├── example/             # Example rendered figure
├── parameters.yaml      # Main experiment sweep
├── main.py              # CLI entrypoint
└── pyproject.toml       # Package and test metadata
```

## Documentation

Start with [docs/index.md](docs/index.md), then use the module that matches the
task:

- [Architecture](docs/architecture.md)
- [Maintainer Guide](docs/maintainer-guide.md)
- [Validation Contract](docs/validation-contract.md)
- [Release Evidence](docs/release-evidence.md)
- [CI and Benchmarks](docs/ci-and-benchmarks.md)
- [Test Strategy](docs/test-strategy.md)
- [Configuration](docs/configuration.md)
- [Running Experiments](docs/running-experiments.md)
- [Artifacts and Data](docs/artifacts-and-data.md)
- [Visualization and Animation](docs/visualization-and-animation.md)
- [Analysis](docs/analysis.md)
- [Reports](docs/reports.md)
- [Testing](docs/testing.md)
- [Reproducibility and Parallelism](docs/reproducibility-and-parallelism.md)
- [API Reference](docs/api-reference.md)
- [Scoped TODO](TODO.md)
- [Agent Instructions](AGENTS.md)

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
