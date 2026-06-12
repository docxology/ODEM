# Validation Contract

Completed run bundles are valid only when the manifest and filesystem agree.

## Manifest Status

`validate_run_bundle(path)` requires manifest status `completed` by default.
Use `require_completed=False` only for tooling that is intentionally inspecting
created or partial bundles.

## Required Data

Completed runs require:

- `combo.json`
- `snapshot.json`
- every raw `.npy` output named by `odem.run_outputs.RAW_OUTPUT_NAMES`
- derived `x_pi_schedule.npy` and `y_pi_schedule.npy`

Every recorded artifact must stay inside the run directory, exist on disk, and
match the manifest byte count and SHA-256 hash.

Validation also enforces hard array-shape contracts used by downstream
analysis and visualization. Core scalar series must be non-empty and agree on
time-step count, `x_clean.npy` must match `x_noisy.npy`, covariance traces must
be square series, and `gen_x_estimates[:, 0, :]` must be compatible with
`x_noisy.npy`.

## Static PDF Contract

When `metadata.requested_outputs.static_plots` is true, validation also
requires the core static PDF set under `plots/`. Runs created with
`--no-static-plots` are valid without those PDFs. Requested dashboards and GIFs
are optional in the required-file set, but any recorded visualization or
animation must pass byte, SHA-256, and visual-artifact verification through
`odem.visual_checks`. Recorded PNGs must open and be nonblank, recorded GIFs
must contain nonblank frames, and recorded PDFs must start with a PDF header.

## Numeric Warnings

Numeric `.npy` artifacts are loaded with safe NumPy loading. Nonfinite numeric
values produce validation warnings and can make summaries ineligible for
ranking even when the manifest structure is otherwise valid.
