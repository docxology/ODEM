# Release Evidence

Release evidence is generated as a local archive, not committed to git and not
uploaded remotely by default.

## Full Sweep Command

Run the 3024-combination sweep under `/tmp/odem-full`:

```bash
python scripts/run_release_sweep.py --config parameters.yaml --results-dir /tmp/odem-full/results --logs-dir /tmp/odem-full/logs --release-dir /tmp/odem-full/release --expected-combos 3024 --chunk-size 84 --workers 2 --best-run-animation --force --no-progress
```

The runner splits the sweep into 36 chunks, keeps static PDFs and dashboards
enabled, skips per-run animations, renders one GIF for the best valid run, then
archives and verifies the sweep. It also writes
`/tmp/odem-full/release/release_progress.jsonl`, with chunk, animation,
archive, and verification events for triage.

## Dry Run

Use dry-run mode to inspect chunk ranges and subprocess commands:

```bash
python scripts/run_release_sweep.py --config parameters.yaml --results-dir /tmp/odem-full/results --logs-dir /tmp/odem-full/logs --release-dir /tmp/odem-full/release --expected-combos 3024 --chunk-size 84 --workers 2 --dry-run
```

## Expected Local Archive

The release directory must contain:

- `/tmp/odem-full/release/release_evidence.json`
- `/tmp/odem-full/release/odem_release_evidence.tar.gz`
- `/tmp/odem-full/release/reports/sweep_summary.png`
- `/tmp/odem-full/release/reports/index.html`
- `/tmp/odem-full/release/verification.json`

The evidence manifest is acceptable only when `completed_runs` is `3024`,
`invalid_runs` is `0`, `complete` is `true`, and the archive byte count plus
SHA-256 match the tarball on disk.

Verify an existing release directory with:

```bash
python scripts/verify_release_evidence.py --release-dir /tmp/odem-full/release --expected-combos 3024
```

The verifier checks manifest counts, required report files, tarball validity,
archive byte/SHA-256 metadata, nonblank PNGs, GIF frames when present, local
gallery links, and the `sweep_summary.json` row count.

## Failure Triage

- Check `/tmp/odem-full/release/chunk_logs/` for chunk stdout/stderr.
- Check `/tmp/odem-full/release/release_progress.jsonl` for the last completed
  orchestration stage.
- Re-run failed explicit ranges with `python main.py run --start-index ...`.
- Do not pass `--allow-incomplete` for release evidence.
- Rebuild the local archive only after validation reports exactly 3024 valid
  completed bundles.
