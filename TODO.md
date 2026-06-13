# ODEM Scoped TODO

No repository-tracked TODO items remain open. The remaining release task is a
local archive generation procedure, documented in
[Release Evidence](docs/release-evidence.md), rather than a source-code change.

Run the full 3024-combination release sweep and write a local archive with:

```bash
python scripts/run_release_sweep.py --config parameters.yaml --results-dir /tmp/odem-full/results --logs-dir /tmp/odem-full/logs --release-dir /tmp/odem-full/release --expected-combos 3024 --chunk-size 84 --workers 2 --best-run-animation --force --no-progress
```

Expected local archive:

- `/tmp/odem-full/release/release_evidence.json`
- `/tmp/odem-full/release/odem_release_evidence.tar.gz`
- `/tmp/odem-full/release/verification.json`
- `/tmp/odem-full/release/release_progress.jsonl`
