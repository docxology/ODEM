from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odem.release_evidence import archive_release_evidence


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Archive validated ODEM sweep outputs and reports as release evidence.")
    parser.add_argument("--results-dir", required=True, help="Directory containing completed run bundles.")
    parser.add_argument("--output-dir", required=True, help="Directory for report outputs, manifest, and archive.")
    parser.add_argument("--expected-combos", type=int, default=None, help="Expected completed-run count.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write evidence even when the completed-run count is below the expected count.")
    args = parser.parse_args(argv)

    manifest = archive_release_evidence(
        args.results_dir,
        args.output_dir,
        expected_combos=args.expected_combos,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(manifest, indent=2))
    if not manifest["complete"] and not args.allow_incomplete:
        print(
            f"completed_runs={manifest['completed_runs']} did not match expected_combos={args.expected_combos}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
