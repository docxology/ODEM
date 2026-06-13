from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odem.release_evidence import verify_release_evidence


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify a local ODEM release-evidence archive and gallery.")
    parser.add_argument("--release-dir", required=True, help="Directory containing release_evidence.json and reports.")
    parser.add_argument("--expected-combos", type=int, default=None, help="Expected completed valid run count.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Treat incomplete counts as warnings instead of errors.")
    args = parser.parse_args(argv)

    verification = verify_release_evidence(
        args.release_dir,
        expected_combos=args.expected_combos,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(verification.to_dict(), indent=2))
    return 0 if verification.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
