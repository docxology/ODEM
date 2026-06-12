from __future__ import annotations

import csv

from odem.analysis import write_summary_csv
from odem.reporting import _sweep_markdown
from odem.summary_schema import SUMMARY_CSV_FIELDNAMES, SWEEP_REPORT_COLUMNS


def test_summary_csv_header_matches_shared_schema(tmp_path):
    row = {field: None for field in SUMMARY_CSV_FIELDNAMES}
    row.update({"run_id": "schema-run", "free_action": 1.0, "valid_for_ranking": True, "issues": []})

    output = write_summary_csv([row], tmp_path / "summary.csv")

    with output.open(encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        assert next(reader) == list(SUMMARY_CSV_FIELDNAMES)


def test_sweep_report_table_header_matches_shared_schema():
    markdown = _sweep_markdown(
        [
            {
                "run_id": "schema-run",
                "combo_index": 0,
                "free_action": 1.0,
                "mse": 0.1,
                "all_numeric_outputs_finite": True,
                "valid_for_ranking": True,
                "issues": [],
            }
        ]
    )

    assert f"| {' | '.join(SWEEP_REPORT_COLUMNS)} |" in markdown
