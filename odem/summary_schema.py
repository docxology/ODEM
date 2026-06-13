from __future__ import annotations


SUMMARY_CSV_FIELDNAMES = (
    "run_id",
    "combo_index",
    "timesteps",
    "free_action",
    "mse",
    "accuracy_total",
    "complexity_total",
    "all_numeric_outputs_finite",
    "valid_bundle",
    "valid_for_ranking",
    "manifest_status",
    "issues",
    "path",
)

SWEEP_REPORT_COLUMNS = ("Rank", "Run ID", "Combo", "Free action", "MSE", "Finite", "Valid", "Issues")
