"""Composable public API for Online Dynamic Expectation Maximisation experiments."""

from odem.analysis import summarize_run, summarize_sweep
from odem.arrays import all_numeric_finite, extract_state_estimate, load_arrays, load_optional_array, state_estimate_issue
from odem.artifacts import RunBundle
from odem.config import ExperimentCombo, ParameterSweep, load_sweep
from odem.model_registry import resolve_model_functions, validate_model_names
from odem.numerics import ensure_finite_tensor, quadratic_form, safe_precision_inverse
from odem.reporting import ReportPaths, create_run_report, create_sweep_report
from odem.release_evidence import ReleaseVerification, archive_release_evidence, verify_release_evidence
from odem.run_outputs import DERIVED_PRECISION_NAMES, OUTPUT_NAMES, RAW_OUTPUT_NAMES, RunOutputs
from odem.slicing import resolve_combo_slice
from odem.summary_schema import SUMMARY_CSV_FIELDNAMES, SWEEP_REPORT_COLUMNS
from odem.validation import BundleValidation, validate_run_bundle

__all__ = [
    "BundleValidation",
    "DERIVED_PRECISION_NAMES",
    "ExperimentCombo",
    "ParameterSweep",
    "OUTPUT_NAMES",
    "RAW_OUTPUT_NAMES",
    "RunBundle",
    "RunOutputs",
    "ReportPaths",
    "ReleaseVerification",
    "SUMMARY_CSV_FIELDNAMES",
    "SWEEP_REPORT_COLUMNS",
    "all_numeric_finite",
    "archive_release_evidence",
    "create_run_report",
    "create_sweep_report",
    "ensure_finite_tensor",
    "extract_state_estimate",
    "load_arrays",
    "load_optional_array",
    "load_sweep",
    "quadratic_form",
    "resolve_model_functions",
    "resolve_combo_slice",
    "safe_precision_inverse",
    "state_estimate_issue",
    "summarize_run",
    "summarize_sweep",
    "validate_run_bundle",
    "validate_model_names",
    "verify_release_evidence",
]
