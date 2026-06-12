from __future__ import annotations

import pytest

from odem.experiment import resolve_combo_slice as experiment_resolve_combo_slice
from odem.slicing import resolve_combo_slice


def test_resolve_combo_slice_returns_explicit_bounds():
    assert resolve_combo_slice(10, start_index=2, end_index=6, use_slurm_env=False) == (2, 6)
    assert resolve_combo_slice(10, start_index=2, end_index=8, max_combos=3, use_slurm_env=False) == (2, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 0},
        {"total": 10, "max_combos": 0},
        {"total": 10, "start_index": -1, "end_index": 2},
        {"total": 10, "start_index": 8, "end_index": 8},
        {"total": 10, "start_index": 11, "end_index": 12},
    ],
)
def test_resolve_combo_slice_rejects_invalid_ranges(kwargs):
    total = kwargs.pop("total")

    with pytest.raises(ValueError):
        resolve_combo_slice(total, use_slurm_env=False, **kwargs)


def test_resolve_combo_slice_uses_zero_based_slurm_environment():
    env = {"SLURM_ARRAY_TASK_ID": "1", "SLURM_ARRAY_TASK_COUNT": "3"}

    assert resolve_combo_slice(10, environ=env) == (4, 7)


def test_resolve_combo_slice_normalizes_one_based_slurm_environment():
    env = {"SLURM_ARRAY_TASK_MIN": "1", "SLURM_ARRAY_TASK_ID": "1", "SLURM_ARRAY_TASK_COUNT": "3"}

    assert resolve_combo_slice(10, environ=env) == (0, 4)


def test_resolve_combo_slice_rejects_empty_or_out_of_range_slurm_environment():
    env = {"SLURM_ARRAY_TASK_ID": "5", "SLURM_ARRAY_TASK_COUNT": "3"}

    with pytest.raises(ValueError, match="empty"):
        resolve_combo_slice(10, environ=env)


def test_experiment_reexports_resolve_combo_slice():
    assert experiment_resolve_combo_slice is resolve_combo_slice
