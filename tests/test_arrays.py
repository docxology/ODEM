from __future__ import annotations

import numpy as np
import pytest

from odem.arrays import (
    all_numeric_finite,
    extract_state_estimate,
    load_arrays,
    load_optional_array,
    state_estimate_issue,
)


def test_load_arrays_uses_npy_stems_and_safe_loading(tmp_path):
    np.save(tmp_path / "vfe.npy", np.array([1.0, 2.0]))
    np.save(tmp_path / "x_noisy.npy", np.array([[1.0, 2.0]]))

    arrays = load_arrays(tmp_path)

    assert sorted(arrays) == ["vfe", "x_noisy"]
    assert np.array_equal(arrays["vfe"], np.array([1.0, 2.0]))
    assert load_optional_array(tmp_path / "missing.npy") is None


def test_all_numeric_finite_rejects_nonfinite_numeric_arrays():
    arrays = {
        "finite": np.array([1.0, 2.0]),
        "bad": np.array([np.inf]),
        "labels": np.array(["ok"], dtype=object),
    }

    assert all_numeric_finite(arrays) is False


def test_extract_state_estimate_returns_truth_and_first_posterior_sample():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    estimates = np.array([[[0.1, 1.1], [9.0, 9.0]], [[2.1, 3.1], [8.0, 8.0]]])

    truth, estimate = extract_state_estimate(x, estimates)

    assert np.array_equal(truth, x)
    assert np.array_equal(estimate, estimates[:, 0, :])
    assert state_estimate_issue(x, estimates) is None


@pytest.mark.parametrize(
    ("x", "estimates", "message"),
    [
        (np.array([1.0, 2.0]), np.zeros((2, 1, 1)), "x_noisy must be a 2D array"),
        (np.zeros((2, 1)), np.zeros((2, 1)), "gen_x_estimates must be a 3D array"),
        (np.empty((0, 2)), np.empty((0, 1, 2)), "x_noisy must contain at least one time step"),
        (np.zeros((2, 2)), np.zeros((2, 0, 2)), "posterior samples"),
        (np.zeros((2, 2)), np.zeros((1, 1, 2)), "must match"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 1, 2)), "x_noisy contains nonfinite values"),
        (np.zeros((1, 2)), np.array([[[np.inf, 0.0]]]), "gen_x_estimates contains nonfinite values"),
    ],
)
def test_extract_state_estimate_rejects_bad_arrays_with_clear_message(x, estimates, message):
    with pytest.raises(ValueError, match=message):
        extract_state_estimate(x, estimates)

    assert message.split(" must ")[0] in state_estimate_issue(x, estimates)
