from __future__ import annotations

import numpy as np
import pytest
import torch

from odem.run_outputs import DERIVED_PRECISION_NAMES, OUTPUT_NAMES, RAW_OUTPUT_NAMES, RunOutputs, scalar_float


def _raw_outputs() -> tuple[object, ...]:
    values: list[object] = []
    for name in OUTPUT_NAMES:
        if name == "free_action":
            values.append(torch.tensor(6.5, dtype=torch.float64, requires_grad=True))
        elif name == "mse":
            values.append(np.float64(0.25))
        elif name == "y_sigma_schedule":
            values.append(np.array([[0.1], [0.2]], dtype=float))
        elif name == "x_sigma_schedule":
            values.append(np.array([[0.05, 0.1], [0.2, 0.4]], dtype=float))
        else:
            values.append(np.ones((2, 1), dtype=float))
    return tuple(values)


def test_run_outputs_maps_tuple_and_precision_schedules():
    outputs = RunOutputs.from_tuple(_raw_outputs())

    assert tuple(outputs.raw_arrays) == RAW_OUTPUT_NAMES
    assert DERIVED_PRECISION_NAMES == ("x_pi_schedule", "y_pi_schedule")
    assert outputs.free_action == 6.5
    assert outputs.mse == 0.25
    assert np.allclose(outputs.derived_arrays["y_pi_schedule"], np.array([[100.0], [25.0]]))
    assert np.allclose(outputs.derived_arrays["x_pi_schedule"], np.array([[400.0, 100.0], [25.0, 6.25]]))
    assert outputs.snapshot_scalars == {"fa": 6.5, "mse": 0.25}


def test_run_outputs_rejects_mismatched_tuple_length():
    with pytest.raises(ValueError, match="ODEM output length"):
        RunOutputs.from_tuple(_raw_outputs()[:-1])


def test_scalar_float_handles_numpy_and_torch_values():
    assert scalar_float(torch.tensor(2.5, dtype=torch.float64, requires_grad=True)) == 2.5
    assert scalar_float(np.array(3.5)) == 3.5

    with pytest.raises(ValueError, match="finite scalar"):
        scalar_float(np.array([1.0, 2.0]))
