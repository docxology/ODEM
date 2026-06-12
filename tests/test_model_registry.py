from __future__ import annotations

import pytest

from odem.model_registry import resolve_model_functions, validate_model_names


def test_model_registry_resolves_supported_model_functions():
    resolved = resolve_model_functions(generative_process="glv", dynamics="lorenz", likelihood="identity")

    assert callable(resolved.process_build)
    assert callable(resolved.dynamics)
    assert callable(resolved.likelihood)
    assert resolved.process_name == "glv"
    assert resolved.dynamics_name == "lorenz"
    assert resolved.likelihood_name == "identity"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"generative_process": "missing", "dynamics": "lorenz", "likelihood": "identity"}, "Unknown generative process"),
        ({"generative_process": "glv", "dynamics": "missing", "likelihood": "identity"}, "Unknown dynamics"),
        ({"generative_process": "glv", "dynamics": "lorenz", "likelihood": "missing"}, "Unknown likelihood"),
    ],
)
def test_model_registry_rejects_unknown_names(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_model_names(**kwargs)
