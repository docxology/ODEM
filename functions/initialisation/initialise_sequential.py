from odem.config import load_sweep


def set(config_name="parameters.yaml"):
    """Return parameter axes and the noise dictionary.

    New code can use :func:`odem.config.load_sweep` for typed combo objects.
    """
    sweep = load_sweep(config_name)
    return sweep.to_parameter_axes(), sweep.noise
