from odem.config import load_sweep


def set(config_name="parameters.yaml"):
    """Return the legacy parameter axes and noise dictionary.

    New code should use :func:`odem.config.load_sweep`, which preserves the
    same sweep ordering while adding validation and typed combo objects.
    """
    sweep = load_sweep(config_name)
    return sweep.to_legacy_parameters(), sweep.noise
