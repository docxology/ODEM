from __future__ import annotations

from functions.plotting.loader import PlotRunData
from functions.plotting.noise import plot_noise_outputs
from functions.plotting.precision import plot_precision_outputs, plot_vfe_outputs
from functions.plotting.state import plot_state_outputs


class PlotDispatcher:
    def __init__(self, data_dir):
        self.data = PlotRunData.load(data_dir)
        self.data_dir = str(self.data.run_dir)
        self.output_dir = str(self.data.output_dir)

    def dispatch(self):
        outputs = []
        outputs.extend(plot_vfe_outputs(self.data))
        outputs.extend(plot_precision_outputs(self.data))
        outputs.extend(plot_state_outputs(self.data))
        outputs.extend(plot_noise_outputs(self.data))
        return outputs
