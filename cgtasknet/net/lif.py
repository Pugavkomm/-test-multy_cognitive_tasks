from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif import LIFParameters, LIFState
from norse.torch.module.exp_filter import ExpFilter

default_tau_filter_inv = 223.1435511314


class SNNLif(torch.nn.Module):
    r"""This net includes one adaptive integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFParameters] = None,
        tau_filter_inv: float = default_tau_filter_inv,
    ) -> None:
        super(SNNLif, self).__init__()
        if neuron_parameters is not None:
            self.lif = snn.LIFRecurrent(feature_size, hidden_size, p=neuron_parameters)
        else:
            self.lif = snn.LIFRecurrent(feature_size, hidden_size)
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)

    def forward(
        self, x: torch.tensor, state: Optional[LIFState] = None
    ) -> Tuple[torch.tensor, LIFState]:
        out, n_state = self.lif(x, state)
        out = self.exp_f(out)
        return (out, n_state)

    @staticmethod
    def type_parameters():
        return LIFParameters

    @staticmethod
    def type_state():
        return LIFState
