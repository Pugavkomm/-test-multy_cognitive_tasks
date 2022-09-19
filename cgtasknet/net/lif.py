from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif import LIFParameters, LIFState
from norse.torch.module.exp_filter import ExpFilter

from cgtasknet.net.save_states import save_states

default_tau_filter_inv = 223.1435511314


class SNNLif(torch.nn.Module):
    r"""This net includes one adaptive integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: LIFParameters = LIFParameters(),
        tau_filter_inv: float = default_tau_filter_inv,
        save_states: bool = False,
        return_spiking: bool = False,
        dropout: float = None,
    ) -> None:
        super(SNNLif, self).__init__()
        self.lif = snn.LIFRecurrent(feature_size, hidden_size, p=neuron_parameters)
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)
        self.save_states = save_states
        self.return_spiking = return_spiking

        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def forward(
        self, x: torch.tensor, state: Optional[LIFState] = None
    ) -> Tuple[torch.tensor, LIFState]:
        outputs, states = save_states(x, self.save_states, self.lif, state)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        outputs = self.exp_f(outputs)

        if self.return_spiking and self.save_states:
            return outputs, states, [s.z for s in states]
        else:
            return outputs, states

    @staticmethod
    def type_parameters():
        return LIFParameters

    @staticmethod
    def type_state():
        return LIFState
