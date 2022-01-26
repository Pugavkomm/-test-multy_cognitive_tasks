from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lsnn import LSNNParameters, LSNNState
from norse.torch.module.exp_filter import ExpFilter

from cgtasknet.net.save_states import save_states

default_tau_filter_inv = 223.1435511314


class SNNAlif(torch.nn.Module):
    r"""This net includes one adaptive integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LSNNParameters] = LSNNParameters(),
        tau_filter_inv: float = default_tau_filter_inv,
        save_states: bool = False,
    ) -> None:
        super(SNNAlif, self).__init__()
        self.alif = snn.LSNNRecurrent(feature_size, hidden_size, p=neuron_parameters)
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)
        self.save_states = save_states

    def forward(
        self, x: torch.tensor, state: Optional[LSNNState] = None
    ) -> Tuple[torch.tensor, LSNNState]:
        outputs, states = save_states(x, self.save_states, self.alif, state)
        outputs = self.exp_f(outputs)
        return (outputs, states)

    @staticmethod
    def type_parameters():
        return LSNNParameters

    @staticmethod
    def type_state():
        return LSNNState
