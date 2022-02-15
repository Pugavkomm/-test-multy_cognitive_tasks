from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif_refrac import LIFRefracParameters, LIFRefracState
from norse.torch.module.exp_filter import ExpFilter

from cgtasknet.net.save_states import save_states


class SNNLifRefrac(torch.nn.Module):
    r"""This net includes one adaptive integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFRefracParameters] = LIFRefracParameters(),
        tau_filter_inv: float = 223.1435511314,
        save_states: bool = False,
    ) -> None:
        super(SNNLifRefrac, self).__init__()

        self.lif_refrac = snn.LIFRefracRecurrent(
            feature_size, hidden_size, p=neuron_parameters
        )

        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)
        self.save_states = save_states

    def forward(
        self, x: torch.tensor, state: Optional[LIFRefracState] = None
    ) -> Tuple[torch.tensor, LIFRefracState]:
        outputs, states = save_states(x, self.save_states, self.lif_refrac, state)
        outputs = self.exp_f(outputs)
        return (outputs, states)

    @staticmethod
    def type_parameters():
        return LIFRefracParameters

    @staticmethod
    def type_state():
        return LIFRefracState
