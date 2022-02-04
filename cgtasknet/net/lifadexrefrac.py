from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch import LIFAdExRefracParameters, LIFAdExRefracState
from norse.torch.module.exp_filter import ExpFilter

from cgtasknet.net.save_states import save_states


class SNNlifadexrefrac(torch.nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[
            LIFAdExRefracParameters
        ] = LIFAdExRefracParameters(),
        tau_filter_inv: float = 223.1435511314,
        input_weights: Optional[torch.Tensor] = None,
        save_states: bool = False,
    ) -> None:
        super(SNNlifadexrefrac, self).__init__()
        self.adexrefrac = snn.LIFAdExRefracRecurrent(
            feature_size,
            hidden_size,
            p=neuron_parameters,
            input_weights=input_weights,
        )
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)
        self.save_states = save_states

    def forward(
        self, x: torch.tensor, state: Optional[LIFAdExRefracState] = None
    ) -> Tuple[torch.tensor, LIFAdExRefracState]:

        outputs, states = save_states(x, self.save_states, self.adexrefrac, state)
        outputs = self.exp_f(outputs)
        return outputs, states

    @property
    def save(self):
        return self.save_states

    @save.setter
    def save(self, new_save_states: bool):
        self.save_states = new_save_states

    @staticmethod
    def type_parameters():
        return LIFAdExRefracParameters

    @staticmethod
    def type_state():
        return LIFAdExRefracState
