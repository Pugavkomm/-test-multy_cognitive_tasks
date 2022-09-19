from typing import List, Optional, Tuple, Union

import norse.torch as snn
import torch
from norse.torch.functional.lif_adex import LIFAdExParameters, LIFAdExState
from norse.torch.module.exp_filter import ExpFilter

from cgtasknet.net.save_states import save_states

default_tau_filter_inv = 223.1435511314


class SNNlifadex(torch.nn.Module):
    r"""This net includes one adaptive exponential integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFAdExParameters] = LIFAdExParameters(),
        tau_filter_inv: float = default_tau_filter_inv,
        input_weights: Optional[torch.Tensor] = None,
        save_states: bool = False,
        return_spiking: bool = False,
    ) -> None:
        super(SNNlifadex, self).__init__()
        self.alif = snn.LIFAdExRecurrent(
            feature_size,
            hidden_size,
            p=neuron_parameters,
            input_weights=input_weights,
        )

        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)
        self.save_states = save_states
        self.return_spiking = return_spiking

    def forward(
        self,
        x: torch.tensor,
        state: Optional[LIFAdExState] = None,
        additional_current: Optional[torch.tensor] = None,
    ) -> Union[
        Tuple[torch.tensor, torch.tensor, List[torch.tensor]],
        Tuple[torch.tensor, torch.tensor],
    ]:
        outputs, states = save_states(
            x,
            self.save_states,
            self.alif,
            state,
            additional_current,
        )

        outputs = self.exp_f(outputs)

        if self.return_spiking and self.save_states:
            return outputs, states, [s.z for s in states]
        else:
            return outputs, states

    @staticmethod
    def type_parameters():
        return LIFAdExParameters

    @staticmethod
    def type_state():
        return LIFAdExState
