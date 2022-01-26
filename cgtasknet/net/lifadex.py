from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif_adex import LIFAdExParameters, LIFAdExState
from norse.torch.module.exp_filter import ExpFilter

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

    def forward(
        self, x: torch.tensor, state: Optional[LIFAdExState] = None
    ) -> Tuple[torch.tensor, LIFAdExState]:
        if self.save_states:
            T = len(x)
            s = state
            states = []
            outputs = []
            for ts in range(T):
                out, s = self.alif(x[ts, :, :], state=s)
                outputs.append(out)
                states.append(s)
            outputs = torch.stack(outputs)
        else:
            outputs, states = self.alif(x, state)
        outputs = self.exp_f(outputs)
        return (outputs, states)

    @staticmethod
    def type_parameters():
        return LIFAdExParameters

    @staticmethod
    def type_state():
        return LIFAdExState
