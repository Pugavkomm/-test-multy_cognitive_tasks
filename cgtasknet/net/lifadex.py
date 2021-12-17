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
        neuron_parameters: Optional[LIFAdExParameters] = None,
        tau_filter_inv: float = default_tau_filter_inv,
        input_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super(SNNlifadex, self).__init__()
        if neuron_parameters is not None:
            self.alif = snn.LIFAdExRecurrent(
                feature_size,
                hidden_size,
                p=neuron_parameters,
                input_weights=input_weights,
            )
        else:
            self.alif = snn.LIFAdExRecurrent(
                feature_size, hidden_size, input_weights=input_weights
            )
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)

    def forward(
        self, x: torch.tensor, state: Optional[LIFAdExState] = None
    ) -> Tuple[torch.tensor, LIFAdExState]:
        out, state = self.alif(x, state=state)
        out = self.exp_f(out)
        return (out, state)

    @staticmethod
    def type_parameters():
        return LIFAdExParameters

    @staticmethod
    def type_state():
        return LIFAdExState
