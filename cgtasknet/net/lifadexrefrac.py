import torch
import norse.torch as snn

from typing import Optional, Tuple
from norse.torch import LIFAdExRefracParameters, LIFAdExRefracState
from norse.torch.module.exp_filter import ExpFilter


class SNNlifadexrefrac(torch.nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFAdExRefracParameters] = None,
        tau_filter_inv: float = 223.1435511314,
        input_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super(SNNlifadexrefrac, self).__init__()
        self.adexrefrac = snn.LIFAdExRefracRecurrent(
            feature_size,
            hidden_size,
            p=neuron_parameters,
            input_weights=input_weights,
        )
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)

    def forward(
        self, x: torch.tensor, state: Optional[LIFAdExRefracState] = None
    ) -> Tuple[torch.tensor, LIFAdExRefracState]:
        out, out_state = self.adexrefrac(x, state=state)
        out = self.exp_f(out)
        return (out, out_state)

    @staticmethod
    def type_parameters():
        return LIFAdExRefracParameters

    @staticmethod
    def type_state():
        return LIFAdExRefracState
