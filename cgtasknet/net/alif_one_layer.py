from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif_adex import LIFAdExParameters

from cgtasknet.instrumetns.exponent_filter import ExpFilter


class SNNALIF(torch.nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFAdExParameters] = None,
        tau_filter_inv: float = 223.1435511314,
    ) -> None:
        super(SNNALIF, self).__init__()
        if neuron_parameters is not None:
            self.alif = snn.LIFAdExRecurrent(
                feature_size, hidden_size, p=neuron_parameters
            )
        else:
            self.alif = snn.LIFAdExRecurrent(feature_size, hidden_size)
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        out, state = self.alif(x)
        out = self.exp_f(out)
        return (out, state)
