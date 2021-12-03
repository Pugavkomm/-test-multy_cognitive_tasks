from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif_adex import LIFAdExParameters

from cgtasknet.instrumetns.exponent_filter import ExpFilter
from cgtasknet.tasks.tasks import TaskCognitive


class SNNALIF(torch.nn.Module):
    r"""
    This net includes one alif layer.
    """

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


class SNNALIFstates:
    def __init__(self, model: SNNALIF, task: TaskCognitive, number_of_tasks) -> None:
        self.model = model
        self.task = task
        self.number_of_tasks = number_of_tasks

    @property
    def number_of_tasks(self):
        return self.number_of_tasks

    @number_of_tasks.setter
    def number_of_tasks(self, new_number_of_tasks):
        self.number_of_tasks = new_number_of_tasks
