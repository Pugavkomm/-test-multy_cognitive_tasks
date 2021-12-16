from typing import Optional, Tuple

import norse.torch as snn
import torch
from norse.torch.functional.lif import LIFState
from norse.torch.functional.lif_refrac import LIFRefracParameters, LIFRefracState
from norse.torch.module.exp_filter import ExpFilter

default_tau_filter_inv = 223.1435511314


class SNNLifRefrac(torch.nn.Module):
    r"""This net includes one adaptive integrate-and-fire layer."""

    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFRefracParameters] = None,
        tau_filter_inv: float = default_tau_filter_inv,
    ) -> None:
        super(SNNLifRefrac, self).__init__()
        if neuron_parameters is not None:
            self.lif_refrac = snn.LIFRefracRecurrent(
                feature_size, hidden_size, p=neuron_parameters
            )
        else:
            self.lif_refrac = snn.LIFRefracRecurrent(feature_size, hidden_size)
        self.exp_f = ExpFilter(hidden_size, output_size, tau_filter_inv)

    def forward(self, x: torch.tensor, state: Optional[LIFRefracState] = None) -> Tuple[torch.tensor, LIFRefracState]:
        out, state = self.lif_refrac(x, state=state)
        out = self.exp_f(out)
        return (out, state)

    @staticmethod
    def type_parameters():
        return LIFRefracParameters

    @staticmethod
    def type_state():
        return LIFRefracState


class SNNLIFRefractOneState(SNNLifRefrac):
    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters: Optional[LIFRefracParameters] = None,
        tau_filter_inv: float = default_tau_filter_inv,
    ) -> None:
        super().__init__(
            feature_size, hidden_size, output_size, neuron_parameters, tau_filter_inv
        )

    def forward(
        self, x: torch.tensor, state: Optional[LIFRefracState] = None
    ) -> Tuple[torch.tensor, LIFRefracState]:
        out, new_state = self.lif_refrac(x, state=state)
        out = self.exp_f(out)
        return (out, new_state)


class SNNLIFRefractStates:
    def __init__(self, model: SNNLIFRefractOneState) -> None:
        self.model = model

    def generate_states(
        self, x: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        T = len(x)
        batch_size = x.shape[1]
        hidden_size = self.model.lif_refrac.hidden_size
        states_v = []
        states_z = []
        outputs = []
        state = LIFRefracState(
            LIFState(
                torch.zeros((batch_size, hidden_size)),
                torch.zeros((batch_size, hidden_size)),
                torch.zeros((batch_size, hidden_size)),
            ),
            torch.zeros((batch_size, hidden_size)),
        )
        for ts in range(0, T - 1, 1):
            out, state = self.model(x[ts : ts + 2], state)
            outputs.append(out.detach().cpu()[1])
            states_v.append(torch.clone(state.lif.v.detach().cpu()))
            states_z.append(torch.clone(state.lif.z.detach().cpu()))
        return torch.stack(outputs), torch.stack(states_v), torch.stack(states_z)

    @property
    def number_of_tasks(self):
        return self.number_of_tasks

    @number_of_tasks.setter
    def number_of_tasks(self, new_number_of_tasks):
        self.number_of_tasks = new_number_of_tasks
