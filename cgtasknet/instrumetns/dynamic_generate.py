from typing import Any, Tuple

import torch


class SNNOneState:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def one_state(self, x: torch.tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        out, new_state = self.model(x)
        return out, new_state


class SNNStates(SNNOneState):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def states(self, x: torch.tensor, initial_state):
        T = len(x)  # timesteps
        state = initial_state
        outputs = []
        states = []
        for ts in range(T):
            out, state = self.model(x[ts : ts + 2], state)
            outputs.append(out)
            states.append(state)
        return torch.stack(outputs), torch.stack(states)
