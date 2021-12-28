from typing import Any, Tuple

import torch


class SNNOneState:
    def __init__(self, model) -> None:
        self.model = model

    def one_state(self, x: torch.tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        out, new_state = self.model(x, state=state)
        return out, new_state


class SNNStates(SNNOneState):
    # TODO change generate states
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def states(self, x, initial_state):
        T = len(x)  # timesteps
        state = initial_state
        outputs = []
        states = []
        for ts in range(T - 2):
            out, state = self.model(x[ts : ts + 2], state)
            outputs.append(out[1])
            states.append(state)
        with torch.no_grad():
            return torch.stack(outputs), states
