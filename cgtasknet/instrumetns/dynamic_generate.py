from typing import Any, Optional, Tuple

import torch


class SNNOneState(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def forward(self, x: torch.tensor, state: Optional[Any] = None) -> Tuple[torch.Tensor, ]:
        out, new_state = self.model(x)
        return out, new_state
    

class SNNStates(SNNOneState):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
    
    def forward(self, x: torch.tensor, initial_state: Optional[Any] = None):
        
