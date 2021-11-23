
import torch
from typing import Optional

@torch.jit.script
def _exp_filter_step_jit(
    old_value: torch.Tensor,
    input_value: torch.Tensor,
    parameter: float) -> torch.Tensor:
    value_new = parameter * old_value + input_value
    return value_new

def exp_filter_step(old_value: torch.Tensor,
    input_value: torch.Tensor,
    parameter: float) -> torch.Tensor:
    return _exp_filter_step_jit(old_value, input_value, parameter)

import torch
from typing import Optional
class ExpFilter(torch.nn.Module):
    def __init__(
                self, 
                input_size:int,
                output_size:int, 
                parameter:float = 0.1,
                input_weights: Optional[torch.Tensor] = None, 
                bias: bool = True
                ) -> None:
        super(ExpFilter, self).__init__()
        self.input_size = torch.as_tensor(input_size)
        self.output_size = torch.as_tensor(output_size)
        self.parameter = parameter
        
        if input_weights is not None:
            self.input_weights = input_weights
        else: 
            k = torch.sqrt(1.0 / self.input_size)
            self.input_weights = -k + 2 * k * torch.rand(output_size, input_size) # from - sqrt(k) to sqrt(k) (like Linear layer)
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        with torch.no_grad():
            self.linear.weight.copy_(self.input_weights)
            
    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}, parameter={self.parameter}"
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.linear(input_tensor)
        T = input_tensor.shape[0]
        outputs = []
        outputs.append(input_tensor[0])
        
        for ts in range(T - 1):
            out = exp_filter_step(outputs[-1], input_tensor[ts + 1], self.parameter)
            outputs.append(out)
        return torch.stack(outputs)


