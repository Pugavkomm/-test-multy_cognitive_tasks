import torch
import time
from typing import Tuple




class ExpFilter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        