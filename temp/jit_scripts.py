import torch


class ExpFilter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
