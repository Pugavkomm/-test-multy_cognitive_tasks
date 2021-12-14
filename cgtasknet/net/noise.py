import torch


class NoiseLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, mean: float = 0.0, sigma: float = 0.01):
        super(NoiseLayer, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear.weight.requires_grad = False
        self.linear.weight.copy_(torch.ones((hidden_size, hidden_size)))
        self._mean = mean
        self._sigma = sigma
        self._hidden_size = hidden_size

    def forward(self, x):
        return self.linear(x) + torch.normal(
            mean=self._mean, std=self._sigma, size=x.shape
        )
