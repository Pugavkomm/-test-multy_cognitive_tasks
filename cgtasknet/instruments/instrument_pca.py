import torch


class PCA:
    def __init__(self, num_components: int = 1, calculate_derivation=False):
        self._num_components = num_components
        self._calculate_derivation = calculate_derivation

    def decompose(self, data: torch.Tensor, center: bool = True):
        X = torch.clone(data)
        if center:
            mean = torch.mean(X, 0)
            X -= mean.expand_as(X)
        U, _, _ = torch.svd(torch.t(X))
        out = torch.mm(X, U[:, : self._num_components])
        if self._calculate_derivation:
            cov = torch.cov(X.T)
            eigs_complex, _ = torch.eig(cov)
            all_dispersion = torch.sum(eigs_complex[:, 0])
            last_dispersion = torch.sum(eigs_complex[self._num_components :, :])
            delta = last_dispersion / all_dispersion
            return out, delta
        return out

    @property
    def num_components(self):
        return self._num_components

    @num_components.setter
    def num_components(self, value):
        self._num_components = value
