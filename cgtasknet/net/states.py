import torch
from norse.torch.functional.lif import LIFState
from norse.torch.functional.lif_adex import LIFAdExState
from norse.torch.functional.lif_refrac import LIFRefracState
from norse.torch.functional.lsnn import LSNNState


class InitialStates:
    def __init__(self, batch_size: int, hidden_size: int) -> None:
        self._batch_size = batch_size
        self._hidden_size = hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size_new):
        self._batch_size = batch_size_new

    @property
    def hidden_size(self):
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, hidden_size_new):
        self._hidden_size = hidden_size_new


class LIFInitState(InitialStates):
    def zero_state(self):
        return LIFState(
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )

    def random_state(self):
        return LIFState(
            torch.rand(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )


class LIFRefracInitState(InitialStates):
    def zero_state(self):
        return LIFRefracState(
            self.lif_state.zero_state(),
            torch.zeros(self._batch_size, self._hidden_size),
        )

    def lif_refrac_random_state(self):
        return LIFRefracState(
            self.lif_state.random_state(),
            torch.zeros(self._batch_size, self._hidden_size),
        )


class LSNNInitState(InitialStates):
    def zero_state(self):
        return LSNNState(
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )

    def random_state(self):
        return LSNNState(
            torch.rand(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )


class LIFAdExInitState(InitialStates):
    def zero_state(self):
        return LIFAdExState(
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )

    def random_state(self):
        return LIFAdExState(
            torch.rand(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
            torch.zeros(self._batch_size, self._hidden_size),
        )
