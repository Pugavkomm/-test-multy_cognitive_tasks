import torch
from norse.torch.functional.lif import LIFState
from norse.torch.functional.lif_adex import LIFAdExState

# from norse.torch.functional.lif_adex_refrac import LIFAdExRefracState
from norse.torch.functional.lif_refrac import LIFRefracState
from norse.torch.functional.lsnn import LSNNState


class InitialStates:
    def __init__(
        self, batch_size: int, hidden_size: int, device=torch.device("cpu")
    ) -> None:
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._device = device

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
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )

    def random_state(self):
        return LIFState(
            torch.rand(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )


class LIFRefracInitState(InitialStates):
    def zero_state(self):
        lif_init_state = LIFInitState(self._batch_size, self._hidden_size)
        return LIFRefracState(
            lif_init_state.zero_state(),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )

    def random_state(self):
        lif_init_state = LIFInitState(self._batch_size, self._hidden_size)
        return LIFRefracState(
            lif_init_state.random_state(),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )


class LSNNInitState(InitialStates):
    def zero_state(self):
        return LSNNState(
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )

    def random_state(self):
        return LSNNState(
            torch.rand(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )


class LIFAdExInitState(InitialStates):
    def zero_state(self):
        return LIFAdExState(
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )

    def random_state(self):
        return LIFAdExState(
            torch.rand(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
        )


# class LIFAdExRefracInitState(InitialStates):
#    def zero_state(self):
#        self.lifadex_init_state = LIFAdExInitState(
#            self._batch_size, self._hidden_size, device=self._device
#        )
#
#        return LIFAdExRefracState(
#            self.lifadex_init_state.zero_state(),
#            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
#        )
#
#    def random_state(self):
#        self.lifadex_init_state = LIFAdExInitState(
#            self._batch_size, self._hidden_size, device=self._device
#        )
#
#        return LIFAdExRefracState(
#            self.lifadex_init_state.random_state(),
#            torch.zeros(self._batch_size, self._hidden_size).to(self._device),
#        )
#
