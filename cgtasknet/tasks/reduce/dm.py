from typing import NamedTuple, Optional, Tuple

import numpy as np

from cgtasknet.tasks.reduce.reduce_task import (
    _generate_random_intervals,
    ReduceTaskCognitive,
    ReduceTaskParameters,
)


class DMTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = 0.75
    answer_time: float = ReduceTaskParameters().answer_time
    value: float = ReduceTaskParameters().value
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class DMTaskRandomModParameters(NamedTuple):
    dm: DMTaskParameters = DMTaskParameters()
    n_mods: int = 2


class DMTask(ReduceTaskCognitive):
    """
    Construct a DMTask class for DMTask .

    Args:
        ReduceTaskCognitive ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        params: Optional[DMTaskParameters] = DMTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize the model .

        Args:
            params (Optional[dict], optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 1.
            mode (str, optional): [description]. Defaults to "random".
        """

        if mode == "value" and params.value is None:
            raise ValueError("params[value] is None")

        super().__init__(
            params,
            batch_size,
            mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._ob_size = 2
        self._act_size = 3
        self._threshold = threshold

    def _identical_batches(self, batch_size: int = 1):
        dt = self._params.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.trial_time,
            self._params.negative_shift_trial_time,
            self._params.positive_shift_trial_time,
        )
        delay = round(self._params.answer_time / dt)
        if self._mode == "random":
            value = np.random.uniform(0, 1, size=batch_size)
        elif self._mode == "value":
            value = np.ones(batch_size) * self._params.value
        else:
            value = np.zeros(batch_size)
        inputs = np.zeros((trial_time + delay, batch_size, self._ob_size))
        inputs[:trial_time, :, 0] = 1
        inputs[:trial_time, :, 1] = value[:]
        target_outputs = np.zeros((trial_time + delay, batch_size, self._act_size))
        target_outputs[:, :, 0] = inputs[:, :, 0]
        target_outputs[trial_time:, :, 1] = value < self._threshold
        target_outputs[trial_time:, :, 2] = value > self._threshold
        return inputs, target_outputs

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with inputs and target outputs .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, outputs]
        """
        if self._uniq_batch:
            return self._unique_every_batch()
        else:
            return self._identical_batches(self._batch_size)

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset()

    @property
    def name(self):
        return "DMTask"


class DMTaskRandomMod(DMTask):
    """
    Class method for DMTask class .

    Args:
        DMTask ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[DMTaskRandomModParameters] = DMTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ) -> None:
        """
        Initialize the model .

        Args:
            batch_size (int): [description]
            params (Optional[dict], optional): [description]. Defaults to None.
            mode (str, optional): [description]. Defaults to "random".
            n_mods (int, optional): [description]. Defaults to 1.
        """

        super().__init__(
            params.dm, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )
        self._n_mods = params.n_mods
        self._ob_size += self._n_mods - 1

    def _one_dataset_mod(self, mode: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, target outputs]
        """
        temp, outputs = self._one_dataset()
        t = temp.shape[0]
        inputs = np.zeros((t, self._batch_size, self._ob_size))
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + mode] = temp[:, :, 1]
        return inputs, outputs

    def one_dataset(self, mode: Optional[int] = None):
        if mode is None:
            mode = np.random.randint(0, self._n_mods)
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "DMTaskRandomMod"

    @property
    def params(self):
        return DMTaskRandomModParameters(self._params, n_mods=self._n_mods)

    @params.setter
    def params(self, new_params: DMTaskRandomModParameters):
        self._params = new_params.dm
        self._n_mods = new_params.n_mods


class DMTask1(DMTaskRandomMod):
    def one_dataset(self, mode=0):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "DMTask1"


class DMTask2(DMTaskRandomMod):
    def one_dataset(self, mode=1):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "DMTask2"
