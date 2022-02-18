from typing import NamedTuple, Optional, Tuple

import numpy as np

from cgtasknet.tasks.reduce.reduce_task import (
    _generate_random_intervals,
    ReduceTaskCognitive,
    ReduceTaskParameters,
)


class RomoTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = 0.25
    answer_time: float = ReduceTaskParameters().answer_time
    value: Tuple[float, float] = (None, None)
    delay: float = 0.15
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time
    negative_shift_delay_time: float = ReduceTaskParameters().negative_shift_delay_time
    positive_shift_delay_time: float = ReduceTaskParameters().positive_shift_delay_time


class RomoTaskRandomModParameters(NamedTuple):
    romo: RomoTaskParameters = RomoTaskParameters()
    n_mods: int = 2


class RomoTask(ReduceTaskCognitive):
    """
    The challenge is for the subjects or the network to
    remember the first stimulus. Then, after the delay time,
    the second stimulus comes. The network must compare this
    incentive and respond correctly.

    Ref: https://www.nature.com/articles/20939


    Args:
        ReduceTaskCognitive ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[RomoTaskParameters] = RomoTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """

        if mode == "value" and (params.value[0] is None or params.value is None):
            raise ValueError("params[values][0]([1]) is None")
        super().__init__(
            params=params,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._ob_size = 2
        self._act_size = 3

    def _unique_every_batch(self):
        max_length = 0
        l_intputs = []
        l_outputs = []
        for _ in range(self._batch_size):
            inputs, outputs = self._identical_batches(batch_size=1)
            l_intputs.append(inputs)
            l_outputs.append(outputs)
            max_length = max(max_length, inputs.shape[0])

        inputs, target_outputs = self._concatenate_batches(
            l_intputs, l_outputs, max_length
        )
        return inputs, target_outputs

    def _identical_batches(self, batch_size: int = 1):
        dt = self._params.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.trial_time,
            self._params.negative_shift_trial_time,
            self._params.positive_shift_trial_time,
        )

        delay = _generate_random_intervals(
            dt,
            self._params.delay,
            self._params.negative_shift_delay_time,
            self._params.positive_shift_delay_time,
        )

        answer_time = int(self._params.answer_time / dt)
        if self._mode == "random":
            values_first = np.random.uniform(0, 1, size=batch_size)
            values_second = np.random.uniform(0, 1, size=batch_size)
        elif self._mode == "value":
            values_first = np.ones(batch_size) * self._params.value[0]
            values_second = np.ones(batch_size) * self._params.value[1]
        else:
            values_first = np.zeros(batch_size)
            values_second = np.zeros(batch_size)
        inputs = np.zeros(
            ((2 * trial_time + delay + answer_time), batch_size, self._ob_size)
        )
        inputs[: 2 * trial_time + delay, :, 0] = 1
        inputs[:trial_time, :, 1] = values_first
        inputs[trial_time + delay : -answer_time, :, 1] = values_second
        target_output = np.zeros(
            ((2 * trial_time + delay + answer_time), batch_size, self._act_size)
        )
        target_output[:, :, 0] = inputs[:, :, 0]
        target_output[2 * trial_time + delay :, :, 1] = values_first < values_second
        target_output[2 * trial_time + delay :, :, 2] = values_second < values_first
        return inputs, target_output

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with the given size and target .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        if self._uniq_batch:
            return self._unique_every_batch()
        else:
            return self._identical_batches(self._batch_size)

    def one_dataset(self):
        """
        Return a single dataset containing only one dataset .

        Returns:
            [type]: [description]
        """
        return self._one_dataset()

    @property
    def name(self):
        return "RomoTask"


class RomoTaskRandomMod(RomoTask):
    """
    Trial task that is used for a random mod .

    Args:
        RomoTask ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[RomoTaskRandomModParameters] = RomoTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
            n_mods (int, optional): [description]. Defaults to 1.
        """

        super().__init__(
            params=params.romo,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )

        self._n_mods = params.n_mods
        self._ob_size += self._n_mods - 1

    def _one_dataset_mod(self, mode: int):
        """
        Generate a single model .

        Returns:
            [type]: [description]
        """
        temp, outputs = self._one_dataset()
        T = temp.shape[0]
        inputs = np.zeros((T, self._batch_size, self._ob_size))
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + mode] = temp[:, :, 1]
        return inputs, outputs

    def one_dataset(self, mode: Optional[int] = None):
        if mode is None:
            mode = np.random.randint(0, self._n_mods)
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "RomoTaskRandomMod"

    @property
    def params(self):
        return RomoTaskRandomModParameters(self._params, n_mods=self._n_mods)

    @params.setter
    def params(self, new_params: RomoTaskRandomModParameters):
        self._params = new_params.romo
        self._n_mods = new_params.n_mods


class RomoTask1(RomoTaskRandomMod):
    def one_dataset(self, mode=0):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "RomoTask1"


class RomoTask2(RomoTaskRandomMod):
    def one_dataset(self, mode=1):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "RomoTask2"
