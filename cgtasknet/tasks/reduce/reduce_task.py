from abc import ABC, abstractmethod
from random import SystemRandom
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np

hard_random = SystemRandom()

modes = ("random", "value")


def _generate_random_intervals(
    dt: float, average: float, left_shift: float, right_shift: float
):
    start = average - left_shift
    stop = average + right_shift
    return round(hard_random.uniform(start, stop) / dt)


def _generate_values(
    mode: str,
    batch_size: int,
    value: Union[float, tuple, list, Iterable],
    distribution=np.random.uniform,
    seed: Optional[int] = None,
) -> np.ndarray:
    mode = str(mode)
    batch_size = int(batch_size)
    if seed is not None:
        np.random.seed(seed)
    if mode not in modes:
        raise ValueError(f"Mode {mode} is not exist, you can use only modes: {modes}")
    if not (isinstance(value, int) or isinstance(value, float)):
        if mode != "random":
            raise ValueError
        if not isinstance(value, Iterable):
            raise TypeError
    else:
        value = float(value)
    if mode == "random":
        if isinstance(value, Iterable):
            return np.random.choice(value, size=batch_size)
        return distribution(0, value, size=batch_size)
    elif mode == "value":
        return np.ones(batch_size, dtype=np.float64) * value


def _concatenate_batches_external(
    l_inputs: List[np.ndarray],
    l_outputs: List[np.ndarray],
    max_length: int,
    batch_size: int,
    enable_fixation_delay: bool,
):
    for i in range(batch_size):
        inputs_bufer = np.zeros((max_length, *l_inputs[0].shape[1:3]))
        outputs_bufer = np.zeros((max_length, *l_outputs[0].shape[1:3]))
        required_length = max_length - l_inputs[i].shape[0]
        if enable_fixation_delay:
            inputs_bufer[:required_length, :, 0] = 1
            outputs_bufer[:required_length, :, 0] = 1
        inputs_bufer[required_length:, ...] = l_inputs[i]
        outputs_bufer[required_length:, ...] = l_outputs[i]
        l_inputs[i] = inputs_bufer
        l_outputs[i] = outputs_bufer

    inputs_plus_rule = np.concatenate(l_inputs, axis=1)
    outputs = np.concatenate(l_outputs, axis=1)
    return inputs_plus_rule, outputs


class ReduceTaskParameters(NamedTuple):
    dt: float = 1e-3
    trial_time: float = 0
    answer_time: float = 0.15
    negative_shift_trial_time: float = 0
    positive_shift_trial_time: float = 0
    negative_shift_delay_time: float = 0
    positive_shift_delay_time: float = 0
    value: float = None
    delay: float = None


class ReduceTaskCognitive(ABC):
    """
    Class method for ReduceTask .
    """

    def __init__(
        self,
        params: Any,
        batch_size: int,
        mode: str,
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        """
        Initialize the instance .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str): [description]
        """
        self._params = params
        self._batch_size = batch_size
        self._ob_size = 0
        self._act_size = 0
        self._mode = mode
        self._enable_fixation_delay = enable_fixation_delay
        self._uniq_batch = uniq_batch

    @abstractmethod
    def one_dataset(self) -> Type[tuple]:
        """
        Return a tuple containing one - dimensional dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """

    def dataset(self, n_trials: int = 1, delay_between=0):
        multy_inputs, multy_outputs = self.one_dataset()
        zeros_array_input = np.zeros(
            (delay_between, multy_inputs.shape[1], multy_inputs.shape[2])
        )
        zeros_array_output = np.zeros(
            (delay_between, multy_outputs.shape[1], multy_outputs.shape[2])
        )
        if self._enable_fixation_delay:
            zeros_array_input[:, :, 0] = 1.0
            zeros_array_output[:, :, 0] = 1.0
        multy_inputs = np.concatenate((zeros_array_input, multy_inputs), axis=0)
        multy_outputs = np.concatenate((zeros_array_output, multy_outputs), axis=0)
        for _ in range(n_trials - 1):
            inputs, outputs = self.one_dataset()
            multy_inputs = np.concatenate(
                (
                    multy_inputs,
                    zeros_array_input,
                ),
                axis=0,
            )
            multy_inputs = np.concatenate((multy_inputs, inputs), axis=0)
            multy_outputs = np.concatenate(
                (
                    multy_outputs,
                    zeros_array_output,
                ),
                axis=0,
            )
            multy_outputs = np.concatenate((multy_outputs, outputs), axis=0)

        return multy_inputs, multy_outputs

    @abstractmethod
    def _identical_batches(self, batch_size: int = 1):
        r"""
        _description_
        """

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

    def _concatenate_batches(
        self, l_intputs: List[np.ndarray], l_outputs: List[np.ndarray], max_length: int
    ):
        return _concatenate_batches_external(
            l_intputs,
            l_outputs,
            max_length,
            self._batch_size,
            self._enable_fixation_delay,
        )

    @property
    def feature_and_act_size(self) -> Tuple[int, int]:
        """
        Returns the feature and action size .

        Returns:
            Tuple[int, int]: [feature size, act_size (output size)]
        """
        return self._ob_size, self._act_size

    @feature_and_act_size.setter
    def feature_and_act_size(self, values: Tuple[int, int]):
        """
        Set the feature and action size .

        Args:
            values (Tuple[int, int]): [feature size, act_size (output size)]
        """
        self._ob_size, self._act_size = values

    @property
    def params(self) -> ReduceTaskParameters:
        """
        A dictionary with the current parameters .

        Returns:
            dict: [description]
        """
        return self._params

    @params.setter
    def params(self, new_params: ReduceTaskParameters):
        """
        Set the new parameters of this query.

        Args:
            new_params (dict): [description]
        """
        self._params = new_params

    @property
    def batch_size(self) -> int:
        """
        Number of batches that have been created .

        Returns:
            int: [description]
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        """
        Set the batch_size of the batch .

        Args:
            new_batch_size (int): [description]
        """
        self._batch_size = new_batch_size
