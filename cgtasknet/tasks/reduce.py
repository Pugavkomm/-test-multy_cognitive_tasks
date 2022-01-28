from __future__ import annotations

"""
In reduced problems, we use two modes, which
are quoted in two different directions. Some
of the tasks can only be transferred to one mod.
The contextual task is transferred to two modes at once.
The network must ignore the wrong mod.
"""
from typing import Optional, Tuple, Union, NamedTuple, Any, Type
from abc import ABC, abstractmethod

import numpy as np


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


class DMTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = 0.75
    answer_time: float = ReduceTaskParameters().answer_time
    value: float = ReduceTaskParameters().value
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class RomoTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time = 0.25
    answer_time: float = ReduceTaskParameters().answer_time
    value: Tuple[float, float] = (None, None)
    delay: float = 0.15
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time
    negative_shift_delay_time: float = ReduceTaskParameters().negative_shift_delay_time
    positive_shift_delay_time: float = ReduceTaskParameters().positive_shift_delay_time


class CtxDMTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = DMTaskParameters().trial_time
    answer_time: float = DMTaskParameters().answer_time
    value: Tuple[float, float] = (None, None)
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class CtxDMTaskRandomModeParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = DMTaskParameters().trial_time
    answer_time: float = DMTaskParameters().answer_time
    value: Tuple[float, float] = (None, None)
    n_mods: int = 2
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class DMTaskRandomModParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = DMTaskParameters().trial_time
    answer_time: float = DMTaskParameters().answer_time
    value: float = DMTaskParameters().value
    n_mods: int = 2
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class RomoTaskRandomModParameters(NamedTuple):
    dt: float = RomoTaskParameters().dt
    trial_time: float = RomoTaskParameters().trial_time
    answer_time: float = RomoTaskParameters().answer_time
    value: Tuple[float, float] = RomoTaskParameters().value
    delay: float = RomoTaskParameters().delay
    n_mods: int = 2
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time
    negative_shift_delay_time: float = ReduceTaskParameters().negative_shift_delay_time
    positive_shift_delay_time: float = ReduceTaskParameters().positive_shift_delay_time


class ReduceTaskCognitive(ABC):
    """
    Class method for ReduceTask .
    """

    def __init__(
        self,
        params: ReduceTaskParameters,
        batch_size: int,
        mode: str,
        enable_fixation_delay: bool = False,
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

    # def set_param(self, name: str, value: int):
    #    if name not in self._params:
    #        raise IndexError(f"{name} is not the parameter")
    #    self._params[name] = value

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


class DMTask(ReduceTaskCognitive):
    """
    Construct a DMTask class for DMTask .

    Args:
        ReduceTaskCognitive ([type]): [description]

    Returns:
        [type]: [description]
    """

    threshold = 0.5

    def __init__(
        self,
        params: Optional[DMTaskParameters] = DMTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
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
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with inputs and target outputs .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, outputs]
        """
        dt = self._params.dt
        trial_time = int(
            np.random.uniform(
                self._params.trial_time - self._params.negative_shift_trial_time,
                self._params.trial_time + self._params.positive_shift_trial_time,
            )
            / dt
        )
        delay = int(self._params.answer_time / dt)
        if self._mode == "random":
            value = np.random.uniform(0, 1, size=self._batch_size)
        elif self._mode == "value":
            value = np.ones(self._batch_size) * self._params.value
        else:
            value = np.zeros(self._batch_size)
        inputs = np.zeros((trial_time + delay, self._batch_size, self._ob_size))
        inputs[:trial_time, :, 0] = 1
        inputs[:trial_time, :, 1] = value[:]
        target_outputs = np.zeros(
            (trial_time + delay, self._batch_size, self._act_size)
        )
        target_outputs[:, :, 0] = inputs[:, :, 0]
        target_outputs[trial_time:, :, 1] = value < self.threshold
        target_outputs[trial_time:, :, 2] = value > self.threshold
        # target_outputs[:, :, 1] = values < self.threshold
        # target_outputs[:, :, 2] = values > self.threshold

        return inputs, target_outputs

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
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
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


class DMTask1(DMTaskRandomMod):
    def __init__(
        self,
        params: Optional[DMTaskRandomModParameters] = DMTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ) -> None:
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self, mode=0):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "DMTask1"


class DMTask2(DMTaskRandomMod):
    def __init__(
        self,
        params: Optional[DMTaskRandomModParameters] = DMTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ) -> None:
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self, mode=0):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "DMTask2"


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
        super().__init__(params, batch_size, mode, enable_fixation_delay)
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with the given size and target .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        dt = self._params.dt
        trial_time = int(
            np.random.uniform(
                self._params.trial_time - self._params.negative_shift_trial_time,
                self._params.trial_time + self._params.positive_shift_trial_time,
            )
            / dt
        )
        delay = int(
            np.random.uniform(
                self._params.delay - self._params.negative_shift_delay_time,
                self._params.delay + self._params.positive_shift_delay_time,
            )
            / dt
        )
        answer_time = int(self._params.answer_time / dt)
        if self._mode == "random":
            values_first = np.random.uniform(0, 1, size=self._batch_size)
            values_second = np.random.uniform(0, 1, size=self._batch_size)
        elif self._mode == "value":
            values_first = np.ones(self._batch_size) * self._params.value[0]
            values_second = np.ones(self._batch_size) * self._params.value[1]
        else:
            values_first = np.zeros(self._batch_size)
            values_second = np.zeros(self._batch_size)
        inputs = np.zeros(
            ((2 * trial_time + delay + answer_time), self._batch_size, self._ob_size)
        )
        inputs[: 2 * trial_time + delay, :, 0] = 1
        inputs[:trial_time, :, 1] = values_first
        inputs[trial_time + delay : -answer_time, :, 1] = values_second
        target_output = np.zeros(
            ((2 * trial_time + delay + answer_time), self._batch_size, self._act_size)
        )
        target_output[:, :, 0] = inputs[:, :, 0]
        target_output[2 * trial_time + delay :, :, 1] = values_first < values_second
        target_output[2 * trial_time + delay :, :, 2] = values_second < values_first
        return inputs, target_output

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
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
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


class RomoTask1(RomoTaskRandomMod):
    def __init__(
        self,
        params: Optional[RomoTaskRandomModParameters] = RomoTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ) -> None:
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self, mode=0):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "RomoTask1"


class RomoTask2(RomoTaskRandomMod):
    def __init__(
        self,
        params: Optional[RomoTaskRandomModParameters] = RomoTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ) -> None:
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self, mode=1):
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "RomoTask2"


class CtxDMTask(ReduceTaskCognitive):
    """
    Context manager for CtxDMTask .

    Args:
        ReduceTaskCognitive ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[CtxDMTaskParameters] = CtxDMTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ):
        """
        Initialize the DMTask .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """

        super().__init__(params, batch_size, mode)

        self.DMTask = DMTask(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )
        self._ob_size = 3
        self._act_size = 3

    def _one_dataset(self, context: int):
        """
        Returns a single batch of inputs for one or more contexts .

        Args:
            context (int): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        inputs_first_context, outputs_first_context = self.DMTask.one_dataset()
        inputs_second_context, outputs_second_context = self.DMTask.one_dataset()
        inputs = np.zeros(
            (inputs_first_context.shape[0], self._batch_size, self._ob_size)
        )
        inputs[:, :, 0] = inputs_first_context[:, :, 0]
        inputs[:, :, 1] = inputs_first_context[:, :, 1]
        inputs[:, :, 2] = inputs_second_context[:, :, 1]
        if context == 0:
            return inputs, outputs_first_context
        if context == 1:
            return inputs, outputs_second_context
        else:
            raise ValueError(f"param: context expected 0 or 1, but actual {context}")

    def one_dataset(self, context: int = np.random.choice([0, 1]), *args, **kwargs):
        """
        Return a single dataset with one context .

        Args:
            context (int, optional): [description]. Defaults to np.random.choice([0, 1]).

        Returns:
            [type]: [description]
        """
        return self._one_dataset(context)

    @property
    def params(self):
        """
        Get a list of parameters .

        Returns:
            [type]: [description]
        """
        return self._params

    @params.setter
    def params(self, new_params: CtxDMTaskParameters):
        """
        Set the parameters of this DMTask .

        Args:
            new_params (dict): [description]
        """
        self._params = new_params
        self.DMTask = DMTask(
            DMTaskParameters(
                dt=new_params.dt,
                trial_time=new_params.trial_time,
                answer_time=new_params.answer_time,
                value=new_params.value,
            ),
            self._batch_size,
            self._mode,
        )

    @property
    def name(self):
        return "CtxDMTask"


class CtxDM1(CtxDMTask):
    """
    A context manager for creating a CtxDM1 code .

    Args:
        CtxDMTask ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[
            CtxDMTaskRandomModeParameters
        ] = CtxDMTaskRandomModeParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ):
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self):
        """
        Return single dataset with only one dataset .

        Returns:
            [type]: [description]
        """
        return self._one_dataset(0)

    @property
    def name(self):
        return "CtxDM1"


class CtxDM2(CtxDMTask):
    """
    A context manager for creating a CtxDMTask class .

    Args:
        CtxDMTask ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[
            CtxDMTaskRandomModeParameters
        ] = CtxDMTaskRandomModeParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
    ):
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        super().__init__(
            params, batch_size, mode, enable_fixation_delay=enable_fixation_delay
        )

    def one_dataset(self):
        """Return a single dataset."""
        return self._one_dataset(1)

    @property
    def name(self):
        return "CtxDM2"


class MultyReduceTasks(ReduceTaskCognitive):
    """
    The class method for creating a multilereduce task .

    Args:
        ReduceTaskCognitive ([list] or [dict]): [If list: list of tasks.
        If dict: dict tasks and their parameters (see [DefaultParameters])]

    Returns:
        [type]: [description]
    """

    task_list = [
        ("DMTask", DMTaskRandomMod),
        ("RomoTask", RomoTaskRandomMod),
        ("CtxDMTask", CtxDMTask),
        ("RomoTask1", RomoTask1),
        ("RomoTask2", RomoTask2),
        ("DMTask1", DMTask1),
        ("DMTask2", DMTask2),
        ("CtxDMTask1", CtxDM1),
        ("CtxDMTask2", CtxDM2),
    ]
    task_list.sort()
    TASKSDICT = dict(task_list)

    def __init__(
        self,
        tasks: Union[dict[str, Any], list[str]],
        batch_size: int = 1,
        mode: str = "random",
        delay_between: int = 0,  # iterations
        number_of_inputs: int = 2,
        enable_fixation_delay: bool = False,
    ):
        """
        Initialize the object with the initial state of the model .

        Args:
            tasks (Union[dict[str, dict[str, float]], list[str]]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            mode (str, optional): [description]. Defaults to "random".
        """
        self._delay_between = delay_between
        self._initial_tasks_list = dict()
        self._enable_fixation_delay = enable_fixation_delay
        if type(tasks) == list:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    batch_size=batch_size,
                    mode=mode,
                    enable_fixation_delay=enable_fixation_delay,
                )
        if type(tasks) == dict:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    params=tasks[task_name],
                    batch_size=batch_size,
                    mode=mode,
                    enable_fixation_delay=enable_fixation_delay,
                )
        self._tasks = tasks
        self._ob_size = 1 + number_of_inputs + len(tasks)
        self._act_size = 3
        self._sorted_tasks()
        self._create_task_list()
        self._batch_size = batch_size

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs one step of the batch .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        current_task = np.random.choice([i for i in range(len(self._task_list))])

        inputs, outputs = self._task_list[current_task].one_dataset()
        inputs_plus_rule = np.zeros((inputs.shape[0], self._batch_size, self._ob_size))
        inputs_plus_rule[:, :, -len(self._task_list) + current_task] = 1
        inputs_plus_rule[:, :, : -len(self._task_list)] = inputs
        if self._delay_between > 0:
            delay_inputs = np.zeros(
                (
                    self._delay_between,
                    inputs_plus_rule.shape[1],
                    inputs_plus_rule.shape[2],
                )
            )
            delay_outputs = np.zeros(
                (self._delay_between, outputs.shape[1], outputs.shape[2])
            )
            inputs_plus_rule = np.concatenate((delay_inputs, inputs_plus_rule), axis=0)
            outputs = np.concatenate((delay_outputs, outputs))
        return inputs_plus_rule, outputs

    def one_dataset(self):
        return self._one_dataset()

    def __getitem__(self, key: int):
        if key > len(self._tasks) or key < 0:
            raise IndexError(f"{key} is not include")
        return self._task_list[key]

    def __len__(self):
        return len(self._task_list)

    def _create_task_list(self):
        self._task_list = [
            self._initial_tasks_list[task_name]
            for task_name in self._initial_tasks_list
        ]

    def _sorted_tasks(self):
        """
        sort the tasks in order to avoid duplicates
        """
        new_dict = dict()
        for task_name in sorted(self._initial_tasks_list):
            new_dict[task_name] = self._initial_tasks_list[task_name]
        self._initial_tasks_list = new_dict

    @property
    def feature_and_act_size(self) -> Tuple[int, int]:
        """
        Return the feature and act size .

        Returns:
            Tuple[int, int]: [description]
        """
        return self._ob_size, self._act_size

    @property
    def feature_and_act_every_task_size(self) -> dict[str, Tuple[int, int]]:
        """
        Returns a dictionary of feature and action for each task.

        Returns:
            dict[str, Tuple[int, int]]: [description]
        """
        ob_and_act_sizes = dict()
        for task_name in self._initial_tasks_list:
            ob_size, act_size = self._initial_tasks_list[task_name].feature_and_act_size
            ob_and_act_sizes[task_name] = (ob_size, act_size)
        return ob_and_act_sizes
