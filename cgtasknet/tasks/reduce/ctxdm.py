from typing import NamedTuple, Optional, Tuple

import numpy as np

from cgtasknet.tasks.reduce.dm import DMTask, DMTaskParameters
from cgtasknet.tasks.reduce.reduce_task import (ReduceTaskCognitive,
                                                ReduceTaskParameters)


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
