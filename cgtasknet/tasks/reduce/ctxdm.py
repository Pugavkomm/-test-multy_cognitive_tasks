from typing import NamedTuple, Optional, Tuple, Union

import numpy as np

from cgtasknet.tasks.reduce.dm import DMTaskParameters
from cgtasknet.tasks.reduce.reduce_task import (
    _generate_random_intervals,
    ReduceTaskCognitive,
)


class CtxDMTaskParameters(NamedTuple):
    dm: DMTaskParameters = DMTaskParameters()
    context: int = None  # if mode is 'value' then generate this context
    value: tuple = (None, None)


class CtxDMTaskRandomModeParameters(NamedTuple):
    ctx: CtxDMTaskParameters = CtxDMTaskParameters()
    n_mods: int = 2


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
        uniq_batch: bool = False,
        threshold: float = 0.5,
        get_context: bool = False,
    ):
        """
        Initialize the DMTask .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """

        if mode == "value" and (
            (params.value[0] is None or params.value[1] is None)
            or params.context is None
        ):
            raise ValueError("params[value] is None")
        if params.context is not None:
            if 0 > params.context > 1:
                raise ValueError(
                    f"params.context should be 0 or 1, but actual: {params.context}"
                )
        super().__init__(
            params=params,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._ob_size = 3
        self._act_size = 3
        self._threshold = threshold
        self._get_context = get_context

    def _identical_batches(
        self, batch_size: int = 1
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
    ]:
        dt = self._params.dm.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.dm.trial_time,
            self._params.dm.negative_shift_trial_time,
            self._params.dm.positive_shift_trial_time,
        )
        answer_time = round(self._params.dm.answer_time / dt)
        if self._mode == "random":
            value_1 = np.random.uniform(0, 1, size=batch_size)
            value_2 = np.random.uniform(0, 1, size=batch_size)

        elif self._mode == "value":
            value_1 = np.ones(batch_size) * self._params.value[0]
            value_2 = np.ones(batch_size) * self._params.value[1]
        else:
            raise ValueError(
                f"mode={self._mode}, but you can use only random and value modes"
            )
        if self._params.context is None:
            contexts = np.random.randint(0, 2, size=batch_size)  # generate two contexts
        else:
            contexts = np.ones(batch_size, dtype=np.int) * self._params.context

        inputs = np.zeros(
            (
                trial_time + answer_time,
                batch_size,
                self._ob_size + (2 if self._get_context else 0),
            )
        )
        inputs[:trial_time, :, 0] = 1
        inputs[:trial_time, :, 1] = value_1[:]
        inputs[:trial_time, :, 2] = value_2[:]
        if self._get_context:
            inputs[:, :, 3] = (
                1 - contexts
            )  # 0 is the first context => 1 - 0 = 1 is first
            inputs[:, :, 4] = contexts  # 1 is the second context => 1 is second
        target_outputs = np.zeros(
            (trial_time + answer_time, batch_size, self._act_size)
        )
        target_outputs[:, :, 0] = inputs[:, :, 0]
        target_outputs[trial_time:, :, 1] = (value_1 < self._threshold) * (
            1 - contexts
        ) + (value_2 < self._threshold) * contexts
        target_outputs[trial_time:, :, 2] = (value_1 > self._threshold) * (
            1 - contexts
        ) + (value_2 > self._threshold) * contexts
        if self._get_context:
            self._contexts = contexts
        return inputs, target_outputs

    def _one_dataset(self):
        """
        Returns a single batch of inputs for one or more contexts .

        Args:
            context (int): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if self._uniq_batch:
            return self._unique_every_batch()
        else:
            return self._identical_batches(self._batch_size)

    def one_dataset(self):
        """
        Return a single dataset with one context .

        Args:
            context (int, optional): [description]. Defaults to np.random.choice([0, 1]).

        Returns:
            [type]: [description]
        """
        return self._one_dataset()

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
        params: Optional[CtxDMTaskParameters] = CtxDMTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
        threshold: float = 0.5,
    ):
        params = CtxDMTaskParameters(dm=params.dm, context=0, value=params.value)
        super().__init__(
            params, batch_size, mode, enable_fixation_delay, uniq_batch, threshold
        )

    def one_dataset(self):
        """
        Return single dataset with only one dataset .

        Returns:
            [type]: [description]
        """
        return self._one_dataset()

    @property
    def name(self):
        return "CtxDM1"


class CtxDM2(CtxDMTask):
    """
    A context manager for creating a CtxDM1 code .

    Args:
        CtxDMTask ([type]): [description]
    """

    def __init__(
        self,
        params: Optional[CtxDMTaskParameters] = CtxDMTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
        threshold: float = 0.5,
    ):
        params = CtxDMTaskParameters(dm=params.dm, context=1, value=params.value)
        super().__init__(
            params, batch_size, mode, enable_fixation_delay, uniq_batch, threshold
        )

    def one_dataset(self):
        """
        Return single dataset with only one dataset .

        Returns:
            [type]: [description]
        """
        return self._one_dataset()

    @property
    def name(self):
        return "CtxDM2"
