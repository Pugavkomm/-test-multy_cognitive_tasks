from typing import NamedTuple, Optional, Tuple, Union

import numpy as np

from cgtasknet.tasks.reduce.reduce_task import (
    _generate_random_intervals,
    _generate_values,
    ReduceTaskCognitive,
    ReduceTaskParameters,
)


class GoTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = 0.75
    answer_time: float = ReduceTaskParameters().answer_time
    value: Union[float, list, tuple] = 1.0

    # task_type: str = "Go"  # Go, Rt, Dl
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class GoRtTaskParameters(NamedTuple):
    dt: float = ReduceTaskParameters().dt
    trial_time: float = 0.75
    answer_time: float = ReduceTaskParameters().answer_time
    negative_shift_answer_time: float = 0.0
    positive_shift_answer_time: float = 0.0
    value: Union[float, list, tuple] = 1.0

    # task_type: str = "Go"  # Go, Rt, Dl
    negative_shift_trial_time: float = ReduceTaskParameters().negative_shift_trial_time
    positive_shift_trial_time: float = ReduceTaskParameters().positive_shift_trial_time


class GoTaskRandomModParameters(NamedTuple):
    go: GoTaskParameters = GoTaskParameters()
    n_mods: int = 2


class GoRtTaskRandomModParameters(NamedTuple):
    go_rt: GoRtTaskParameters = GoRtTaskParameters()
    n_mods: int = 2


class GoDlTaskParameters(NamedTuple):
    go: GoTaskParameters = GoTaskParameters()
    delay: float = 1.0
    negative_shift_delay_time: float = 0.0
    positive_shift_delay_time: float = 0.0


class GoDlTaskRandomModParameters(NamedTuple):
    go_dl: GoDlTaskParameters = GoDlTaskParameters()
    n_mods: int = 2


class GoTask(ReduceTaskCognitive):
    def __init__(
        self,
        params: GoTaskParameters = GoTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )

        self._ob_size = 2
        self._act_size = 2

    def _identical_batches(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._params.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.trial_time,
            self._params.negative_shift_trial_time,
            self._params.positive_shift_trial_time,
        )
        answer_time = round(self._params.answer_time / dt)
        inputs = np.zeros((trial_time + answer_time, batch_size, self._ob_size))
        target_outputs = np.zeros(
            (trial_time + answer_time, batch_size, self._act_size)
        )
        values = _generate_values(self._mode, batch_size, self._params.value)
        inputs[:trial_time, :, 0] = 1
        inputs[:, :, 1] = values
        target_outputs[:, :, 0] = inputs[:, :, 0]
        target_outputs[trial_time:, :, 1] = values

        return inputs, target_outputs

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._uniq_batch:
            return self._unique_every_batch()
        else:
            return self._identical_batches(self._batch_size)

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        return self._one_dataset()

    @property
    def name(self) -> str:
        return "Go"


class GoRtTask(GoTask):
    def __init__(
        self,
        params: GoRtTaskParameters = GoRtTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )

        self._ob_size = 2
        self._act_size = 2

    def _identical_batches(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._params.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.trial_time,
            self._params.negative_shift_trial_time,
            self._params.positive_shift_trial_time,
        )
        answer_time = _generate_random_intervals(
            dt,
            self._params.answer_time,
            self._params.negative_shift_answer_time,
            self._params.positive_shift_answer_time,
        )
        inputs = np.zeros((trial_time + answer_time, batch_size, self._ob_size))
        target_outputs = np.zeros(
            (trial_time + answer_time, batch_size, self._act_size)
        )
        values = _generate_values(self._mode, batch_size, self._params.value)
        inputs[:, :, 0] = 1
        inputs[trial_time:, :, 1] = values
        target_outputs[:trial_time, :, 0] = 1
        target_outputs[trial_time:, :, 1] = values

        return inputs, target_outputs

    @property
    def name(self) -> str:
        return "GoRt"


class GoDlTask(GoTask):
    def __init__(
        self,
        params: Union[GoDlTaskParameters, GoRtTaskParameters] = GoDlTaskParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )

        self._ob_size = 2
        self._act_size = 2

    def _identical_batches(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._params.go.dt
        trial_time = _generate_random_intervals(
            dt,
            self._params.go.trial_time,
            self._params.go.negative_shift_trial_time,
            self._params.go.positive_shift_trial_time,
        )
        answer_time = round(self._params.go.answer_time / dt)
        delay_time = _generate_random_intervals(
            dt,
            self._params.delay,
            self._params.negative_shift_delay_time,
            self._params.positive_shift_delay_time,
        )

        inputs = np.zeros(
            (trial_time + answer_time + delay_time, batch_size, self._ob_size)
        )
        target_outputs = np.zeros(
            (trial_time + answer_time + delay_time, batch_size, self._act_size)
        )
        values = _generate_values(self._mode, batch_size, self._params.go.value)
        inputs[: trial_time + delay_time, :, 0] = 1
        inputs[:trial_time, :, 1] = values
        target_outputs[:, :, 0] = inputs[:, :, 0]
        target_outputs[trial_time + delay_time :, :, 1] = values
        return inputs, target_outputs

    @property
    def name(self) -> str:
        return "GoDl"


class GoTaskRandomMod(GoTask):
    def __init__(
        self,
        params: GoTaskRandomModParameters = GoTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ):
        super().__init__(
            params=params.go,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._n_mods = params.n_mods
        self._ob_size += self._n_mods - 1
        self._act_size += self._n_mods - 1

    def _one_dataset_mod(self, mod: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, target outputs]
        """
        temp, temp_outputs = self._one_dataset()
        t = temp.shape[0]
        inputs = np.zeros((t, self._batch_size, self._ob_size))
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + mod] = temp[:, :, 1]
        target_outputs = np.zeros((t, self._batch_size, self._act_size))
        target_outputs[:, :, 0] = temp_outputs[:, :, 0]
        target_outputs[:, :, 1 + mod] = temp_outputs[:, :, 1]
        return inputs, target_outputs

    def one_dataset(self, mode: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if mode is None:
            mode = np.random.randint(0, self._n_mods)
        return self._one_dataset_mod(mode)

    @property
    def name(self):
        return "GoTaskRandomMod"


class GoRtTaskRandomMod(GoRtTask):
    def __init__(
        self,
        params: GoRtTaskRandomModParameters = GoRtTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ):
        super().__init__(
            params=params.go_rt,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._n_mods = params.n_mods
        self._ob_size += self._n_mods - 1
        self._act_size += self._n_mods - 1

    def _one_dataset_mod(self, mod: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, target outputs]
        """
        temp, temp_outputs = self._one_dataset()
        t = temp.shape[0]
        inputs = np.zeros((t, self._batch_size, self._ob_size))
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + mod] = temp[:, :, 1]
        target_outputs = np.zeros((t, self._batch_size, self._act_size))
        target_outputs[:, :, 0] = temp_outputs[:, :, 0]
        target_outputs[:, :, 1 + mod] = temp_outputs[:, :, 1]
        return inputs, target_outputs

    def one_dataset(self, mod: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if mod is None:
            mod = np.random.randint(0, self._n_mods)
        return self._one_dataset_mod(mod)

    @property
    def name(self) -> str:
        return "GoRtTaskRandomMod"


class GoDlTaskRandomMod(GoDlTask):
    def __init__(
        self,
        params: GoDlTaskRandomModParameters = GoDlTaskRandomModParameters(),
        batch_size: int = 1,
        mode: str = "random",
        enable_fixation_delay: bool = False,
        uniq_batch: bool = False,
    ):
        super().__init__(
            params=params.go_dl,
            batch_size=batch_size,
            mode=mode,
            enable_fixation_delay=enable_fixation_delay,
            uniq_batch=uniq_batch,
        )
        self._n_mods = params.n_mods
        self._ob_size += self._n_mods - 1
        self._act_size += self._n_mods - 1

    def _one_dataset_mod(self, mod: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, target outputs]
        """
        temp, temp_outputs = self._one_dataset()
        t = temp.shape[0]
        inputs = np.zeros((t, self._batch_size, self._ob_size))
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + mod] = temp[:, :, 1]
        target_outputs = np.zeros((t, self._batch_size, self._act_size))
        target_outputs[:, :, 0] = temp_outputs[:, :, 0]
        target_outputs[:, :, 1 + mod] = temp_outputs[:, :, 1]
        return inputs, target_outputs

    def one_dataset(self, mod: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if mod is None:
            mod = np.random.randint(0, self._n_mods)
        return self._one_dataset_mod(mod)

    @property
    def name(self) -> str:
        return "GoDlTaskRandomMod"


class GoTask1(GoTaskRandomMod):
    def one_dataset(self, mod=0) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)


class GoTask2(GoTaskRandomMod):
    def one_dataset(self, mod=1) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)


class GoRtTask1(GoRtTaskRandomMod):
    def one_dataset(self, mod=0) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)


class GoRtTask2(GoRtTaskRandomMod):
    def one_dataset(self, mod=1) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)


class GoDlTask1(GoDlTaskRandomMod):
    def one_dataset(self, mod=0) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)


class GoDlTask2(GoDlTaskRandomMod):
    def one_dataset(self, mod=1) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset_mod(mod)
