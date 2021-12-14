from typing import Tuple

import numpy as np


class DefaultParams:
    def __init__(self, task: str):
        self._task = task

    def generate_params(self):
        if self._task == "DM":
            return dict([("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.75)])
        elif self._task == "Romo":
            return dict([("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.25)])
        return None


class ReduceTaskCognitive:
    def __init__(self, params: dict, batch_size: int, mode: str) -> None:
        self._params = params
        self._batch_size = batch_size
        self._ob_size = 0
        self._act_size = 0
        self._mode = mode

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return Tuple[np.ndarray, np.ndarray]

    @property
    def feature_and_act_size(self):
        return self._ob_size, self._act_size

    @feature_and_act_size.setter
    def feature_and_act_size(self, values: Tuple[int, int]):
        self._ob_size, self._act_size = values

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, new_params: dict):
        self._params = new_params

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        self._batch_size = new_batch_size


class DMTask(ReduceTaskCognitive):
    threshold = 0.5

    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        super().__init__(params, batch_size, mode)
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._params["dt"]
        trial_time = int(self._params["trial_time"] / dt)
        delay = int(self._params["delay"] / dt)
        if self._mode == "random":
            values = np.random.uniform(0, 1, size=(self._batch_size))
            inputs = np.zeros((trial_time + delay, self._batch_size, self._ob_size))
            inputs[:trial_time, :, 0] = 1
            inputs[:trial_time, :, 1] = values[:]
            target_outputs = np.zeros(
                (trial_time + delay, self._batch_size, self._act_size)
            )
            target_outputs[:, :, 0] = inputs[:, :, 0]
            target_outputs[trial_time:, :, 1] = values < self.threshold
            target_outputs[trial_time:, :, 2] = values > self.threshold
            return inputs, target_outputs

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._one_dataset()


class DMTaskRandomMod(DMTask):
    def __init__(
        self, params: dict, batch_size: int, mode: str = "random", n_mods=1
    ) -> None:
        super().__init__(params, batch_size, mode)
        self._n_mods = n_mods

    def one_dataset(self):
        temp, outputs = self._one_dataset()
        T = temp.shape[0]
        inputs = np.zeros((T, self._batch_size, self._ob_size + self._n_mods - 1))
        curent_mod = np.random.randint(0, self._n_mods)
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + curent_mod] = temp[:, :, 1]
        return inputs, outputs


class RomoTask(ReduceTaskCognitive):
    def __init__(self, params: dict, batch_size: int, mode: str = "random") -> None:
        super().__init__(params, batch_size, mode)
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        dt = self.params["dt"]
        trial_time = int(self._params["trial_time"] / dt)
        delay = int(self._params["delay"] / dt)
        if self._mode == "random":
            values_first = np.random.uniform(0, 1, size=(self._batch_size))
            values_second = np.random.uniform(0, 1, size=(self._batch_size))
            # TODO: добавить проверку на совпадения (хотя это маловероятно)
            inputs = np.zeros(
                (2 * (trial_time + delay), self._batch_size, self._ob_size)
            )
            inputs[: 2 * trial_time + delay, :, 0] = 1
            inputs[:trial_time, :, 1] = values_first
            inputs[trial_time + delay: -delay, :, 1] = values_second
            target_output = np.zeros(
                (2 * (trial_time + delay), self._batch_size, self._act_size)
            )
            target_output[:, :, 0] = inputs[:, :, 0]
            target_output[2 * trial_time + delay:, :, 1] = values_first < values_second
            target_output[2 * trial_time + delay:, :, 2] = values_second < values_first
            return inputs, target_output

    def one_dataset(self):
        return self._one_dataset()


class RomoTaskRandomMod(RomoTask):
    def __init__(
        self, params: dict, batch_size: int, mode: str = "random", n_mods=1
    ) -> None:
        super().__init__(params, batch_size, mode)
        self._n_mods = n_mods

    def one_dataset(self):
        temp, outputs = self._one_dataset()
        T = temp.shape[0]
        inputs = np.zeros((T, self._batch_size, self._ob_size + self._n_mods - 1))
        curent_mod = np.random.randint(0, self._n_mods)
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + curent_mod] = temp[:, :, 1]
        return inputs, outputs


class CtxDM(ReduceTaskCognitive):
    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        super().__init__(params, batch_size, mode)
        self.DMTask = DMTask(params, batch_size, mode)

    def _one_dataset(self, context: int):
        inputs_first_context, outputs_first_context = self.DMTask.one_dataset()
        inputs_second_context, outputs_second_context = self.DMTask.one_dataset()
        inputs = np.zeros(
            (inputs_first_context.shape[0], self._batch_size, self._ob_size + 1))
        inputs[:, :, 0] = inputs_first_context[:, :, 0]
        inputs[:, :, 1] = inputs_first_context[:, :, 1]
        inputs[:, :, 2] = inputs_second_context[:, :, 1]
        if context == 0:
            return inputs, outputs_first_context
        if context == 1:
            return inputs, outputs_second_context
        else:
            raise ValueError(f'param: context expected 0 or 1, but actual {context}')

    def one_dataset(self, context: int = np.random.choice([0, 1]), *args, **kwargs):
        return self._one_dataset(context)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params: dict):
        self._params = new_params
        self.DMTask = DMTask(new_params, self._batch_size, self._mode)


class CtxDM1(CtxDM):
    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        super().__init__(params, batch_size, mode)

    def one_dataset(self):
        return self._one_dataset(0)


class CtxDM2(CtxDM):
    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        super().__init__(params, batch_size, mode)

    def one_dataset(self):
        return self._one_dataset(1)


class MultyTask:

    def __init__(self):
        return None
