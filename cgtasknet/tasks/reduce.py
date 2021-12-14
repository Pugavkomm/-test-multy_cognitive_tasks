from typing import Optional, Tuple, Union

import numpy as np


class DefaultParams:
    """
    This method is used to create a new Params class .
    """

    def __init__(self, task: str):
        """
        Initialize the task.

        Args:
            task (str): [Name of task]
        """
        self._task = task

    def generate_params(self):
        """
        Generate parameters for the task .

        Returns:
            [type]: [description]
        """
        if self._task == "DMTask":
            return dict([("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.75)])
        elif self._task == "RomoTask":
            return dict([("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.25)])
        elif self._task == "CtxDMTask":
            return dict([("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.75)])
        elif self._task == "DMTaskRandomMod":
            return dict(
                [("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.75), ("n_mods", 2)]
            )
        elif self._task == "RomoTaskRandomMod":
            return dict(
                [("dt", 1e-3), ("delay", 0.3), ("trial_time", 0.25), ("n_mods", 2)]
            )

        return None


class ReduceTaskCognitive:
    """
    Class method for ReduceTask .
    """

    def __init__(self, params: dict, batch_size: int, mode: str) -> None:
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

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a tuple containing one - dimensional dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        return Tuple[np.ndarray, np.ndarray]

    def dataset(self, n_trials: int = 1):
        multy_inputs, multy_outputs = self.one_dataset()
        for _ in range(n_trials - 1):
            inputs, outputs = self.one_dataset()
            multy_inputs = np.concatenate((multy_inputs, inputs), axis=0)
            multy_outputs = np.concatenate((multy_outputs, outputs), axis=0)
        return multy_inputs, multy_outputs

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
    def params(self) -> dict:
        """
        A dictionary with the current parameters .

        Returns:
            dict: [description]
        """
        return self._params

    @params.setter
    def params(self, new_params: dict):
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
        self, params: Optional[dict] = None, batch_size: int = 1, mode: str = "random"
    ) -> None:
        """
        Initialize the model .

        Args:
            params (Optional[dict], optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 1.
            mode (str, optional): [description]. Defaults to "random".
        """
        if params is None:
            params = DefaultParams("DMTask").generate_params()
        super().__init__(params, batch_size, mode)
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with inputs and target outputs .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, outputs]
        """
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
    """
    Class method for DMTask class .

    Args:
        DMTask ([type]): [description]
    """

    def __init__(
        self, params: Optional[dict] = None, batch_size: int = 1, mode: str = "random"
    ) -> None:
        """
        Initialize the model .

        Args:
            batch_size (int): [description]
            params (Optional[dict], optional): [description]. Defaults to None.
            mode (str, optional): [description]. Defaults to "random".
            n_mods (int, optional): [description]. Defaults to 1.
        """
        if params is None:
            params = DefaultParams("DMTaskRandomMod").generate_params()
        super().__init__(params, batch_size, mode)
        self._n_mods = params["n_mods"]
        self._ob_size += self._n_mods - 1

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single dataset .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, target outputs]
        """
        temp, outputs = self._one_dataset()
        T = temp.shape[0]
        inputs = np.zeros((T, self._batch_size, self._ob_size))
        curent_mod = np.random.randint(0, self._n_mods)
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + curent_mod] = temp[:, :, 1]
        return inputs, outputs


class RomoTask(ReduceTaskCognitive):
    """
    Constructs a RiveTask .

    Args:
        ReduceTaskCognitive ([type]): [description]
    """

    def __init__(
        self, params: Optional[dict] = None, batch_size: int = 1, mode: str = "random"
    ) -> None:
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        if params is None:
            params = DefaultParams("RomoTask").generate_params()
        super().__init__(params, batch_size, mode)
        self._ob_size = 2
        self._act_size = 3

    def _one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a single dataset with the given size and target .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
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
            inputs[trial_time + delay : -delay, :, 1] = values_second
            target_output = np.zeros(
                (2 * (trial_time + delay), self._batch_size, self._act_size)
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


class RomoTaskRandomMod(RomoTask):
    """
    Trial task that is used for a random mod .

    Args:
        RomoTask ([type]): [description]
    """

    def __init__(
        self, params: Optional[dict] = None, batch_size: int = 1, mode: str = "random"
    ) -> None:
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
            n_mods (int, optional): [description]. Defaults to 1.
        """
        if params is None:
            params = DefaultParams("RomoTaskRandomMod").generate_params()
        super().__init__(params, batch_size, mode)
        self._n_mods = params["n_mods"]
        self._ob_size += self._n_mods - 1

    def one_dataset(self):
        """
        Generate a single model .

        Returns:
            [type]: [description]
        """
        temp, outputs = self._one_dataset()
        T = temp.shape[0]
        inputs = np.zeros((T, self._batch_size, self._ob_size))
        curent_mod = np.random.randint(0, self._n_mods)
        inputs[:, :, 0] = temp[:, :, 0]
        inputs[:, :, 1 + curent_mod] = temp[:, :, 1]
        return inputs, outputs


class CtxDMTask(ReduceTaskCognitive):
    """
    Context manager for CtxDMTask .

    Args:
        ReduceTaskCognitive ([type]): [description]
    """

    def __init__(
        self, params: Optional[dict] = None, batch_size: int = 1, mode: str = "random"
    ):
        """
        Initialize the DMTask .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        super().__init__(params, batch_size, mode)
        params = DefaultParams("CtxDMTask").generate_params()
        self.DMTask = DMTask(params, batch_size, mode)
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
    def params(self, new_params: dict):
        """
        Set the parameters of this DMTask .

        Args:
            new_params (dict): [description]
        """
        self._params = new_params
        self.DMTask = DMTask(new_params, self._batch_size, self._mode)


class CtxDM1(CtxDMTask):
    """
    A context manager for creating a CtxDM1 code .

    Args:
        CtxDMTask ([type]): [description]
    """

    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        super().__init__(params, batch_size, mode)

    def one_dataset(self):
        """
        Return single dataset with only one dataset .

        Returns:
            [type]: [description]
        """
        return self._one_dataset(0)


class CtxDM2(CtxDMTask):
    """
    A context manager for creating a CtxDMTask class .

    Args:
        CtxDMTask ([type]): [description]
    """

    def __init__(self, params: dict, batch_size: int, mode: str = "random"):
        """
        Initialize the model .

        Args:
            params (dict): [description]
            batch_size (int): [description]
            mode (str, optional): [description]. Defaults to "random".
        """
        super().__init__(params, batch_size, mode)

    def one_dataset(self):
        """Return a single dataset."""
        return self._one_dataset(1)


class MultyReduceTasks(ReduceTaskCognitive):
    task_list = [
        ("DMTask", DMTaskRandomMod),
        ("RomoTask", RomoTaskRandomMod),
        ("CtxDMTask", CtxDMTask),
    ]
    task_list.sort()
    TASKSDICT = dict(task_list)
    n_mods = 2

    def __init__(
        self,
        tasks: Union[dict[str, dict[str, float]], list[str]],
        batch_size: int = 1,
        mode: str = "random",
    ):
        """
        Initialize the object with the initial state of the model .

        Args:
            tasks (Union[dict[str, dict[str, float]], list[str]]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            mode (str, optional): [description]. Defaults to "random".
        """
        self._initial_tasks_list = dict()
        if type(tasks) == list:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    batch_size=batch_size, mode=mode
                )
        if type(tasks) == dict:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    params=tasks[task_name], batch_size=batch_size, mode=mode
                )
        self._tasks = tasks
        self._ob_size, self._act_size = self.feature_and_act_size
        self._ob_size += len(tasks)
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
        return inputs_plus_rule, outputs

    def one_dataset(self):
        return self._one_dataset()

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
        ob_size = 3
        act_size = 3
        return ob_size, act_size

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
