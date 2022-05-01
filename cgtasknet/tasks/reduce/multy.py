from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from cgtasknet.tasks.reduce.ctxdm import CtxDM1, CtxDM2, CtxDMTask
from cgtasknet.tasks.reduce.dm import DMTask1, DMTask2, DMTaskRandomMod
from cgtasknet.tasks.reduce.go import (
    GoDlTask1,
    GoDlTask2,
    GoRtTask1,
    GoRtTask2,
    GoTask1,
    GoTask2,
)
from cgtasknet.tasks.reduce.reduce_task import ReduceTaskCognitive
from cgtasknet.tasks.reduce.romo import RomoTask1, RomoTask2, RomoTaskRandomMod


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
        ("GoTask1", GoTask1),
        ("GoTask2", GoTask2),
        ("GoRtTask1", GoRtTask1),
        ("GoRtTask2", GoRtTask2),
        ("GoDlTask1", GoDlTask1),
        ("GoDlTask2", GoDlTask2),
    ]
    task_list.sort()
    TASKSDICT = OrderedDict(task_list)

    def __init__(
        self,
        tasks: Union[Dict[str, Any], List[str], OrderedDict],
        batch_size: int = 1,
        mode: str = "random",
        delay_between: int = 0,  # iterations
        number_of_inputs: int = 2,
        enable_fixation_delay: bool = False,
        task_number: int = None,
    ):
        """
        Initialize the object with the initial state of the model .

        Args:
            tasks (Union[dict[str, dict[str, float]], list[str]]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            mode (str, optional): [description]. Defaults to "random".
            delay_between (int, optional): [description]. Number of iteration without signal
        """
        self._delay_between = delay_between
        self._initial_tasks_list = OrderedDict()
        self._enable_fixation_delay = enable_fixation_delay
        if isinstance(tasks, list):
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    batch_size=1,
                    mode=mode,
                    enable_fixation_delay=enable_fixation_delay,
                )
        if isinstance(tasks, dict):
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    params=tasks[task_name],
                    batch_size=1,
                    mode=mode,
                    enable_fixation_delay=enable_fixation_delay,
                )
        self._tasks = tasks
        self._ob_size = 1 + number_of_inputs + len(tasks)
        self._act_size = 3
        self._sorted_tasks()
        self._create_task_list()
        self._batch_size = batch_size
        self._correct_task_number(task_number)
        self._task_number = task_number

    def _identical_batches(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs one step of the batch .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        if self._task_number is None:

            numbers_of_tasks = np.random.randint(
                0, len(self._task_list), size=self._batch_size
            )
        else:
            numbers_of_tasks = np.ones(self._batch_size, dtype=int) * self._task_number
        max_length = 0
        l_intputs = []
        l_outputs = []
        for i in range(batch_size):
            inputs, outputs = self._task_list[numbers_of_tasks[i]].one_dataset()
            input_bufer = np.zeros((inputs.shape[0], 1, self._ob_size))
            input_bufer[:, :, : inputs.shape[-1]] = inputs[...]
            input_bufer[:, :, inputs.shape[-1] + numbers_of_tasks[i]] = 1
            l_intputs.append(input_bufer)
            l_outputs.append(outputs)
            max_length = max(max_length, inputs.shape[0])
        inputs_plus_rule, outputs = self._concatenate_batches(
            l_intputs, l_outputs, max_length
        )
        input_shape = self._ob_size - len(self._task_list)
        indexes_rules = np.where(inputs_plus_rule[-1, :, input_shape:] == 1)
        inputs_plus_rule[:, indexes_rules[0], indexes_rules[1] + input_shape] = 1
        return inputs_plus_rule, outputs

    def one_dataset(self):
        """
        Returns a single dataset with inputs and target outputs .

        Returns:
            Tuple[np.ndarray, np.ndarray]: [inputs, outputs]
        """
        return self._identical_batches(self._batch_size)

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
        new_dict = OrderedDict()
        for task_name in sorted(self._initial_tasks_list):
            new_dict[task_name] = self._initial_tasks_list[task_name]
        self._initial_tasks_list = new_dict

    def _correct_task_number(self, task_number: Union[None, int]):
        if task_number is not None:
            if not isinstance(task_number, int):
                raise TypeError(f"task_number has type: {type(task_number)}")
            if task_number >= len(self._tasks) or task_number < 0:
                raise IndexError(
                    f"task_number must be from 0 to {len(self._tasks)}, but actual task_number = {task_number}"
                )

    @property
    def feature_and_act_size(self) -> Tuple[int, int]:
        """
        Return the feature and act size .

        Returns:
            Tuple[int, int]: [description]
        """
        return self._ob_size, self._act_size

    @property
    def feature_and_act_every_task_size(self) -> Dict[str, Tuple[int, int]]:
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

    @property
    def task_number(self):
        return self._task_number

    @task_number.setter
    def task_number(self, new_task_number):
        self._correct_task_number(new_task_number)
        self._task_number = new_task_number
