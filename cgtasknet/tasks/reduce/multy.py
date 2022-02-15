from typing import Any, Dict, List, Tuple, Union

import numpy as np

from cgtasknet.tasks.reduce.ctxdm import CtxDM1, CtxDM2, CtxDMTask
from cgtasknet.tasks.reduce.dm import DMTask1, DMTask2, DMTaskRandomMod
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
    ]
    task_list.sort()
    TASKSDICT = dict(task_list)

    def __init__(
        self,
        tasks: Union[Dict[str, Any], List[str]],
        batch_size: int = 1,
        mode: str = "random",
        delay_between: int = 0,  # iterations
        number_of_inputs: int = 2,
        enable_fixation_delay: bool = False,
        sequence_bathces: bool = False,
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
        self._initial_tasks_list = dict()
        self._enable_fixation_delay = enable_fixation_delay
        self._sequence_batchers = sequence_bathces
        if type(tasks) == list:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    batch_size=batch_size if (not sequence_bathces) else 1,
                    mode=mode,
                    enable_fixation_delay=enable_fixation_delay,
                )
        if type(tasks) == dict:
            for task_name in tasks:
                self._initial_tasks_list[task_name] = self.TASKSDICT[task_name](
                    params=tasks[task_name],
                    batch_size=batch_size if (not sequence_bathces) else 1,
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
        if self._sequence_batchers:
            numbers_of_tasks = np.random.randint(
                0, len(self._task_list), size=self._batch_size
            )
            max_length = 0
            l_intputs = []
            l_outputs = []
            for i in range(self._batch_size):
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

        else:
            current_task = np.random.randint(0, len(self._task_list))
            inputs, outputs = self._task_list[current_task].one_dataset()
            inputs_plus_rule = np.zeros(
                (inputs.shape[0], self._batch_size, self._ob_size)
            )
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
