from typing import Tuple

import numpy as np


def _compare_time(f_time, interval):
    """
    Compares time with interval less than interval

    Args:
        f_time ([type]): [description]
        interval ([type]): [description]

    Returns:
        [type]: [description]
    """
    if f_time < interval:
        f_time = interval
    elif f_time % interval != 0:
        f_time -= f_time % interval
    return f_time


class TaskCognitive:
    """
    A class method to create a TaskCognitive class .

    Returns:
        [type]: [description]
    """

    ob_size = 0
    act_size = 0

    def __init__(self, params: dict, batch_size: int) -> None:
        self._params = params
        self._batch_size = batch_size

    def one_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return Tuple[np.ndarray, np.ndarray]

    def dataset(self, number_of_trials: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        inputs = []
        outputs = []
        for _ in range(number_of_trials):
            one_trial_input, one_trial_output = self.one_dataset()
            inputs.append(one_trial_input)
            outputs.append(one_trial_output)
        inputs = np.concatenate(inputs, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        return inputs, outputs

    @property
    def feature_and_act_size(self):
        return self.ob_size, self.act_size

    @property
    def task_parameters(self):
        """Property to return the Tasks parameters ."""
        return self._params

    @task_parameters.setter
    def task_parameters(self, params: dict):
        """Setter for task parameters ."""
        self._params = params

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size


class ContextDM(TaskCognitive):

    ob_size = 5  # number of inputs
    act_size = 3  # number of outputs

    def __init__(
        self,
        params: dict = dict(
            [
                ("sigma", 0),
                ("fixation", 0.3),
                ("target", 0.35),
                ("delay", 0.3),
                ("trial", 0.75),
                ("dt", 1e-3),
            ]
        ),
        batch_size: int = 1,
    ) -> None:
        super().__init__(params, batch_size)

    def one_dataset(self):
        sigma = self._params["sigma"]
        t_fixation = self._params["fixation"]
        t_target = self._params["target"]
        t_delay = self._params["delay"]
        t_trial = self._params["trial"]
        batch_size = self._batch_size
        dt = self._params["dt"]

        # two stimuly and two (m.b. one) context signal = 4 (3) inputs and fixation

        fixation = int(t_fixation / dt)
        target = int(t_target / dt)
        delay = int(t_delay / dt)
        trial = int(t_trial / dt)
        full_interval = fixation + target + delay + trial
        full_interval_and_delay = full_interval + delay
        context = np.zeros((2, batch_size))
        # inputs = np.zeros((0, batch_size, self.ob_size))
        # outputs = np.zeros((0, batch_size, self.act_size))

        context[0, :] = np.random.choice([0, 1], size=batch_size)
        context[1, :] = 1 - context[0, :]
        move_average = np.random.uniform(0, 1, size=batch_size)
        color_average = np.random.uniform(0, 1, size=batch_size)
        move_average_label = move_average > 0.5
        move_average_label = move_average_label.astype(np.longlong)
        color_average_label = color_average > 0.5
        color_average_label = color_average_label.astype(np.longlong)
        fixation_array = np.ones(
            (full_interval, batch_size, 1)
        )  # m.b full_interva - time of between trials
        context_one = np.ones((full_interval, batch_size, 1))
        context_one[:, :, 0] *= context[0]
        context_two = np.ones((full_interval, batch_size, 1))
        context_two[:, :, 0] *= context[1]
        input_one = np.zeros((full_interval, batch_size, 1))
        input_two = np.zeros((full_interval, batch_size, 1))
        output_one = np.zeros((full_interval, batch_size, 1))
        output_two = np.zeros((full_interval, batch_size, 1))

        target_fixation = np.zeros((full_interval_and_delay, batch_size, 1))
        target_fixation[0:full_interval, ...] = fixation_array[...]

        indexes_context = np.where(context == 0)[0].astype(bool)  # list 0 1 0 1 1 0
        for j in range(batch_size):
            if sigma == 0:
                input_one[:, j] += np.ones((full_interval, 1)) * move_average[j]
                input_two[:, j] += np.ones((full_interval, 1)) * color_average[j]
            else:
                input_one[:, j] += np.random.normal(
                    move_average[j], sigma, size=(full_interval, 1)
                )
                input_two[:, j] += np.random.normal(
                    color_average[j], sigma, size=(full_interval, 1)
                )

            if indexes_context[j]:
                output_one[:, j] += move_average_label[j]
                output_two[:, j] += 1 - output_one[:, j]
            else:
                output_one[:, j] += color_average_label[j]
                output_two[:, j] += 1 - output_one[:, j]
        inputs = np.concatenate(
            (fixation_array, input_one, input_two, context_one, context_two), axis=-1
        )
        inputs = np.concatenate(
            (inputs, np.zeros((delay, self._batch_size, self.ob_size)))
        )
        outputs = np.concatenate((fixation_array, output_one, output_two), axis=-1)
        outputs = np.concatenate(
            (outputs, np.zeros((delay, self._batch_size, self.act_size)))
        )
        return inputs, outputs


class WorkingMemory(TaskCognitive):

    r"""
    Neuronal correlates of parametric working memory in the prefrontal cortex
    Ranulfo Romo, Carlos D. Brody, Adria ́n Herna ́ndez & Luis Lemus
    Instituto de Fisiologı ́a Celular, Universidad Nacional Autono ́ma de Me ́xico,
    Me ́xico D.F. 04510, Me ́xico
    """

    ob_size = 2  # number of inputs: fix + stimul
    act_size = 3  # number of outputs: fix + two outputs

    def __init__(
        self,
        params: dict = dict(
            [
                ("dt", 1e-3),  # step 1ms
                ("delay", 0.5),  # 500 ms
                ("trial", 0.5),  # 500 ms
                ("KU", 0.05),  # 50 ms
                ("PB", 0.05),  # 50 ms
                ("min", 0),
                ("max", 1),
                ("first", -1),
                ("second", -1),
            ]
        ),
        batch_size: int = 1,
    ) -> None:
        super().__init__(params, batch_size)

    def one_dataset(self):
        dt = self._params["dt"]
        delay = int(self._params["delay"] / dt)
        trial = int(self._params["trial"] / dt)
        KU = int(self._params["KU"] / dt)
        PB = int(self._params["PB"] / dt)
        full_interval = delay + 2 * trial + KU + PB
        start_base = 0
        end_base = trial
        start_compare = trial + delay
        end_compare = start_compare + trial
        start_act = end_compare - trial
        fixation_interval = end_compare  # full_interval - PB
        min_stimulus = self._params["min"]
        max_stimulus = self._params["max"]
        fixation = np.zeros((full_interval, self._batch_size, 1))
        fixation[0:fixation_interval] = 1
        if self._params["first"] == -1:
            base_stimulus = np.random.uniform(
                min_stimulus, max_stimulus, size=self._batch_size
            )
        else:
            base_stimulus = np.ones(self._batch_size) * self._params["first"]
        if self._params["second"] == -1:
            comparison = np.random.uniform(
                min_stimulus, max_stimulus, size=self._batch_size
            )
        else:
            comparison = np.ones(self._batch_size) * self._params["second"]

        trial_input = np.zeros((full_interval, self._batch_size, 1))
        trial_output = np.zeros((full_interval, self._batch_size, 2))
        for batch in range(self._batch_size):
            base = base_stimulus[batch]
            compare = comparison[batch]
            input_stimulus = np.zeros((full_interval, 1, 1))
            input_stimulus[start_base:end_base] = base
            input_stimulus[start_compare:end_compare] = compare
            trial_input[:, batch, 0] = input_stimulus[:, 0, 0]
            act_output = np.zeros((full_interval, 1, 2))
            act_output[start_act:, 0, int(compare > base)] = 1
            trial_output[:, batch, :] = act_output[:, 0, :]
        inputs = np.concatenate((fixation, trial_input), axis=-1)
        outputs = np.concatenate((fixation, trial_output), axis=-1)
        inputs = np.concatenate(
            (inputs, np.zeros((delay * 2, inputs.shape[1], inputs.shape[2])))
        )
        outputs = np.concatenate(
            (outputs, np.zeros((delay * 2, outputs.shape[1], outputs.shape[2])))
        )
        return inputs, outputs


class CompareObjects(TaskCognitive):
    ob_size = 2  # number of inputs (fixation + one input)
    act_size = 2  # number of outputs (fixation + one output)
    examples = (0.1, 0.3, 0.5, 0.9)

    def __init__(
        self,
        params: dict = dict(
            [("dt", 1e-3), ("delay", 1), ("trial", 0.5), ("time_object", 0.3)]
        ),
        batch_size: int = 1,
    ) -> None:
        super().__init__(params, batch_size)

    def one_dataset(self):
        dt = self._params["dt"]
        delay = int(self._params["delay"] / dt)
        trial = int(self._params["trial"] / dt)
        time_object = int(self._params["time_object"] / dt)
        tasks_number = [0, 1]
        full_interval = (
            time_object + (len(tasks_number) + 1) * delay + len(tasks_number) * trial
        )  # example, 6 delay, 5 trial (sum)
        fixation = np.zeros((full_interval, 1))
        stimul = np.zeros((full_interval, 1))
        target_output = np.zeros((full_interval, 1))
        object_for_batch = np.random.uniform(0, 1, size=(self._batch_size))
        choice_correct = np.random.choice(tasks_number, size=(self._batch_size, 1))
        inputs = np.zeros((full_interval, self._batch_size, self.ob_size))
        outputs = np.zeros((full_interval, self._batch_size, self.act_size))
        for batch in range(self._batch_size):
            fixation *= 0
            stimul *= 0
            target_output *= 0
            fixation[
                0 : time_object
                + (1 + choice_correct[batch, 0]) * (trial + delay)
                + trial
                + delay
                - int(delay / 100),
                0,
            ] = 1
            start_out = (
                time_object + choice_correct[batch, 0] * (trial + delay) + 2 * trial
            )
            target_output[
                start_out : start_out + trial + delay - int(delay / 100), 0
            ] = 1
            stimul[0:time_object, 0] = object_for_batch[batch]
            for j in range(choice_correct[batch, 0]):
                curent_example = np.random.uniform(0, 1)
                while curent_example == object_for_batch[batch]:
                    curent_example = np.random.uniform(0, 1)
                stimul[
                    time_object
                    + delay
                    + j * (trial + delay) : time_object
                    + delay
                    + j * (trial + delay)
                    + trial
                ] = curent_example
            stimul[
                time_object
                + delay
                + choice_correct[batch, 0] * (trial + delay) : time_object
                + delay
                + choice_correct[batch, 0] * (trial + delay)
                + trial,
                0,
            ] = object_for_batch[batch]
            inputs[:, batch, 0] = fixation[:, 0]
            inputs[:, batch, 1] = stimul[:, 0]
            outputs[:, batch, 0] = fixation[:, 0]
            outputs[:, batch, 1] = target_output[:, 0]
        inputs = np.concatenate(
            (inputs, np.zeros((delay, self._batch_size, self.ob_size))), axis=0
        )
        outputs = np.concatenate(
            (outputs, np.zeros((delay, self._batch_size, self.act_size))), axis=0
        )

        return inputs, outputs


class MultyTask:
    task_list = [
        ("ContextDM", ContextDM),
        ("CompareObjects", CompareObjects),
        ("WorkingMemory", WorkingMemory),
    ]
    task_list.sort()
    TASKSDICT = dict(task_list)

    def __init__(self, tasks: dict[str, dict], batch_size: int = 1) -> None:
        # tasks : dict(task_name -> parameters)
        for name in tasks:
            if not (name in self.TASKSDICT):
                raise KeyError(f'"{name}" not supported')
        self._tasks = tasks
        self._sorted_tasks()
        self._batch_size = batch_size
        self._task_list = []
        self._init_tasks()  # init all tasks

    def _sorted_tasks(self):
        new_dict = dict()
        for key in sorted(self._tasks):
            new_dict[key] = self._tasks[key]
        self._tasks = new_dict

    def _init_tasks(self):
        self._task_list.clear()
        for key in self._tasks:
            if len(self._tasks[key]) > 0:
                self._task_list.append(
                    self.TASKSDICT[key](self._tasks[key], self._batch_size)
                )
            else:
                self._task_list.append(self.TASKSDICT[key](batch_size=self._batch_size))

    def dataset(self, number_of_generations: int = 1) -> tuple[np.ndarray, np.ndarray]:
        number_of_tasks = len(self._tasks)
        choice_tasks = [i for i in range(number_of_tasks)]
        all_inputs, all_outputs = self._count_feature_and_act_size()
        rules = np.eye(number_of_tasks)
        inputs = np.zeros((0, self._batch_size, all_inputs))
        outputs = np.zeros((0, self._batch_size, all_outputs))
        sizes_all_tasks = self._feature_and_act_size_every_task()
        start_input_tasks = [1 + number_of_tasks]
        start_output_tasks = [1]

        size_input_tasks = []
        size_output_tasks = []
        for key in sizes_all_tasks:
            n_inputs, n_outputs = sizes_all_tasks[key]
            n_inputs -= 1  # -fix
            n_outputs -= 1  # -fix
            size_input_tasks.append(n_inputs)
            size_output_tasks.append(n_outputs)
            start_input_tasks.append(n_inputs + start_input_tasks[-1])
            start_output_tasks.append(n_outputs + start_output_tasks[-1])

        for _ in range(number_of_generations):
            task_number = np.random.choice(choice_tasks)
            task_inputs, task_outputs = self._task_list[task_number].dataset()
            # 1. expansion of matrices
            inputs = np.concatenate(
                (inputs, np.zeros((task_inputs.shape[0], self._batch_size, all_inputs)))
            )
            outputs = np.concatenate(
                (
                    outputs,
                    np.zeros((task_outputs.shape[0], self._batch_size, all_outputs)),
                )
            )
            # 2. put fixations
            inputs[-task_inputs.shape[0] :, :, 0] = task_inputs[
                -task_inputs.shape[0] :, :, 0
            ]
            outputs[-task_inputs.shape[0] :, :, 0] = task_outputs[
                -task_outputs.shape[0] :, :, 0
            ]
            # 3. put rule
            inputs[-task_inputs.shape[0] :, :, 1 : 1 + number_of_tasks] += rules[
                :, task_number
            ]
            # 4. put stimuly and outputs
            start_input = start_input_tasks[task_number]
            stop_input = start_input + size_input_tasks[task_number]
            start_output = start_output_tasks[task_number]
            stop_output = start_output + size_output_tasks[task_number]
            inputs[-task_inputs.shape[0] :, :, start_input:stop_input] = task_inputs[
                :, :, 1:
            ]
            outputs[
                -task_outputs.shape[0] :, :, start_output:stop_output
            ] = task_outputs[:, :, 1:]
        return inputs, outputs

    @property
    def feature_and_act_size(self) -> tuple[tuple[int, int], dict]:
        return (
            self._count_feature_and_act_size(),
            self._feature_and_act_size_every_task(),
        )

    def _count_feature_and_act_size(self) -> tuple[int, int]:
        all_inputs = 0
        all_outputs = 0
        for key in self._tasks:
            all_inputs += self.TASKSDICT[key].ob_size - 1  # minus fix
            all_outputs += self.TASKSDICT[key].act_size - 1  # minus_fix

        all_inputs += 1 + len(self._tasks)  # fix + rule vector
        all_outputs += 1  # fix
        return (all_inputs, all_outputs)

    def _feature_and_act_size_every_task(self):
        sizes = dict()
        for key in self._tasks:
            sizes[key] = (self.TASKSDICT[key].ob_size, self.TASKSDICT[key].act_size)
        return sizes

    @property
    def tasks(self) -> dict:
        return self._tasks

    @tasks.setter
    def tasks(self, tasks) -> None:
        self.__init__(tasks)

    def get_task(self, key) -> dict:
        if not (key in self._tasks):
            raise KeyError()
        return self._tasks[key]

    def set_task(self, key: str, params: dict):
        if not (key in self._tasks):
            raise KeyError()
        self._tasks[key] = params
        self._init_tasks()

    def __getitem__(self, index: int) -> tuple:
        if index < 0 and index > len(self._tasks) - 1:
            raise IndexError(f"index not include in [{0}, {len(self._tasks)}]")
        for i, key in enumerate(self._tasks):
            if index == i:
                return key, self._tasks[key]

    def __setitem__(self, index: int, new_task: tuple):
        if index < 0 and index > len(self._tasks) - 1:
            raise IndexError(f"index not include in [{0}, {len(self._tasks)}]")
        new_name, new_parameters = new_task
        if not (new_name in self.TASKSDICT):
            raise KeyError(f'"{new_name}" not supported')
        for i, key in enumerate(self._tasks):
            if index == i:
                old_key = key
                break
        del self._tasks[old_key]
        self._tasks[new_name] = new_parameters
        self._init_tasks()

    def __len__(self):
        return len(self._tasks)
