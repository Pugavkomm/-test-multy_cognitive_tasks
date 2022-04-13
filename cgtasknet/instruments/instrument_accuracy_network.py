from typing import Iterable

import torch

from .correct_answer import _is_correct_output_batches


def correct_answer(
    output: torch.tensor, target_output: torch.tensor, fixation: torch.tensor
) -> torch.tensor:
    answer_indexes = torch.where(fixation == 0)[0]

    return _is_correct_output_batches(
        output[answer_indexes], target_output[answer_indexes]
    )


def _check_labels_correct(labels, output, end_fixations, mode):
    output_m = torch.zeros((output.shape[1], output.shape[2]))
    for i in range(output_m.shape[0]):
        if end_fixations[i] >= output.shape[0] - 1:

            output_m[i, :] = 0
        else:
            if mode == "mean":
                output_m[i, :] = output[end_fixations[i] :, i, :].mean(axis=0)
            elif mode == "max":
                output_m[i, :] = output[end_fixations[i] :, i, :].max(axis=0)[0]
    decisions = torch.zeros(len(labels))
    for i in range(len(labels)):
        max_v, max_i = output_m[i, :].max(), output_m[i, :].argmax()
        if len(torch.where(output_m[i, :] == max_v)[0]) > 1:
            decisions[i] = -1
        else:
            decisions[i] = max_i

    return torch.sum(labels == decisions).item()


def _first_value(x, value, axis=0, max_v=0):
    equ = x < value
    result = ((equ.cumsum(axis) == 1) & equ).max(axis)[1]
    result[torch.where(result == 0)[0]] = max_v
    return result


class CorrectAnswerNetwork:
    def __init__(
        self,
        choice_task_indexes: Iterable = None,
        repeated_task_indexes: Iterable = None,
        accuracy_repeated: float = None,
        mode: str = "mean",
    ) -> None:
        if choice_task_indexes is None:
            choice_task_indexes = []
        if repeated_task_indexes is None:
            repeated_task_indexes = []
        self._choice_task_indexes = choice_task_indexes
        self._repeated_task_indexes = repeated_task_indexes
        if accuracy_repeated is not None:
            accuracy_repeated = float(accuracy_repeated)
        self._accuracy_repeated = accuracy_repeated
        self._mode = mode

    def run(
        self,
        t_fixation: torch.tensor,
        fixation: torch.tensor,
        t_output: torch.tensor,
        output: torch.tensor,
        task_labels: Iterable[int],
    ) -> int:
        task_labels = [el for el in task_labels]
        time_steps, batch_size, out_size = t_output.shape

        if len(t_output.shape) != len(output.shape):
            raise ValueError

        if len(t_output) != len(output):
            raise ValueError

        if len(t_fixation) != len(fixation):
            raise ValueError

        fixation_stop = torch.max(
            t_fixation.argmin(axis=0), _first_value(fixation, 0.5, 0, len(fixation))
        )
        choices_tasks = []
        values_tasks = []
        for i in range(batch_size):
            if task_labels[i] in self._choice_task_indexes:
                choices_tasks.append(i)
            elif task_labels[i] in self._repeated_task_indexes:
                values_tasks.append(i)
        labels = t_output[-1, choices_tasks, :].argmax(axis=1)
        values, _ = t_output[-1, values_tasks, :].max(axis=1)
        sum_result_decisions = 0
        if len(choices_tasks):
            sum_result_decisions = _check_labels_correct(
                labels,
                output[:, choices_tasks, :],
                fixation_stop[choices_tasks],
                self._mode,
            )
        sum_result_values = 0
        if len(values_tasks):
            sum_result_values = self._check_values_correct(
                values, output[:, values_tasks, :], fixation_stop[values_tasks]
            )
        return int(sum_result_decisions + sum_result_values)

    def _check_values_correct(self, values, output, end_fixations):
        output_m = torch.zeros((output.shape[1], output.shape[2]))
        for i in range(output_m.shape[0]):
            if self._mode == "mean":
                output_m[i, :] = output[end_fixations[i] :, i, :].mean(axis=0)
            elif self._mode == "max":
                output_m[i, :] = output[end_fixations[i] :, i, :].max(axis=0)[0]
        real_values = output_m.max(dim=1)[0]
        return torch.sum(
            torch.abs(values - real_values) < self._accuracy_repeated
        ).item()
