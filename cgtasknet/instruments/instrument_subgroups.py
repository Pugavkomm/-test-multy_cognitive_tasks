from typing import Tuple

import torch


def _is_correct_output(real_output: torch.Tensor, target_output: torch.Tensor) -> bool:
    # compute mean:
    if len(real_output.shape) == 3:
        real_output = real_output.reshape(real_output.shape[0], real_output.shape[2])
    if len(target_output.shape) == 3:
        target_output = target_output.reshape(
            target_output.shape[0], target_output.shape[2]
        )
    target_mean = torch.mean(target_output, axis=0)
    real_mean = torch.mean(real_output, axis=0)
    return torch.argmax(target_mean).item() == torch.argmax(real_mean).item()


class SubgroupFinder:
    def __init__(self, dt: float = 1e-3) -> None:
        self._dt = dt
        self._average_freq_fixation = None
        self._average_freq_answer = None
        self._number_of_trials = 0

    def _average_frequency_update(self, s: torch.Tensor) -> torch.Tensor:
        freq = torch.sum(s, axis=0) / s.shape[0]
        # freq = freq.reshape(freq.shape[1])
        return freq

    def compute_average_freq(
        self,
        s: list,
        fixations: list,
        outputs: list,
        target_outputs: list,
        criterion=_is_correct_output,
    ) -> None:
        # s: list[s_first_trial, s_second_trial, ..., s_last_trial]
        # fixations: list[fixation_first_trial, fixation_second_trial, fixation_last_trial]
        # outputs: list[output_first_trial, output_second_trial, ..., output_last_trial]
        # s_..._trial - torch.tensor -- shape[0] -- timestep, shape[1] -- number of neurons
        # fixation_..._trial -||-
        # outputs_..._trial -||-
        # if len(s) != len(fixations) != len(outputs):

        for i in range(len(s)):
            (
                (start_fixation, stop_fixation),
                (start_answer, stop_answer),
            ) = self.find_fixation_start_stop(fixations[i])

            if criterion(
                outputs[i][start_answer:stop_answer],
                target_outputs[i][start_answer:stop_answer],
            ):
                if self._average_freq_fixation is None:
                    self._average_freq_fixation = torch.zeros(s[0].shape[1])

                if self._average_freq_answer is None:
                    self._average_freq_answer = torch.zeros(s[0].shape[1])
                self._average_freq_fixation += self._average_frequency_update(
                    s[i][start_fixation:stop_fixation, :]
                )
                self._average_freq_answer += self._average_frequency_update(
                    s[i][start_answer:stop_answer, :]
                )
                self._number_of_trials += 1

    def find_fixation_start_stop(
        self, fixation: torch.Tensor
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        fixation_indexes = torch.where(fixation == 1)
        start_fixation = int(fixation_indexes[0][0].item())
        stop_fixation = int(fixation_indexes[0][-1].item())

        non_fixation_indexes = torch.where(fixation == 0)
        start_answer = int(non_fixation_indexes[0][0].item())
        stop_answer = int(non_fixation_indexes[0][-1].item())

        return ((start_fixation, stop_fixation), (start_answer, stop_answer))

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    def get_average_freq(self):
        if (self._average_freq_answer is None and self._average_freq_fixation is None):
            return (None, None)
        return (
            self._average_freq_fixation / self._number_of_trials / self._dt,
            self._average_freq_answer / self._number_of_trials / self._dt,
        )
