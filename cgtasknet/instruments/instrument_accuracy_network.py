import torch

from .correct_answer import _is_correct_output_batches


def correct_answer(
    output: torch.tensor, target_output: torch.tensor, fixation: torch.tensor
) -> torch.tensor:
    answer_indexes = torch.where(fixation == 0)[0]
    return _is_correct_output_batches(
        output[answer_indexes], target_output[answer_indexes]
    )
