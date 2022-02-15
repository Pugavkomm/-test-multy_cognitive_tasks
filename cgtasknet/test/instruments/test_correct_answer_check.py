import torch

from cgtasknet.instruments.correct_answer import (
    _is_correct_output,
    _is_correct_output_batches,
)


def test_correct_output_good_answer():
    real_outputs = torch.zeros(10, 1, 2)
    target_outputs = torch.zeros(10, 1, 2)

    real_outputs[:, 0, 0] = 1
    target_outputs[:, 0, 0] = 1
    assert _is_correct_output(real_outputs, target_outputs)


def test_correct_output_bad_answer():
    real_outputs = torch.zeros(10, 1, 2)
    target_outputs = torch.zeros(10, 1, 2)

    real_outputs[:, 0, 0] = 1
    target_outputs[:, 0, 1] = 1
    assert not _is_correct_output(real_outputs, target_outputs)


def test_correct_output_batch_answers():
    real_outputs = torch.zeros(10, 3, 2)
    target_outputs = torch.zeros(10, 3, 2)
    real_outputs[:, :, 0] = 1
    target_outputs[:, :, 0] = 1
    target_outputs[:, 1, 0] = 0
    target_outputs[:, 1, 1] = 1
    assert torch.allclose(
        _is_correct_output_batches(real_outputs, target_outputs),
        torch.tensor([True, False, True]),
    )
