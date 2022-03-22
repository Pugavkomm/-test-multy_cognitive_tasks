import torch

from cgtasknet.instruments.correct_answer import (
    _is_correct_output,
    _is_correct_output_batches,
)
from cgtasknet.instruments.instrument_accuracy_network import CorrectAnswerNetwork


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


def test_smart_correct_answer_test():
    cn = CorrectAnswerNetwork([0, 1, 3], [2, 4], 0.1)
    t_fixation = torch.zeros(10, 5)
    t_fixation[0:5, :] = 1
    fixation = torch.zeros_like(t_fixation)
    fixation[0:6, :] = 1
    t_out = torch.zeros((10, 5, 3))
    t_out[-1, 0, 0] = 1
    t_out[-1, 1, 2] = 1
    t_out[:, 2, 1] = 2

    t_out[:, 3, 2] = 2.0
    t_out[:, 4, 0] = 0.9  # + torch.rand(10)
    out = torch.clone(t_out)
    result = cn.run(t_fixation, fixation, t_out, out, [1, 0, 1, 2, 4])
    assert result == 5

    out[:, 4, 0] = 10
    result = cn.run(t_fixation, fixation, t_out, out, [1, 0, 1, 2, 4])
    assert result == 4

    out[-1, 0, 1] = 1
    out[-1, 0, 0] = 0
    result = cn.run(t_fixation, fixation, t_out, out, [1, 0, 1, 2, 4])
    assert result == 3


def test_smart_all_wrong_answers():
    cn = CorrectAnswerNetwork([0, 1], [2, 4, 3], 0.1)
    t_fixation = torch.zeros(10, 5)
    t_fixation[0:5, :] = 1
    fixation = t_fixation * 0
    t_out = torch.zeros((10, 5, 3))
    t_out[-1, 0, 0] = 1
    t_out[-1, 1, 2] = 1
    t_out[:, 2, 1] = 2

    t_out[:, 3, 2] = 2.0
    t_out[:, 4, 0] = 0.9  # + torch.rand(10)
    out = 1 - torch.clone(t_out)
    result = cn.run(t_fixation, fixation, t_out, out, [1, 4, 1, 2, 4])
    assert result == 0


def test_smart_all_wrong_fixations():
    cn = CorrectAnswerNetwork([0, 1, 3], [2, 4], 0.1)
    t_fixation = torch.zeros(10, 5)
    t_fixation[0:5, :] = 1
    fixation = torch.ones_like(t_fixation)
    t_out = torch.zeros((10, 5, 3))
    t_out[-1, 0, 0] = 1
    t_out[-1, 1, 2] = 1
    t_out[:, 2, 1] = 2

    t_out[:, 3, 2] = 2.0
    t_out[:, 4, 0] = 0.9  # + torch.rand(10)
    out = torch.clone(t_out)
    result = cn.run(t_fixation, fixation, t_out, out, [1, 0, 1, 2, 4])
    assert result == 0
