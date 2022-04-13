import torch

from cgtasknet.instruments.instrument_accuracy_network import CorrectAnswerNetwork


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


def test_difficult_go_signal():
    fixation_signal = torch.ones(150)
    fixation_signal[70:] = 1 - torch.tensor([*range(80)]) / 80
    fixation_signal[100:120] = torch.cos(torch.tensor([*range(20)]) * 0.05) * 0.7

    fixation_signal += (
        torch.randn(len(fixation_signal)) * 0.05 * torch.rand(len(fixation_signal)) * 5
    )
    stimulus = torch.zeros(150)
    stimulus[70:] = 0.6
    stimulus[100:120] *= torch.sin(torch.tensor([*range(20)]) * 0.05)

    stimulus += torch.randn(len(fixation_signal)) * torch.sin(
        torch.tensor([*range(150)]) * 0.001
    ) - torch.sin(torch.tensor(torch.randn(len(fixation_signal))) * 0.001)

    t_fixation = torch.ones((150, 1))  # 100 time step, 1 batch
    t_fixation[70:, 0] = 0
    fixation = torch.zeros((150, 1))
    fixation[:, 0] = fixation_signal[:]
    t_out = torch.zeros((150, 1, 1))
    out = torch.zeros_like(t_out)
    out[:, 0, 0] = stimulus
    t_out[70:] = 0.6
    cn = CorrectAnswerNetwork(None, [0], 0.2)
    result = cn.run(t_fixation, fixation, t_out, out, [0])
    assert result == 1
