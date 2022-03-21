import torch

from cgtasknet.instruments.instrument_subgroups import SubgroupFinder


def test_create_subgroup_finder():
    sgf = SubgroupFinder(dt=1e-2)
    assert sgf.dt == 1e-2


def test_compute_average_freq_zeros():
    sgf = SubgroupFinder()
    s = torch.zeros(100, 4)
    fixation = torch.zeros(100)
    fixation[0:50] = 1
    outputs = torch.zeros(100, 1, 4)
    outputs[:, 0, 0] = fixation[:]
    target_outputs = torch.zeros(100, 1, 4)
    target_outputs[:, 0, 0] = fixation
    s = [s]
    fixation = [fixation]
    outputs = [outputs]
    target_outputs = [target_outputs]
    sgf.compute_average_freq(s, fixation, outputs, target_outputs)
    freq_fixation, freq_answer = sgf.get_average_freq()

    assert torch.allclose(freq_fixation, torch.zeros(4))
    assert torch.allclose(freq_answer, torch.zeros(4))


# def test_incorrect_output():
#    sgf = SubgroupFinder()
#    s = torch.zeros(100, 4)
#    fixation = torch.zeros(100, 1)
#    fixation[0:50, 0] = 1
#    outputs = torch.zeros(100, 1,  2)
#    target_outputs = torch.zeros(100, 1,  2)
#    outputs[:, 0, 0] = fixation[:, 0]
#    target_outputs[:, 0,  0] = 1
#    s = [s]
#    outputs = [outputs]
#    target_outputs = [target_outputs]
#    fixation = [fixation]
#    sgf.compute_average_freq(s, fixation, outputs, target_outputs)
#    freq_fixation, freq_answer = sgf.get_average_freq()
#
#    assert freq_answer is None
#    assert freq_fixation is None
