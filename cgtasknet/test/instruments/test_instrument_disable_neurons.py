import torch

from cgtasknet.instruments.instrument_disable_neurons import DisableSomeNeurons
from cgtasknet.net.lif import SNNLif


def test_init_disables():
    net = SNNLif(3, 10, 3)
    DisableSomeNeurons(net)


def test_output_disables():
    net = SNNLif(3, 10, 3)
    number_of_neurons = (0, 1, 2)
    disables = DisableSomeNeurons(net)
    disables.disable_output_weights(number_of_neurons)
    for name, params in net.named_parameters():
        if name == "exp_f.linear.weight":
            with torch.no_grad():
                for value in number_of_neurons:
                    assert torch.allclose(params[:, value], torch.zeros(3))


def test_input_disables():
    net = SNNLif(3, 10, 3)
    number_of_neurons = (0, 1, 2)
    disables = DisableSomeNeurons(net)
    disables.disable_input_weights(number_of_neurons)
    for name, params in net.named_parameters():
        if name == "lif.input_weights":
            with torch.no_grad():
                for value in number_of_neurons:
                    assert torch.allclose(params[value, :], torch.zeros(3))


def test_hidden_disables():
    net = SNNLif(3, 5, 3)
    number_of_neurons = (0, 1, 2)
    disables = DisableSomeNeurons(net)
    disables.disable_hidden_weights(number_of_neurons)
    for name, params in net.named_parameters():
        if name == "lif.recurrent_weights":
            with torch.no_grad():
                for value in number_of_neurons:
                    assert torch.allclose(params[:, value], torch.zeros(5))
        if name == "exp_f.linear.weight":
            with torch.no_grad():
                for value in number_of_neurons:
                    assert torch.allclose(params[:, value], torch.zeros(3))
