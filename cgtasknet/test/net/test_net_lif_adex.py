import torch
from norse.torch.functional.lif_adex import LIFAdExParameters, LIFAdExState

from cgtasknet.net.lifadex import SNNlifadex


def test_init_lif_adex():
    input_s, hidden_s, output_s = 100, 100, 100
    SNNlifadex(input_s, hidden_s, output_s)


def test_set_custom_params_lif_adex():
    input_s, hidden_s, output_s = 100, 100, 100
    p = LIFAdExParameters()
    SNNlifadex(input_s, hidden_s, output_s, neuron_parameters=p)


def test_run_lif_adex():
    input_s, hidden_s, output_s = 100, 100, 10
    batch_size = 100
    data = torch.zeros((1000, batch_size, input_s))
    model = SNNlifadex(input_s, hidden_s, output_s)
    outputs, state = model(data)
    assert outputs.shape == (1000, batch_size, output_s)
    assert isinstance(state, LIFAdExState)


def test_run_save_state_lif_adex():
    input_s, hidden_s, output_s = 100, 100, 10
    batch_size = 100
    data = torch.zeros((10, batch_size, input_s))
    model = SNNlifadex(input_s, hidden_s, output_s, save_states=True)
    outputs, state = model(data)
    assert outputs.shape == (10, batch_size, output_s)
    assert len(state) == 10
    for i in range(10):
        assert isinstance(state[i], LIFAdExState)


def test_additon_current_lif_adex():
    input_s, hidden_s, output_s = 100, 100, 10
    batch_size = 100
    data = torch.zeros((10, batch_size, input_s)) + 1
    model = SNNlifadex(input_s, hidden_s, output_s, save_states=True)
    state = LIFAdExState(
        torch.zeros((batch_size, hidden_s)),
        torch.zeros((batch_size, hidden_s)),
        torch.zeros((batch_size, hidden_s)),
        torch.zeros((batch_size, hidden_s)),
    )
    outputs, state = model(
        data, state, additional_current=torch.ones(batch_size, hidden_s) * 1000
    )
    print(state[9].i)
    assert outputs.shape == (10, batch_size, output_s)
    assert len(state) == 10

    for i in range(10):
        assert isinstance(state[i], LIFAdExState)
