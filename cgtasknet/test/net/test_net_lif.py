import torch
from norse.torch.functional.lif import LIFParameters, LIFState

from cgtasknet.net.lif import SNNLif


def test_init_lif():
    input_s, hidden_s, output_s = 100, 100, 100
    SNNLif(input_s, hidden_s, output_s)


def test_set_custom_params_lif():
    input_s, hidden_s, output_s = 100, 100, 100
    p = LIFParameters(v_th=torch.as_tensor(0.2))
    SNNLif(input_s, hidden_s, output_s, neuron_parameters=p)


def test_run_lif():
    input_s, hidden_s, output_s = 100, 100, 10
    batch_size = 100
    data = torch.zeros((1000, batch_size, input_s))
    model = SNNLif(input_s, hidden_s, output_s)
    outputs, state = model(data)
    assert outputs.shape == (1000, batch_size, output_s)
    assert isinstance(state, LIFState)


def test_run_save_state_lif():
    input_s, hidden_s, output_s = 100, 100, 10
    batch_size = 100
    data = torch.zeros((10, batch_size, input_s))
    model = SNNLif(input_s, hidden_s, output_s, save_states=True)
    outputs, state = model(data)
    assert outputs.shape == (10, batch_size, output_s)
    assert len(state) == 10
    for i in range(10):
        assert isinstance(state[i], LIFState)
