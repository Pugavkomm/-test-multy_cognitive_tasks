import torch

from cgtasknet.net import LIFAdExInitState
from cgtasknet.net.lifadex import SNNlifadex


def test_init_lif_refrac():
    input_s, hidden_s, output_s = 500, 100, 1
    model = SNNlifadex(input_s, hidden_s, output_s)
    inputs = torch.rand((300, 1, input_s))
    init_state = LIFAdExInitState(1, hidden_s)
    zero_state = init_state.zero_state()
    outputs_without_save_states = model(inputs, zero_state)[0]
    model.save = True
    outputs_with_save_states = model(inputs, zero_state)[0]
    assert torch.allclose(outputs_without_save_states, outputs_with_save_states)
