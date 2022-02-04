import inputs as inputs

from cgtasknet.net import LIFAdExRefracInitState
from cgtasknet.net.lifadexrefrac import SNNlifadexrefrac
import torch
from norse.torch.functional.lif_adex_refrac import (
    LIFAdExRefracParameters,
    LIFAdExRefracState,
)


def test_init_lif_refrac():
    input_s, hidden_s, output_s = 500, 100, 1
    model = SNNlifadexrefrac(input_s, hidden_s, output_s)
    inputs = torch.rand((300, 1, input_s))
    init_state = LIFAdExRefracInitState(1, hidden_s)
    zero_state = init_state.zero_state()
    outputs_without_save_states = model(inputs, zero_state)[0]
    model.save = True
    outputs_with_save_states = model(inputs, zero_state)[0]
    assert torch.allclose(outputs_without_save_states, outputs_with_save_states)
