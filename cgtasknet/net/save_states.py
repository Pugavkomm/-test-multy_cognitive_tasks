import torch


def save_states(x, save_states: bool, layer, state):
    if save_states:
        T = len(x)
        s = state
        states = []
        outputs = []
        for ts in range(T):
            out, s = layer(x[ts : ts + 1, :, :], state=s)
            outputs.append(out)
            states.append(s)
        outputs = torch.concat(outputs, axis=0)
    else:
        outputs, states = layer(x, state)
    return outputs, states
