import torch
from norse.torch import LIFAdExState


def save_states(x, save_states: bool, layer, state, additional_current=None):
    if additional_current is not None:
        if additional_current.shape[0] == 1:
            tmp_additional_current = torch.zeros(
                (len(x), additional_current.shape[1])
            ).to(additional_current.device)
            tmp_additional_current[:, :] = additional_current[:, :]
            additional_current = tmp_additional_current
    if save_states or additional_current is not None:
        T = len(x)
        s = state
        states = []
        outputs = []
        for ts in range(T):
            # if additional_current is not None:
            #    s = (
            #        LIFAdExState(s.z, s.v, s.i + additional_current[ts], s.a)
            #        if s is not None
            #        else None
            #    )
            out, s = layer(x[ts : ts + 1, :, :], state=s)
            outputs.append(out)
            if save_states:
                states.append(s)

            else:
                states = s
        outputs = torch.concat(outputs, axis=0)
    else:
        outputs, states = layer(x, state)
    return outputs, states
