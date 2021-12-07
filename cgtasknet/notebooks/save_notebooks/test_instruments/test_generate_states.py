import matplotlib.pyplot as plt
import torch

from cgtasknet.instrumetns.dynamic_generate import SNNStates
from cgtasknet.net.lif import SNNLif
from cgtasknet.net.states import LIFInitState

feature_size = 1
output_size = 1
hidden_size = 10
batch_size = 1
model = SNNLif(feature_size, hidden_size, output_size)

init_state = LIFInitState(batch_size, hidden_size)
first_state = init_state.zero_state()
second_state = init_state.random_state()
print(f"init state zero: {first_state}")
print(f"random state: {second_state}")
data = torch.zeros(50, batch_size, feature_size)
data[0] = 10
model(data)
states_generator = SNNStates(model)
out, states = states_generator.states(data, first_state)
print(f"\nout.shape: {out.shape}")
print(f"s: {states[0]}")
v = []
s = []
i = []
for j in range(len(states)):
    v.append(states[j].v)
    s.append(states[j].z)
    i.append(states[j].i)
v = torch.stack(v)
s = torch.stack(s)
i = torch.stack(i)
plt.plot(v.detach().numpy()[:, 0, 0])
plt.plot(v.detach().numpy()[:, 0, 1])
plt.show()
