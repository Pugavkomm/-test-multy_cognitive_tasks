import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from cgtasknet.instrumetns.dynamic_generate import SNNStates
from cgtasknet.instrumetns.instrument_pca import PCA
from cgtasknet.net.lifadex import SNNlifadex
from cgtasknet.net.states import LIFAdExInitState
from cgtasknet.tasks.tasks import WorkingMemory

number_of_tasks = 1
task_list = [("WorkingMemory", dict()), ("ContextDM", dict())]
tasks = dict(task_list)
params = dict(
    [
        ("dt", 1e-3),  # step 1ms
        ("delay", 0.5),  # 500 ms
        ("trial", 0.5),  # 500 ms
        ("KU", 0.05),  # 50 ms
        ("PB", 0.05),  # 50 ms
        ("min", 0),
        ("max", 1),
        ("first", 0.1),
        ("second", 0.9),
    ]
)

params2 = dict(
    [
        ("dt", 1e-3),  # step 1ms
        ("delay", 0.5),  # 500 ms
        ("trial", 0.5),  # 500 ms
        ("KU", 0.05),  # 50 ms
        ("PB", 0.05),  # 50 ms
        ("min", 0),
        ("max", 1),
        ("first", 0.9),
        ("second", 0.1),
    ]
)
Task = WorkingMemory(params)  # MultyTask(tasks=tasks, batch_size=1)
Task2 = WorkingMemory(params2)
feature_size = 8
output_size = 5
hidden_size = 400
batch_size = 1
model = SNNlifadex(feature_size, hidden_size, output_size)
if True:
    model.load_state_dict(
        torch.load(
            "./cgtasknet/notebooks/save_notebooks/test_instruments/lif_adex_romo_and_ctx_1510_iterations_steps_lr_1e-3_copy"
        )
    )
init_state = LIFAdExInitState(batch_size, hidden_size)
first_state = init_state.zero_state()
second_state = init_state.random_state()
one_trajectory_time = int(5198 / 2) - 1
v_mean = torch.zeros((one_trajectory_time, batch_size, hidden_size))
number_of_trials = 500
for trial in tqdm(range(number_of_trials)):
    inputs, target_out = Task.dataset(1)
    data = np.zeros((inputs.shape[0], batch_size, feature_size))
    data[:, :, 0] = inputs[:, :, 0]
    data[:, :, 2] = 1
    data[:, :, 7] = inputs[:, :, 1]
    data += np.random.normal(0, 0.01, size=(data.shape))
    data = torch.from_numpy(data).type(torch.float32)
    # inputs, target_out = Task2.dataset(1)
    # data2 = np.zeros((inputs.shape[0], batch_size, feature_size))
    # data2[:, :, 0] = inputs[:, :, 0]
    # data2[:, :, 2] = 1
    # data2[:, :, 7] = inputs[:, :, 1]
    # data2 += np.random.normal(0, 0.01, size=(data.shape))
    # data2 = torch.from_numpy(data2).type(torch.float32)
    # data = torch.concat((data, data2), axis=0)
    # target_out = torch.from_numpy(target_out).type(torch.float)

    states_generator = SNNStates(model)
    out, states = states_generator.states(data, second_state)
    v = []
    s = []
    i = []
    for j in range(len(states)):
        v.append(states[j].v)
        s.append(states[j].z)
        i.append(states[j].i)
    v = torch.stack(v).detach()
    # s = torch.stack(s).detach()
    # i = torch.stack(i).detach()
    # plt.plot(data[:, 0, 7])
    # plt.plot(out.detach().cpu().numpy()[:, 0, 3])
    # plt.plot(out.detach().cpu().numpy()[:, 0, 4])
    # plt.show()
    v_mean += v
v_mean /= float(number_of_trials)
print(f"v_mean = {v_mean.shape}")
v_mean = v_mean.reshape(v_mean.shape[0], v_mean.shape[2])

_, derivation1 = PCA(1, True).decompose(v_mean)
_, derivation2 = PCA(2, True).decompose(v_mean)
_, derivation3 = PCA(3, True).decompose(v_mean)
pca, derivation4 = PCA(4, True).decompose(v_mean)
print(f"derivation 1: {derivation1 * 100}; derivation2 = {derivation2 * 100};")
print(f"derivation 3: {derivation3 * 100}; derivation4 = {derivation4 * 100};")
print(f"pca.shape={pca.shape}")


def plot_pca(x, y, times=(0, 0, 0, 0), cmap="jet"):
    c = np.arange(0, len(x))
    plt.plot(x, y, "--", linewidth=1)
    plt.scatter(x, y, c=c, s=4, cmap=cmap)
    plt.plot(x[times[0]], y[times[0]], "*", markersize=15, label="Start")
    for i in range(1, len(times)):
        plt.plot(
            x[times[i]], y[times[i]], "*", markersize=15, label=f"time = {times[i]}"
        )


times = (0, 500, 1000, 1500)
plt.subplot(221)
plt.title("x - 1, y - 2")
plot_pca(pca[:, 0], pca[:, 1], times=times)
plt.subplot(222)
plt.title("x - 2, y - 3")
plot_pca(pca[:, 1], pca[:, 2], times=times)
plt.subplot(223)
plt.title("x - 3, y - 4")
plot_pca(pca[:, 2], pca[:, 3], times=times)
plt.subplot(224)
plt.title("x - 1, y - 3")
plot_pca(pca[:, 0], pca[:, 2], times=times)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(pca.numpy()[:, 0], "--", linewidth=1)

plt.show()
