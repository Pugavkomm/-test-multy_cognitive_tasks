import matplotlib.pyplot as plt
import numpy as np

pi = np.pi


def stimuli_i(i, psi, n_each_ring, gamma=1):
    psi_i = 2 * pi / n_each_ring * i
    return gamma * 0.8 * np.exp(-0.5 * (8 * np.abs(psi - psi_i) / 2) ** 2)


n_each_ring = 32
psi = (2 * pi / n_each_ring * 10 + 2 * pi / n_each_ring * 11) / 2
directions = np.zeros((n_each_ring, 100))
for i in range(n_each_ring):
    directions[i] = stimuli_i(i + 1, psi, n_each_ring)
directions += np.random.normal(0, 1, size=directions.shape) * np.sqrt(2 * 5) * 0.01
plt.imshow(directions, aspect="auto", origin="lower")
plt.colorbar()
plt.savefig("directions.svg")

plt.show()
