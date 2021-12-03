import matplotlib.pyplot as plt

from .net_cognitive_tasks.tasks.tasks import WorkingMemory

task = WorkingMemory()
inputs, outputs = task.dataset(1)
print(f"inputs: {inputs.shape} outputs: {outputs.shape}")
plt.plot(inputs[:, 0, 1])
plt.plot(outputs[:, 0, 1])
plt.plot(outputs[:, 0, 2])
plt.plot(inputs[:, 0, 0], c="r")
plt.show()
