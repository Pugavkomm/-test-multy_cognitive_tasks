import matplotlib.pyplot as plt

from cgtasknet.tasks.tasks import ContextDM

task = ContextDM()
inputs, outputs = task.dataset(1)
print(f"inputs: {inputs.shape} outputs: {outputs.shape}")
plt.plot(inputs[:, 0, 0])
plt.show()
