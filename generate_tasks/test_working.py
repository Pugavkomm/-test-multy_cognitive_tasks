from cognitive_task import WorkingMemory
import matplotlib.pyplot as plt
task = WorkingMemory()
inputs, outputs = task.dataset(1)
print(f'inputs: {inputs.shape} outputs: {outputs.shape}')
plt.plot(inputs[:, 0, 0])
plt.show()