#tasks = MultyTask()
import matplotlib.pyplot as plt
from cognitive_task import MultyTask

task_list = [('WorkingMemory', dict()),
             (('ContextDM', dict()))]
tasks = dict(task_list)
TASKS = MultyTask(tasks)
inputs, outputs = TASKS.dataset(10)
print(f'inputs.shape: {inputs.shape}, outputs.shape: {outputs.shape}')
plt.subplot(411)
plt.plot(inputs[:, 0, 0], label='input: Fixation')
plt.legend()
plt.subplot(412)
plt.plot(inputs[:, 0, 1], label='input: Rule 1')
plt.plot(inputs[:, 0, 2], label='input: Rule 2')
plt.legend()

plt.subplot(413)
plt.plot(inputs[:, 0, 3], label='input: 1')
plt.plot(inputs[:, 0, 4], label='input: 2')
plt.plot(inputs[:, 0, 5], label='input: 3')
plt.plot(inputs[:, 0, 6], label='input: 4')
plt.plot(inputs[:, 0, 7], label='input: 5')
plt.legend()

plt.subplot(414)
plt.plot(inputs[:, 0, 0], label='input: 1')
plt.plot(outputs[:, 0, 0], label='output: 1')

plt.legend()
plt.show()