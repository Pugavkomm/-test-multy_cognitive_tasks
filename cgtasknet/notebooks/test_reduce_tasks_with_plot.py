import matplotlib.pyplot as plt

from cgtasknet.tasks.reduce import (
    CtxDMTask,
    DMTask,
    DMTaskRandomMod,
    MultyReduceTasks,
    RomoTask,
    RomoTaskRandomMod,
)

print("Start.output.sizes".center(40, "."))
task = DMTask()
print(f"DMTask: sizes = {task.feature_and_act_size}")
task = DMTaskRandomMod()
print(f"DMTaskRandomMod: n_mods = {2}, sizes = {task.feature_and_act_size}")
task = RomoTask()
print(f"RomoTask: sizes = {task.feature_and_act_size}")
task = RomoTaskRandomMod()
print(f"RomoTaskRandomMod: n_mods = {2}, sizes = {task.feature_and_act_size}")
task = CtxDMTask()
print(f"CtxDMTask: sizes = {task.feature_and_act_size}")
tasks_list = ["DMTask", "RomoTask", "CtxDMTask"]
task = MultyReduceTasks(tasks_list)
print(f"\nMultyTask: sizes = {task.feature_and_act_size}")
print("Every task in MultyTasks has size:")
for task_name in task.feature_and_act_every_task_size:
    print(
        f"{task_name}".ljust(10),
        ":",
        f"{task.feature_and_act_every_task_size[task_name]}",
    )
print("End.output.sizes".center(40, "."))
print("Start.plot".center(40, "."))
inputs, outputs = task.dataset(10)
plt.subplot(211)
for i in range(inputs.shape[2]):
    plt.plot(inputs[:, :, i])

plt.subplot(212)
for i in range(outputs.shape[2]):
    plt.plot(outputs[:, :, i])
plt.show()

print("End.plot".center(40, "."))
