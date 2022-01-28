from cgtasknet.tasks.reduce import DefaultParams, MultyReduceTasks


def test_init_multy_task():
    batch_size = 20
    task_list = [
        "DMTask",
        "RomoTask",
        "DMTask1",
        "DMTask2",
        "RomoTask1",
        "RomoTask2",
        "CtxDMTask1",
        "CtxDMTask2",
    ]
    MultyReduceTasks(tasks=task_list, batch_size=batch_size)


def test_size_mylty_task():
    batch_size = 20
    task_list = [
        "RomoTask",
        "DMTask1",
        "DMTask2",
        "DMTask",
        "RomoTask1",
        "RomoTask2",
        "CtxDMTask1",
        "CtxDMTask2",
    ]
    task = MultyReduceTasks(tasks=task_list, batch_size=batch_size)
    size = task.feature_and_act_size
    task[2].set_param('dtasd', 100)
    for i in range(len(task)):
        print(f"{task[i].name}: {task[i].params}")
    assert size == (3 + len(task_list), 3)


test_size_mylty_task()


def test_run_mylty_task():
    batch_size = 20
    task_list = [
        "DMTask",
        "RomoTask",
        "DMTask1",
        "DMTask2",
        "RomoTask1",
        "RomoTask2",
        "CtxDMTask1",
        "CtxDMTask2",
    ]
    task = MultyReduceTasks(tasks=task_list, batch_size=batch_size)
    for _ in range(100):
        inputs, outputs = task.dataset(10)
        assert inputs.shape[1] == batch_size
        assert outputs.shape[1] == batch_size
        assert inputs.shape[2] == task.feature_and_act_size[0]
        assert outputs.shape[2] == task.feature_and_act_size[1]


def test_shape_for_one_input_of_mods():
    romoparams = DefaultParams("RomoTask").generate_params()
    romoparams["n_mods"] = 1
    dmparams = DefaultParams("DMTask").generate_params()
    dmparams["n_mods"] = 1
    task_list = ["DMTask1", "RomoTask1"]
    tasks_params = dict([(task_list[1], romoparams), (task_list[0], dmparams)])
    task = MultyReduceTasks(tasks=tasks_params, batch_size=20, number_of_inputs=1)
    inputs, outputs = task.dataset(10)
    assert inputs.shape[1] == 20
    assert inputs.shape[2] == 4  # 2 inpyuts + 2 dim rule one-hot vector
    assert outputs.shape[0] == inputs.shape[0]
    assert outputs.shape[1] == 20
    assert outputs.shape[2] == 3


def test_shape_for_two_inputs_of_mods():
    task_list = ["DMTask1", "RomoTask1"]
    task = MultyReduceTasks(tasks=task_list, batch_size=20, number_of_inputs=2)
    inputs, outputs = task.dataset(10)
    assert inputs.shape[1] == 20
    assert inputs.shape[2] == 5
    assert outputs.shape[0] == inputs.shape[0]
    assert outputs.shape[1] == 20
    assert outputs.shape[2] == 3
