from ..tasks.reduce import MultyReduceTasks


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
    size = task.feature_and_act_size
    assert size == (3 + len(task_list), 3)


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
