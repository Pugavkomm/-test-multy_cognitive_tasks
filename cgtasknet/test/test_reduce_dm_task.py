from ..tasks.reduce import DefaultParams, DMTask, DMTaskRandomMod


def test_dm_task_size():
    assert DMTask().feature_and_act_size == (2, 3)


def test_dm_task_run_one_data_set():
    DMTask().one_dataset()


def test_dm_task_run_some_datasets():
    inputs, outputs = DMTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[2] == 3


def test_dm_task_get_params():
    def_params = DefaultParams("DMTask").generate_params()
    assert DMTask().params == def_params
    assert (DMTask(batch_size=10).batch_size) == 10


def test_dm_rm_task_size():
    assert DMTaskRandomMod().feature_and_act_size == (3, 3)


def test_dm_rm_task_run_one_data_set():
    DMTaskRandomMod().one_dataset()


def test_dm_rm_task_run_some_datasets():
    inputs, outputs = DMTaskRandomMod(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 3
    assert outputs.shape[2] == 3


def test_dm_rm_task_get_params():
    def_params = DefaultParams("DMTaskRandomMod").generate_params()
    assert DMTaskRandomMod().params == def_params
    assert (DMTaskRandomMod(batch_size=10).batch_size) == 10


def test_correct_target_dm_task():
    task = DMTask(batch_size=10)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(10):
            assert (inputs[0, j, 1] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 1] < 0.5) != (outputs[-1, j, 2] == 1)
