from cgtasknet.tasks.reduce import (
    DMTask,
    DMTaskParameters,
    DMTaskRandomMod,
    DMTaskRandomModParameters,
)


def test_dm_task_size():
    assert DMTask().feature_and_act_size == (2, 3)


def test_dm_task_run_one_data_set():
    DMTask().one_dataset()


def test_dm_task_run_just_run_some_data_set():
    DMTask().dataset(12)


def test_dm_task_run_some_datasets():
    inputs, outputs = DMTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[2] == 3


def test_dm_task_get_params():
    def_params = DMTaskParameters()
    assert DMTask().params == def_params
    assert DMTask(batch_size=10).batch_size == 10


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
    def_params = DMTaskRandomModParameters()
    assert DMTaskRandomMod().params == def_params
    assert DMTaskRandomMod(batch_size=10).batch_size == 10


def test_correct_target_dm_task():
    task = DMTask(batch_size=20)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(10):
            assert (inputs[0, j, 1] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 1] < 0.5) != (outputs[-1, j, 2] == 1)


def test_value_mode():
    params = DMTaskParameters(value=1)
    task = DMTask(params=params, batch_size=10, mode="value")
    task.dataset(10)


def test_value_mode_correct_generate_value_equal_1():
    params = DMTaskParameters(value=1)
    task = DMTask(params=params, batch_size=10, mode="value")
    inputs, outputs = task.dataset(1)
    for i in range(10):
        assert inputs[0, i, 0] == 1
        assert inputs[0, i, 1] == 1
        assert outputs[0, i, 0] == 1
        assert outputs[0, i, 1] == 0


def test_value_mode_correct_generate_value_equal_0_1():
    params = DMTaskParameters(value=0.1)
    task = DMTask(params=params, batch_size=10, mode="value")
    inputs, outputs = task.dataset(1)
    for i in range(10):
        assert inputs[0, i, 0] == 1
        assert inputs[0, i, 1] == 0.1
        assert outputs[0, i, 0] == 1
        assert outputs[0, i, 1] == 0


def test_dm_shift_trial_time():
    def_params = DMTaskParameters(
        negative_shift_trial_time=0.1, positive_shift_trial_time=-0.1
    )
    task = DMTask(params=def_params)
    expected_time = int(
        (
            def_params.trial_time
            - def_params.negative_shift_trial_time
            + def_params.answer_time
        )
        / def_params.dt
    )
    assert len(task.dataset(1)[0]) == expected_time
