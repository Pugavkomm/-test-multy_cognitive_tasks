from cgtasknet.tasks.reduce import (
    RomoTask,
    RomoTaskParameters,
    RomoTaskRandomMod,
    RomoTaskRandomModParameters,
)


def test_romo_task_size():
    assert RomoTask().feature_and_act_size == (2, 3)


def test_romo_task_run_one_data_set():
    RomoTask().one_dataset()


def test_romo_task_run_some_datasets():
    inputs, outputs = RomoTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[2] == 3


def test_romo_task_get_params():
    def_params = RomoTaskParameters()
    assert RomoTask().params == def_params
    assert (RomoTask(batch_size=10).batch_size) == 10


def test_romo_rm_task_size():
    assert RomoTaskRandomMod().feature_and_act_size == (3, 3)


def test_romo_rm_task_run_one_data_set():
    RomoTaskRandomMod().one_dataset()


def test_romo_rm_task_run_some_datasets():
    inputs, outputs = RomoTaskRandomMod(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 3
    assert outputs.shape[2] == 3


def test_romo_rm_task_get_params():
    def_params = RomoTaskRandomModParameters()
    assert RomoTaskRandomMod().params == def_params
    assert (RomoTaskRandomMod(batch_size=10).batch_size) == 10


def test_regime_values():
    def_params = RomoTaskParameters(value=(0, 1))
    task = RomoTask(params=def_params, batch_size=10, mode="value")
    task.dataset()


def test_romo_shift_trial_interval():
    def_params = RomoTaskParameters(
        negative_shift_trial_time=0.1, positive_shift_trial_time=-0.1
    )
    task = RomoTask(params=def_params)
    expected_time = int(
        (
            2 * def_params.trial_time
            - 2 * def_params.negative_shift_trial_time
            + def_params.delay
            + def_params.answer_time
        )
        / def_params.dt
    )
    assert expected_time == len(task.dataset(1)[0])


def test_romo_shit_delay_interval():
    def_params = RomoTaskParameters(
        negative_shift_delay_time=0.1,
        positive_shift_delay_time=-0.1,
    )
    task = RomoTask(params=def_params)
    expected_time = int(
        (
            2 * def_params.trial_time
            + def_params.delay
            - def_params.negative_shift_trial_time
            + def_params.answer_time
        )
        / def_params.dt
    )
    assert len(task.dataset(1)[0]) != expected_time
