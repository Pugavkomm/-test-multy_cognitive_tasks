from cgtasknet.tasks.reduce import DefaultParams, RomoTask, RomoTaskRandomMod


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
    def_params = DefaultParams("RomoTask").generate_params()
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
    def_params = DefaultParams("RomoTaskRandomMod").generate_params()
    assert RomoTaskRandomMod().params == def_params
    assert (RomoTaskRandomMod(batch_size=10).batch_size) == 10


def test_regime_values():
    def_params = DefaultParams("RomoTask").generate_params()
    def_params["values"] = (0, 1)
    task = RomoTask(params=def_params, batch_size=10, mode="value")
    task.dataset()
