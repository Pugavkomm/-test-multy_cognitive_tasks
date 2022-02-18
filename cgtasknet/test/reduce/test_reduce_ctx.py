from cgtasknet.tasks.reduce import (
    CtxDMTask,
    CtxDM1,
    CtxDM2,
    CtxDMTaskParameters,
)


def test_dm_task_size():
    assert CtxDMTask().feature_and_act_size == (3, 3)
    assert CtxDM1().feature_and_act_size == (3, 3)
    assert CtxDM2().feature_and_act_size == (3, 3)


def test_ctx_task_run_one_data_set():
    CtxDMTask().one_dataset()


def test_ctx_1_task_run_one_data_set():
    CtxDM1().one_dataset()


def test_ctx_2_task_run_one_data_set():
    CtxDM2().one_dataset()


def test_ctx_rm_task_run_some_datasets():
    inputs, outputs = CtxDMTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert outputs.shape[1] == 10
    assert inputs.shape[2] == 3
    assert outputs.shape[2] == 3


def test_ctx_rm_task_get_params():
    def_params = CtxDMTaskParameters()
    assert CtxDMTask().params == def_params
    assert CtxDMTask(batch_size=10).batch_size == 10


def test_ctx_1_def_params_context():
    assert CtxDM1().params.context == 0


def test_ctx_2_def_params_context():
    assert CtxDM2().params.context == 1


def test_ctx_value_mode():
    def_params = CtxDMTaskParameters(value=(1, 0), context=1)
    task = CtxDMTask(params=def_params, batch_size=100, mode="value")
    inputs, _ = task.dataset(1)
    for i in range(100):
        assert inputs[0, i, 0] == 1
        assert inputs[0, i, 1] == 1
        assert inputs[0, i, 2] == 0
        assert task.params.context == 1


def test_ctx_value_mode_second_input():
    def_params = CtxDMTaskParameters(value=(0, 1), context=1)
    task = CtxDMTask(params=def_params, batch_size=100, mode="value")
    inputs, outputs = task.dataset(1)
    for i in range(100):
        assert inputs[0, i, 0] == 1
        assert inputs[0, i, 1] == 0
        assert inputs[0, i, 2] == 1
        assert task.params.context == 1


def test_correct_target_ctx_params_context_1_task():
    def_params = CtxDMTaskParameters(context=0)
    task = CtxDMTask(params=def_params, batch_size=200)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(200):
            assert (inputs[0, j, 1] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 1] > 0.5) == (outputs[-1, j, 2] == 1)


def test_correct_target_ctx_params_context_2_task():
    def_params = CtxDMTaskParameters(context=1)
    task = CtxDMTask(params=def_params, batch_size=200)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(1):
            assert (inputs[0, j, 2] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 2] > 0.5) == (outputs[-1, j, 2] == 1)


def test_correct_target_ctx_1_task():

    task = CtxDM1(batch_size=200)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(200):
            assert (inputs[0, j, 1] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 1] > 0.5) == (outputs[-1, j, 2] == 1)


def test_correct_target_ctx_2_task():

    task = CtxDM2(batch_size=200)
    for _ in range(100):
        inputs, outputs = task.dataset(1)
        for j in range(1):
            assert (inputs[0, j, 2] < 0.5) == (outputs[-1, j, 1] == 1)
            assert (inputs[0, j, 2] > 0.5) == (outputs[-1, j, 2] == 1)


def test_correct_target_test_get_context_shape_random_context_task():
    task = CtxDMTask(batch_size=1000, get_context=True)
    inputs, outputs = task.dataset(1)
    assert inputs.shape[1] == 1000
    assert inputs.shape[2] == 5


def test_correct_target_random_context_task():
    task = CtxDMTask(batch_size=100, get_context=True)
    for _ in range(100):
        inputs, outputs = task.dataset()

        for batch in range(100):
            assert inputs[0, batch, 0] == 1
            assert bool((inputs[0, batch, 1] < 0.5) * inputs[0, batch, 3]) == bool(
                (outputs[-1, batch, 1] == 1) * inputs[0, batch, 3]
            )
            assert bool((inputs[0, batch, 1] > 0.5) * inputs[0, batch, 3]) == bool(
                (outputs[-1, batch, 2] == 1) * inputs[0, batch, 3]
            )
            assert bool((inputs[0, batch, 2] < 0.5) * inputs[0, batch, 4]) == bool(
                (outputs[-1, batch, 1] == 1) * inputs[0, batch, 4]
            )
            assert bool((inputs[0, batch, 2] > 0.5) * inputs[0, batch, 4]) == bool(
                (outputs[-1, batch, 2] == 1) * inputs[0, batch, 4]
            )
