import numpy as np

from cgtasknet.tasks.reduce.go import (
    GoDlTask,
    GoDlTask1,
    GoDlTask2,
    GoDlTaskParameters,
    GoDlTaskRandomMod,
    GoDlTaskRandomModParameters,
    GoRtTask,
    GoRtTask1,
    GoRtTask2,
    GoRtTaskParameters,
    GoRtTaskRandomMod,
    GoRtTaskRandomModParameters,
    GoTask,
    GoTask1,
    GoTask2,
    GoTaskParameters,
    GoTaskRandomMod,
    GoTaskRandomModParameters,
)


def test_dm_task_size():
    assert GoTask().feature_and_act_size == (2, 2)
    assert GoRtTask().feature_and_act_size == (2, 2)
    assert GoDlTask().feature_and_act_size == (2, 2)


def test_go_task_run_one_dataset():
    GoTask().one_dataset()


def test_gort_task_run_one_dataset():
    GoRtTask().one_dataset()


def test_godl_task_run_one_dataset():
    GoDlTask().one_dataset()


def test_go_task_get_parameters():
    def_params = GoTaskParameters()
    assert GoTask().params == def_params
    assert GoTask(batch_size=10).batch_size == 10


def test_gort_get_parameters():
    def_params = GoRtTaskParameters()
    assert GoRtTask().params == def_params
    assert GoRtTask(batch_size=10).batch_size == 10


def test_go_task_size_run_some_dataset():
    inputs, outputs = GoTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[1] == 10
    assert outputs.shape[2] == 2


def test_gort_task_size_run_some_dataset():
    inputs, outputs = GoRtTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[1] == 10
    assert outputs.shape[2] == 2


def test_godl_task_size_run_some_dataset():
    inputs, outputs = GoDlTask(batch_size=10).dataset(10)
    assert inputs.shape[1] == 10
    assert inputs.shape[2] == 2
    assert outputs.shape[1] == 10
    assert outputs.shape[2] == 2


def test_go_times_run_some_dataset():
    inputs, outputs = GoTask(batch_size=10).dataset(1)
    def_params = GoTaskParameters()
    dt = def_params.dt
    trial_time = round(def_params.trial_time / dt)
    answer_time = round(def_params.answer_time / dt)
    assert inputs.shape[0] == answer_time + trial_time
    assert outputs.shape[0] == answer_time + trial_time


def test_gort_times_run_some_dataset():
    inputs, outputs = GoRtTask(batch_size=10).dataset(1)
    def_params = GoTaskParameters()
    dt = def_params.dt
    trial_time = round(def_params.trial_time / dt)
    answer_time = round(def_params.answer_time / dt)
    assert inputs.shape[0] == answer_time + trial_time
    assert outputs.shape[0] == answer_time + trial_time


def test_godl_times_run_some_dataset():
    inputs, outputs = GoDlTask(batch_size=10).dataset(1)
    def_params = GoDlTaskParameters()
    dt = def_params.go.dt
    trial_time = round(def_params.go.trial_time / dt)
    answer_time = round(def_params.go.answer_time / dt)
    delay_time = round(def_params.delay / dt)
    print(delay_time)
    assert inputs.shape[0] == answer_time + trial_time + delay_time
    assert outputs.shape[0] == answer_time + trial_time + delay_time


def test_go_task_values_dataset():
    inputs, outputs = GoTask(batch_size=1, mode="value").dataset()
    assert inputs[0, 0, 0] == 1
    assert inputs[-1, 0, 0] == 0
    assert inputs[0, 0, 1] == 1
    assert inputs[-1, 0, 1] == 1

    assert outputs[0, 0, 0] == 1
    assert outputs[-1, 0, 0] == 0
    assert outputs[0, 0, 1] == 0
    assert outputs[-1, 0, 1] == 1


def test_go_task_values_batch_size_dataset():
    batch_size = 100
    inputs, outputs = GoTask(batch_size=batch_size, mode="value").dataset()
    for i in range(batch_size):
        assert inputs[0, i, 0] == 1
        assert inputs[-1, i, 0] == 0
        assert inputs[0, i, 1] == 1
        assert inputs[-1, i, 1] == 1

        assert outputs[0, i, 0] == 1
        assert outputs[-1, i, 0] == 0
        assert outputs[0, i, 1] == 0
        assert outputs[-1, i, 1] == 1


def test_gort_task_values_dataset():
    inputs, outputs = GoRtTask(batch_size=1, mode="value").dataset()
    assert inputs[0, 0, 0] == 1
    assert inputs[-1, 0, 0] == 1
    assert inputs[0, 0, 1] == 0
    assert inputs[-1, 0, 1] == 1

    assert outputs[0, 0, 0] == 1
    assert outputs[-1, 0, 0] == 0
    assert outputs[0, 0, 1] == 0
    assert outputs[-1, 0, 1] == 1


def test_gort_task_values_batch_size_dataset():
    batch_size = 100
    params = GoRtTaskParameters(value=0.5)
    inputs, outputs = GoRtTask(
        params=params, batch_size=batch_size, mode="value"
    ).dataset()
    for i in range(batch_size):
        assert inputs[0, i, 0] == 1
        assert inputs[-1, i, 0] == 1
        assert inputs[0, i, 1] == 0
        assert inputs[-1, i, 1] == 0.5

        assert outputs[0, i, 0] == 1
        assert outputs[-1, i, 0] == 0
        assert outputs[0, i, 1] == 0
        assert outputs[-1, i, 1] == 0.5


def test_godl_task_values_dataset():
    inputs, outputs = GoDlTask(batch_size=10, mode="value").dataset()
    assert inputs[0, 0, 0] == 1
    assert inputs[-1, 0, 0] == 0
    assert inputs[0, 0, 1] == 1
    assert inputs[-1, 0, 1] == 0

    assert outputs[0, 0, 0] == 1
    assert outputs[-1, 0, 0] == 0
    assert outputs[0, 0, 1] == 0
    assert outputs[-1, 0, 1] == 1


def test_godl_task_values_batch_size_dataset():
    batch_size = 100
    inputs, outputs = GoDlTask(batch_size=batch_size, mode="value").dataset()
    for i in range(batch_size):
        assert inputs[0, i, 0] == 1
        assert inputs[-1, i, 0] == 0
        assert inputs[0, i, 1] == 1
        assert inputs[-1, i, 1] == 0

        assert outputs[0, i, 0] == 1
        assert outputs[-1, i, 0] == 0
        assert outputs[0, i, 1] == 0
        assert outputs[-1, i, 1] == 1


def test_go_run_uniq_batches():
    GoTask(uniq_batch=True).one_dataset()
    GoRtTask(uniq_batch=True).one_dataset()
    GoDlTask(uniq_batch=True).one_dataset()


def test_go_random_mode_n_mods_1():
    params = GoTaskRandomModParameters(n_mods=1)
    task = GoTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 2
    assert outputs.shape[-1] == 2


def test_go_random_mode_n_mods_2():
    params = GoTaskRandomModParameters(n_mods=2)
    task = GoTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 3
    assert outputs.shape[-1] == 3


def test_gort_random_mode_n_mods_1():
    params = GoRtTaskRandomModParameters(n_mods=1)
    task = GoRtTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 2
    assert outputs.shape[-1] == 2


def test_gort_random_mode_n_mods_2():
    params = GoRtTaskRandomModParameters(n_mods=2)
    task = GoRtTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 3
    assert outputs.shape[-1] == 3


def test_godl_random_mode_n_mods_1():
    params = GoDlTaskRandomModParameters(n_mods=1)
    task = GoDlTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 2
    assert outputs.shape[-1] == 2


def test_godl_random_mode_n_mods_2():
    params = GoDlTaskRandomModParameters(n_mods=2)
    task = GoDlTaskRandomMod(params=params)
    inputs, outputs = task.dataset()
    assert inputs.shape[-1] == 3
    assert outputs.shape[-1] == 3


def test_go_1():
    params = GoTaskRandomModParameters()
    task = GoTask1(params=params)
    task.dataset()


def test_go_2():
    params = GoTaskRandomModParameters()
    task = GoTask2(params=params)
    task.dataset()


def test_gort_1():
    params = GoRtTaskRandomModParameters()
    task = GoRtTask1(params=params)
    task.dataset()


def test_gort_2():
    params = GoRtTaskRandomModParameters()
    task = GoRtTask2(params=params)
    task.dataset()


def test_godl_1():
    params = GoDlTaskRandomModParameters()
    task = GoDlTask1(params=params)
    task.dataset()


def test_godl_2():
    params = GoDlTaskRandomModParameters()
    task = GoDlTask2(params=params)
    task.dataset()


def test_go_run_list_values():
    values = [0.0, 0.1]
    check_values = []
    params = GoTaskParameters(value=values)
    task = GoTask(params=params)
    test_loop_count = 1000
    for _ in range(test_loop_count):
        data, _ = task.dataset()
        if data[0, 0, 1] not in check_values:
            check_values.append(data[0, 0, 1])
    assert len(check_values) == len(values)
    assert np.allclose(sorted(check_values), values)


def test_gort_run_list_values():
    values = [0.0, 0.1, 1.1]
    check_values = []
    params = GoRtTaskParameters(value=values)
    task = GoRtTask(params=params)
    test_loop_count = 1000
    for _ in range(test_loop_count):
        data, _ = task.dataset()
        if data[-1, 0, 1] not in check_values:
            check_values.append(data[-1, 0, 1])
    assert len(check_values) == len(values)
    assert np.allclose(sorted(check_values), values)


def test_godl_run_list_values():
    values = [0.0, 0.1]
    check_values = []
    params = GoDlTaskParameters(GoTaskParameters(value=values))
    task = GoDlTask(params=params)
    test_loop_count = 1000
    for _ in range(test_loop_count):
        data, _ = task.dataset()
        if data[0, 0, 1] not in check_values:
            check_values.append(data[0, 0, 1])
    assert len(check_values) == len(values)
    assert np.allclose(sorted(check_values), values)
