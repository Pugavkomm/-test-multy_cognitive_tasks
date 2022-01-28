from cgtasknet.tasks.reduce import DefaultParams


def test_default_params_dm_task():
    assert DefaultParams("DMTask").generate_params() == dict(
        [("dt", 1e-3), ("trial_time", 0.75), ("answer_time", 0.15), ("value", None)]
    )


def test_default_params_dm_task_random_mod():
    assert DefaultParams("DMTaskRandomMod").generate_params() == dict(
        [("dt", 1e-3), ("trial_time", 0.75), ("answer_time", 0.15), ("n_mods", 2)]
    )


def test_default_params_romo_task():
    assert DefaultParams("RomoTask").generate_params() == dict(
        [
            ("dt", 1e-3),
            ("delay", 0.3),
            ("trial_time", 0.25),
            ("answer_time", 0.15),
            ("values", (None, None)),
        ]
    )


def test_default_params_romo_task_random_mod():
    assert DefaultParams("RomoTaskRandomMod").generate_params() == dict(
        [
            ("dt", 1e-3),
            ("delay", 0.3),
            ("trial_time", 0.25),
            ("answer_time", 0.15),
            ("n_mods", 2),
        ]
    )


def test_default_params_ctx_dm_task():
    assert DefaultParams("CtxDMTask").generate_params() == dict(
        [("dt", 1e-3), ("trial_time", 0.75), ("answer_time", 0.15)]
    )
