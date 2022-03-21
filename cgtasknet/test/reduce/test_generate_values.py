import numpy as np

from cgtasknet.tasks.reduce import _generate_values


def test_generate_value_mode():
    mode = "value"
    values = [0, 1, 2, 3, 4]
    size = 10
    for value in values:
        assert np.allclose(
            np.ones(size, dtype=np.float64) * value, _generate_values(mode, size, value)
        )


def test_generate_random():
    mode = "random"
    value = 1
    size = 10
    seed = 1024
    np.random.seed(seed)
    assert np.allclose(
        np.random.uniform(0, value, size=size),
        _generate_values(mode, size, value, seed=seed),
    )


def test_generate_list_values():
    mode = "random"
    values = [0, 1, 2, 3, 4]
    size = 150
    seed = 30125
    np.random.seed(seed)
    assert np.allclose(
        np.random.choice(values, size=size),
        _generate_values(mode, size, values, seed=seed),
    )
