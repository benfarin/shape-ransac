import numpy as np


def random_point(d):
    return np.random.default_rng().uniform(0, 1, size=d)


def random_unit_vector(d):
    v = np.random.default_rng().standard_normal(d)
    return v / np.linalg.norm(v)


def uniform_box(n, d):
    return np.random.default_rng().uniform(0, 1, size=(n, d))
