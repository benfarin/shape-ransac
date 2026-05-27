import numpy as np
import pytest

from src.utils.random_utils import random_point, random_unit_vector, uniform_box


def test_random_point_shape_and_range():
    for _ in range(10):
        p = random_point(2)
        assert p.shape == (2,)
        assert (0 <= p).all() and (p <= 1).all()


def test_random_unit_vector_is_unit():
    for d in (2, 3):
        v = random_unit_vector(d)
        assert v.shape == (d,)
        assert np.linalg.norm(v) == pytest.approx(1.0)


def test_uniform_box_shape_and_range():
    box = uniform_box(50, 3)
    assert box.shape == (50, 3)
    assert (0 <= box).all() and (box <= 1).all()
