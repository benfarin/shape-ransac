import pytest

from src.shapes import Circle2D, Line2D
from src.utils.metrics import model_error


def test_zero_for_overlapping_lines():
    a = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    b = Line2D(direction=[1.0, 0.0], point=[0.5, 0.0])

    assert model_error(a, b) == pytest.approx(0.0, abs=1e-12)


def test_matches_perpendicular_offset_between_parallel_lines():
    gt = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    parallel = Line2D(direction=[1.0, 0.0], point=[0.0, 0.3])

    assert model_error(gt, parallel) == pytest.approx(0.3, abs=1e-6)


def test_rejects_mismatched_shape_types():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    circle = Circle2D(center=[0.0, 0.0], radius=1.0)

    with pytest.raises(ValueError):
        model_error(line, circle)
