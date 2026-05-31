import numpy as np
import pytest

from src.shapes.circle2d import Circle2D


def test_min_points_required():
    assert Circle2D.min_points_required() == 3


def test_from_points_recovers_unit_circle():
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    circle = Circle2D.from_points(pts)

    for p in pts:
        assert circle.distance_to_point(p) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_radial_offset():
    circle = Circle2D(center=[0.0, 0.0], radius=1.0)

    assert circle.distance_to_point(np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert circle.distance_to_point(np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert circle.distance_to_point(np.array([1.5, 0.0])) == pytest.approx(0.5)


def test_distance_batched():
    circle = Circle2D(center=[0.0, 0.0], radius=1.0)
    pts = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])

    np.testing.assert_allclose(circle.distance_to_point(pts), [0.0, 1.0, 1.0])


def test_sample_points_with_zero_noise_lie_on_circle():
    circle = Circle2D(center=[0.5, 0.5], radius=0.3)
    pts = circle.sample_points(50, randomness=0.0)

    np.testing.assert_allclose(circle.distance_to_point(pts), 0.0, atol=1e-12)


def test_to_dict_round_trip():
    original = Circle2D(center=[0.5, 0.5], radius=0.3)
    restored = Circle2D.from_dict(original.to_dict())

    assert restored.distance_to_point(np.array([0.8, 0.5])) == pytest.approx(0.0, abs=1e-12)


def test_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        Circle2D(center=[0.0, 0.0], radius=0.0)
    with pytest.raises(ValueError):
        Circle2D(center=[0.0, 0.0], radius=-0.5)
