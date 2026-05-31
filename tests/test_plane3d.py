import numpy as np
import pytest

from src.shapes.plane3d import Plane3D


def test_min_points_required():
    assert Plane3D.min_points_required() == 3


def test_from_points_recovers_xy_plane():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    plane = Plane3D.from_points(pts)

    for p in pts:
        assert plane.distance_to_point(p) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_normal_offset():
    plane = Plane3D(normal=[0.0, 0.0, 1.0], point=[0.0, 0.0, 0.0])

    assert plane.distance_to_point(np.array([0.5, 0.5, 0.4])) == pytest.approx(0.4)
    assert plane.distance_to_point(np.array([0.5, 0.5, -0.3])) == pytest.approx(0.3)
    assert plane.distance_to_point(np.array([1.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)


def test_distance_batched():
    plane = Plane3D(normal=[0.0, 0.0, 1.0], point=[0.0, 0.0, 0.0])
    pts = np.array([[0.0, 0.0, 0.1], [1.0, 2.0, -0.5], [0.0, 0.0, 0.0]])

    np.testing.assert_allclose(plane.distance_to_point(pts), [0.1, 0.5, 0.0])


def test_sample_points_with_zero_noise_lie_on_plane():
    plane = Plane3D(normal=[1.0, 2.0, 3.0], point=[0.5, 0.5, 0.5])
    pts = plane.sample_points(50, randomness=0.0)

    np.testing.assert_allclose(plane.distance_to_point(pts), 0.0, atol=1e-12)


def test_to_dict_round_trip():
    original = Plane3D(normal=[1.0, 2.0, 3.0], point=[0.5, 0.5, 0.5])
    restored = Plane3D.from_dict(original.to_dict())

    assert restored.distance_to_point(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.0, abs=1e-12)


def test_rejects_zero_normal():
    with pytest.raises(ValueError):
        Plane3D(normal=[0.0, 0.0, 0.0], point=[0.0, 0.0, 0.0])
