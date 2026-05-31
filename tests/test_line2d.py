import numpy as np
import pytest

from src.shapes.line2d import Line2D


def test_min_points_required():
    assert Line2D.min_points_required() == 2


def test_from_points_passes_through_both():
    p1 = np.array([0.2, 0.3])
    p2 = np.array([0.7, 0.8])
    line = Line2D.from_points(np.stack([p1, p2]))

    assert line.distance_to_point(p1) == pytest.approx(0.0, abs=1e-12)
    assert line.distance_to_point(p2) == pytest.approx(0.0, abs=1e-12)


def test_distance_matches_perpendicular_offset():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])

    assert line.distance_to_point(np.array([0.5, 0.3])) == pytest.approx(0.3)
    assert line.distance_to_point(np.array([0.0, -0.4])) == pytest.approx(0.4)


def test_distance_batched():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    pts = np.array([[0.0, 0.1], [1.0, -0.2], [-2.0, 0.0]])

    np.testing.assert_allclose(line.distance_to_point(pts), [0.1, 0.2, 0.0])


def test_sample_points_with_zero_noise_lie_on_line():
    line = Line2D(direction=[3.0, 4.0], point=[0.5, 0.5])
    pts = line.sample_points(50, randomness=0.0)

    np.testing.assert_allclose(line.distance_to_point(pts), 0.0, atol=1e-12)


def test_to_dict_round_trip():
    original = Line2D(direction=[3.0, 4.0], point=[0.5, 0.5])
    restored = Line2D.from_dict(original.to_dict())
    on_line = np.array([0.5, 0.5]) + 0.7 * np.array([3.0, 4.0]) / 5.0

    assert restored.distance_to_point(on_line) == pytest.approx(0.0, abs=1e-12)


def test_rejects_zero_direction():
    with pytest.raises(ValueError):
        Line2D(direction=[0.0, 0.0], point=[0.5, 0.5])
