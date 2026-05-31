import numpy as np
import pytest

from src.estimator.ransac import RansacEstimator, RansacParams
from src.shapes import Circle2D, Line2D, Line3D, Plane3D
from src.utils.exceptions import EstimationError


def test_recovers_clean_line2d():
    line = Line2D(direction=[1.0, 2.0], point=[0.5, 0.5])
    points = line.sample_points(50, randomness=0.0)

    result = RansacEstimator().fit(points, Line2D)

    np.testing.assert_allclose(result.model.distance_to_point(points), 0.0, atol=1e-12)


def test_recovers_clean_circle2d():
    circle = Circle2D(center=[0.5, 0.5], radius=0.3)
    points = circle.sample_points(50, randomness=0.0)

    result = RansacEstimator().fit(points, Circle2D)

    np.testing.assert_allclose(result.model.distance_to_point(points), 0.0, atol=1e-12)


def test_recovers_clean_line3d():
    line = Line3D(direction=[1.0, 2.0, 3.0], point=[0.5, 0.5, 0.5])
    points = line.sample_points(50, randomness=0.0)

    result = RansacEstimator().fit(points, Line3D)

    np.testing.assert_allclose(result.model.distance_to_point(points), 0.0, atol=1e-12)


def test_recovers_clean_plane3d():
    plane = Plane3D(normal=[0.0, 0.0, 1.0], point=[0.0, 0.0, 0.5])
    points = plane.sample_points(50, randomness=0.0)

    result = RansacEstimator().fit(points, Plane3D)

    np.testing.assert_allclose(result.model.distance_to_point(points), 0.0, atol=1e-12)


def test_rejects_outliers_far_from_line():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    inliers = line.sample_points(50, randomness=0.0)
    outliers = np.random.uniform(0, 1, size=(20, 2)) + np.array([0.0, 1.0])
    points = np.vstack([inliers, outliers])

    result = RansacEstimator().fit(points, Line2D)

    assert result.score == 50


def test_returns_boolean_inlier_mask_matching_points():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    inliers = line.sample_points(50, randomness=0.0)
    outliers = np.random.uniform(0, 1, size=(10, 2)) + np.array([0.0, 1.0])
    points = np.vstack([inliers, outliers])

    result = RansacEstimator().fit(points, Line2D)

    assert result.inlier_mask.shape == (60,)
    assert result.inlier_mask.dtype == bool


def test_raises_when_too_few_points_for_shape():
    points = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError):
        RansacEstimator().fit(points, Line2D)


def test_raises_when_min_inlier_ratio_not_met():
    points = np.random.uniform(0, 1, size=(50, 2))
    params = RansacParams(threshold=1e-6, min_inlier_ratio=0.9)

    with pytest.raises(EstimationError):
        RansacEstimator(params).fit(points, Line2D)
