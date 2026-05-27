import numpy as np
import pytest

from src.generator.config import Config
from src.generator.generator import Generator
from src.shapes import ShapeType
from src.utils.exceptions import ConfigurationError


def test_produces_expected_point_count_per_shape():
    config = Config(
        shapes={ShapeType.LINE2D: 2, ShapeType.CIRCLE2D: 1},
        num_points=100,
        randomness=0.3,
    )

    result = Generator(config).generate()

    assert len(result.shapes) == 3
    assert result.points.shape == (300, 2)
    assert result.inlier_mask.shape == (300,)


def test_inlier_outlier_split_is_80_20_per_shape():
    config = Config(
        shapes={ShapeType.LINE2D: 1},
        num_points=100,
        randomness=0.0,
    )

    result = Generator(config).generate()

    assert int(result.inlier_mask.sum()) == 80
    assert int((~result.inlier_mask).sum()) == 20


def test_inliers_lie_on_ground_truth_when_randomness_zero():
    config = Config(
        shapes={ShapeType.LINE2D: 1},
        num_points=100,
        randomness=0.0,
    )

    result = Generator(config).generate()
    inliers = result.points[result.inlier_mask]

    np.testing.assert_allclose(
        result.shapes[0].shape.distance_to_point(inliers), 0.0, atol=1e-12
    )


def test_shape_instances_have_unique_typed_ids():
    config = Config(
        shapes={ShapeType.LINE2D: 3},
        num_points=10,
        randomness=0.5,
    )

    result = Generator(config).generate()

    assert [inst.id for inst in result.shapes] == ["Line2D_0", "Line2D_1", "Line2D_2"]


def test_rejects_mixed_dimension_shapes():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.PLANE3D: 1},
        num_points=10,
        randomness=0.0,
    )

    with pytest.raises(ConfigurationError):
        Generator(config).generate()


def test_generates_3d_points_for_plane3d():
    config = Config(
        shapes={ShapeType.PLANE3D: 1},
        num_points=50,
        randomness=0.0,
    )

    result = Generator(config).generate()

    assert result.points.shape == (50, 3)
