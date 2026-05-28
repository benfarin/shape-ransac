import numpy as np
import pytest

from src.estimator.estimator import Estimator
from src.generator.config import Config
from src.generator.generator import Generator
from src.shapes import ShapeType


def test_estimates_expected_shape_count():
    config = Config(
        shapes={ShapeType.LINE2D: 2, ShapeType.CIRCLE2D: 1},
        num_points=200,
        randomness=0.3,
    )
    result = Generator(config).generate()

    estimation = Estimator().estimate(result.points, config)

    assert len(estimation.estimated_shapes) == 3


def test_recovers_known_line2d_from_generator_output():
    config = Config(
        shapes={ShapeType.LINE2D: 1},
        num_points=200,
        randomness=0.0,
    )
    result = Generator(config).generate()

    estimation = Estimator().estimate(result.points, config)
    gt_point = result.shapes[0].shape.point
    est_line = estimation.estimated_shapes[0].shape

    assert est_line.distance_to_point(gt_point) == pytest.approx(0.0, abs=0.05)


def test_inlier_masks_are_disjoint_between_rounds():
    config = Config(
        shapes={ShapeType.LINE2D: 2},
        num_points=100,
        randomness=0.0,
    )
    result = Generator(config).generate()

    estimation = Estimator().estimate(result.points, config)
    mask_a = estimation.estimated_shapes[0].inlier_mask
    mask_b = estimation.estimated_shapes[1].inlier_mask

    assert not (mask_a & mask_b).any()


def test_inlier_masks_match_original_point_count():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.CIRCLE2D: 1},
        num_points=80,
        randomness=0.2,
    )
    result = Generator(config).generate()

    estimation = Estimator().estimate(result.points, config)

    for est in estimation.estimated_shapes:
        assert est.inlier_mask.shape == (len(result.points),)
