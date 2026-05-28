import numpy as np
import pytest

from src.estimator.estimated_shape import EstimatedShape
from src.estimator.estimation_result import EstimationResult
from src.estimator.estimator import Estimator
from src.generator.config import Config
from src.generator.generation_result import GenerationResult
from src.generator.generator import Generator
from src.generator.shape_instance import ShapeInstance
from src.shapes import Circle2D, Line2D, ShapeType
from src.tester.tester import Tester


def test_matches_each_ground_truth_shape_with_an_estimate():
    config = Config(
        shapes={ShapeType.LINE2D: 2, ShapeType.CIRCLE2D: 1},
        num_points=200,
        randomness=0.0,
    )
    gen_result = Generator(config).generate()
    est_result = Estimator().estimate(gen_result.points, config)

    test_result = Tester().evaluate(gen_result, est_result)

    assert len(test_result.matches) == 3


def test_error_stays_low_on_clean_data():
    config = Config(shapes={ShapeType.LINE2D: 1}, num_points=100, randomness=0.0)
    gen_result = Generator(config).generate()
    est_result = Estimator().estimate(gen_result.points, config)

    test_result = Tester().evaluate(gen_result, est_result)

    assert test_result.mean_error < 0.05


def test_matched_pair_has_same_shape_type():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.CIRCLE2D: 1},
        num_points=200,
        randomness=0.0,
    )
    gen_result = Generator(config).generate()
    est_result = Estimator().estimate(gen_result.points, config)

    test_result = Tester().evaluate(gen_result, est_result)

    for match in test_result.matches:
        assert match.gt_shape.shape.shape_type() == match.est_shape.shape.shape_type()


def test_mean_and_median_match_per_pair_errors():
    config = Config(shapes={ShapeType.LINE2D: 2}, num_points=10, randomness=0.0)

    gt_lines = [
        Line2D(direction=[1.0, 0.0], point=[0.0, 0.0]),
        Line2D(direction=[1.0, 0.0], point=[0.0, 1.0]),
    ]
    est_lines = [
        Line2D(direction=[1.0, 0.0], point=[0.0, 0.1]),
        Line2D(direction=[1.0, 0.0], point=[0.0, 1.3]),
    ]

    gt = GenerationResult(
        shapes=[
            ShapeInstance(shape=gt_lines[0], id="Line2D_0"),
            ShapeInstance(shape=gt_lines[1], id="Line2D_1"),
        ],
        points=np.empty((0, 2)),
        inlier_mask=np.empty(0, dtype=bool),
        config=config,
    )
    est = EstimationResult(
        estimated_shapes=[
            EstimatedShape(shape=est_lines[0], inlier_mask=np.empty(0, dtype=bool), score=0),
            EstimatedShape(shape=est_lines[1], inlier_mask=np.empty(0, dtype=bool), score=0),
        ],
        config=config,
    )

    test_result = Tester().evaluate(gt, est)

    errors = sorted(m.error for m in test_result.matches)
    np.testing.assert_allclose(errors, [0.1, 0.3], atol=1e-6)
    assert test_result.mean_error == pytest.approx(0.2)


def test_empty_inputs_yield_zero_error():
    config = Config(shapes={ShapeType.LINE2D: 1}, num_points=10, randomness=0.0)
    gt = GenerationResult(
        shapes=[],
        points=np.empty((0, 2)),
        inlier_mask=np.empty(0, dtype=bool),
        config=config,
    )
    est = EstimationResult(estimated_shapes=[], config=config)

    test_result = Tester().evaluate(gt, est)

    assert test_result.matches == []
    assert test_result.mean_error == 0.0
    assert test_result.median_error == 0.0
