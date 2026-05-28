import numpy as np

from src.estimator.estimated_shape import EstimatedShape
from src.estimator.estimation_result import EstimationResult
from src.estimator.serializer import EstimationSerializer
from src.generator.config import Config
from src.shapes import Circle2D, Line2D, ShapeType


def make_result():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.CIRCLE2D: 1},
        num_points=100,
        randomness=0.3,
    )
    estimated = [
        EstimatedShape(
            shape=Line2D(direction=[1.0, 0.0], point=[0.0, 0.0]),
            inlier_mask=np.array([True, False, True, True]),
            score=3,
        ),
        EstimatedShape(
            shape=Circle2D(center=[0.5, 0.5], radius=0.3),
            inlier_mask=np.array([False, True, False, False]),
            score=1,
        ),
    ]
    return EstimationResult(estimated_shapes=estimated, config=config)


def test_save_and_load_round_trip(tmp_path):
    original = make_result()
    path = tmp_path / "estimation.json"

    EstimationSerializer.save(original, path)
    restored = EstimationSerializer.load(path)

    assert restored.config == original.config
    assert len(restored.estimated_shapes) == 2
    assert [e.score for e in restored.estimated_shapes] == [3, 1]
    for est_orig, est_restored in zip(original.estimated_shapes, restored.estimated_shapes):
        np.testing.assert_array_equal(est_restored.inlier_mask, est_orig.inlier_mask)


def test_shape_types_round_trip(tmp_path):
    original = make_result()
    path = tmp_path / "estimation.json"

    EstimationSerializer.save(original, path)
    restored = EstimationSerializer.load(path)

    assert restored.estimated_shapes[0].shape.shape_type() == ShapeType.LINE2D
    assert restored.estimated_shapes[1].shape.shape_type() == ShapeType.CIRCLE2D
