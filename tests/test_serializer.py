import numpy as np

from src.generator.config import Config
from src.generator.generation_result import GenerationResult
from src.generator.serializer import Serializer
from src.generator.shape_instance import ShapeInstance
from src.shapes import Circle2D, Line2D, ShapeType


def make_result():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.CIRCLE2D: 1},
        num_points=4,
        randomness=0.3,
    )
    shapes = [
        ShapeInstance(shape=Line2D(direction=[1.0, 0.0], point=[0.0, 0.0]), id="Line2D_0"),
        ShapeInstance(shape=Circle2D(center=[0.5, 0.5], radius=0.3), id="Circle2D_0"),
    ]
    points = np.array([[0.1, 0.0], [0.2, 0.0], [0.5, 0.8], [0.9, 0.9]])
    inlier_mask = np.array([True, True, True, False])
    return GenerationResult(shapes=shapes, points=points, inlier_mask=inlier_mask, config=config)


def test_save_and_load_full_round_trip(tmp_path):
    original = make_result()
    path = tmp_path / "scene.json"

    Serializer.save(original, path)
    restored = Serializer.load_full(path)

    assert restored.config == original.config
    assert [inst.id for inst in restored.shapes] == ["Line2D_0", "Circle2D_0"]
    np.testing.assert_allclose(restored.points, original.points)
    np.testing.assert_array_equal(restored.inlier_mask, original.inlier_mask)


def test_load_points_and_schema_round_trips_points_and_config(tmp_path):
    original = make_result()
    path = tmp_path / "scene.json"
    Serializer.save(original, path)

    points, config = Serializer.load_points_and_schema(path)

    np.testing.assert_allclose(points, original.points)
    assert config == original.config


def test_load_points_and_schema_returns_only_points_and_config(tmp_path):
    original = make_result()
    path = tmp_path / "scene.json"
    Serializer.save(original, path)

    result = Serializer.load_points_and_schema(path)

    assert len(result) == 2
