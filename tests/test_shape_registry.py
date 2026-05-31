import numpy as np
import pytest

from src.shapes import (
    Circle2D,
    Line2D,
    Line3D,
    Plane3D,
    ShapeType,
    class_for,
    shape_from_dict,
)


def test_class_for_each_shape_type():
    assert class_for(ShapeType.LINE2D) is Line2D
    assert class_for(ShapeType.CIRCLE2D) is Circle2D
    assert class_for(ShapeType.LINE3D) is Line3D
    assert class_for(ShapeType.PLANE3D) is Plane3D


def test_round_trip_line2d():
    original = Line2D(direction=[1.0, 1.0], point=[0.2, 0.3])
    restored = shape_from_dict(original.to_dict())

    assert isinstance(restored, Line2D)
    assert restored.distance_to_point(np.array([0.2, 0.3])) == pytest.approx(0.0, abs=1e-12)


def test_round_trip_circle2d():
    original = Circle2D(center=[0.5, 0.5], radius=0.3)
    restored = shape_from_dict(original.to_dict())

    assert isinstance(restored, Circle2D)
    assert restored.distance_to_point(np.array([0.8, 0.5])) == pytest.approx(0.0, abs=1e-12)


def test_round_trip_line3d():
    original = Line3D(direction=[1.0, 1.0, 1.0], point=[0.2, 0.3, 0.4])
    restored = shape_from_dict(original.to_dict())

    assert isinstance(restored, Line3D)
    assert restored.distance_to_point(np.array([0.2, 0.3, 0.4])) == pytest.approx(0.0, abs=1e-12)


def test_round_trip_plane3d():
    original = Plane3D(normal=[1.0, 2.0, 3.0], point=[0.0, 0.0, 0.0])
    restored = shape_from_dict(original.to_dict())

    assert isinstance(restored, Plane3D)
    assert restored.distance_to_point(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-12)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        shape_from_dict({"type": "Triangle3D"})
