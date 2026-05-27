"""Geometric shape abstractions and concrete implementations."""

from .shape import Shape
from .shape_type import ShapeType
from .line2d import Line2D
from .circle2d import Circle2D
from .plane3d import Plane3D


_REGISTRY = {
    ShapeType.LINE2D: Line2D,
    ShapeType.CIRCLE2D: Circle2D,
    ShapeType.PLANE3D: Plane3D,
}


def class_for(shape_type):
    return _REGISTRY[shape_type]


def shape_from_dict(data):
    return _REGISTRY[ShapeType(data["type"])].from_dict(data)


__all__ = [
    "Shape",
    "ShapeType",
    "Line2D",
    "Circle2D",
    "Plane3D",
    "class_for",
    "shape_from_dict",
]
