from enum import Enum


class ShapeType(Enum):
    """Supported shape categories.

    Values match the keys used in the JSON config file so that
    ``ShapeType(value)`` parses a config entry directly.
    """

    LINE2D = "Line2D"
    CIRCLE2D = "Circle2D"
    LINE3D = "Line3D"
    PLANE3D = "Plane3D"
