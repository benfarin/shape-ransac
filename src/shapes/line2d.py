import numpy as np

from src.utils.random_utils import random_point, random_unit_vector

from .shape import Shape
from .shape_type import ShapeType


NOISE_SCALE = 0.05


class Line2D(Shape):
    """A 2D line parameterized by a unit direction and a point on the line."""

    def __init__(self, direction, point):
        direction = np.asarray(direction, dtype=float).reshape(2)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("direction must be non-zero")
        self._direction = direction / norm
        self._point = np.asarray(point, dtype=float).reshape(2)

    @property
    def direction(self):
        return self._direction

    @property
    def point(self):
        return self._point

    @classmethod
    def min_points_required(cls):
        return 2

    @classmethod
    def from_points(cls, points):
        p1, p2 = np.asarray(points, dtype=float)
        return cls(direction=p2 - p1, point=p1)

    def distance_to_point(self, points):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        delta = pts - self._point
        cross = delta[:, 0] * self._direction[1] - delta[:, 1] * self._direction[0]
        d = np.abs(cross)
        return d[0] if single else d

    def sample_points(self, n, randomness):
        rng = np.random.default_rng()
        t = rng.uniform(-0.5, 0.5, size=n)
        base = self._point + t[:, None] * self._direction
        normal = np.array([-self._direction[1], self._direction[0]])
        noise = rng.normal(0, NOISE_SCALE * randomness, size=n)
        return base + noise[:, None] * normal

    def to_dict(self):
        return {
            "type": ShapeType.LINE2D.value,
            "direction": self._direction.tolist(),
            "point": self._point.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(direction=data["direction"], point=data["point"])

    @classmethod
    def shape_type(cls):
        return ShapeType.LINE2D

    @classmethod
    def random(cls):
        return cls(direction=random_unit_vector(2), point=random_point(2))
