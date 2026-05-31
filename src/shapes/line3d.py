import numpy as np

from src.utils.random_utils import random_point, random_unit_vector

from .shape import Shape
from .shape_type import ShapeType


NOISE_SCALE = 0.05


class Line3D(Shape):
    """A 3D line parameterized by a unit direction and a point on the line."""

    def __init__(self, direction, point):
        direction = np.asarray(direction, dtype=float).reshape(3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("direction must be non-zero")
        self._direction = direction / norm
        self._point = np.asarray(point, dtype=float).reshape(3)

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
        cross = np.cross(delta, self._direction)
        d = np.linalg.norm(cross, axis=1)
        return d[0] if single else d

    def sample_points(self, n, randomness):
        rng = np.random.default_rng()
        t = rng.uniform(-0.5, 0.5, size=n)
        base = self._point + t[:, None] * self._direction
        u_axis, v_axis = self._perpendicular_basis()
        a = rng.normal(0, NOISE_SCALE * randomness, size=n)
        b = rng.normal(0, NOISE_SCALE * randomness, size=n)
        return base + a[:, None] * u_axis + b[:, None] * v_axis

    def _perpendicular_basis(self):
        helper = np.array([1.0, 0.0, 0.0]) if abs(self._direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = helper - (helper @ self._direction) * self._direction
        u /= np.linalg.norm(u)
        v = np.cross(self._direction, u)
        return u, v

    def to_dict(self):
        return {
            "type": ShapeType.LINE3D.value,
            "direction": self._direction.tolist(),
            "point": self._point.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(direction=data["direction"], point=data["point"])

    @classmethod
    def shape_type(cls):
        return ShapeType.LINE3D

    @classmethod
    def random(cls):
        return cls(direction=random_unit_vector(3), point=random_point(3))
