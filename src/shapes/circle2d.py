import numpy as np

from src.utils.random_utils import random_point

from .shape import Shape
from .shape_type import ShapeType


NOISE_SCALE = 0.05


class Circle2D(Shape):
    """A 2D circle defined by a center and a radius."""

    def __init__(self, center, radius):
        radius = float(radius)
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._center = np.asarray(center, dtype=float).reshape(2)
        self._radius = radius

    @classmethod
    def min_points_required(cls):
        return 3

    @classmethod
    def from_points(cls, points):
        a, b, c = np.asarray(points, dtype=float)
        m = 2 * np.stack([b - a, c - a])
        rhs = np.array([b @ b - a @ a, c @ c - a @ a])
        center = np.linalg.solve(m, rhs)
        radius = np.linalg.norm(center - a)
        return cls(center=center, radius=radius)

    def distance_to_point(self, points):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        radial = np.linalg.norm(pts - self._center, axis=1)
        d = np.abs(radial - self._radius)
        return d[0] if single else d

    def sample_points(self, n, randomness):
        rng = np.random.default_rng()
        theta = rng.uniform(0, 2 * np.pi, size=n)
        unit = np.column_stack([np.cos(theta), np.sin(theta)])
        base = self._center + self._radius * unit
        noise = rng.normal(0, NOISE_SCALE * randomness, size=n)
        return base + noise[:, None] * unit

    def to_dict(self):
        return {
            "type": ShapeType.CIRCLE2D.value,
            "center": self._center.tolist(),
            "radius": self._radius,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(center=data["center"], radius=data["radius"])

    @classmethod
    def shape_type(cls):
        return ShapeType.CIRCLE2D

    @classmethod
    def random(cls):
        radius = float(np.random.default_rng().uniform(0.1, 0.3))
        return cls(center=random_point(2), radius=radius)
