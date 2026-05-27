import numpy as np

from src.utils.random_utils import random_point, random_unit_vector

from .shape import Shape
from .shape_type import ShapeType


NOISE_SCALE = 0.05


class Plane3D(Shape):
    """A plane in 3D space defined by a unit normal and a point on the plane."""

    def __init__(self, normal, point):
        normal = np.asarray(normal, dtype=float).reshape(3)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("normal must be non-zero")
        self._normal = normal / norm
        self._point = np.asarray(point, dtype=float).reshape(3)

    @property
    def normal(self):
        return self._normal

    @property
    def point(self):
        return self._point

    @classmethod
    def min_points_required(cls):
        return 3

    @classmethod
    def from_points(cls, points):
        a, b, c = np.asarray(points, dtype=float)
        normal = np.cross(b - a, c - a)
        return cls(normal=normal, point=a)

    def distance_to_point(self, points):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        d = np.abs((pts - self._point) @ self._normal)
        return d[0] if single else d

    def sample_points(self, n, randomness):
        rng = np.random.default_rng()
        u_axis, v_axis = self.plane_basis()
        u = rng.uniform(-0.5, 0.5, size=n)
        v = rng.uniform(-0.5, 0.5, size=n)
        base = self._point + u[:, None] * u_axis + v[:, None] * v_axis
        noise = rng.normal(0, NOISE_SCALE * randomness, size=n)
        return base + noise[:, None] * self._normal

    def plane_basis(self):
        helper = np.array([1.0, 0.0, 0.0]) if abs(self._normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = helper - (helper @ self._normal) * self._normal
        u /= np.linalg.norm(u)
        v = np.cross(self._normal, u)
        return u, v

    def to_dict(self):
        return {
            "type": ShapeType.PLANE3D.value,
            "normal": self._normal.tolist(),
            "point": self._point.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(normal=data["normal"], point=data["point"])

    @classmethod
    def shape_type(cls):
        return ShapeType.PLANE3D

    @classmethod
    def random(cls):
        return cls(normal=random_unit_vector(3), point=random_point(3))
