from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .shape_type import ShapeType


class Shape(ABC):
    """Behavioral interface for geometric shapes.

    Concrete shapes own their own parameter representation (a 2D line stores
    a direction and a point, a circle stores a center and a radius, and so on).
    The base class deliberately defines what a shape can *do* — sample points,
    measure perpendicular distance, fit from a minimal sample, serialize —
    rather than imposing a shared data layout. This keeps RANSAC, the generator,
    and the tester completely shape-agnostic.
    """

    @classmethod
    @abstractmethod
    def min_points_required(cls) -> int:
        """Number of points RANSAC must sample to determine a candidate model."""

    @classmethod
    @abstractmethod
    def from_points(cls, points: np.ndarray) -> Shape:
        """Fit a shape from a minimal sample of exactly ``min_points_required()`` points."""

    @abstractmethod
    def distance_to_point(self, points: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each point to the shape.

        Accepts a single point of shape ``(D,)`` or a batch ``(N, D)`` and
        returns the matching scalar or ``(N,)`` array.
        """

    @abstractmethod
    def sample_points(self, n: int, randomness: float) -> np.ndarray:
        """Generate ``n`` noisy inlier points around the shape.

        ``randomness`` is clamped to ``[0, 1]`` and scales the additive noise.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization including a ``"type"`` tag for round-tripping."""

    @classmethod
    @abstractmethod
    def shape_type(cls) -> ShapeType:
        """The :class:`ShapeType` enum value for this concrete shape."""
