from dataclasses import dataclass

import numpy as np

from src.shapes import Shape
from src.utils.exceptions import EstimationError


@dataclass
class RansacParams:
    threshold: float = 0.05
    max_iterations: int = 200
    min_inlier_ratio: float = 0.1


@dataclass
class RansacResult:
    model: Shape
    inlier_mask: np.ndarray
    score: int
    iterations: int


class RansacEstimator:

    def __init__(self, params=None):
        self.params = params or RansacParams()

    def fit(self, points, shape_cls):
        sample_size = shape_cls.min_points_required()
        if len(points) < sample_size:
            raise ValueError(
                f"need at least {sample_size} points to fit {shape_cls.__name__}, "
                f"got {len(points)}"
            )

        rng = np.random.default_rng()
        best_model = None
        best_mask = None
        best_score = -1
        best_iter = 0

        for i in range(self.params.max_iterations):
            idx = rng.choice(len(points), size=sample_size, replace=False)
            try:
                candidate = shape_cls.from_points(points[idx])
            except (ValueError, np.linalg.LinAlgError):
                continue

            distances = candidate.distance_to_point(points)
            mask = distances < self.params.threshold
            score = int(mask.sum())
            if score > best_score:
                best_model = candidate
                best_mask = mask
                best_score = score
                best_iter = i + 1

        if best_model is None or best_score < self.params.min_inlier_ratio * len(points):
            raise EstimationError(
                f"RANSAC failed to find a {shape_cls.__name__} model "
                f"(best score = {max(best_score, 0)})"
            )

        return RansacResult(
            model=best_model,
            inlier_mask=best_mask,
            score=best_score,
            iterations=best_iter,
        )
