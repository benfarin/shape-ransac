import numpy as np

from src.estimator.estimated_shape import EstimatedShape
from src.estimator.estimation_result import EstimationResult
from src.estimator.ransac import RansacEstimator
from src.shapes import class_for
from src.utils.exceptions import EstimationError


class Estimator:

    def __init__(self, params=None):
        self.ransac = RansacEstimator(params)

    def estimate(self, points, config):
        remaining_indices = np.arange(len(points))
        estimated = []

        for shape_type, count in config.shapes.items():
            shape_cls = class_for(shape_type)
            for _ in range(count):
                if len(remaining_indices) < shape_cls.min_points_required():
                    raise EstimationError(
                        f"too few residual points to fit another {shape_cls.__name__}"
                    )
                residual = points[remaining_indices]
                result = self.ransac.fit(residual, shape_cls)

                original_inliers = remaining_indices[result.inlier_mask]
                full_mask = np.zeros(len(points), dtype=bool)
                full_mask[original_inliers] = True

                estimated.append(EstimatedShape(
                    shape=result.model,
                    inlier_mask=full_mask,
                    score=result.score,
                ))

                remaining_indices = remaining_indices[~result.inlier_mask]

        return EstimationResult(estimated_shapes=estimated, config=config)
