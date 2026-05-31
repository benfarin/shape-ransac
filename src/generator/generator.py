import numpy as np

from src.generator.generation_result import GenerationResult
from src.generator.shape_instance import ShapeInstance
from src.shapes import class_for
from src.utils.exceptions import ConfigurationError


OUTLIER_RATE = 0.2
OUTLIER_BOUNDS = (-0.2, 1.2)


class Generator:

    def __init__(self, config):
        self.config = config

    def generate(self):
        instances = self._random_shape_instances()
        dim = self._verify_uniform_dim(instances)

        per_shape = self.config.num_points
        n_outliers = int(round(per_shape * OUTLIER_RATE))
        n_inliers = per_shape - n_outliers

        chunks_points = []
        chunks_mask = []
        for inst in instances:
            inliers = inst.shape.sample_points(n_inliers, self.config.randomness)
            outliers = self._random_outliers(n_outliers, dim)
            chunks_points.extend([inliers, outliers])
            chunks_mask.extend([
                np.ones(n_inliers, dtype=bool),
                np.zeros(n_outliers, dtype=bool),
            ])

        points = np.vstack(chunks_points)
        inlier_mask = np.concatenate(chunks_mask)

        order = np.random.default_rng().permutation(len(points))
        return GenerationResult(
            shapes=instances,
            points=points[order],
            inlier_mask=inlier_mask[order],
            config=self.config,
        )

    def _random_shape_instances(self):
        instances = []
        for shape_type, count in self.config.shapes.items():
            cls = class_for(shape_type)
            for i in range(count):
                instances.append(
                    ShapeInstance(shape=cls.random(), id=f"{shape_type.value}_{i}")
                )
        return instances

    def _random_outliers(self, n, dim):
        low, high = OUTLIER_BOUNDS
        return np.random.default_rng().uniform(low, high, size=(n, dim))

    def _verify_uniform_dim(self, instances):
        dims = {inst.shape.sample_points(1, 0.0).shape[1] for inst in instances}
        if len(dims) > 1:
            raise ConfigurationError(
                "shapes in a config must share the same ambient dimension"
            )
        return dims.pop()
