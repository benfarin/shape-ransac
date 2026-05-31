import numpy as np

from src.estimator.estimated_shape import EstimatedShape
from src.estimator.estimation_result import EstimationResult
from src.generator.config import Config
from src.shapes import shape_from_dict
from src.utils.io_utils import load_json, save_json


class EstimationSerializer:

    @staticmethod
    def save(result, path):
        data = {
            "config": result.config.to_dict(),
            "estimated_shapes": [
                {
                    "shape": est.shape.to_dict(),
                    "inlier_mask": est.inlier_mask.tolist(),
                    "score": est.score,
                }
                for est in result.estimated_shapes
            ],
        }
        save_json(data, path)

    @staticmethod
    def load(path):
        data = load_json(path)
        config = Config.from_dict(data["config"])
        estimated = [
            EstimatedShape(
                shape=shape_from_dict(item["shape"]),
                inlier_mask=np.array(item["inlier_mask"], dtype=bool),
                score=item["score"],
            )
            for item in data["estimated_shapes"]
        ]
        return EstimationResult(estimated_shapes=estimated, config=config)
