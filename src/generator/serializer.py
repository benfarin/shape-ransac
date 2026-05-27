import numpy as np

from src.generator.config import Config
from src.generator.generation_result import GenerationResult
from src.generator.shape_instance import ShapeInstance
from src.shapes import ShapeType, shape_from_dict
from src.utils.io_utils import load_json, save_json


class Serializer:

    @staticmethod
    def save(result, path):
        data = {
            "config": _config_to_dict(result.config),
            "shapes": [
                {"id": inst.id, "shape": inst.shape.to_dict()}
                for inst in result.shapes
            ],
            "points": result.points.tolist(),
            "inlier_mask": result.inlier_mask.tolist(),
        }
        save_json(data, path)

    @staticmethod
    def load_full(path):
        data = load_json(path)
        config = _config_from_dict(data["config"])
        shapes = [
            ShapeInstance(shape=shape_from_dict(item["shape"]), id=item["id"])
            for item in data["shapes"]
        ]
        points = np.array(data["points"], dtype=float)
        inlier_mask = np.array(data["inlier_mask"], dtype=bool)
        return GenerationResult(
            shapes=shapes,
            points=points,
            inlier_mask=inlier_mask,
            config=config,
        )

    @staticmethod
    def load_points_and_schema(path):
        data = load_json(path)
        config = _config_from_dict(data["config"])
        points = np.array(data["points"], dtype=float)
        return points, config


def _config_to_dict(config):
    return {
        "shapes": {st.value: count for st, count in config.shapes.items()},
        "num_points": config.num_points,
        "randomness": config.randomness,
    }


def _config_from_dict(data):
    return Config(
        shapes={ShapeType(k): v for k, v in data["shapes"].items()},
        num_points=data["num_points"],
        randomness=data["randomness"],
    )
