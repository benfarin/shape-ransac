from src.generator.config import Config
from src.shapes import ShapeType
from src.utils.exceptions import ConfigurationError
from src.utils.io_utils import load_json


class ConfigLoader:

    @staticmethod
    def load_config(path):
        raw = load_json(path)
        try:
            shapes = {ShapeType(k): int(v) for k, v in raw["shapes"].items()}
            num_points = int(raw["num_points"])
            randomness = float(raw["randomness"])
        except (KeyError, ValueError) as e:
            raise ConfigurationError(f"invalid config at {path}: {e}") from e

        if not shapes:
            raise ConfigurationError("'shapes' must be non-empty")
        if any(count <= 0 for count in shapes.values()):
            raise ConfigurationError("shape counts must be positive")
        if num_points <= 0:
            raise ConfigurationError("'num_points' must be positive")
        if not 0 <= randomness <= 1:
            raise ConfigurationError("'randomness' must be in [0, 1]")

        return Config(shapes=shapes, num_points=num_points, randomness=randomness)
