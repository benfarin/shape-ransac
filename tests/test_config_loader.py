import json

import pytest

from src.generator.config_loader import ConfigLoader
from src.shapes import ShapeType
from src.utils.exceptions import ConfigurationError


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def test_loads_valid_config(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {"Line2D": 2, "Circle2D": 1},
        "num_points": 500,
        "randomness": 0.5,
    })

    config = ConfigLoader.load_config(path)

    assert config.shapes == {ShapeType.LINE2D: 2, ShapeType.CIRCLE2D: 1}
    assert config.num_points == 500
    assert config.randomness == 0.5


def test_rejects_unknown_shape_name(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {"Triangle": 2},
        "num_points": 500,
        "randomness": 0.5,
    })

    with pytest.raises(ConfigurationError):
        ConfigLoader.load_config(path)


def test_rejects_empty_shapes(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {},
        "num_points": 100,
        "randomness": 0.5,
    })

    with pytest.raises(ConfigurationError):
        ConfigLoader.load_config(path)


def test_rejects_non_positive_count(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {"Line2D": 0},
        "num_points": 100,
        "randomness": 0.5,
    })

    with pytest.raises(ConfigurationError):
        ConfigLoader.load_config(path)


def test_rejects_non_positive_num_points(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {"Line2D": 1},
        "num_points": 0,
        "randomness": 0.5,
    })

    with pytest.raises(ConfigurationError):
        ConfigLoader.load_config(path)


def test_rejects_randomness_out_of_range(tmp_path):
    path = write_config(tmp_path, {
        "shapes": {"Line2D": 1},
        "num_points": 100,
        "randomness": 1.5,
    })

    with pytest.raises(ConfigurationError):
        ConfigLoader.load_config(path)
