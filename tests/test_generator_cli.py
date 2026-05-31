from pathlib import Path

import generator as cli
from src.generator.serializer import Serializer


def test_writes_scene_to_output_path(tmp_path):
    config_path = Path("configs/example.json").resolve()
    output_path = tmp_path / "scene.json"

    cli.main(["--config_path", str(config_path), "--output_path", str(output_path)])

    assert output_path.exists()


def test_output_matches_example_config(tmp_path):
    config_path = Path("configs/example.json").resolve()
    output_path = tmp_path / "scene.json"

    cli.main(["--config_path", str(config_path), "--output_path", str(output_path)])
    result = Serializer.load_full(output_path)

    assert len(result.shapes) == 3
    assert result.points.shape == (600, 2)
