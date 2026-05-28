import estimator as cli
from src.estimator.serializer import EstimationSerializer
from src.generator.config_loader import ConfigLoader
from src.generator.generator import Generator
from src.generator.serializer import Serializer


def test_writes_estimation_from_scene(tmp_path):
    config = ConfigLoader.load_config("configs/example.json")
    gen_result = Generator(config).generate()
    scene_path = tmp_path / "scene.json"
    Serializer.save(gen_result, scene_path)

    est_path = tmp_path / "est.json"
    cli.main(["--input_path", str(scene_path), "--output_path", str(est_path)])

    assert est_path.exists()


def test_output_has_expected_shape_count(tmp_path):
    config = ConfigLoader.load_config("configs/example.json")
    gen_result = Generator(config).generate()
    scene_path = tmp_path / "scene.json"
    Serializer.save(gen_result, scene_path)

    est_path = tmp_path / "est.json"
    cli.main(["--input_path", str(scene_path), "--output_path", str(est_path)])
    est_result = EstimationSerializer.load(est_path)

    assert len(est_result.estimated_shapes) == 3
