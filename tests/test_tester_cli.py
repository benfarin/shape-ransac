import tester as cli
from src.estimator.estimator import Estimator
from src.estimator.serializer import EstimationSerializer
from src.generator.config_loader import ConfigLoader
from src.generator.generator import Generator
from src.generator.serializer import Serializer


def setup_pipeline(tmp_path):
    config = ConfigLoader.load_config("configs/example.json")
    gen_result = Generator(config).generate()
    scene_path = tmp_path / "scene.json"
    Serializer.save(gen_result, scene_path)

    est_result = Estimator().estimate(gen_result.points, config)
    est_path = tmp_path / "est.json"
    EstimationSerializer.save(est_result, est_path)

    return scene_path, est_path, gen_result


def test_prints_aggregate_score(tmp_path, capsys):
    scene_path, est_path, _ = setup_pipeline(tmp_path)

    cli.main([
        "--gt_input_path", str(scene_path),
        "--est_input_path", str(est_path),
    ])

    out = capsys.readouterr().out
    assert "Final score" in out


def test_prints_per_shape_errors(tmp_path, capsys):
    scene_path, est_path, gen_result = setup_pipeline(tmp_path)

    cli.main([
        "--gt_input_path", str(scene_path),
        "--est_input_path", str(est_path),
    ])

    out = capsys.readouterr().out
    for inst in gen_result.shapes:
        assert inst.id in out
