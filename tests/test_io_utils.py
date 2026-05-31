from src.utils.io_utils import load_json, save_json


def test_save_and_load_round_trip(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"d": "hello"}}
    path = tmp_path / "out" / "data.json"

    save_json(data, path)
    assert load_json(path) == data


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "data.json"
    save_json({"x": 1}, path)

    assert path.exists()
