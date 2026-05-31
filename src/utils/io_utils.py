import json
from pathlib import Path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)
