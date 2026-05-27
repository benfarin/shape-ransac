from dataclasses import dataclass

from src.shapes import ShapeType


@dataclass
class Config:
    shapes: dict[ShapeType, int]
    num_points: int
    randomness: float

    def to_dict(self):
        return {
            "shapes": {st.value: count for st, count in self.shapes.items()},
            "num_points": self.num_points,
            "randomness": self.randomness,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            shapes={ShapeType(k): v for k, v in data["shapes"].items()},
            num_points=data["num_points"],
            randomness=data["randomness"],
        )
