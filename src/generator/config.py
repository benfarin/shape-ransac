from dataclasses import dataclass

from src.shapes import ShapeType


@dataclass
class Config:
    shapes: dict[ShapeType, int]
    num_points: int
    randomness: float
