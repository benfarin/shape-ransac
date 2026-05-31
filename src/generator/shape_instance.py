from dataclasses import dataclass

from src.shapes import Shape


@dataclass
class ShapeInstance:
    shape: Shape
    id: str
