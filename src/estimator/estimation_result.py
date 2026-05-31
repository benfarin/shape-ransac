from dataclasses import dataclass

from src.estimator.estimated_shape import EstimatedShape
from src.generator.config import Config


@dataclass
class EstimationResult:
    estimated_shapes: list[EstimatedShape]
    config: Config
