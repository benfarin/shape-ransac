from dataclasses import dataclass

from src.estimator.estimated_shape import EstimatedShape
from src.generator.shape_instance import ShapeInstance


@dataclass
class MatchResult:
    gt_shape: ShapeInstance
    est_shape: EstimatedShape
    error: float
