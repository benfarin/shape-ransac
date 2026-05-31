from dataclasses import dataclass

import numpy as np

from src.generator.config import Config
from src.generator.shape_instance import ShapeInstance


@dataclass
class GenerationResult:
    shapes: list[ShapeInstance]
    points: np.ndarray
    inlier_mask: np.ndarray
    config: Config
