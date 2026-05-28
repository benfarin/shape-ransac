from dataclasses import dataclass

import numpy as np

from src.shapes import Shape


@dataclass
class EstimatedShape:
    shape: Shape
    inlier_mask: np.ndarray
    score: int
