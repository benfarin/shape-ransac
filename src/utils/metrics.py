import numpy as np


def model_error(gt, est, n_samples=200):
    if gt.shape_type() != est.shape_type():
        raise ValueError("model_error requires shapes of the same type")
    points = gt.sample_points(n_samples, randomness=0.0)
    return float(np.mean(est.distance_to_point(points)))
