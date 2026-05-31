import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.generator.config import Config
from src.generator.generator import Generator
from src.shapes import Circle2D, Line2D, Plane3D, ShapeType
from src.visualization.plotter import Plotter


def test_plot_generator_debug_2d_does_not_raise():
    config = Config(
        shapes={ShapeType.LINE2D: 1, ShapeType.CIRCLE2D: 1},
        num_points=30,
        randomness=0.2,
    )
    result = Generator(config).generate()

    Plotter().plot_generator_debug(result)


def test_plot_generator_debug_3d_does_not_raise():
    config = Config(shapes={ShapeType.PLANE3D: 1}, num_points=30, randomness=0.2)
    result = Generator(config).generate()

    Plotter().plot_generator_debug(result)


def test_plot_estimator_debug_does_not_raise():
    line = Line2D(direction=[1.0, 0.0], point=[0.0, 0.0])
    circle = Circle2D(center=[0.5, 0.5], radius=0.3)
    points = np.random.uniform(0, 1, size=(20, 2))

    Plotter().plot_estimator_debug(points, [line, circle])


def test_plot_tester_debug_does_not_raise():
    config = Config(shapes={ShapeType.LINE2D: 1}, num_points=30, randomness=0.2)
    result = Generator(config).generate()
    estimated = [Line2D(direction=[1.0, 0.1], point=[0.0, 0.05])]

    Plotter().plot_tester_debug(result, estimated)


def test_plot_tester_debug_3d_does_not_raise():
    config = Config(shapes={ShapeType.PLANE3D: 1}, num_points=30, randomness=0.2)
    result = Generator(config).generate()
    estimated = [Plane3D(normal=[0.1, 0.1, 1.0], point=[0.5, 0.5, 0.5])]

    Plotter().plot_tester_debug(result, estimated)
