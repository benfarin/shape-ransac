import argparse

from src.estimator.estimator import Estimator
from src.estimator.serializer import EstimationSerializer
from src.generator.serializer import Serializer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate shape parameters from a noisy point cloud using RANSAC."
    )
    parser.add_argument("--input_path", required=True, help="path to a scene file produced by generator.py")
    parser.add_argument("--output_path", required=True, help="path for the estimated shapes")
    parser.add_argument("--debug", action="store_true", help="plot the estimated shapes")
    args = parser.parse_args(argv)

    points, config = Serializer.load_points_and_schema(args.input_path)
    result = Estimator().estimate(points, config)
    EstimationSerializer.save(result, args.output_path)

    if args.debug:
        from src.visualization.plotter import Plotter
        Plotter().plot_estimator_debug(points, [est.shape for est in result.estimated_shapes])


if __name__ == "__main__":
    main()
