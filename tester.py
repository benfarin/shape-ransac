import argparse

from src.estimator.serializer import EstimationSerializer
from src.generator.serializer import Serializer
from src.tester.tester import Tester


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare estimated shapes against the ground truth."
    )
    parser.add_argument("--gt_input_path", required=True, help="ground-truth scene from generator.py")
    parser.add_argument("--est_input_path", required=True, help="estimated shapes from estimator.py")
    parser.add_argument("--debug", action="store_true", help="plot ground truth, estimates, and points")
    args = parser.parse_args(argv)

    gt_result = Serializer.load_full(args.gt_input_path)
    est_result = EstimationSerializer.load(args.est_input_path)
    test_result = Tester().evaluate(gt_result, est_result)

    for match in test_result.matches:
        print(f"  {match.gt_shape.id}: error = {match.error:.4f}")
    print()
    print(f"Mean error:   {test_result.mean_error:.4f}")
    print(f"Median error: {test_result.median_error:.4f}")

    if args.debug:
        from src.visualization.plotter import Plotter
        estimated_shapes = [match.est_shape.shape for match in test_result.matches]
        Plotter().plot_tester_debug(gt_result, estimated_shapes)


if __name__ == "__main__":
    main()
