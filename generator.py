import argparse

from src.generator.config_loader import ConfigLoader
from src.generator.generator import Generator
from src.generator.serializer import Serializer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a noisy point cloud from a shape config."
    )
    parser.add_argument("--config_path", required=True, help="path to config JSON file")
    parser.add_argument("--output_path", required=True, help="path for the generated scene")
    parser.add_argument("--debug", action="store_true", help="plot the generated scene")
    args = parser.parse_args(argv)

    config = ConfigLoader.load_config(args.config_path)
    result = Generator(config).generate()
    Serializer.save(result, args.output_path)

    if args.debug:
        from src.visualization.plotter import Plotter
        Plotter().plot_generator_debug(result)


if __name__ == "__main__":
    main()
