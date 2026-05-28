# shape-ransac

A small system that generates noisy point clouds around random shapes, recovers the shape parameters from the noise with RANSAC, and grades the recovery against the ground truth.

Built as the three parts laid out in the assignment:

1. **Generator** — reads a config, places random shapes in the scene, and emits a noisy point cloud (80% inliers per shape + 20% outliers).
2. **Estimator** — runs RANSAC on the points to recover each shape. Only sees the points and the schema (shape types + counts), never the ground-truth parameters.
3. **Tester** — matches each ground-truth shape to its closest estimate of the same type and prints per-shape errors plus a single aggregate score.

Supported shapes: `Line2D`, `Circle2D`, `Plane3D`.

The design diagram is in `design.drawio` (PNG export in `design.png`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pytest`.

## Run

The three scripts at the repo root are thin CLI wrappers around the corresponding classes in `src/`.

### Generator

```bash
python generator.py \
    --config_path configs/example.json \
    --output_path out/scene.json \
    [--debug]
```

`--debug` plots the generated scene (inliers in green, outliers in red, ground-truth shapes overlaid).

### Estimator

```bash
python estimator.py \
    --input_path out/scene.json \
    --output_path out/estimation.json \
    [--debug]
```

Reads the generator's output through `Serializer.load_points_and_schema`, which returns only the points and the config — the estimator code physically cannot reach the ground-truth shape parameters.

### Tester

```bash
python tester.py \
    --gt_input_path out/scene.json \
    --est_input_path out/estimation.json \
    [--debug]
```

Prints the per-shape error and the mean / median aggregate score. Example output:

```
  Line2D_0:   error = 0.0022
  Line2D_1:   error = 0.0048
  Circle2D_0: error = 0.0062

Mean error:   0.0044
Median error: 0.0048
```

## Config file

```json
{
  "shapes": {"Line2D": 2, "Circle2D": 1},
  "num_points": 200,
  "randomness": 0.5
}
```

- `shapes`: instances per shape type. All shapes in a single config must share an ambient dimension (don't mix 2D and 3D).
- `num_points`: total points generated per shape instance (split 80% inliers / 20% outliers).
- `randomness`: noise level for inliers, in `[0, 1]`.

## Project layout

```
shape-ransac/
├── generator.py            # CLI: generate a scene
├── estimator.py            # CLI: estimate shapes from a scene
├── tester.py               # CLI: compare estimates against ground truth
├── design.drawio           # UML class diagram (full system design)
├── design.png              # rendered design diagram
├── docs/                   # assignment PDF
├── configs/example.json    # sample config
├── requirements.txt
└── src/
    ├── shapes/             # Shape ABC + Line2D, Circle2D, Plane3D, ShapeType, registry
    ├── generator/          # Generator, Config, ConfigLoader, GenerationResult, Serializer
    ├── estimator/          # RansacEstimator, Estimator, EstimationResult, serializer
    ├── tester/             # Tester, MatchResult, TestResult
    ├── utils/              # random helpers, JSON io, exceptions, metrics
    └── visualization/      # Plotter — the only module that imports matplotlib
```

## Design notes

The whole system hangs off a single behavioral abstraction: every shape implements a tiny interface (`min_points_required`, `from_points`, `distance_to_point`, `sample_points`, `random`, `to_dict`, `from_dict`, `shape_type`). RANSAC, the generator, the tester, and the serializer all talk to this interface and only this interface — adding a new shape is a one-file change plus a single entry in the registry.

The Plotter is the only place in the codebase that imports matplotlib. The CLI scripts construct it lazily, inside the `if args.debug` branch, so non-debug runs don't touch matplotlib at all.

## Tests

```bash
python -m pytest
```

78 unit and integration tests covering the shapes, RANSAC, the generator pipeline, the estimator orchestration, the tester, all three CLIs, and the Plotter smoke paths.
