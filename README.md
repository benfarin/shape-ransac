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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pytest`.

## Quick start

Run the whole pipeline in one block. The `out/` folder is created automatically.

```bash
python3 generator.py --config_path configs/example.json --output_path out/scene.json --debug
python3 estimator.py --input_path out/scene.json --output_path out/estimation.json --debug
python3 tester.py    --gt_input_path out/scene.json --est_input_path out/estimation.json --debug
```

Drop `--debug` from any command to skip its plot (the JSON outputs are written regardless).

If you'd rather keep each command on one line, you can also write them with backslash continuation, but make sure the backslash is the **last** character on the line (no trailing space):

```bash
python3 generator.py \
    --config_path configs/example.json \
    --output_path out/scene.json \
    --debug
```

## What each step shows

### Generator (`--debug`)

A 2D scatter plot:
- **Green dots** — inliers, ≈ 80% of `num_points` per shape, clustered tightly around the ground-truth shapes.
- **Red dots** — outliers, ≈ 20%, scattered uniformly across the scene.
- **Blue lines / circles** — the ground-truth shapes themselves.

For a 3D config (e.g. `Plane3D`), the plot is a rotatable 3D scatter with a translucent blue plane.

### Estimator (`--debug`)

The same points, this time all in **gray** (the estimator doesn't know which were inliers). **Orange** lines/curves are RANSAC's recovered shapes. They should sit on top of the green clusters from the previous plot.

### Tester (`--debug`)

The points in gray, **blue** ground-truth shapes, **orange** estimated shapes. On a good run they overlap so closely you can barely tell them apart.

The terminal prints:

```
  Line2D_0:   error = 0.0079
  Line2D_1:   error = 0.0121
  Circle2D_0: error = 0.0092

Mean error:   0.0098
Median error: 0.0092
```

Anything below `0.05` (the RANSAC inlier threshold) is good. Below `0.01` is essentially perfect.

## Config file

```json
{
  "shapes": {"Line2D": 2, "Circle2D": 1},
  "num_points": 200,
  "randomness": 0.5
}
```

- `shapes`: instances per shape type. All shapes in a single config must share an ambient dimension — don't mix 2D and 3D.
- `num_points`: total points generated per shape instance (split 80% inliers / 20% outliers).
- `randomness`: noise level for inliers, in `[0, 1]`.

A few configs to try:

```bash
# sparse — two shapes, low overlap, very consistent recovery
echo '{"shapes":{"Line2D":1,"Circle2D":1},"num_points":300,"randomness":0.3}' > configs/sparse.json

# 3D — one tilted plane
echo '{"shapes":{"Plane3D":1},"num_points":300,"randomness":0.4}' > configs/plane.json

# noisy — large randomness, recoveries still reasonable
echo '{"shapes":{"Line2D":2,"Circle2D":1},"num_points":300,"randomness":0.9}' > configs/noisy.json
```

## A note on overlapping scenes

The estimator uses simplest sequential RANSAC (per the spec) — fit one shape, remove its inliers, fit the next one. When several ground-truth shapes happen to overlap in the same region, the first RANSAC round can grab points from multiple shapes at once, leaving the next round with too few points to recover the second shape cleanly. You'll see one badly-fit shape in the tester plot and a higher mean error (e.g. ~0.1 instead of ~0.01).

This is a known limitation of simplest sequential RANSAC, not a bug. Two ways to avoid it:

- **Re-run.** Each run draws new random ground-truth shapes; most layouts have low overlap.
- **Use a sparser config** (fewer shapes or higher `num_points` so each shape is dense enough to dominate).

## Tests

```bash
python3 -m pytest
```

78 unit and integration tests covering the shapes, RANSAC, the generator pipeline, the estimator orchestration, the tester, all three CLIs, and the Plotter smoke paths.

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
