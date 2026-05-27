# shape-ransac

Generic RANSAC framework for detecting geometric shapes in noisy 2D point data.

The project is structured around a clean separation of three layers:

1. **Generator** — produces synthetic point clouds from a config: shape instances + inliers + outliers, plus serialization.
2. **Estimator** — RANSAC-based fitting that recovers shape parameters from the generated points. Shape-agnostic: new shapes are added by implementing a small `Shape` interface.
3. **Tester** — compares estimated shapes against ground truth, computes match metrics, and plots results.

## Layout

```
shape-ransac/
├── generator.py     # Generator, ConfigLoader, Serializer, GenerationResult, ShapeInstance, Config
├── estimator.py     # Estimator, RansacEstimator, EstimatedShape, RansacParams, RansacResult
├── tester.py        # Tester, Plotter, EstimationResult, MatchResult, TestResult
├── shapes/          # Shape abstraction + concrete shapes (line, circle, …)
├── configs/         # YAML/JSON configs describing scenes to generate
└── tests/
```

## Usage (planned)

```bash
# Generate a scene from a config
python -m shape_ransac.generator --config configs/example.yaml --out data/scene.npz

# Run RANSAC on a saved scene
python -m shape_ransac.estimator --in data/scene.npz --out data/estimates.json

# End-to-end test (generate → estimate → match → plot)
python -m shape_ransac.tester --config configs/example.yaml
```

## Design

See `../uml_design.drawio` for the full class diagram and `../shape_generator.drawio` for the generator design.

Core idea: the `Shape` abstraction owns its own sampling, distance, and minimal-sample fitting — so RANSAC stays generic and adding a new shape is a single-file change.
