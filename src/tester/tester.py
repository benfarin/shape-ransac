import numpy as np

from src.tester.match_result import MatchResult
from src.tester.test_result import TestResult
from src.utils.metrics import model_error


class Tester:

    def evaluate(self, gt_result, est_result):
        matches = self._match_by_type(gt_result.shapes, est_result.estimated_shapes)
        errors = [m.error for m in matches]

        return TestResult(
            matches=matches,
            mean_error=float(np.mean(errors)) if errors else 0.0,
            median_error=float(np.median(errors)) if errors else 0.0,
        )

    def _match_by_type(self, gt_instances, est_shapes):
        matches = []
        used = set()

        for gt_inst in gt_instances:
            candidates = [
                (i, est) for i, est in enumerate(est_shapes)
                if i not in used
                and est.shape.shape_type() == gt_inst.shape.shape_type()
            ]
            if not candidates:
                continue

            best_i, best_est, best_error = min(
                (
                    (i, est, model_error(gt_inst.shape, est.shape))
                    for i, est in candidates
                ),
                key=lambda t: t[2],
            )
            matches.append(MatchResult(
                gt_shape=gt_inst,
                est_shape=best_est,
                error=best_error,
            ))
            used.add(best_i)

        return matches
