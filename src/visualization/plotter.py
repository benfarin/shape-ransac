import matplotlib.pyplot as plt
import numpy as np

from src.shapes import ShapeType


class Plotter:

    GT_COLOR = "blue"
    EST_COLOR = "orange"

    def plot_generator_debug(self, result):
        shapes = [inst.shape for inst in result.shapes]
        self._plot(
            points=result.points,
            inlier_mask=result.inlier_mask,
            overlays=[(shapes, self.GT_COLOR)],
            title="Generated scene (green=inlier, red=outlier)",
        )

    def plot_estimator_debug(self, points, estimated_shapes):
        self._plot(
            points=points,
            inlier_mask=None,
            overlays=[(estimated_shapes, self.EST_COLOR)],
            title="Estimated shapes",
        )

    def plot_tester_debug(self, gt_result, estimated_shapes):
        gt_shapes = [inst.shape for inst in gt_result.shapes]
        self._plot(
            points=gt_result.points,
            inlier_mask=None,
            overlays=[
                (gt_shapes, self.GT_COLOR),
                (estimated_shapes, self.EST_COLOR),
            ],
            title="Ground truth (blue) vs estimated (orange)",
        )

    def _plot(self, points, inlier_mask, overlays, title):
        is_3d = points.shape[1] == 3
        fig = plt.figure(figsize=(7, 7))

        if is_3d:
            ax = fig.add_subplot(projection="3d")
            self._scatter_3d(ax, points, inlier_mask)
            for shapes, color in overlays:
                for shape in shapes:
                    self._overlay_3d(ax, shape, color)
        else:
            ax = fig.add_subplot()
            ax.set_aspect("equal", adjustable="box")
            self._scatter_2d(ax, points, inlier_mask)
            for shapes, color in overlays:
                for shape in shapes:
                    self._overlay_2d(ax, shape, color)

        ax.set_title(title)
        plt.show()
        plt.close(fig)

    def _scatter_2d(self, ax, points, mask):
        if mask is None:
            ax.scatter(points[:, 0], points[:, 1], s=6, color="gray")
            return
        ax.scatter(points[mask, 0], points[mask, 1], s=6, color="green", label="inliers")
        ax.scatter(points[~mask, 0], points[~mask, 1], s=6, color="red", label="outliers")
        ax.legend(loc="upper right")

    def _scatter_3d(self, ax, points, mask):
        if mask is None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, color="gray")
            return
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], s=6, color="green")
        ax.scatter(points[~mask, 0], points[~mask, 1], points[~mask, 2], s=6, color="red")

    def _overlay_2d(self, ax, shape, color):
        drawer = _OVERLAY_2D.get(shape.shape_type())
        if drawer is not None:
            drawer(ax, shape, color)

    def _overlay_3d(self, ax, shape, color):
        drawer = _OVERLAY_3D.get(shape.shape_type())
        if drawer is not None:
            drawer(ax, shape, color)


def _draw_line2d(ax, line, color):
    t = np.linspace(-1.0, 1.0, 50)
    pts = line.point + t[:, None] * line.direction
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)


def _draw_circle2d(ax, circle, color):
    theta = np.linspace(0, 2 * np.pi, 80)
    pts = circle.center + circle.radius * np.column_stack([np.cos(theta), np.sin(theta)])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)


def _draw_plane3d(ax, plane, color):
    u_axis, v_axis = plane.plane_basis()
    grid = np.linspace(-0.5, 0.5, 8)
    u, v = np.meshgrid(grid, grid)
    x = plane.point[0] + u * u_axis[0] + v * v_axis[0]
    y = plane.point[1] + u * u_axis[1] + v * v_axis[1]
    z = plane.point[2] + u * u_axis[2] + v * v_axis[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.25)


def _draw_line3d(ax, line, color):
    t = np.linspace(-1.0, 1.0, 50)
    pts = line.point + t[:, None] * line.direction
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.5)


_OVERLAY_2D = {
    ShapeType.LINE2D: _draw_line2d,
    ShapeType.CIRCLE2D: _draw_circle2d,
}

_OVERLAY_3D = {
    ShapeType.LINE3D: _draw_line3d,
    ShapeType.PLANE3D: _draw_plane3d,
}
