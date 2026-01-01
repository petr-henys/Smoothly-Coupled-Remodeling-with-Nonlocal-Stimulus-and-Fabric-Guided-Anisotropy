from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple
import json
import logging

import numpy as np
import pyvista as pv
from scipy.optimize import least_squares
from scipy.spatial import KDTree

from femur.paths import FemurPaths

logger = logging.getLogger(__name__)

NDArrayF = np.ndarray


def load_json_points(json_file: str | Path) -> NDArrayF:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = [p["position"] for p in data["markups"][0]["controlPoints"]]
    return np.asarray(pts, dtype=float)


def _fit_femoral_head(femur: pv.PolyData, head_line: NDArrayF, file: str | Path | None = None) -> Tuple[NDArrayF, float]:
    c0 = head_line.mean(axis=0)
    r0 = np.linalg.norm(head_line[0] - head_line[1]) / 2.0

    surf_pts = femur.extract_surface().points
    idx = KDTree(surf_pts).query_ball_point(c0, r0 * 1.1, workers=-1)
    roi = surf_pts[idx]

    def resid(params: NDArrayF) -> NDArrayF:
        c, r = params[:3], params[3]
        return np.linalg.norm(roi - c, axis=1) - r

    sol = least_squares(resid, np.concatenate([c0, [r0]]), method="lm")
    center, radius = sol.x[:3], float(sol.x[3])

    if file is not None:
        pv.Sphere(center=center, radius=radius, theta_resolution=30, phi_resolution=30).save(str(file))
    return center, radius


def _unit(v: NDArrayF) -> NDArrayF:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Vector magnitude is zero – cannot normalise.")
    return v / n


class FemurCSS:
    """Femur coordinate system: Y along neck, Z mediolateral, origin at head center."""

    def __init__(
        self,
        femur: pv.PolyData,
        head_line: NDArrayF,
        le_me: NDArrayF,
        *,
        side: Literal["left", "right"] = "left",
        axis_labels: dict[str, str] | None = None,
        save_head_sphere: str | Path | None = None,
    ) -> None:
        self.femur = femur
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side: Literal["left", "right"] = side

        # Used for visualization / reporting (not required for computation).
        self.axis_labels = axis_labels or {"x": "Anterior", "y": "Superior", "z": "Medial"}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.fhc, self.head_radius = _fit_femoral_head(femur, head_line, save_head_sphere)
        self._build_axes(le_me, side=self.side)

    def forward_transform(self, mesh: pv.PolyData) -> pv.PolyData:
        M = self._transformation_matrix(world_to_css=True)
        cp = mesh.copy()
        cp.transform(M, inplace=True)
        return cp

    def inverse_transform(self, mesh: pv.PolyData) -> pv.PolyData:
        M = self._transformation_matrix(world_to_css=False)
        cp = mesh.copy()
        cp.transform(M, inplace=True)
        return cp

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_axes(self, le_me: NDArrayF, *, side: Literal["left", "right"]) -> None:
        """Compute Y, Z and X = Y × Z."""
        y = _unit(self.fhc - le_me.mean(axis=0))

        # vector along epicondylar line, projected to ⟂ Y
        le, me = le_me  # expected order [LE, ME]
        line_vec = me - le
        line_vec -= np.dot(line_vec, y) * y
        z = _unit(line_vec)

        # Enforce consistent anatomical "medial" axis direction across sides.
        if side == "right":
            z = -z

        x = _unit(np.cross(y, z))  # Y × Z ensures right‑handed system

        # sanity check determinant
        R = np.vstack([x, y, z])
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(f"Coordinate system not right‑handed (det={det:.3f})")

        self.axes: dict[str, NDArrayF] = {"x": x, "y": y, "z": z}
        self.logger.info("CSS built – axes orthonormal, det=+1")

    # ------------------------------------------------------------------

    def _transformation_matrix(self, *, world_to_css: bool) -> NDArrayF:
        """Return a 4×4 homogeneous transform matrix.

        Args:
            world_to_css: If True, return world→CSS; otherwise CSS→world.
        """
        R = np.vstack([self.axes[a] for a in ("x", "y", "z")])
        T = np.eye(4)
        if world_to_css:
            T[:3, :3] = R
            T[:3, 3] = -R @ self.fhc
        else:
            T[:3, :3] = R.T
            T[:3, 3] = self.fhc
        return T

    # ------------------------------------------------------------------
    # Convenience methods ------------------------------------------------
    # ------------------------------------------------------------------

    def css_to_world_vector(self, v_css: NDArrayF) -> NDArrayF:
        """Rotate a direction vector from CSS → world coords (no translation)."""
        R = np.vstack([self.axes[a] for a in ("x", "y", "z")])
        return R.T @ v_css

    def world_to_css_vector(self, v_w: NDArrayF) -> NDArrayF:
        """Rotate a direction vector from world → CSS (no translation)."""
        R = np.vstack([self.axes[a] for a in ("x", "y", "z")])
        return R @ v_w

    def css_to_world_point(self, p_css: NDArrayF) -> NDArrayF:
        """Transform a point from CSS → world coords (rotation + translation)."""
        R = np.vstack([self.axes[a] for a in ("x", "y", "z")])
        return R.T @ p_css + self.fhc

    def world_to_css_point(self, p_w: NDArrayF) -> NDArrayF:
        """Transform a point from world → CSS (rotation + translation)."""
        R = np.vstack([self.axes[a] for a in ("x", "y", "z")])
        return R @ (p_w - self.fhc)

    def save_axes_vtk(self, filename: str | Path) -> None:
        """Save a VTK PolyData with axes vectors as point data arrays."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        pd = pv.PolyData(self.fhc.reshape(1, 3))
        pd["x"] = self.axes["x"].reshape(1, 3)
        pd["y"] = self.axes["y"].reshape(1, 3)
        pd["z"] = self.axes["z"].reshape(1, 3)
        pd.save(str(path))

###############################################################################
# CLI demo (for manual testing) – run `python femur_css.py <femur.vtk> <line1.json> <line2.json>`
###############################################################################

if __name__ == "__main__":
    femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
    head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
    le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)

    css = FemurCSS(femur_mesh, head_line, le_me_line, side="left")
    css.save_axes_vtk(FemurPaths.CSS_AXES_VTK)
