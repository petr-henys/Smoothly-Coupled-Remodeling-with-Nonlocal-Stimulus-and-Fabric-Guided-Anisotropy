#!/usr/bin/env python3
"""Visualize femoral coordinate system (CSS) for left proximal femur.

Generates VTK files for visualization in ParaView:
  - head_line.vtk / head_line_points.vtk  - line through femoral head
  - le_me_line.vtk / le_me_line_points.vtk - lateral-medial epicondyle line
  - head_sphere.vtk    - fitted sphere on femoral head
  - css_axes_arrows.vtm - CSS axes as arrows (MultiBlock)
  - femur_world.vtk    - femur in world coordinates
  - femur_in_css.vtk   - femur transformed to CSS coordinates
  - fhc_origin.vtk     - femoral head center point
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from femur.paths import FemurPaths, ANATOMY_PROCESSED_DIR
from femur.css import FemurCSS, load_json_points


def create_axes_arrows(
    origin: np.ndarray,
    axes: dict[str, np.ndarray],
    axis_labels: dict[str, str],
    length: float = 40.0,
) -> pv.MultiBlock:
    """Create CSS axes as arrows in a MultiBlock.
    
    Args:
        origin: Origin point (FHC).
        axes: Dict {"x": vector, "y": vector, "z": vector}.
        axis_labels: Dict {"x": "Anterior", "y": "Superior", "z": "Medial"}.
        length: Arrow length in mm.
        
    Returns:
        MultiBlock with named blocks (e.g., "Anterior", "Superior", "Medial").
    """
    blocks = pv.MultiBlock()
    for axis_name, direction in axes.items():
        arrow = pv.Arrow(start=origin, direction=direction, scale=length)
        blocks.append(arrow)
        blocks.set_block_name(len(blocks) - 1, axis_labels[axis_name])
    return blocks


def save_vtk(mesh: pv.DataSet, path: Path) -> None:
    """Save mesh and print confirmation."""
    mesh.save(str(path))
    print(f"  ✓ {path.name}")


def main() -> None:
    output_dir = ANATOMY_PROCESSED_DIR / "css_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load inputs
    print("Loading inputs...")
    femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
    head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
    le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
    
    # Axis labels (left femur convention)
    axis_labels = {"x": "Anterior", "y": "Superior", "z": "Medial"}
    
    # Build CSS
    head_sphere_path = output_dir / "head_sphere.vtk"
    css = FemurCSS(
        femur=femur_mesh,
        head_line=head_line,
        le_me=le_me_line,
        axis_labels=axis_labels,
        save_head_sphere=head_sphere_path,
    )
    
    # Print CSS info
    print(f"\nCSS parameters:")
    print(f"  FHC: {css.fhc}")
    print(f"  Head radius: {css.head_radius:.2f} mm")
    print(f"  Axes:")
    for name, vec in css.axes.items():
        print(f"    {name.upper()} ({css.axis_labels[name]}): {vec}")
    
    # Save outputs
    print(f"\nSaving to {output_dir}:")
    save_vtk(pv.Line(head_line[0], head_line[1]), output_dir / "head_line.vtk")
    save_vtk(pv.PolyData(head_line), output_dir / "head_line_points.vtk")
    save_vtk(pv.Line(le_me_line[0], le_me_line[1]), output_dir / "le_me_line.vtk")
    
    le_me_pts = pv.PolyData(le_me_line)
    le_me_pts["labels"] = np.array(["LE", "ME"])
    save_vtk(le_me_pts, output_dir / "le_me_line_points.vtk")
    
    print(f"  ✓ head_sphere.vtk")  # already saved by FemurCSS
    
    # Mechanical axis: line from LE-ME midpoint to FHC
    le_me_midpoint = le_me_line.mean(axis=0)
    save_vtk(pv.Line(le_me_midpoint, css.fhc), output_dir / "mechanical_axis.vtk")
    
    axes_arrows = create_axes_arrows(css.fhc, css.axes, css.axis_labels)
    axes_arrows.save(str(output_dir / "css_axes_arrows.vtm"))
    print(f"  ✓ css_axes_arrows.vtm")
    
    save_vtk(femur_mesh, output_dir / "femur_world.vtk")
    save_vtk(css.forward_transform(femur_mesh), output_dir / "femur_in_css.vtk")
    save_vtk(pv.PolyData(css.fhc.reshape(1, 3)), output_dir / "fhc_origin.vtk")
    
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
