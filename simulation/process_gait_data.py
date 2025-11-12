from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

from .logging_config import get_logger
from .paths import GaitPaths

logger = get_logger(__name__)

def load_xy_datasets(xlsx_path: str | Path, sheet: str | int | None = 0, flip_y: bool = False) -> Dict[str, np.ndarray]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=[0, 1], engine="openpyxl")

    datasets = {}
    for dataset_name in df.columns.levels[0]:
        # Skip NaN and unnamed columns (pandas adds these for index column)
        if pd.isna(dataset_name) or str(dataset_name).startswith('Unnamed:'):
            continue

        try:
            x_full = df[(dataset_name, "X")].to_numpy(dtype=float)
            y_full = df[(dataset_name, "Y")].to_numpy(dtype=float)
        except KeyError as err:
            raise ValueError(f"Dataset '{dataset_name}' missing X/Y columns.") from err
        
        mask = ~np.isnan(x_full) & ~np.isnan(y_full)
        x, y = x_full[mask], y_full[mask]

        if flip_y:
            y = -y

        datasets[dataset_name] = np.column_stack((x, y))

    return datasets


def segment_curves_grid(points: np.ndarray, n_vertical: int, m_horizontal: int, *, jitter: float = 1e-9, eps_y = 0.2) -> List[np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N,2)")
    
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min() - eps_y, points[:, 1].max()

    x_edges = np.linspace(min_x - jitter, max_x + jitter, n_vertical)
    y_edges = np.linspace(min_y - jitter, max_y + jitter, m_horizontal)

    col_idx = np.digitize(points[:, 0], x_edges) - 1
    row_idx = np.digitize(points[:, 1], y_edges) - 1

    curves: List[np.ndarray] = []

    for j in range(m_horizontal - 1):
        for i in range(n_vertical - 1):
            mask = (row_idx == j) & (col_idx == i)
            if not np.any(mask):
                continue
            cell_pts = points[mask]
            order = np.argsort(cell_pts[:, 0])
            curves.append(cell_pts[order])
    return curves


def plot_curves(curves: List[np.ndarray], title: str = "Segmented Curves", figsize: tuple = (12, 8)) -> None:
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))
    
    for i, curve in enumerate(curves):
        if len(curve) == 0:
            continue
            
        label = f"Curve {i+1} ({len(curve)} pts)"
        plt.scatter(curve[:, 0], -curve[:, 1], c=[colors[i]], 
                   s=30, alpha=0.8, label=label)
    
    plt.xlabel('X (Gait Cycle %)')
    plt.ylabel('Y (Force/Moment)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def parse_hip_file(hip_path: str | Path, validate_physics: bool = True, fix_physics: bool = True) -> dict:
    hip_path = Path(hip_path)
    
    if not hip_path.exists():
        raise FileNotFoundError(f"HIP file not found: {hip_path}")
    
    with open(hip_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_start_idx = None
    metadata = {'file_name': hip_path.name}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Cycle [%]"):
            column_names = [col.strip() for col in line.split('\t')]
            data_start_idx = i + 1
            break
        elif "Peak Resultant Force" in line:
            force_match = re.search(r'F = ([\d.]+)N', line)
            if force_match:
                metadata['peak_force'] = float(force_match.group(1))
    
    if data_start_idx is None:
        raise ValueError("Could not find data section in HIP file")
    
    data_rows = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if line:
            try:
                values = [float(val.strip()) for val in line.split('\t')]
                if len(values) == len(column_names):
                    data_rows.append(values)
            except ValueError:
                continue
    
    if not data_rows:
        raise ValueError("No valid data rows found")
    
    data = np.array(data_rows)
    physics_warnings = []
    
    if len(column_names) >= 6:
        fx, fy, fz, f_given = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        f_calculated = np.sqrt(fx**2 + fy**2 + fz**2)
        
        if validate_physics:
            violations = f_given < (f_calculated - 1e-6)
            n_violations = np.sum(violations)
            
            if n_violations > 0:
                physics_warnings.append(f"Found {n_violations} physics violations where F < sqrt(Fx²+Fy²+Fz²)")
                max_error = np.max(f_calculated[violations] - f_given[violations])
                physics_warnings.append(f"Max error: {max_error:.2f} N")
        
        if fix_physics:
            data[:, 4] = f_calculated
            physics_warnings.append("Recalculated resultant force F from components")
    
    return {
        'data': data,
        'metadata': metadata,
        'column_names': column_names,
        'physics_warnings': physics_warnings
    }


def hip_to_xy_datasets(hip_path: str | Path, components: List[str] = None, vs_time: bool = False) -> Dict[str, np.ndarray]:
    result = parse_hip_file(hip_path)
    data = result['data']
    
    if components is None:
        components = ['Fx', 'Fy', 'Fz', 'F']
    
    x_data = data[:, 5] if vs_time else data[:, 0]
    x_label = "Time" if vs_time else "Cycle"
    
    component_map = {'Fx': 1, 'Fy': 2, 'Fz': 3, 'F': 4}
    
    datasets = {}
    for comp in components:
        if comp in component_map:
            y_data = data[:, component_map[comp]]
            datasets[f"{comp}_vs_{x_label}"] = np.column_stack((x_data, y_data))
        else:
            logger.warning(f"Unknown component: {comp}")
    
    return datasets


def plot_hip_data(hip_path: str | Path, components: List[str] = None, vs_time: bool = False, show_warnings: bool = True) -> None:
    if components is None:
        components = ['Fx', 'Fy', 'Fz', 'F']
    
    result = parse_hip_file(hip_path)
    datasets = hip_to_xy_datasets(hip_path, components, vs_time)
    
    if show_warnings and result['physics_warnings']:
        logger.warning("Physics warnings found:")
        for warning in result['physics_warnings']:
            logger.warning(f"  - {warning}")
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (name, data) in enumerate(datasets.items()):
        component = name.split('_vs_')[0]
        color = colors[i % len(colors)]
        plt.plot(data[:, 0], data[:, 1], color=color, linewidth=2, 
                label=f"{component} [N]", marker='o', markersize=2, alpha=0.8)
    
    x_label = "Time [s]" if vs_time else "Gait Cycle [%]"
    plt.xlabel(x_label)
    plt.ylabel('Force [N]')
    
    title = f"Hip Joint Forces - {result['metadata']['file_name']}"
    if 'peak_force' in result['metadata']:
        title += f" (Peak: {result['metadata']['peak_force']:.0f}N)"
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def rescale_curve(curve_poins, x_scale=(0., 1.), y_scale=(0., 1.)):
    """Rescale curve points to a given range. Handles degenerate cases (single point, uniform values)."""
    x_min, x_max = np.min(curve_poins[:, 0]), np.max(curve_poins[:, 0])
    y_min, y_max = np.min(curve_poins[:, 1]), np.max(curve_poins[:, 1])
    
    # Handle degenerate cases (x_max == x_min or y_max == y_min)
    # Map to center of target range
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range > 0:
        x_rescaled = (curve_poins[:, 0] - x_min) / x_range * (x_scale[1] - x_scale[0]) + x_scale[0]
    else:
        x_rescaled = np.full_like(curve_poins[:, 0], (x_scale[0] + x_scale[1]) / 2)
    
    if y_range > 0:
        y_rescaled = (curve_poins[:, 1] - y_min) / y_range * (y_scale[1] - y_scale[0]) + y_scale[0]
    else:
        y_rescaled = np.full_like(curve_poins[:, 1], (y_scale[0] + y_scale[1]) / 2)
    
    return np.column_stack((x_rescaled, y_rescaled))

def demo_hip_parser() -> None:
    hip_file = GaitPaths.HIP99_WALKING
    
    if not hip_file.exists():
        logger.error(f"HIP file not found: {hip_file}")
        return
    
    logger.info("=== HIP Parser Demo ===")
    
    result = parse_hip_file(hip_file)
    data = result['data']
    
    logger.info(f"File: {result['metadata']['file_name']}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {result['column_names']}")

    fx, fy, fz, f = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    logger.info("Force Statistics:")
    logger.info(f"  Max |Fx|: {np.max(np.abs(fx)):.1f} N")
    logger.info(f"  Max |Fy|: {np.max(np.abs(fy)):.1f} N")
    logger.info(f"  Max |Fz|: {np.max(np.abs(fz)):.1f} N")
    logger.info(f"  Max |F|:  {np.max(f):.1f} N")
    
    logger.info("--- Plotting All Force Components ---")
    plot_hip_data(hip_file, show_warnings=False)


def main() -> int:
    try:
        from .logging_config import setup_logging
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from logging_config import setup_logging
    
    setup_logging(level="INFO")
    
    logger.info("=== Excel Data Processing Demo ===")
    xlsx_file = GaitPaths.AMIRI_EXCEL
    n_vertical, m_horizontal = 4, 9

    points = load_xy_datasets(xlsx_file, flip_y=True)
    points = points["Dataset_WS"]

    curves = segment_curves_grid(points, n_vertical, m_horizontal)
    logger.info(f"Detected {len(curves)} curves on a {n_vertical}×{m_horizontal} grid")

    plot_curves(curves, title=f"Grid {n_vertical}×{m_horizontal}")
    
    logger.info("=" * 50)
    demo_hip_parser()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
