"""Filesystem paths for anatomy inputs and results."""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent

# Anatomy input directories.
ANATOMY_DIR = PROJECT_ROOT / "anatomy"
ANATOMY_RAW_DIR = ANATOMY_DIR / "raw"
ANATOMY_PROCESSED_DIR = ANATOMY_DIR / "processed"

FEMUR_ANATOMY_DIR = ANATOMY_RAW_DIR / "femur_anatomy"
GAIT_DATA_DIR = ANATOMY_RAW_DIR / "gait_data"
PROXIMAL_FEMUR_DIR = ANATOMY_RAW_DIR / "proximal_femur"

RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

class FemurPaths:
    FEMUR_MESH_VTK = PROXIMAL_FEMUR_DIR / "model.vtk"
    FEMUR_MESH_FEB = PROXIMAL_FEMUR_DIR / "model.feb"
    
    HEAD_LINE_JSON = FEMUR_ANATOMY_DIR / "infsup_head_line.mrk.json"
    LE_ME_LINE_JSON = FEMUR_ANATOMY_DIR / "LE-ME_line.mrk.json"
    VASTUS_LATERALIS_JSON = FEMUR_ANATOMY_DIR / "VASTUS_lateralis_short.mrk.json"
    VASTUS_MEDIALIS_JSON = FEMUR_ANATOMY_DIR / "VASTUS_medialis_short.mrk.json"
    VASTUS_INTERMEDIUS_JSON = FEMUR_ANATOMY_DIR / "VASTUS_intermedius_short.mrk.json"
    PSOAS_JSON = FEMUR_ANATOMY_DIR / "PSOAS.mrk.json"
    GL_MED_JSON = FEMUR_ANATOMY_DIR / "GL_med.mrk.json"
    GL_MAX_JSON = FEMUR_ANATOMY_DIR / "GL_max.mrk.json"
    GL_MIN_JSON = FEMUR_ANATOMY_DIR / "GL_min.mrk.json"
    
    CSS_AXES_VTK = RESULTS_DIR / "css.vtk"
    FACET_MARKERS_VTK = RESULTS_DIR / "facet_markers.vtk"


class GaitPaths:
    AMIRI_EXCEL = GAIT_DATA_DIR / "Amiri" / "gait_data_amiri.xlsx"
    HIP99_WALKING = GAIT_DATA_DIR / "HIP99" / "Walking_Average.HIP"


def ensure_directories():
    directories = [
        ANATOMY_DIR,
        ANATOMY_RAW_DIR,
        ANATOMY_PROCESSED_DIR,
        FEMUR_ANATOMY_DIR,
        GAIT_DATA_DIR,
        PROXIMAL_FEMUR_DIR,
        RESULTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_output_path(filename: str, subdir: str = "", clean: bool = False) -> Path:
    if subdir:
        output_dir = RESULTS_DIR / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for item in output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        return output_dir / filename
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if clean:
            for item in RESULTS_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        return RESULTS_DIR / filename


def get_hip_traction_path(index: int) -> Path:
    """Get path for hip traction output file."""
    return get_output_path(f"hip_traction_{index:03d}.vtk")


ensure_directories()
