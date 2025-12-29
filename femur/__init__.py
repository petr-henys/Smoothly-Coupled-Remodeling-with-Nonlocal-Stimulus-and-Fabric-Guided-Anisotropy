"""Femur model package for bone remodeling simulations.

Provides mesh parsing, coordinate system, loading, and scenarios for femur specimens.
"""

from femur.febio_parser import FEBio2Dolfinx
from femur.css import FemurCSS, load_json_points, _fit_femoral_head, _unit
from femur.loads import (
    GaussianSurfaceLoad,
    HIPJointLoad,
    MuscleLoad,
    build_load,
    vector_from_angles,
    gait_interpolator,
    orthoload2ISB,
)
from femur.loader import (
    Loader,
    LoadingCase,
    HipLoadSpec,
    MuscleLoadSpec,
    CachedTraction,
    MUSCLE_PATHS,
)
from femur.scenarios import get_standard_gait_cases, load_scenarios_from_yaml
from femur.paths import (
    FemurPaths,
    GaitPaths,
    PROJECT_ROOT,
    ANATOMY_DIR,
    ANATOMY_RAW_DIR,
    ANATOMY_PROCESSED_DIR,
    FEMUR_ANATOMY_DIR,
    GAIT_DATA_DIR,
    PROXIMAL_FEMUR_DIR,
    RESULTS_DIR,
    ARCHIVE_DIR,
    ensure_directories,
    get_output_path,
    get_hip_traction_path,
)

__all__ = [
    # FEBio parser
    "FEBio2Dolfinx",
    # CSS
    "FemurCSS",
    "load_json_points",
    "_fit_femoral_head",
    "_unit",
    # Loads
    "GaussianSurfaceLoad",
    "HIPJointLoad",
    "MuscleLoad",
    "build_load",
    "vector_from_angles",
    "gait_interpolator",
    "orthoload2ISB",
    # Loader
    "Loader",
    "LoadingCase",
    "HipLoadSpec",
    "MuscleLoadSpec",
    "CachedTraction",
    "MUSCLE_PATHS",
    # Scenarios
    "get_standard_gait_cases",
    "load_scenarios_from_yaml",
    # Paths
    "FemurPaths",
    "GaitPaths",
    "PROJECT_ROOT",
    "ANATOMY_DIR",
    "ANATOMY_RAW_DIR",
    "ANATOMY_PROCESSED_DIR",
    "FEMUR_ANATOMY_DIR",
    "GAIT_DATA_DIR",
    "PROXIMAL_FEMUR_DIR",
    "RESULTS_DIR",
    "ARCHIVE_DIR",
    "ensure_directories",
    "get_output_path",
    "get_hip_traction_path",
]
