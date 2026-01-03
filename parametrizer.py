"""Parameter sweep framework for bone remodeling simulations.

Supports:
- Dot-notation parameter paths (e.g., "material.E0", "density.k_rho_form")
- Cartesian product sweeps with hash-based output directories
- Integration with run_box_model/run_model via Config patching
- MPI-aware execution with progress tracking

Example usage:
    from parametrizer import ParameterSweep, Parametrizer, run_box_simulation

    sweep = ParameterSweep(
        params={
            "material.E0": [5000.0, 7500.0, 10000.0],
            "density.k_rho_form": [0.01, 0.05, 0.1],
            "time.total_time": [100.0],
        },
        base_output_dir=Path("./sweep_results"),
    )
    
    parametrizer = Parametrizer(sweep, run_box_simulation, MPI.COMM_WORLD)
    parametrizer.run()
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace, fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Protocol

from mpi4py import MPI

from box import (
    BoxSolverFactory,
    BoxLoader,
    BoxGeometry,
    BoxMeshBuilder,
    get_parabolic_pressure_case,
)
from simulation.config import Config
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import (
    DensityParams,
    FabricParams,
    GeometryParams,
    MaterialParams,
    NumericsParams,
    OutputParams,
    SolverParams,
    StimulusParams,
    TimeParams,
)
from simulation.progress import ProgressReporter, SweepProgressReporter

ParamValue = int | float | str | bool | None
ParamDict = dict[str, list[ParamValue] | ParamValue]


class SimulationRunner(Protocol):
    """Protocol for simulation runner functions.
    
    Runners receive parameter point, output path, communicator, and an optional
    progress reporter for unified sweep progress display.
    """
    def __call__(
        self,
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None: ...


# Mapping from group name to dataclass type
PARAM_GROUPS: dict[str, type] = {
    "material": MaterialParams,
    "density": DensityParams,
    "stimulus": StimulusParams,
    "fabric": FabricParams,
    "solver": SolverParams,
    "time": TimeParams,
    "numerics": NumericsParams,
    "output": OutputParams,
    "geometry": GeometryParams,
}


def is_config_param_path(path: str) -> bool:
    """Check if path is a Config parameter path (group.field format)."""
    return "." in path


def parse_param_path(path: str) -> tuple[str, str]:
    """Parse 'group.field' into (group, field). E.g., 'material.E0' → ('material', 'E0')."""
    parts = path.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid parameter path '{path}'. Use 'group.field' format (e.g., 'material.E0').")
    group, field_name = parts
    if group not in PARAM_GROUPS:
        raise ValueError(f"Unknown parameter group '{group}'. Valid: {list(PARAM_GROUPS.keys())}")
    return group, field_name


def validate_param_path(path: str, strict: bool = True) -> bool:
    """Validate that a parameter path refers to an existing field.
    
    Args:
        path: Parameter path (e.g., 'material.E0' or 'N').
        strict: If True, raise for invalid paths. If False, return False.
    
    Returns:
        True if valid Config parameter, False if custom parameter.
    
    Raises:
        ValueError: If strict=True and path is invalid Config parameter.
    """
    if not is_config_param_path(path):
        # Custom parameter (e.g., "N", "dt_days") - not validated
        return False
    
    try:
        group, field_name = parse_param_path(path)
        param_cls = PARAM_GROUPS[group]
        valid_fields = {f.name for f in dataclass_fields(param_cls)}
        if field_name not in valid_fields:
            if strict:
                raise ValueError(f"Unknown field '{field_name}' in {group}. Valid: {sorted(valid_fields)}")
            return False
        return True
    except ValueError:
        if strict:
            raise
        return False


def patch_param_group(param_obj: Any, field_name: str, value: ParamValue) -> Any:
    """Return a new dataclass instance with one field replaced."""
    return replace(param_obj, **{field_name: value})


@dataclass
class ParameterSweep:
    """Parameter sweep specification with Cartesian product and hash-based paths.
    
    Attributes:
        params: Dict mapping parameter paths to values. Supports two formats:
            - Config paths (group.field): "material.E0", "density.k_rho_form"
            - Custom parameters: "N", "dt_days" (not validated against Config)
            Values can be single items or lists for sweeping.
        base_output_dir: Root directory for all sweep outputs.
        metadata: Optional metadata stored in sweep summary.
        validate_config_params: If True, validate Config parameter paths.
    """
    params: ParamDict
    base_output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    validate_config_params: bool = True
    
    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError("params dictionary cannot be empty")
        
        # Normalize single values to lists and optionally validate paths
        normalized: dict[str, list[ParamValue]] = {}
        for key, values in self.params.items():
            if self.validate_config_params and is_config_param_path(key):
                validate_param_path(key, strict=True)
            normalized[key] = values if isinstance(values, list) else [values]
        self.params = normalized
        self.base_output_dir = Path(self.base_output_dir)
    
    def generate_points(self) -> list[dict[str, ParamValue]]:
        """Generate Cartesian product of all parameter values."""
        param_names = list(self.params.keys())
        param_values = [self.params[k] for k in param_names]
        return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]
    
    def format_output_path(self, param_point: dict[str, ParamValue]) -> Path:
        """Generate output path via 8-char hash of parameter point."""
        param_str = json.dumps(param_point, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.sha1(param_str.encode("utf-8")).hexdigest()[:8]
        return self.base_output_dir / params_hash
    
    def total_runs(self) -> int:
        """Calculate total number of runs in sweep."""
        total = 1
        for values in self.params.values():
            total *= len(values)
        return total


# Type alias for simulation runner functions
SimulationRunner = Callable[[dict[str, ParamValue], Path, MPI.Comm], None]


class Parametrizer:
    """Execute parameter sweeps with configurable simulation runners.
    
    Args:
        sweep: Parameter sweep specification.
        runner: Callable that runs a single simulation. Signature:
            runner(param_point: dict[str, ParamValue], output_path: Path, comm: MPI.Comm) -> None
        comm: MPI communicator.
    """
    
    def __init__(
        self,
        sweep: ParameterSweep,
        runner: SimulationRunner,
        comm: MPI.Comm,
    ) -> None:
        self.sweep = sweep
        self.runner = runner
        self.comm = comm
        self.logger = get_logger(comm, name="Parametrizer")
    
    def run(self) -> list[dict[str, Any]]:
        """Execute full parameter sweep and save summary.
        
        Returns:
            List of run metadata dicts (output_dir + param values).
        """
        param_points = self.sweep.generate_points()
        total_runs = len(param_points)
        
        if self.comm.rank == 0:
            self.logger.info(f"Starting parameter sweep: {total_runs} runs")
            self.logger.info(f"Parameters: {list(self.sweep.params.keys())}")
        
        runs_data: list[dict[str, Any]] = []
        
        # Create unified sweep progress reporter (handles all 3 levels)
        reporter = SweepProgressReporter(self.comm, total_runs)
        reporter.start()
        
        try:
            for idx, param_point in enumerate(param_points):
                output_path = self.sweep.format_output_path(param_point)
                
                # Format params for display and signal start of run
                params_info = " ".join(f"{k}={v}" for k, v in param_point.items())
                
                # Start run updates sweep bar and resets inner bars
                # Note: total_time is set by runner via start_run() call
                reporter.start_run(run_idx=idx, total_time=100.0, params_info=params_info)
                
                # Run simulation with unified reporter
                self.runner(param_point, output_path, self.comm, reporter)
                
                runs_data.append({
                    "run_id": idx,
                    # Store directory name (hash) so analysis can locate runs via base_output_dir/output_dir.
                    # Keep output_path for convenience/debugging.
                    "output_dir": output_path.name,
                    "output_path": str(output_path),
                    **param_point
                })
                
                # Mark run as complete in sweep progress
                reporter.finish_run()
        finally:
            reporter.stop()
        
        self._save_summary(runs_data)
        
        if self.comm.rank == 0:
            self.logger.info(f"Sweep complete. Summary: {self.sweep.base_output_dir / 'sweep_summary.csv'}")
        
        return runs_data
    
    def run_single(self, param_point: dict[str, ParamValue], output_path: Path | None = None) -> None:
        """Execute a single run with given parameters."""
        if output_path is None:
            output_path = self.sweep.format_output_path(param_point)
        self.runner(param_point, output_path, self.comm)
    
    def _save_summary(self, runs_data: list[dict[str, Any]]) -> None:
        """Save parameter sweep summary (rank 0 only)."""
        if self.comm.rank != 0 or not runs_data:
            return
        
        self.sweep.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = self.sweep.base_output_dir / "sweep_summary.csv"
        json_file = self.sweep.base_output_dir / "sweep_summary.json"
        
        # Write CSV summary
        header = list(runs_data[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(runs_data)
        
        # Write JSON summary
        summary = {
            "metadata": self.sweep.metadata,
            "total_runs": len(runs_data),
            "parameters_swept": list(self.sweep.params.keys()),
            "runs": runs_data
        }
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Box Model Runner
# =============================================================================

@dataclass
class BoxModelConfig:
    """Configuration for box model parameter sweeps.
    
    Non-swept parameters are set here. Swept parameters override these via param_point.
    """
    # Geometry [mm]
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 30.0
    nx: int = 10
    ny: int = 10
    nz: int = 30
    
    # Loading
    pressure: float = 1.0
    gradient_axis: int = 0
    center_factor: float = 2.0
    edge_factor: float = 0.3


def apply_param_point(cfg: Config, param_point: dict[str, ParamValue]) -> None:
    """Apply parameter overrides to Config in-place.
    
    Only applies Config parameters (group.field format). Custom parameters
    (e.g., "N", "dt_days") are skipped - the runner handles them directly.
    
    Args:
        cfg: Config instance to modify.
        param_point: Dict of 'group.field' → value overrides.
    """
    # Group overrides by param group for efficient patching
    grouped: dict[str, dict[str, ParamValue]] = {}
    for path, value in param_point.items():
        if not is_config_param_path(path):
            # Skip custom parameters (not part of Config)
            continue
        group, field_name = parse_param_path(path)
        grouped.setdefault(group, {})[field_name] = value
    
    # Apply patches to each group
    for group, overrides in grouped.items():
        current = getattr(cfg, group)
        patched = replace(current, **overrides)
        setattr(cfg, group, patched)


def create_box_runner(box_config: BoxModelConfig | None = None) -> SimulationRunner:
    """Create a simulation runner for box model sweeps.
    
    Args:
        box_config: Base configuration for non-swept parameters.
                   If None, uses default BoxModelConfig.
    
    Returns:
        Runner function compatible with Parametrizer.
    """
    if box_config is None:
        box_config = BoxModelConfig()
    
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        """Run single box model simulation with parameter overrides."""
        # Create mesh
        geometry = BoxGeometry(
            Lx=box_config.Lx, Ly=box_config.Ly, Lz=box_config.Lz,
            nx=box_config.nx, ny=box_config.ny, nz=box_config.nz,
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create base config with output path
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            output=OutputParams(results_dir=str(output_path)),
            geometry=GeometryParams(
                fix_tag=BoxMeshBuilder.TAG_BOTTOM,
                load_tag=BoxMeshBuilder.TAG_TOP,
            ),
        )
        
        # Apply parameter overrides
        apply_param_point(cfg, param_point)
        
        # Re-validate after patching
        cfg.validate()
        
        # Update reporter with correct total_time for this run
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=cfg.time.total_time)
        
        # Create loader and loading cases
        loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)
        loading_cases = [get_parabolic_pressure_case(
            pressure=box_config.pressure,
            gradient_axis=box_config.gradient_axis,
            center_factor=box_config.center_factor,
            edge_factor=box_config.edge_factor,
            box_extent=(0.0, box_config.Lx),
            name="parabolic_compression",
        )]
        
        # Create factory and run with sweep reporter (or create standalone)
        factory = BoxSolverFactory(cfg)
        
        with Remodeller(cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            if reporter is not None:
                # Use sweep reporter (unified 3-level progress)
                remodeller.simulate(reporter=reporter)
            else:
                # Standalone run: create own progress reporter
                with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as standalone_reporter:
                    remodeller.simulate(reporter=standalone_reporter)
    
    return runner


# Convenience function for direct use
def run_box_simulation(
    param_point: dict[str, ParamValue],
    output_path: Path,
    comm: MPI.Comm,
    reporter: SweepProgressReporter | None = None,
) -> None:
    """Run a box model simulation with default configuration.
    
    This is a convenience wrapper using default BoxModelConfig.
    For custom base parameters, use create_box_runner(BoxModelConfig(...)).
    """
    runner = create_box_runner()
    runner(param_point, output_path, comm, reporter)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: sweep over formation rate and reference SED
    comm = MPI.COMM_WORLD
    
    sweep = ParameterSweep(
        params={
            "density.k_rho_form": [0.01, 0.05],
            "stimulus.psi_ref_trab": [1e-5, 5e-5],
            "time.total_time": [50.0],  # Short runs for testing
        },
        base_output_dir=Path("./sweep_test"),
        metadata={"description": "Formation rate vs psi_ref_trab sweep"},
    )
    
    # Create runner with custom box config
    box_cfg = BoxModelConfig(
        Lx=10.0, Ly=10.0, Lz=20.0,
        nx=5, ny=5, nz=10,  # Coarse mesh for fast testing
    )
    runner = create_box_runner(box_cfg)
    
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
