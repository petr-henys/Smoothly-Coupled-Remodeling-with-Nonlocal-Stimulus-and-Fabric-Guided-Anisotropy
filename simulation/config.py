from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from dolfinx import mesh
import ufl

from simulation.params import (
    MaterialParams,
    DensityParams,
    StimulusParams,
    FabricParams,
    SolverParams,
    TimeParams,
    NumericsParams,
    OutputParams,
    GeometryParams,
    params_to_dict,
)


@dataclass
class Config:
    """Simulation parameters (mm, day, MPa, g/cm³).

    Parameters are organized into logical groups via nested dataclasses.
    Access parameters via their group: cfg.material.E0, cfg.density.rho_min, etc.
    """

    # Grouped parameter dataclasses
    material: MaterialParams = field(default_factory=MaterialParams)
    density: DensityParams = field(default_factory=DensityParams)
    stimulus: StimulusParams = field(default_factory=StimulusParams)
    fabric: FabricParams = field(default_factory=FabricParams)
    solver: SolverParams = field(default_factory=SolverParams)
    time: TimeParams = field(default_factory=TimeParams)
    numerics: NumericsParams = field(default_factory=NumericsParams)
    output: OutputParams = field(default_factory=OutputParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)

    # Runtime state (not serialized)
    domain: mesh.Mesh | None = field(default=None, repr=False)
    facet_tags: mesh.MeshTags | None = field(default=None, repr=False)
    dx: ufl.Measure | None = field(init=False, default=None, repr=False)
    ds: ufl.Measure | None = field(init=False, default=None, repr=False)
    dt: float = field(init=False, default=1.0)

    # -------------------------------------------------------------------------
    # Initialization and validation
    # -------------------------------------------------------------------------

    def __post_init__(self):
        if self.domain is None:
            raise ValueError("Config requires a valid 'domain' (dolfinx.mesh.Mesh).")

        # Resolve log_file relative to results_dir.
        self._log_file = str(Path(self.output.results_dir) / self.output.log_file)

        self.validate()
        self._build_measures()
        self._ensure_results_dir()
        self.update_config_json()

    def validate(self):
        """Validate all parameter groups."""
        self.material.validate()
        self.density.validate()
        self.stimulus.validate()
        self.fabric.validate()
        self.solver.validate()
        self.time.validate()
        self.numerics.validate()
        self.output.validate()
        self.geometry.validate()

    def _build_measures(self):
        """Create UFL measures with the configured quadrature degree."""
        metadata = {"quadrature_degree": int(self.numerics.quadrature_degree)}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=self.facet_tags,
            metadata=metadata,
        )

    def _ensure_results_dir(self) -> None:
        """Create results directory on rank 0 and barrier."""
        outdir = Path(self.output.results_dir)
        if self.domain.comm.rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
        self.domain.comm.Barrier()

    def set_dt(self, dt_days: float):
        """Set the timestep `dt` in days."""
        if dt_days <= 0:
            raise ValueError(f"Timestep dt_days={dt_days} must be positive.")
        self.dt = float(dt_days)

    def update_config_json(self):
        """Write `config.json` in `results_dir` (rank 0 only)."""
        if self.domain.comm.rank == 0:
            path = Path(self.output.results_dir) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_json_dict(), f, indent=2)
        self.domain.comm.Barrier()

    def rebuild(self, domain: mesh.Mesh, facet_tags: mesh.MeshTags | None = None):
        """Update mesh and rebuild integration measures."""
        self.domain = domain
        self.facet_tags = facet_tags
        self._build_measures()

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize all parameter groups to a JSON-compatible dict."""
        result = {}
        result["material"] = params_to_dict(self.material)
        result["density"] = params_to_dict(self.density)
        result["stimulus"] = params_to_dict(self.stimulus)
        result["fabric"] = params_to_dict(self.fabric)
        result["solver"] = params_to_dict(self.solver)
        result["time"] = params_to_dict(self.time)
        result["numerics"] = params_to_dict(self.numerics)
        result["output"] = params_to_dict(self.output)
        return result

    @property
    def log_file(self) -> str:
        """Absolute path to log file (computed from output.results_dir + output.log_file)."""
        return self._log_file

    @log_file.setter
    def log_file(self, value: str) -> None:
        self._log_file = value
