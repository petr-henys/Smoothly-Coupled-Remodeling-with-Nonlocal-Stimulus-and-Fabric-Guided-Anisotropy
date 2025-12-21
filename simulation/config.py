from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any

import numpy as np

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
    params_to_dict,
)


@dataclass
class Config:
    """Simulation parameters (mm, day, MPa, g/cm³).

    Parameters are organized into logical groups via nested dataclasses.
    For backward compatibility, individual parameters are also accessible
    as properties on the Config object itself.
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

    # Runtime state (not serialized)
    domain: mesh.Mesh | None = field(default=None, repr=False)
    facet_tags: mesh.MeshTags | None = field(default=None, repr=False)
    dx: ufl.Measure | None = field(init=False, default=None, repr=False)
    ds: ufl.Measure | None = field(init=False, default=None, repr=False)
    dt: float = field(init=False, default=1.0)

    # -------------------------------------------------------------------------
    # Backward-compatible property accessors
    # These allow existing code to access cfg.E0 instead of cfg.material.E0
    # -------------------------------------------------------------------------

    # Material properties
    @property
    def E0(self) -> float:
        return self.material.E0

    @property
    def nu0(self) -> float:
        return self.material.nu0

    @property
    def n_trab(self) -> float:
        return self.material.n_trab

    @property
    def n_cort(self) -> float:
        return self.material.n_cort

    @property
    def rho_trab_max(self) -> float:
        return self.material.rho_trab_max

    @property
    def rho_cort_min(self) -> float:
        return self.material.rho_cort_min

    @property
    def stiff_pE(self) -> float:
        return self.material.stiff_pE

    @property
    def stiff_pG(self) -> float:
        return self.material.stiff_pG

    # Density properties
    @property
    def rho_min(self) -> float:
        return self.density.rho_min

    @property
    def rho_max(self) -> float:
        return self.density.rho_max

    @property
    def rho0(self) -> float:
        return self.density.rho0

    @property
    def rho_ref(self) -> float:
        return self.density.rho_ref

    @property
    def k_rho_form(self) -> float:
        return self.density.k_rho_form

    @property
    def k_rho_resorb(self) -> float:
        return self.density.k_rho_resorb

    @property
    def D_rho(self) -> float:
        return self.density.D_rho

    @property
    def rho_tissue(self) -> float:
        return self.density.rho_tissue

    @property
    def surface_use(self) -> bool:
        return self.density.surface_use

    @property
    def surface_A_min(self) -> float:
        return self.density.surface_A_min

    @property
    def surface_S0(self) -> float:
        return self.density.surface_S0

    # Stimulus properties
    @property
    def stimulus_power_p(self) -> float:
        return self.stimulus.stimulus_power_p

    @property
    def psi_ref(self) -> float:
        return self.stimulus.psi_ref

    @property
    def stimulus_tau(self) -> float:
        return self.stimulus.stimulus_tau

    @property
    def stimulus_D(self) -> float:
        return self.stimulus.stimulus_D

    @property
    def stimulus_S_max(self) -> float:
        return self.stimulus.stimulus_S_max

    @property
    def stimulus_kappa(self) -> float:
        return self.stimulus.stimulus_kappa

    @property
    def stimulus_delta0(self) -> float:
        return self.stimulus.stimulus_delta0

    # Fabric properties
    @property
    def fabric_tau(self) -> float:
        return self.fabric.fabric_tau

    @property
    def fabric_D(self) -> float:
        return self.fabric.fabric_D

    @property
    def fabric_cA(self) -> float:
        return self.fabric.fabric_cA

    @property
    def fabric_gammaF(self) -> float:
        return self.fabric.fabric_gammaF

    @property
    def fabric_epsQ(self) -> float:
        return self.fabric.fabric_epsQ

    @property
    def fabric_m_min(self) -> float:
        return self.fabric.fabric_m_min

    @property
    def fabric_m_max(self) -> float:
        return self.fabric.fabric_m_max

    # Solver properties
    @property
    def ksp_type(self) -> str:
        return self.solver.ksp_type

    @property
    def pc_type(self) -> str:
        return self.solver.pc_type

    @property
    def ksp_rtol(self) -> float:
        return self.solver.ksp_rtol

    @property
    def ksp_atol(self) -> float:
        return self.solver.ksp_atol

    @property
    def ksp_max_it(self) -> int:
        return self.solver.ksp_max_it

    @property
    def ksp_reuse_pc(self) -> bool:
        return self.solver.ksp_reuse_pc

    @property
    def accel_type(self) -> str:
        return self.solver.accel_type

    @property
    def m(self) -> int:
        return self.solver.m

    @property
    def beta(self) -> float:
        return self.solver.beta

    @property
    def lam(self) -> float:
        return self.solver.lam

    @property
    def gamma(self) -> float:
        return self.solver.gamma

    @property
    def safeguard(self) -> bool:
        return self.solver.safeguard

    @property
    def backtrack_max(self) -> int:
        return self.solver.backtrack_max

    @property
    def coupling_tol(self) -> float:
        return self.solver.coupling_tol

    @property
    def restart_on_reject_k(self) -> int:
        return self.solver.restart_on_reject_k

    @property
    def restart_on_stall(self) -> float:
        return self.solver.restart_on_stall

    @property
    def restart_on_cond(self) -> float:
        return self.solver.restart_on_cond

    @property
    def step_limit_factor(self) -> float:
        return self.solver.step_limit_factor

    @property
    def max_subiters(self) -> int:
        return self.solver.max_subiters

    @property
    def min_subiters(self) -> int:
        return self.solver.min_subiters

    # Time properties
    @property
    def total_time(self) -> float:
        return self.time.total_time

    @property
    def dt_initial(self) -> float:
        return self.time.dt_initial

    @property
    def adaptive_dt(self) -> bool:
        return self.time.adaptive_dt

    @property
    def adaptive_rtol(self) -> float:
        return self.time.adaptive_rtol

    @property
    def adaptive_atol(self) -> float:
        return self.time.adaptive_atol

    @property
    def dt_min(self) -> float:
        return self.time.dt_min

    @property
    def dt_max(self) -> float:
        return self.time.dt_max

    # Numerics properties
    @property
    def quadrature_degree(self) -> int:
        return self.numerics.quadrature_degree

    @property
    def smooth_eps(self) -> float:
        return self.numerics.smooth_eps

    # Output properties
    @property
    def saving_interval(self) -> int:
        return self.output.saving_interval

    @property
    def results_dir(self) -> str:
        return self.output.results_dir

    @property
    def log_file(self) -> str:
        return self._log_file

    @log_file.setter
    def log_file(self, value: str) -> None:
        self._log_file = value

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

    # -------------------------------------------------------------------------
    # Backward-compatible factory method
    # -------------------------------------------------------------------------

    @classmethod
    def from_flat_kwargs(
        cls,
        domain: mesh.Mesh,
        facet_tags: mesh.MeshTags | None = None,
        **kwargs,
    ) -> "Config":
        """Create Config from flat keyword arguments (backward compatibility).

        This method allows creating a Config using the old-style flat API:
            Config.from_flat_kwargs(domain=mesh, n_trab=2.0, E0=7500.0, ...)

        Parameters are automatically routed to the correct nested dataclass.
        """
        # Parameter mapping: flat name -> (group_name, param_name)
        # Material
        material_keys = {
            "E0", "nu0", "n_trab", "n_cort", "rho_trab_max", "rho_cort_min",
            "stiff_pE", "stiff_pG",
        }
        # Density
        density_keys = {
            "rho_min", "rho_max", "rho0", "rho_ref", "k_rho_form", "k_rho_resorb",
            "D_rho", "rho_tissue", "surface_use", "surface_A_min", "surface_S0",
        }
        # Stimulus
        stimulus_keys = {
            "stimulus_power_p", "psi_ref", "stimulus_tau", "stimulus_D",
            "stimulus_S_max", "stimulus_kappa", "stimulus_delta0",
        }
        # Fabric
        fabric_keys = {
            "fabric_tau", "fabric_D", "fabric_cA", "fabric_gammaF",
            "fabric_epsQ", "fabric_m_min", "fabric_m_max",
        }
        # Solver
        solver_keys = {
            "ksp_type", "pc_type", "ksp_rtol", "ksp_atol", "ksp_max_it", "ksp_reuse_pc",
            "accel_type", "m", "beta", "lam", "gamma", "safeguard", "backtrack_max",
            "coupling_tol", "restart_on_reject_k", "restart_on_stall", "restart_on_cond",
            "step_limit_factor", "max_subiters", "min_subiters",
        }
        # Time
        time_keys = {
            "total_time", "dt_initial", "adaptive_dt", "adaptive_rtol",
            "adaptive_atol", "dt_min", "dt_max",
        }
        # Numerics
        numerics_keys = {"quadrature_degree", "smooth_eps"}
        # Output
        output_keys = {"saving_interval", "results_dir", "log_file"}

        # Sort kwargs into groups
        material_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in material_keys}
        density_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in density_keys}
        stimulus_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in stimulus_keys}
        fabric_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in fabric_keys}
        solver_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in solver_keys}
        time_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in time_keys}
        numerics_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in numerics_keys}
        output_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in output_keys}

        # Warn about unknown kwargs
        if kwargs:
            import warnings
            warnings.warn(f"Unknown Config kwargs ignored: {list(kwargs.keys())}")

        return cls(
            domain=domain,
            facet_tags=facet_tags,
            material=MaterialParams(**material_kw) if material_kw else MaterialParams(),
            density=DensityParams(**density_kw) if density_kw else DensityParams(),
            stimulus=StimulusParams(**stimulus_kw) if stimulus_kw else StimulusParams(),
            fabric=FabricParams(**fabric_kw) if fabric_kw else FabricParams(),
            solver=SolverParams(**solver_kw) if solver_kw else SolverParams(),
            time=TimeParams(**time_kw) if time_kw else TimeParams(),
            numerics=NumericsParams(**numerics_kw) if numerics_kw else NumericsParams(),
            output=OutputParams(**output_kw) if output_kw else OutputParams(),
        )
