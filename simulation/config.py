from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, TYPE_CHECKING, Any, Dict, Union

from dolfinx import mesh
import ufl

if TYPE_CHECKING:
    from simulation.telemetry import Telemetry


@dataclass
class Config:
    """Global configuration for bone remodeling simulation.

    Units (model-wide):
    - Length: millimetres [mm]
    - Mass: tonnes [t] (1 t = 1000 kg)
    - Time: days [day]
    - Stress/Energy: megapascals [MPa] (1 MPa = 1 N/mm²)
    - Density: relative [-] in [0, 1]
    """

    # =========================================================================
    # Material Properties
    # =========================================================================
    # Density-stiffness relationship: E = E0 * rho^n(rho)
    n_power: float = 4.        # Exponent for stimulus calculation (proximal femur, fatigue-like)
    n_trab: float = 2.0         # Exponent for trabecular bone
    n_cort: float = 1.2         # Exponent for cortical bone
    rho_trab_max: float = 0.6   # Max density for trabecular regime
    rho_cort_min: float = 0.9   # Min density for cortical regime

    # =========================================================================
    # Density Evolution (Remodeling)
    # =========================================================================
    rho_min: float = 0.05       # Min relative density (very low trabecular)
    rho_max: float = 1.00       # Max relative density
    rho0: float = 0.4           # Initial relative density (proximal femur: mostly trabecular)
    k_rho: float = 0.02        # Density remodeling rate [1/day] (half-time ~1 year)

    # Density diffusion [mm^2/day]
    beta: float = 0.01       # Isotropic diffusion [mm^2/day]

    # =========================================================================
    # Stimulus (Local)
    # =========================================================================
    psi_ref: float = 20.     # Reference effective stress [MPa] for daily stimulus in proximal femur
    distal_damping_height: float = 1.0     # Height of distal damping zone [mm] (cut shaft of femur)
    distal_damping_transition: float = 0.5    # Transition width of distal damping zone [mm]

    # Standard Zysset parameters (approximate)
    k_stiff: float = 1.9        # Density exponent for stiffness (often close to 2)
    
    # Base moduli [MPa]
    E0: float = 15000.0       # Young's modulus
    nu0: float = 0.3          # Poisson ratio

    # =========================================================================
    # Gait & Loading
    # =========================================================================
    gait_cycles_per_day: float = 1.   # Daily equivalent hip loading cycles (walking, stairs)
    load_scale: float = 1.0      # Keep loads from musculoskeletal model unscaled
    gait_samples: int = 9
    body_mass_tonnes: float = 0.075   # 0.075 t ≈ 75 kg (proximal femur subject)

    # =========================================================================
    # Numerics & I/O
    # =========================================================================
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    verbose: Union[bool, str] = True
    log_file: str = "simulation.log"

    # Linear Solver
    ksp_type: str = "minres"        # Changed from minres to cg for SPD elasticity
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-7
    ksp_max_it: int = 100

    # Nonlinear Solver (Anderson/Picard)
    accel_type: str = "anderson"
    m: int = 8                  # Anderson history size
    beta: float = 1.0           # Mixing parameter
    lam: float = 1e-8          # Regularization
    gamma: float = 0.05         # Safeguard tolerance
    safeguard: bool = True
    backtrack_max: int = 6
    coupling_tol: float = 1e-4
    
    # Restart heuristics
    restart_on_reject_k: int = 2
    restart_on_stall: float = 1.10
    restart_on_cond: float = 1e12
    step_limit_factor: float = 2.0

    # Nitsche (KUBC)
    nitsche_alpha: float = 30.0
    nitsche_theta: float = 1.0

    # Subiterations
    max_subiters: int = 50
    min_subiters: int = 1

    # Diagnostics
    smooth_eps: float = 5e-7    # Regularization for abs/max/PSD

    # =========================================================================
    # Internal State (Runtime)
    # =========================================================================
    domain: Optional[mesh.Mesh] = field(default=None, repr=False)
    facet_tags: Optional[mesh.MeshTags] = field(default=None, repr=False)
    
    telemetry: Optional["Telemetry"] = field(init=False, default=None, repr=False)

    # UFL Measures
    dx: Optional[ufl.Measure] = field(init=False, default=None, repr=False)
    ds: Optional[ufl.Measure] = field(init=False, default=None, repr=False)

    # State
    dt: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.domain is None:
            raise ValueError("Config requires a valid 'domain' (dolfinx.mesh.Mesh).")
        
        # Resolve log_file path relative to results_dir
        from pathlib import Path
        self.log_file = str(Path(self.results_dir) / self.log_file)

        self.validate()
        self._build_measures()
        self._init_telemetry()

    def validate(self):
        """Validate configuration parameters."""
        # Material
        if self.n_trab <= 0 or self.n_cort <= 0:
            raise ValueError("n_trab and n_cort must be positive.")
        
        # Density
        if not (0.0 <= self.rho_min < self.rho_max <= 1.0):
            raise ValueError("rho_min/max must satisfy 0 <= rho_min < rho_max <= 1.")
        if not (self.rho_min <= self.rho_trab_max <= self.rho_cort_min <= self.rho_max):
            raise ValueError("rho_trab_max and rho_cort_min must satisfy rho_min <= rho_trab_max <= rho_cort_min <= rho_max.")
        if not (self.rho_min <= self.rho0 <= self.rho_max):
            raise ValueError("rho0 must lie within [rho_min, rho_max].")
        if self.beta < 0:
            raise ValueError("beta must be non-negative.")
        
        # Stimulus
        if self.psi_ref <= 0:
            raise ValueError("Reference value psi_ref must be positive.")
        
        # Gait
        if self.body_mass_tonnes <= 0:
            raise ValueError("body_mass_tonnes must be positive (tonnes).")
        if self.gait_cycles_per_day <= 0:
            raise ValueError("gait_cycles_per_day must be positive.")
        if self.gait_samples < 2:
            raise ValueError("gait_samples must be at least 2.")
        if self.load_scale < 0:
            raise ValueError("load_scale must be non-negative.")
        
        # Solver
        if self.accel_type not in ("anderson", "picard"):
            raise ValueError(f"accel_type must be 'anderson' or 'picard', got {self.accel_type!r}")
            
        # Elasticity
        if self.E0 <= 0:
            raise ValueError("Young's modulus E0 must be positive.")
        if not (-1.0 < self.nu0 < 0.5):
            raise ValueError("Poisson ratio nu0 must be in (-1.0, 0.5).")

    def _build_measures(self):
        """Create UFL integration measures with quadrature degree."""
        metadata = {"quadrature_degree": int(self.quadrature_degree)}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=self.facet_tags,
            metadata=metadata,
        )

    def _init_telemetry(self) -> None:
        """Initialize telemetry and persist config.json (rank-0 only)."""
        from simulation.telemetry import Telemetry

        # If verbose is "progressbar", we want Telemetry to be quiet on console (verbose=False)
        telemetry_verbose = self.verbose
        if telemetry_verbose == "progressbar":
            telemetry_verbose = False

        self.telemetry = Telemetry(
            comm=self.domain.comm,
            outdir=self.results_dir,
            verbose=telemetry_verbose,
        )
        self.update_config_json()

    def set_dt(self, dt_days: float):
        """Update timestep in days."""
        if dt_days <= 0:
            raise ValueError(f"Timestep dt_days={dt_days} must be positive.")
        self.dt = float(dt_days)

    def update_config_json(self):
        """Re-write config.json with current parameters (rank-0 only)."""
        if self.telemetry is None:
            return
        self.telemetry.write_metadata(
            self.to_json_dict(),
            filename="config.json",
            overwrite=True,
        )

    def rebuild(self, domain: mesh.Mesh, facet_tags: Optional[mesh.MeshTags] = None):
        """Rebuild measures after domain change."""
        self.domain = domain
        self.facet_tags = facet_tags
        self._build_measures()

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize init-time parameters to JSON-compatible dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.init and f.repr and isinstance(getattr(self, f.name), (int, float, bool, str, type(None)))
        }
