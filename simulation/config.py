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
    n_power: float = 1.0        # Exponent for stimulus calculation
    n_trab: float = 2.0         # Exponent for trabecular bone
    n_cort: float = 1.2         # Exponent for cortical bone
    rho_trab_max: float = 0.6   # Max density for trabecular regime
    rho_cort_min: float = 0.9   # Min density for cortical regime

    # =========================================================================
    # Density Evolution (Remodeling)
    # =========================================================================
    rho_min: float = 0.1        # Min relative density
    rho_max: float = 1.00       # Max relative density
    rho0: float = 0.5           # Initial relative density

    k_rho: float = 0.001          # Density remodeling rate [1/day]

    # Density diffusion [mm^2/day]
    beta_par: float = 0.05       # Parallel to fabric
    beta_perp: float = 0.05      # Perpendicular to fabric

    # =========================================================================
    # Stimulus (Reaction-Diffusion)
    # =========================================================================
    psi_ref: float = 0.04      # Reference value (Stress [MPa], Strain [-], or SED [MPa])
    
    cS: float = 1.0             # Signaling capacity
    tauS: float = 0.2           # Relaxation time [day] (was decay rate 5.0)
    kappaS: float = 1.0         # Diffusion coefficient [mm^2/day]
    distal_damping_height: float = 1       # Height of distal damping zone [mm]
    distal_damping_transition: float = 5.0    # Transition width of distal damping zone [mm]

    # =========================================================================
    # Fabric Tensor Evolution
    # =========================================================================
    cA: float = 1.0             # Orientation capacity
    tauA: float = 200.0         # Relaxation time [day]
    ell: float = 2.0            # Diffusion length [mm]
    
    # Standard Zysset parameters (approximate)
    k_stiff: float = 1.9        # Density exponent for stiffness (often close to 2)
    p_stiff: float = 0.5        # Fabric exponent (linear or quadratic)
    
    # Base moduli [MPa]
    E0_z: float = 15000.0       # Axial modulus
    G0_z: float = 5000.0        # Shear modulus
    nu0_z: float = 0.3          # Poisson ratio

    # =========================================================================
    # Gait & Loading
    # =========================================================================
    gait_cycles_per_day: float = 1.0
    load_scale: float = 1.0
    gait_samples: int = 9
    body_mass_tonnes: float = 0.075   # 0.075 t ≈ 75 kg

    # =========================================================================
    # Numerics & I/O
    # =========================================================================
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    verbose: Union[bool, str] = True

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
    lam: float = 1e-10          # Regularization
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
        if self.beta_par < 0 or self.beta_perp < 0:
            raise ValueError("beta_par/beta_perp must be non-negative.")
        
        # Stimulus
        if self.psi_ref <= 0:
            raise ValueError("Reference value psi_ref must be positive.")
        if self.cS <= 0 or self.tauS <= 0 or self.kappaS < 0:
            raise ValueError("cS>0, tauS>0, kappaS>=0 required.")
        
        # Fabric
        if self.cA <= 0 or self.tauA < 0 or self.ell <= 0:
            raise ValueError("cA>0, tauA>=0, ell>0 required.")
        
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

        self.telemetry = Telemetry(
            comm=self.domain.comm,
            outdir=self.results_dir,
            verbose=self.verbose,
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
