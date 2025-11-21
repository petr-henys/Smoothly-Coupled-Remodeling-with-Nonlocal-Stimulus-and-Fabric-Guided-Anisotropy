from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, TYPE_CHECKING

from dolfinx import mesh, default_scalar_type
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

    # --- Material properties ---
    E0: float = 15e3            # Young's modulus [MPa] at rho=1
    nu: float = 0.3             # Poisson's ratio [-]

    # Density-stiffness relationship: E = E0 * rho^n(rho)
    n_power: float = 2.0        # Exponent for stimulus calculation
    n_trab: float = 2.0         # Exponent for trabecular bone
    n_cort: float = 1.2         # Exponent for cortical bone
    rho_trab_max: float = 0.6   # Max density for trabecular regime
    rho_cort_min: float = 0.9   # Min density for cortical regime

    xi_aniso: float = 0.3       # Anisotropic reinforcement factor

    # --- Density evolution ---
    rho_min: float = 0.1        # Min relative density
    rho_max: float = 1.00       # Max relative density
    rho0: float = 0.5           # Initial relative density
    lambda_rho: float = 0.05    # Remodeling rate [1/day]

    # Frost-style mechanostat (two-threshold, smooth Heaviside)
    S_form_th: float = 0.2     # Formation threshold in S units (steady S≈ψ-1)
    S_resorb_th: float = -0.2  # Resorption threshold in S units
    k_step: float = 6.0        # Step steepness for smooth thresholds
    lambda_form: float = 0.05  # Formation rate [1/day]
    lambda_resorb: float = 0.08 # Resorption rate [1/day]

    # Mechanostat parameters
    k_mech: float = 4.0         # Steepness of equilibrium curve
    S_shift: float = 0.0        # Setpoint shift
    S_lazy: float = 0.1         # Lazy zone width

    # Density diffusion [mm^2/day]
    beta_par: float = 1.0       # Parallel to fabric
    beta_perp: float = 0.1      # Perpendicular to fabric

    # --- Stimulus (Reaction-Diffusion) ---
    psi_ref: float = 3.0       # Reference stress/energy [MPa]
    cS: float = 1.0             # Signaling capacity
    tauS: float = 1.0           # Decay rate [1/day]
    kappaS: float = 5.0         # Diffusion coefficient [mm^2/day]
    rS_gain: float = 1.0        # Transduction gain [1/day]

    # --- Fabric Tensor Evolution ---
    cA: float = 1.0             # Orientation capacity
    tauA: float = 100.0         # Relaxation time [day]
    ell: float = 2.0            # Diffusion length [mm]

    # --- Gait & Loading ---
    gait_cycles_per_day: float = 1.0
    load_scale: float = 1.0
    gait_samples: int = 20
    body_mass_tonnes: float = 0.075   # 0.075 t ≈ 75 kg

    # --- Numerics & I/O ---
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    verbose: bool = True

    # Linear Solver
    ksp_type: str = "minres"
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
    coupling_eps: float = 1e-3
    coupling_each_iter: bool = False
    smooth_eps: float = 5e-7    # Regularization for abs/max/PSD

    # --- Internal Fields ---
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
        
        # Validate parameter ranges
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"Poisson ratio nu={self.nu} must be in range (-1, 0.5).")
        
        if self.E0 <= 0:
            raise ValueError(f"Young's modulus E0={self.E0} must be positive.")
        if self.n_trab <= 0 or self.n_cort <= 0:
            raise ValueError("n_trab and n_cort must be positive.")
        if not (0.0 <= self.rho_min < self.rho_max <= 1.0):
            raise ValueError("rho_min/max must satisfy 0 <= rho_min < rho_max <= 1.")
        if not (self.rho_min <= self.rho_trab_max <= self.rho_cort_min <= self.rho_max):
            raise ValueError("rho_trab_max and rho_cort_min must satisfy rho_min <= rho_trab_max <= rho_cort_min <= rho_max.")
        if not (self.rho_min <= self.rho0 <= self.rho_max):
            raise ValueError("rho0 must lie within [rho_min, rho_max].")
        if self.lambda_rho < 0:
            raise ValueError("lambda_rho must be non-negative.")
        if self.k_mech <= 0:
            raise ValueError("k_mech must be positive.")
        if self.S_lazy < 0:
            raise ValueError("S_lazy must be non-negative.")
        if self.beta_par < 0 or self.beta_perp < 0:
            raise ValueError("beta_par/beta_perp must be non-negative.")
        if self.cS <= 0 or self.tauS < 0 or self.kappaS < 0 or self.rS_gain < 0:
            raise ValueError("cS>0, tauS>=0, kappaS>=0, rS_gain>=0 required.")
        if self.cA <= 0 or self.tauA < 0 or self.ell <= 0:
            raise ValueError("cA>0, tauA>=0, ell>0 required.")
        if self.body_mass_tonnes <= 0:
            raise ValueError("body_mass_tonnes must be positive (tonnes).")
        if self.gait_cycles_per_day <= 0:
            raise ValueError("gait_cycles_per_day must be positive.")
        if self.gait_samples < 2:
            raise ValueError("gait_samples must be at least 2.")
        if self.load_scale < 0:
            raise ValueError("load_scale must be non-negative.")
        
        if self.accel_type not in ("anderson", "picard"):
            raise ValueError(f"accel_type must be 'anderson' or 'picard', got {self.accel_type!r}")
        
        self._build_measures()
        self._init_telemetry()

    @staticmethod
    def _cast(val: float):
        """Cast to DOLFINx default scalar type."""
        return default_scalar_type(val)

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

        outdir = self.results_dir
        self.telemetry = Telemetry(
            comm=self.domain.comm,
            outdir=outdir,
            verbose=bool(getattr(self, "verbose", True)),
        )

        cfg_dict = self.to_json_dict()
        self.telemetry.write_metadata(
            cfg_dict,
            filename="config.json",
            overwrite=True,
        )

    def set_dt(self, dt_days: float):
        """Update timestep in days."""
        if dt_days <= 0:
            raise ValueError(f"Timestep dt_days={dt_days} must be positive.")
        self.dt = float(dt_days)

    def update_config_json(self):
        """Re-write config.json with current parameters (rank-0 only)."""
        if self.telemetry is None:
            return
        cfg_dict = self.to_json_dict()
        self.telemetry.write_metadata(
            cfg_dict,
            filename="config.json",
            overwrite=True,
        )

    def rebuild(self, domain: mesh.Mesh, facet_tags: Optional[mesh.MeshTags] = None):
        """Rebuild measures after domain change."""
        self.domain = domain
        self.facet_tags = facet_tags
        self._build_measures()

    def to_json_dict(self) -> dict:
        """Serialize init-time parameters to JSON-compatible dict."""
        cfg = {}
        for f in fields(self):
            if not f.init or not f.repr:
                continue
            name = f.name
            val = getattr(self, name)
            if isinstance(val, (int, float, bool, str)) or val is None:
                cfg[name] = val
        return cfg
