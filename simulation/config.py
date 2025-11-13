from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, TYPE_CHECKING

from dolfinx import fem, mesh, default_scalar_type
import ufl

if TYPE_CHECKING:
    from simulation.telemetry import Telemetry


@dataclass
class Config:
    """Global configuration for bone remodeling simulation.
    
    All parameters use SI units:
    - Length: meters [m]
    - Mass: kilograms [kg]
    - Time: seconds [s] (or days [day] where explicitly noted)
    - Stress/Pressure: Pascals [Pa]
    - Density: kg/m³
    """

    # --- Material properties ---
    E0: float = 6.5e9             # Young's modulus [Pa]
    nu: float = 0.3               # Poisson's ratio [-]
    n_power: float = 2.0          # density-stiffness power law exponent [-]
    xi_aniso: float = 0.2         # anisotropic reinforcement factor [-]

    # --- Density bounds ---
    rho_min: float = 350.0        # minimum density [kg/m³]
    rho_max: float = 1850.0       # maximum density [kg/m³]
    rho0: float = 1200.0          # initial density [kg/m³]

    # --- Density: anisotropic diffusion ---
    beta_par: float = 2.8e-6      # parallel diffusion [m²/day]
    beta_perp: float = 8.5e-7     # perpendicular diffusion [m²/day]

    # --- Stimulus S: reaction-diffusion ---
    psi_ref: float = 300.0        # reference energy density [Pa]
    cS: float = 32.0              # signaling capacity [Pa·day]
    tauS: float = 0.04            # decay rate [1/day] → 25-day time constant
    kappaS: float = 2.5e-4        # diffusion [m²/day]
    rS: float = 2.0e-7            # mechano-transduction gain [1/(Pa·day)]

    # --- Orientation A: fabric tensor evolution ---
    cA: float = 1.4               # orientation capacity [-]
    tauA: float = 0.6             # orientation relaxation time [day]
    ell: float = 0.35             # orientation diffusion length [m]

    # --- Load scaling ---
    t_p: float = 3e6              # reference peak stress [Pa]

    # --- Load scaling ---
    t_p: float = 3e6              # reference peak stress [Pa]

    # --- Numerics / I-O ---
    quadrature_degree: int = 6
    saving_interval: int = 1
    results_dir: str = ".results"
    verbose: bool = True

    # Global linear solver defaults (tighter for verification-grade solves)
    ksp_type: str = "minres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-9
    ksp_atol: float = 1e-11
    ksp_max_it: int = 100

    # Convergence acceleration (Anderson/Picard)
    accel_type: str = "anderson"             # "anderson" | "picard" | "none"
    m: int = 8                               # Anderson window size
    beta: float = 1.0                        # damping for newest residual
    lam: float = 1e-10                       # Tikhonov regularization
    gamma: float = 0.05                      # safeguard tolerance
    safeguard: bool = True                   # enable backtracking
    backtrack_max: int = 6                   # max backtracking steps
    coupling_tol: float = 1e-6               # fixed-point tolerance
    # Anderson restarts and step limiting
    restart_on_reject_k: int = 2             # restart after k rejections
    restart_on_stall: float = 1.10           # restart if residual stalls by this factor
    restart_on_cond: float = 1e12            # restart if cond(Gram) exceeds this
    step_limit_factor: float = 2.0           # limit ||Anderson step|| to factor*||Picard residual||

    # Nitsche method parameters for KUBC homogenizer
    nitsche_alpha: float = 30.0
    nitsche_theta: float = 1.0

    # Subiteration bounds per external step
    max_subiters: int = 50
    min_subiters: int = 1

    # Coupling diagnostics (optional)
    coupling_eps: float = 1e-3
    coupling_each_iter: bool = False         # compute Jacobian each subiteration (expensive)

    # Smoothness controls
    smooth_eps: float = 5e-7                 # C∞ regularization for abs, max, PSD

    # Gait / remodeling runtime defaults
    gait_cycles_per_day: float = 7000.0      # average steps/day (2 contacts/cycle)
    load_scale: float = 1.0                  # dimensionless load multiplier
    gait_samples: int = 9                    # quadrature points across gait cycle
    body_mass_kg: float = 75.0
    sim_dt_days: float = 10.0
    sim_total_days: float = 1000.0

    # --- FE / I-O ---
    domain: Optional[mesh.Mesh] = field(default=None, repr=False)
    facet_tags: Optional[mesh.MeshTags] = field(default=None, repr=False)
    
    telemetry: Optional["Telemetry"] = field(init=False, default=None, repr=False)

    # --- UFL measures ---
    dx: Optional[ufl.Measure] = field(init=False, default=None, repr=False)
    ds: Optional[ufl.Measure] = field(init=False, default=None, repr=False)

    # --- UFL Constants (SI units) ---
    E0_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    dt_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    psi_ref_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)

    beta_par_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    beta_perp_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)

    cS_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    tauS_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    kappaS_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    rS_gain_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)

    cA_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    tauA_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    ell_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)

    xi_aniso_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    nu_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)
    n_power_c: Optional[fem.Constant] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.domain is None:
            raise ValueError("Config requires a valid 'domain' (dolfinx.mesh.Mesh).")
        
        # Validate parameter ranges
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"Poisson ratio nu={self.nu} must be in range (-1, 0.5) for physical validity.")
        
        if self.E0 <= 0:
            raise ValueError(f"Young's modulus E0={self.E0} must be positive.")
        
        self._build_measures()
        self._build_constants()
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

    def _build_constants(self):
        """Build UFL Constants from SI-unit parameters."""
        # Convert time from days to seconds for consistency
        DAY_TO_SEC = 86400.0
        
        # Mechanics (already in SI)
        self.E0_c = fem.Constant(self.domain, self._cast(self.E0))
        self.nu_c = fem.Constant(self.domain, self._cast(self.nu))
        self.n_power_c = fem.Constant(self.domain, self._cast(self.n_power))

        # Time step (will be set later in seconds)
        self.dt_c = fem.Constant(self.domain, self._cast(1.0))

        # Energy reference [Pa]
        self.psi_ref_c = fem.Constant(self.domain, self._cast(self.psi_ref))

        # Diffusion [m²/s] - convert from m²/day
        self.beta_par_c = fem.Constant(self.domain, self._cast(self.beta_par / DAY_TO_SEC))
        self.beta_perp_c = fem.Constant(self.domain, self._cast(self.beta_perp / DAY_TO_SEC))

        # Stimulus (convert from per-day to per-second)
        self.cS_c = fem.Constant(self.domain, self._cast(self.cS / DAY_TO_SEC))
        self.tauS_c = fem.Constant(self.domain, self._cast(self.tauS / DAY_TO_SEC))
        self.kappaS_c = fem.Constant(self.domain, self._cast(self.kappaS / DAY_TO_SEC))
        self.rS_gain_c = fem.Constant(self.domain, self._cast(self.rS / DAY_TO_SEC))

        # Orientation (convert tauA from days to seconds)
        self.cA_c = fem.Constant(self.domain, self._cast(self.cA))
        self.tauA_c = fem.Constant(self.domain, self._cast(self.tauA / DAY_TO_SEC))
        self.ell_c = fem.Constant(self.domain, self._cast(self.ell))

        # Anisotropy
        self.xi_aniso_c = fem.Constant(self.domain, self._cast(self.xi_aniso))

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

    def set_dt(self, dt_seconds: float):
        """Update Δt constant from value in seconds."""
        if dt_seconds <= 0:
            raise ValueError(f"Timestep dt_seconds={dt_seconds} must be positive.")
        self.dt_c.value = self._cast(dt_seconds)

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
        """Rebuild measures and constants after domain change."""
        self.domain = domain
        self.facet_tags = facet_tags
        self._build_measures()
        self._build_constants()

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
