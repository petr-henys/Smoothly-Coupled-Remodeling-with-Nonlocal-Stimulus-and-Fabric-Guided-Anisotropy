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

    Unit convention (consistent and explicit):
    - Length: millimeters [mm]
    - Mass: tonnes [t] (1000 kg)
    - Time: days [day]
    - Stress/Energy density: Megapascals [MPa] = [N/mm²]
    - Density ρ: dimensionless relative density [-] in [0, 1]
      (use as structural fraction; E = E0·ρⁿ). If you need physical density
      [kg/m³], normalize it outside this model.

    Notes:
    - Using mm for length implies stresses/energy densities are naturally in MPa.
    - All stress-like quantities (E, σ, ψ, tractions) are in MPa.
    - Time is in days for remodeling simulations.
    - Variant A (chosen): Stimulus S is dimensionless; parameters have units
      cS [-], τS [1/day], κS [mm²/day], rS_gain [1/(MPa·day)], ψ, ψ_ref [MPa].
    - Orientation A uses cA [-], tauA [day], ell [mm]; effective diffusivity
      D_A = ell² / tauA [mm²/day].
    """

    # --- Material properties ---
    E0: float = 15e3            # Young's modulus [MPa] at ρ=1 (≈15 GPa)
    nu: float = 0.3             # Poisson's ratio [-]

    # Density–stiffness law:
    # E(ρ) = E0 · ρ^{n(ρ)}, where n(ρ) transitions smoothly
    # from trabecular to cortical values between rho_trab_max and rho_cort_min.
    n_power: float = 1.0          # exponent for gait energy driver (ψ/ψ_ref)^n in stimulus
    n_trab: float = 2.0           # trabecular density–stiffness exponent [-]
    n_cort: float = 1.2           # cortical density–stiffness exponent [-]
    rho_trab_max: float = 0.6     # upper ρ for trabecular regime [-]
    rho_cort_min: float = 0.9     # lower ρ for cortical regime [-]

    xi_aniso: float = 0.3         # anisotropic reinforcement factor [-]

    # --- Density bounds and remodeling kinetics ---
    rho_min: float = 0.1          # minimum relative density [-]
    rho_max: float = 1.00         # maximum relative density [-]
    rho0: float = 0.5             # initial relative density [-]
    lambda_rho: float = 0.02      # baseline remodeling rate [1/day] in soft mechanostat (time constant ~100 days when |S| ≈ 1)

    # Soft mechanostat: ρ_eq(S) and lazy zone
    k_mech: float = 2.0           # steepness of logistic ρ_eq(S) in S-space [-]
    S_shift: float = 0.0          # shift of mechanostat setpoint in S [-]
    S_lazy: float = 0.2           # |S|-scale for lazy zone (small |S| → slow remodeling) [-]

    # --- Density: anisotropic diffusion ---
    beta_par: float = 1.          # parallel density diffusion [mm²/day] (O(0.1–10) mm²/day typical)
    beta_perp: float = 0.1         # perpendicular density diffusion [mm²/day] (usually ≤ beta_par)

    # --- Stimulus S: reaction-diffusion ---
    psi_ref: float = 20.0       # reference daily equivalent energy density [MPa]
    cS: float = 1.0             # signaling capacity [-]
    tauS: float = 1.0           # decay rate [1/day]
    kappaS: float = 2.5         # diffusion [mm²/day]
    rS_gain: float = 0.1       # mechano-transduction gain [1/(MPa·day)]

    # --- Orientation A: fabric tensor evolution ---
    cA: float = 1.0               # orientation capacity [-]
    tauA: float = 100.0             # orientation relaxation time [day] (order 1–30 days; smaller = faster reorientation)
    ell: float = 2.0              # orientation diffusion length [mm] (on order of trabecular spacing / microstructural length)

    # Gait / remodeling runtime defaults
    gait_cycles_per_day: float = 7000.0      # average steps/day (2 contacts/cycle)
    load_scale: float = 1.0                  # dimensionless load multiplier
    gait_samples: int = 9                    # quadrature points across gait cycle
    body_mass_kg: float = 75.0

    # --- Numerics / I-O ---
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    verbose: bool = True

    # Global linear solver defaults (tighter for verification-grade solves)
    ksp_type: str = "minres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-7
    ksp_max_it: int = 100

    # Convergence acceleration (Anderson/Picard)
    accel_type: str = "anderson"             # "anderson" | "picard"
    m: int = 8                               # Anderson window size
    beta: float = 1.0                        # damping for newest residual
    lam: float = 1e-10                       # Tikhonov regularization
    gamma: float = 0.05                      # safeguard tolerance
    safeguard: bool = True                   # enable backtracking
    backtrack_max: int = 6                   # max backtracking steps
    coupling_tol: float = 1e-4               # fixed-point tolerance
    
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

    # --- FE / I-O ---
    domain: Optional[mesh.Mesh] = field(default=None, repr=False)
    facet_tags: Optional[mesh.MeshTags] = field(default=None, repr=False)
    
    telemetry: Optional["Telemetry"] = field(init=False, default=None, repr=False)

    # --- UFL measures ---
    dx: Optional[ufl.Measure] = field(init=False, default=None, repr=False)
    ds: Optional[ufl.Measure] = field(init=False, default=None, repr=False)

    # --- Timestep (days) ---
    dt: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.domain is None:
            raise ValueError("Config requires a valid 'domain' (dolfinx.mesh.Mesh).")
        
        # Validate parameter ranges
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"Poisson ratio nu={self.nu} must be in range (-1, 0.5) for physical validity.")
        
        if self.E0 <= 0:
            raise ValueError(f"Young's modulus E0={self.E0} must be positive.")
        if self.n_trab <= 0 or self.n_cort <= 0:
            raise ValueError("n_trab and n_cort must be positive.")
        if not (0.0 <= self.rho_min < self.rho_max <= 1.0):
            raise ValueError("rho_min/max must satisfy 0 ≤ rho_min < rho_max ≤ 1 (relative density).")
        if not (self.rho_min <= self.rho_trab_max <= self.rho_cort_min <= self.rho_max):
            raise ValueError("rho_trab_max and rho_cort_min must satisfy rho_min ≤ rho_trab_max ≤ rho_cort_min ≤ rho_max.")
        if not (self.rho_min <= self.rho0 <= self.rho_max):
            raise ValueError("rho0 must lie within [rho_min, rho_max].")
        if self.lambda_rho < 0:
            raise ValueError("lambda_rho must be non-negative [1/day].")
        if self.k_mech <= 0:
            raise ValueError("k_mech must be positive.")
        if self.S_lazy < 0:
            raise ValueError("S_lazy must be non-negative.")
        if self.beta_par < 0 or self.beta_perp < 0:
            raise ValueError("beta_par/beta_perp must be non-negative [mm²/day].")
        if self.cS <= 0 or self.tauS < 0 or self.kappaS < 0 or self.rS_gain < 0:
            raise ValueError("cS>0, tauS≥0, kappaS≥0, rS_gain≥0 required.")
        if self.cA <= 0 or self.tauA < 0 or self.ell <= 0:
            raise ValueError("cA>0, tauA≥0, ell>0 required.")
        
        # Validate acceleration type
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