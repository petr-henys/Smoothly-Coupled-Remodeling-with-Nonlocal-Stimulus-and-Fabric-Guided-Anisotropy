from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any

from dolfinx import mesh
import ufl


@dataclass
class Config:
    """Simulation parameters (mm, day, MPa, g/cm^3)."""

    # Material: density-dependent exponent k(rho) blended from trabecular to cortical:
    # k = n_trab*(1-w) + n_cort*w, w = smoothstep01((rho-rho_trab_max)/(rho_cort_min-rho_trab_max)).
    E0: float = 7500.0
    n_trab: float = 2.0
    n_cort: float = 1.3
    rho_trab_max: float = 1.0
    rho_cort_min: float = 1.25
    nu0: float = 0.3  # Poisson ratio

    # Density [g/cm^3].
    rho_min: float = 0.1  # Lower bound
    rho_max: float = 2.0  # Upper bound
    rho0: float = 1.0  # Initial value
    rho_ref: float = 1.0  # Reference value for nondimensionalization

    # Density update (see DensitySolver): implicit Euler diffusion + stimulus-driven source with soft bounds.
    k_rho_form: float = 7e-03   # Formation rate [1/day] scaling positive stimulus
    k_rho_resorb: float = 7e-03 # Resorption rate [1/day] scaling negative stimulus
    D_rho: float = 1e-2  # Diffusion coefficient [mm^2/day]

    # Stimulus (osteocyte-inspired signal). Units: S is dimensionless.
    #
    # Mechanostat: m = psi/rho_safe, m_ref = psi_ref/rho_ref, delta = (m-m_ref)/m_ref.
    # Stimulus PDE (see StimulusSolver): tau_S dS/dt = D_S ΔS - S + S_max tanh(delta/kappa).
    stimulus_power_p: float = 2.0  # Power-mean exponent (1=mean; higher→peak-biased)
    psi_ref: float = 0.02         # Reference SED [MPa] used via m_ref = psi_ref / rho_ref
    stimulus_tau: float = 25.0      # tau_S [days]; tau_S=0 gives quasi-static stimulus (no time derivative)
    stimulus_D: float = 10.0         # D_S [mm^2/day]; nonlocal length ~ sqrt(D_S * tau_S)
    stimulus_S_max: float = 1.0     # S_max (dimensionless): cap on |S|
    stimulus_kappa: float = 0.5     # kappa: saturation width in tanh(delta/kappa)

    # Time stepping
    total_time: float = 500.0     # Total time [days]
    dt_initial: float = 25.0      # Initial timestep [days]
    adaptive_dt: bool = False     # Enable adaptive time stepping
    adaptive_rtol: float = 1e-2   # Relative error tolerance
    adaptive_atol: float = 1e-3   # Absolute error tolerance
    dt_min: float = 1e-4          # Min timestep [days]
    dt_max: float = 100           # Max timestep [days]

    # Numerics
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    log_file: str = "simulation.log"

    # Linear solver
    ksp_type: str = "minres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-7
    ksp_max_it: int = 100

    # Fixed-point iteration (Anderson/Picard)
    accel_type: str = "anderson"
    m: int = 5                    # History size
    beta: float = 1.0             # Mixing parameter
    lam: float = 1e-9             # Regularization
    gamma: float = 0.05           # Safeguard tolerance
    safeguard: bool = True
    backtrack_max: int = 5
    coupling_tol: float = 1e-4
    restart_on_reject_k: int = 2
    restart_on_stall: float = 1.10
    restart_on_cond: float = 1e12
    step_limit_factor: float = 2.0
    max_subiters: int = 25
    min_subiters: int = 2

    # Regularization
    smooth_eps: float = 1e-6

    # Runtime state (not serialized)
    domain: mesh.Mesh | None = field(default=None, repr=False)
    facet_tags: mesh.MeshTags | None = field(default=None, repr=False)
    dx: ufl.Measure | None = field(init=False, default=None, repr=False)
    ds: ufl.Measure | None = field(init=False, default=None, repr=False)
    dt: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.domain is None:
            raise ValueError("Config requires a valid 'domain' (dolfinx.mesh.Mesh).")
        
        # Resolve log_file relative to results_dir.
        self.log_file = str(Path(self.results_dir) / self.log_file)

        self.validate()
        self._build_measures()
        self._ensure_results_dir()
        self.update_config_json()

    def validate(self):
        """Validate parameter ranges and basic invariants."""
        # Material
        if self.n_trab <= 0 or self.n_cort <= 0:
            raise ValueError("Material exponents must satisfy n_trab>0 and n_cort>0.")
        if not (self.rho_trab_max < self.rho_cort_min):
            raise ValueError("Require rho_trab_max < rho_cort_min for smoothstep transition.")
        
        # Density
        if not (0.0 <= self.rho_min < self.rho_max):
            raise ValueError("rho_min/max must satisfy 0 <= rho_min < rho_max.")

        if not (self.rho_min <= self.rho0 <= self.rho_max):
            raise ValueError("rho0 must be within [rho_min, rho_max].")
        if self.D_rho < 0:
            raise ValueError("D_rho must be >= 0 (diffusion coefficient).")
        if self.k_rho_form < 0 or self.k_rho_resorb < 0:
            raise ValueError("k_rho_form and k_rho_resorb must be >= 0 (remodeling rates).")

        
        # Stimulus
        if self.psi_ref <= 0:
            raise ValueError("Reference value psi_ref must be positive.")
        if self.stimulus_tau < 0:
            raise ValueError("stimulus_tau must be >= 0 (tau_S in days).")
        if self.stimulus_D < 0:
            raise ValueError("stimulus_D must be >= 0 (diffusion coefficient, mm^2/day).")
        if self.stimulus_S_max <= 0:
            raise ValueError("stimulus_S_max must be > 0 (dimensionless cap on |S|).")
        if self.stimulus_kappa <= 0:
            raise ValueError("stimulus_kappa must be > 0 (dimensionless saturation width).")
            
        # Elasticity
        if self.E0 <= 0:
            raise ValueError("Young's modulus E0 must be positive.")
        if not (-1.0 < self.nu0 < 0.5):
            raise ValueError("Poisson ratio nu0 must be in range (-1, 0.5).")
        
        # Solver
        if self.accel_type not in ("anderson", "picard"):
            raise ValueError("accel_type must be 'anderson' or 'picard'.")
        if self.stimulus_power_p < 1.0:
            raise ValueError("stimulus_power_p must be >= 1.0 (1=mean; larger biases toward peaks).")

    def _build_measures(self):
        """Create UFL measures with the configured quadrature degree."""
        metadata = {"quadrature_degree": int(self.quadrature_degree)}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=self.facet_tags,
            metadata=metadata,
        )

    def _ensure_results_dir(self) -> None:
        """Create results directory on rank 0 and barrier."""
        outdir = Path(self.results_dir)
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
            path = Path(self.results_dir) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_json_dict(), f, indent=2)
        self.domain.comm.Barrier()

    def rebuild(self, domain: mesh.Mesh, facet_tags: mesh.MeshTags | None = None):
        """Update mesh and rebuild integration measures."""
        self.domain = domain
        self.facet_tags = facet_tags
        self._build_measures()

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize init-time parameters to a JSON-compatible dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.init and f.repr and isinstance(getattr(self, f.name), (int, float, bool, str, type(None)))
        }