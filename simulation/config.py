from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, TYPE_CHECKING, Any, Dict

from dolfinx import mesh
import ufl

if TYPE_CHECKING:
    from simulation.telemetry import Telemetry


@dataclass
class Config:
    """Simulation parameters. Units: mm, day, MPa, g/cm³."""

    # Material: E = E0 * (rho/rho_ref)^n
    n: float = 2.              # Power-law exponent
    E0: float = 6500.0         # Reference Young's modulus [MPa]
    nu0: float = 0.3           # Poisson ratio

    # Density bounds and initial value
    rho_min: float = 0.1       # Min density [g/cm³]
    rho_max: float = 2.        # Max density [g/cm³]
    rho0: float = 1.5          # Initial density [g/cm³]
    rho_ref: float = 1.0       # Reference density [g/cm³]
    
    # Remodeling: dρ/dt = k_rho * S, where S = (Ψ - Ψ_ref) / Ψ_ref
    k_rho: float = 0.01        # Remodeling rate [1/day]
    D_rho: float = 0.03        # Diffusion coefficient [mm²/day]
    psi_ref: float = 0.1       # Reference SED [MPa]

    # Helmholtz filter: (ρ_filt, v) + L²(∇ρ_filt, ∇v) = (ρ_raw, v)
    # Physical length scale ~0.3mm (osteocyte mechanosensing range)
    helmholtz_L: float = 0.3       # Filter length [mm]

    # Time stepping
    total_time: float = 1000.0    # Total time [days]
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
    domain: Optional[mesh.Mesh] = field(default=None, repr=False)
    facet_tags: Optional[mesh.MeshTags] = field(default=None, repr=False)
    telemetry: Optional["Telemetry"] = field(init=False, default=None, repr=False)
    dx: Optional[ufl.Measure] = field(init=False, default=None, repr=False)
    ds: Optional[ufl.Measure] = field(init=False, default=None, repr=False)
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
        if self.n <= 0:
            raise ValueError("Exponent n must be positive.")
        
        # Density
        if not (0.0 <= self.rho_min < self.rho_max):
            raise ValueError("rho_min/max must satisfy 0 <= rho_min < rho_max.")
        
        # Stimulus
        if self.psi_ref <= 0:
            raise ValueError("Reference value psi_ref must be positive.")
            
        # Elasticity
        if self.E0 <= 0:
            raise ValueError("Young's modulus E0 must be positive.")
        if not (-1.0 < self.nu0 < 0.5):
            raise ValueError("Poisson ratio nu0 must be in range (-1, 0.5).")
        
        # Solver
        if self.accel_type not in ("anderson", "picard"):
            raise ValueError("accel_type must be 'anderson' or 'picard'.")

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