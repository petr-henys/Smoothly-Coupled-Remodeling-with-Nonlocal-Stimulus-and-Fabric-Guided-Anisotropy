from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, TYPE_CHECKING, Any, Dict

from dolfinx import mesh
import ufl

if TYPE_CHECKING:
    from simulation.telemetry import Telemetry


@dataclass
class Config:
    """
    Simulation configuration with material, solver, and I/O parameters.
    
    Units Convention:
        - Length: mm
        - Time: day  
        - Stress/Modulus: MPa (N/mm²)
        - Density: g/cm³ (equivalent to kg/L)
    
    Note on Dimensional Consistency:
        The remodeling equation dρ/dt = k_rho × f(Ψ) currently has dimensional issues.
        k_rho [1/day] × Ψ [MPa] does not yield [g/cm³/day] without a conversion constant.
        This is a known limitation - see code review documentation.
    """

    # =========================================================================
    # Material Properties (Updated based on Bensel et al., 2024, Table 2)
    # =========================================================================
    # Density-stiffness relationship: E = E0 * (rho/rho_max)^n
    # Article uses p=2 (Eq. 3), so we unify trab/cort exponents.
    n_trab: float = 2.0         # Exponent for trabecular bone (p=2 in article)
    n_cort: float = 2.0         # Exponent for cortical bone (p=2 in article)
    
    # Smooth step transition parameters (irrelevant if n_trab == n_cort)
    rho_trab_max: float = 1.2   # Max density for trabecular regime [g/cm^3]
    rho_cort_min: float = 1.7   # Min density for cortical regime [g/cm^3]

    # =========================================================================
    # Density Evolution (Remodeling)
    # =========================================================================
    rho_min: float = 0.1        # Min physical density [g/cm³] - prevents singular stiffness
    rho_max: float = 2.0        # Max physical density [g/cm³] - cortical bone limit
    rho0: float = 1.0           # Initial density [g/cm³]
    rho_ref: float = 1.0        # Reference density for E(ρ) power law [g/cm³]
    
    # Rate constant for relaxation formulation
    # WARNING: Dimensional analysis issue!
    # For dρ/dt = k_rho × (Ψ - Ψ_ref), we need [k_rho] = [g/cm³/(MPa·day)]
    # Current interpretation: k_rho as dimensionless rate coefficient
    # Actual physical rate depends on how source term is formulated.
    k_rho: float = 2            # Remodeling rate coefficient [1/day] (heuristic)

    # Density diffusion coefficient [mm²/day]
    # Provides spatial regularization (prevents checkerboard patterns)
    # Physical interpretation: nonlocal averaging of remodeling signal
    # Typical range: 0.01-1.0 mm²/day (larger = more smoothing)
    D_rho: float = 0.1          # Isotropic diffusion [mm²/day]

    # =========================================================================
    # Stimulus (Strain Energy Density)
    # =========================================================================
    # Reference SED (homeostatic setpoint)
    # Ψ = ½ σ:ε [MPa] - computed from mechanics solution
    # When Ψ > Ψ_ref: bone formation (ρ increases toward ρ_max)
    # When Ψ < Ψ_ref: bone resorption (ρ decreases toward ρ_min)
    # Typical cortical bone: Ψ_ref ~ 0.001-0.01 MPa
    psi_ref: float = 0.01          # Reference SED [MPa]      
        
    # Base moduli [MPa]
    E0: float = 6500.0          # Young's modulus (Table 2)
    nu0: float = 0.3            # Poisson ratio (Table 2)

    # =========================================================================
    # Adaptive Time Stepping
    # =========================================================================
    adaptive_rtol: float = 1e-2
    adaptive_atol: float = 1e-3
    dt_min: float = 1e-4
    dt_max: float = 100

    # =========================================================================
    # Numerics & I/O
    # =========================================================================
    quadrature_degree: int = 4
    saving_interval: int = 1
    results_dir: str = ".results"
    log_file: str = "simulation.log"

    # Linear Solver (RESTORED TO ORIGINAL SETTINGS)
    ksp_type: str = "minres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-7
    ksp_max_it: int = 100

    # Nonlinear Solver (Anderson/Picard)
    accel_type: str = "anderson"
    m: int = 5                  # History size
    beta: float = 1.0           # Mixing parameter
    lam: float = 1e-9           # Regularization
    gamma: float = 0.05         # Safeguard tolerance
    safeguard: bool = True
    backtrack_max: int = 5
    coupling_tol: float = 1e-4
    
    # Restart heuristics
    restart_on_reject_k: int = 2
    restart_on_stall: float = 1.10
    restart_on_cond: float = 1e12
    step_limit_factor: float = 2.0

    # Subiterations
    max_subiters: int = 25
    min_subiters: int = 2

    # Diagnostics
    smooth_eps: float = 1e-6    # Regularization for abs/max/PSD

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