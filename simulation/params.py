"""Parameter dataclasses for simulation configuration (mm, day, MPa, g/cm³)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any


@dataclass
class MaterialParams:
    """Material law: E(rho) = E0*(rho/rho_ref)^k with k blended trabecular→cortical."""

    # Young's modulus reference [MPa]
    E0: float = 7500.0

    # Poisson's ratio (isotropic)
    nu0: float = 0.3

    # Power-law exponent for trabecular bone
    n_trab: float = 2.0

    # Power-law exponent for cortical bone
    n_cort: float = 1.3

    # Transition zone: trabecular → cortical [g/cm³]
    rho_trab_max: float = 1.0
    rho_cort_min: float = 1.25

    # Anisotropy stiffness exponents (fabric-based)
    stiff_pE: float = 1.0  # Axial modulus exponent
    stiff_pG: float = 1.0  # Shear modulus exponent

    def validate(self) -> None:
        """Validate material parameter constraints."""
        if self.E0 <= 0:
            raise ValueError("Young's modulus E0 must be positive.")
        if not (-1.0 < self.nu0 < 0.5):
            raise ValueError("Poisson ratio nu0 must be in (-1, 0.5).")
        if self.n_trab <= 0 or self.n_cort <= 0:
            raise ValueError("Material exponents must satisfy n_trab>0 and n_cort>0.")
        if not (self.rho_trab_max < self.rho_cort_min):
            raise ValueError("Require rho_trab_max < rho_cort_min for smoothstep transition.")
        if self.stiff_pE < 0 or self.stiff_pG < 0:
            raise ValueError("stiff_pE and stiff_pG must be >= 0.")


@dataclass
class DensityParams:
    """Density evolution: ∂ρ/∂t = D_ρ∇²ρ + formation - resorption."""

    # Density bounds [g/cm³]
    rho_min: float = 0.1
    rho_max: float = 2.0

    # Initial density [g/cm³]
    rho0: float = 1.0

    # Reference density for nondimensionalization [g/cm³]
    rho_ref: float = 1.0

    # Formation rate gain [g/cm³/day]
    k_rho_form: float = 2e-02

    # Resorption rate gain [g/cm³/day]
    k_rho_resorb: float = 2e-02

    # Diffusion coefficient [mm²/day]
    D_rho: float = 2e-2

    # Tissue density used for a porosity proxy f = 1 - rho/rho_tissue
    rho_tissue: float = 2.0  # Fully mineralized matrix density [g/cm³]

    # Surface availability scaling
    surface_use: bool = True
    surface_A_min: float = 0.02  # Residual remodeling activity as ρ → ρ_tissue
    surface_S0: float = 1.0      # Specific-surface scale [1/mm] (half-saturation at S_v=S0)

    def validate(self) -> None:
        """Validate density parameter constraints."""
        if not (0.0 <= self.rho_min < self.rho_max):
            raise ValueError("rho_min/max must satisfy 0 <= rho_min < rho_max.")
        if not (self.rho_min <= self.rho0 <= self.rho_max):
            raise ValueError("rho0 must satisfy rho_min <= rho0 <= rho_max.")
        if self.k_rho_form < 0 or self.k_rho_resorb < 0:
            raise ValueError("k_rho_form and k_rho_resorb must be >= 0.")
        if self.D_rho < 0:
            raise ValueError("D_rho must be >= 0 (diffusion coefficient, mm²/day).")
        if self.rho_tissue <= 0:
            raise ValueError("rho_tissue must be > 0 (fully mineralized matrix density).")
        if self.surface_A_min < 0 or self.surface_A_min >= 1.0:
            raise ValueError("surface_A_min must be in [0, 1).")
        if self.surface_S0 <= 0:
            raise ValueError("surface_S0 must be > 0.")


@dataclass
class StimulusParams:
    """Stimulus evolution: diffusion + decay toward mechanostat drive."""

    # Power-mean exponent for multi-load SED averaging (1=mean; higher→peak-biased)
    stimulus_power_p: float = 4.0

    # Reference strain energy density [MPa]
    psi_ref: float = 0.01

    # Time constant [days]; τ_S=0 gives quasi-static stimulus
    stimulus_tau: float = 25.0

    # Diffusion coefficient [mm²/day]
    stimulus_D: float = 1.0

    # Maximum stimulus magnitude (dimensionless)
    stimulus_S_max: float = 1.0

    # Saturation width in tanh (dimensionless)
    stimulus_kappa: float = 0.5

    # Lazy-zone half-width (dimensionless)
    stimulus_delta0: float = 0.10

    def validate(self) -> None:
        """Validate stimulus parameter constraints."""
        if self.psi_ref <= 0:
            raise ValueError("Reference value psi_ref must be positive.")
        if self.stimulus_tau < 0:
            raise ValueError("stimulus_tau must be >= 0 (τ_S in days).")
        if self.stimulus_D < 0:
            raise ValueError("stimulus_D must be >= 0 (diffusion coefficient, mm²/day).")
        if self.stimulus_S_max <= 0:
            raise ValueError("stimulus_S_max must be > 0 (dimensionless cap on |S|).")
        if self.stimulus_kappa <= 0:
            raise ValueError("stimulus_kappa must be > 0 (dimensionless saturation width).")
        if self.stimulus_delta0 < 0:
            raise ValueError("stimulus_delta0 must be >= 0 (dimensionless lazy-zone half-width).")
        if self.stimulus_power_p < 1.0:
            raise ValueError("stimulus_power_p must be >= 1.0 (1=mean; larger biases toward peaks).")


@dataclass
class FabricParams:
    """Log-fabric tensor L: reaction-diffusion toward L_target(Q̄)."""

    # Time constant [days]
    fabric_tau: float = 50.0

    # Diffusion coefficient [mm²/day]
    fabric_D: float = 1.0

    # Coupling strength
    fabric_cA: float = 1.0

    # Power-law exponent for fabric eigenvalues
    fabric_gammaF: float = 1.0

    # Regularization for Q̄ eigenvalue computation [MPa²] (added as epsQ * I to Q̄)
    fabric_epsQ: float = 1e-12

    # Activity-gate width for near-isotropic loading (dimensionless)
    fabric_aniso_eps: float = 1e-4

    # Eigenvalue ratio bounds
    fabric_m_min: float = 0.2
    fabric_m_max: float = 5.0

    def validate(self) -> None:
        """Validate fabric parameter constraints."""
        if self.fabric_tau <= 0:
            raise ValueError("fabric_tau must be > 0.")
        if self.fabric_D < 0:
            raise ValueError("fabric_D must be >= 0.")
        if self.fabric_cA <= 0:
            raise ValueError("fabric_cA must be > 0.")
        if self.fabric_gammaF <= 0:
            raise ValueError("fabric_gammaF must be > 0.")
        if self.fabric_epsQ <= 0:
            raise ValueError("fabric_epsQ must be > 0.")
        if self.fabric_aniso_eps <= 0:
            raise ValueError("fabric_aniso_eps must be > 0.")
        if self.fabric_m_min <= 0:
            raise ValueError("fabric_m_min must be > 0.")
        if self.fabric_m_max <= self.fabric_m_min:
            raise ValueError("fabric_m_max must be > fabric_m_min.")


@dataclass
class SolverParams:
    """KSP and fixed-point solver settings."""

    # PETSc KSP solver type (default MINRES; CG is usually best for SPD operators)
    ksp_type: str = "minres"

    # PETSc preconditioner type (GAMG with Chebyshev/Jacobi smoothing for elasticity)
    pc_type: str = "gamg"

    # Relative tolerance for KSP
    ksp_rtol: float = 1e-7

    # Absolute tolerance for KSP
    ksp_atol: float = 1e-8

    # Maximum KSP iterations
    ksp_max_it: int = 200

    # Reuse preconditioner across solves
    ksp_reuse_pc: bool = False

    # Fixed-point acceleration type: 'anderson' or 'picard'
    accel_type: str = "anderson"

    # Anderson history size
    m: int = 5

    # Mixing parameter (relaxation)
    beta: float = 1.0

    # Relative Tikhonov regularization for Anderson (scaled by average Gram diagonal)
    lam: float = 1e-2

    # Safeguard tolerance (residual improvement threshold)
    gamma: float = 0.05

    # Enable safeguard with backtracking
    safeguard: bool = False

    # Maximum backtrack attempts
    backtrack_max: int = 5

    # Fixed-point convergence tolerance
    coupling_tol: float = 1e-6

    # Restart Anderson after k consecutive rejections
    restart_on_reject_k: int = 2

    # Restart on stall (ratio threshold)
    restart_on_stall: float = 1.10

    # Restart on ill-conditioning
    restart_on_cond: float = 1e12

    # Step size limit factor
    step_limit_factor: float = 2.0

    # Maximum sub-iterations per timestep
    max_subiters: int = 25

    # Minimum sub-iterations per timestep
    min_subiters: int = 2

    def validate(self) -> None:
        """Validate solver parameter constraints."""
        if self.accel_type not in ("anderson", "picard"):
            raise ValueError("accel_type must be 'anderson' or 'picard'.")
        if self.m < 0:
            raise ValueError("Anderson history m must be >= 0.")
        if self.coupling_tol <= 0:
            raise ValueError("coupling_tol must be > 0.")
        if self.max_subiters < self.min_subiters:
            raise ValueError("max_subiters must be >= min_subiters.")


@dataclass
class TimeParams:
    """Time stepping and adaptive control."""

    # Total simulation time [days]
    total_time: float = 500

    # Initial timestep [days]
    dt_initial: float = 25

    # Enable adaptive time stepping
    adaptive_dt: bool = False

    # Relative error tolerance for time stepping
    adaptive_rtol: float = 1e-2

    # Absolute error tolerance for time stepping
    adaptive_atol: float = 1e-3

    # Minimum timestep [days]
    dt_min: float = 1e-4

    # Maximum timestep [days]
    dt_max: float = 100.0

    # --- PI Controller parameters (Gustafsson) ---
    # Safety factor for step size adjustment
    pi_safety: float = 0.9

    # Maximum growth factor per step
    pi_growth_max: float = 5.0

    # Minimum shrink factor per step
    pi_shrink_min: float = 0.1

    # Exponent for simple rejection formula: h_new = h * (1/err)^(1/k_exp)
    pi_k_exp: float = 1.5

    # Proportional gain (error ratio term)
    pi_kp: float = 0.20

    # Integral gain (current error term)
    pi_ki: float = 0.40

    def validate(self) -> None:
        """Validate time stepping parameter constraints."""
        if self.total_time <= 0:
            raise ValueError("total_time must be > 0.")
        if self.dt_initial <= 0:
            raise ValueError("dt_initial must be > 0.")
        if self.dt_min <= 0 or self.dt_max <= 0:
            raise ValueError("dt_min and dt_max must be > 0.")
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max.")
        if self.adaptive_rtol <= 0 or self.adaptive_atol <= 0:
            raise ValueError("adaptive_rtol and adaptive_atol must be > 0.")
        # PI controller validation
        if not (0.0 < self.pi_safety <= 1.0):
            raise ValueError("pi_safety must be in (0, 1].")
        if self.pi_growth_max < 1.0:
            raise ValueError("pi_growth_max must be >= 1.0.")
        if not (0.0 < self.pi_shrink_min < 1.0):
            raise ValueError("pi_shrink_min must be in (0, 1).")
        if self.pi_k_exp <= 0:
            raise ValueError("pi_k_exp must be > 0.")
        if self.pi_kp < 0 or self.pi_ki < 0:
            raise ValueError("pi_kp and pi_ki must be >= 0.")


@dataclass
class NumericsParams:
    """Quadrature and smoothing settings."""

    # Quadrature degree for integration
    quadrature_degree: int = 4

    # Smoothing epsilon for C¹ approximations (absolute width, interpreted in the units of the smoothed quantity)
    smooth_eps: float = 1e-6

    def validate(self) -> None:
        """Validate numerics parameter constraints."""
        if self.quadrature_degree < 1:
            raise ValueError("quadrature_degree must be >= 1.")
        if self.smooth_eps <= 0:
            raise ValueError("smooth_eps must be > 0.")


@dataclass
class OutputParams:
    """Results directory and saving interval."""

    # Output interval (every N steps)
    saving_interval: int = 1

    # Results directory path
    results_dir: str = ".results"

    # Log file name (relative to results_dir)
    log_file: str = "simulation.log"

    def validate(self) -> None:
        """Validate output parameter constraints."""
        if self.saving_interval < 1:
            raise ValueError("saving_interval must be >= 1.")


@dataclass
class GeometryParams:
    """Facet tag IDs for boundary conditions."""

    # Facet tag for fixed boundary conditions (Dirichlet)
    fix_tag: int = 1

    # Facet tag for loading surface (Neumann)
    load_tag: int = 2

    def validate(self) -> None:
        """Validate geometry parameter constraints."""
        if self.fix_tag < 0:
            raise ValueError("fix_tag must be >= 0.")
        if self.load_tag < 0:
            raise ValueError("load_tag must be >= 0.")


def params_to_dict(params) -> dict[str, Any]:
    """Convert a params dataclass to a JSON-serializable dict."""
    result = {}
    for f in dataclass_fields(params):
        val = getattr(params, f.name)
        if isinstance(val, (int, float, bool, str, type(None))):
            result[f.name] = val
    return result


def load_default_params(json_path: str) -> dict[str, Any]:
    """Load parameters from JSON file.
    
    Args:
        json_path: Path to JSON config file (required, no default).
    
    Returns:
        Dict with param dataclass instances keyed by section name.
        Modify the returned params directly before calling create_config().
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist.
        ValueError: If required section or field is missing in JSON.
        
    Example:
        params = load_default_params("default_params_box.json")
        params["time"].total_time = 100.0
        params["density"].k_rho_form = 0.1
        cfg = create_config(domain, facet_tags, params)
    """
    import json
    from pathlib import Path
    
    path = Path(json_path)
    if not path.exists():
        # Try relative to this file's directory (for imports from subdirs)
        path = Path(__file__).parent.parent / json_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    
    with open(path) as f:
        data = json.load(f)
    
    # Map section names to dataclass types
    section_to_class = {
        "material": MaterialParams,
        "density": DensityParams,
        "stimulus": StimulusParams,
        "fabric": FabricParams,
        "solver": SolverParams,
        "time": TimeParams,
        "numerics": NumericsParams,
        "output": OutputParams,
        "geometry": GeometryParams,
    }
    
    result: dict[str, Any] = {}
    for section, cls in section_to_class.items():
        if section not in data:
            raise ValueError(f"Missing required section '{section}' in {json_path}")
        
        # Get expected fields from dataclass
        expected_fields = {f.name for f in dataclass_fields(cls)}
        provided_fields = set(data[section].keys())
        
        # Check for missing fields
        missing = expected_fields - provided_fields
        if missing:
            raise ValueError(
                f"Missing fields in '{section}' section of {json_path}: {sorted(missing)}"
            )
        
        # Check for unknown fields (warn but allow)
        unknown = provided_fields - expected_fields
        if unknown:
            import warnings
            warnings.warn(
                f"Unknown fields in '{section}' section of {json_path}: {sorted(unknown)}. "
                "These will be ignored."
            )
        
        # Create dataclass with only valid fields
        filtered = {k: v for k, v in data[section].items() if k in expected_fields}
        result[section] = cls(**filtered)
    
    # Pass through non-dataclass sections (like "box") as raw dicts
    for section in data:
        if section not in section_to_class and not section.startswith("_"):
            result[section] = data[section]
    
    return result


def create_config(domain, facet_tags, params: dict[str, Any]):
    """Create Config from loaded params dict.
    
    Args:
        domain: DOLFINx mesh.
        facet_tags: MeshTags for boundaries.
        params: Dict from load_default_params() - must contain all required sections.
    
    Returns:
        Config instance.
        
    Raises:
        KeyError: If required section is missing from params.
        
    Example:
        params = load_default_params("default_params_box.json")
        params["time"].total_time = 100.0
        params["output"].results_dir = ".results_box"
        cfg = create_config(domain, facet_tags, params)
    """
    # Import here to avoid circular import
    from simulation.config import Config
    
    required_sections = ["material", "density", "stimulus", "fabric", 
                         "solver", "time", "numerics", "output", "geometry"]
    missing = [s for s in required_sections if s not in params]
    if missing:
        raise KeyError(f"Missing required param sections: {missing}")
    
    return Config(
        domain=domain,
        facet_tags=facet_tags,
        material=params["material"],
        density=params["density"],
        stimulus=params["stimulus"],
        fabric=params["fabric"],
        solver=params["solver"],
        time=params["time"],
        numerics=params["numerics"],
        output=params["output"],
        geometry=params["geometry"],
    )
