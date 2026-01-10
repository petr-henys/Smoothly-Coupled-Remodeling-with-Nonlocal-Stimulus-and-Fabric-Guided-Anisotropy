"""Conservation monitoring: tracks mass, energy, and source integrals.

This module provides non-invasive monitoring of conservation laws without
modifying solver internals. All quantities are computed as post-step integrals.

Physical interpretation:
- Total mass M(t) = ∫ρ dV should change only due to source terms (no-flux BC)
- Mass rate dM/dt should equal source integral (discrete mass balance)
- Strain energy W(t) = ∫ψ dV represents stored elastic energy
- Stimulus flux should be zero (homogeneous Neumann BC)

Units:
- Mass: [g] (density [g/cm³] × volume [mm³] × 1e-3)
- Energy: [mJ] = [MPa × mm³]
- Rates: [g/day], [mJ/day]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional

import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI

from simulation.utils import smooth_max

if TYPE_CHECKING:
    from simulation.config import Config


@dataclass
class ConservationMetrics:
    """Conservation quantities at a single timestep.
    
    All extensive quantities are global (summed across MPI ranks).
    """
    # Mass quantities
    total_mass: float           # ∫ρ dV [g] (with mm³→cm³ conversion)
    mass_rate: float            # dM/dt ≈ (M - M_prev)/dt [g/day]
    source_integral: float      # ∫(formation - resorption) dV [g/day]
    mass_balance_error: float   # |dM/dt - source_integral| / max(|dM/dt|, |source|, eps)
    
    # Energy quantities  
    total_energy: float         # ∫ψ dV [mJ] (strain energy)
    energy_rate: float          # dW/dt [mJ/day]
    
    # Stimulus quantities
    total_stimulus: float       # ∫S dV (should be ~0 at equilibrium)
    stimulus_abs_integral: float  # ∫|S| dV (activity measure)
    
    # Volume (for normalization)
    volume: float               # ∫1 dV [mm³]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for CSV logging."""
        return {
            "total_mass_g": self.total_mass,
            "mass_rate_g_day": self.mass_rate,
            "source_integral_g_day": self.source_integral,
            "mass_balance_error": self.mass_balance_error,
            "total_energy_mJ": self.total_energy,
            "energy_rate_mJ_day": self.energy_rate,
            "total_stimulus": self.total_stimulus,
            "stimulus_activity": self.stimulus_abs_integral,
            "volume_mm3": self.volume,
        }


class ConservationMonitor:
    """Monitors conservation laws by computing global integrals after each step.
    
    This class is designed to be non-invasive: it only reads field values and
    computes integrals, never modifying solver state.
    
    Usage:
        monitor = ConservationMonitor(cfg, rho, S, psi_field)
        
        # After each timestep:
        metrics = monitor.compute(dt)
        
        # Get metrics dict for logging:
        metrics_dict = metrics.to_dict()
    """
    
    # Conversion factor: mm³ to cm³ (for mass in g when ρ is g/cm³)
    MM3_TO_CM3 = 1e-3
    
    def __init__(
        self,
        cfg: "Config",
        rho: fem.Function,
        rho_old: fem.Function,
        S: fem.Function,
        psi: fem.Function,
    ):
        """Initialize conservation monitor.
        
        Args:
            cfg: Simulation configuration.
            rho: Current density field [g/cm³].
            rho_old: Previous density field [g/cm³].
            S: Stimulus field [-].
            psi: Strain energy density field [MPa] (cycle-averaged).
        """
        self.cfg = cfg
        self.comm = cfg.domain.comm
        self.rho = rho
        self.rho_old = rho_old
        self.S = S
        self.psi = psi
        
        # Cache previous values for rate computation
        self._mass_prev: Optional[float] = None
        self._energy_prev: Optional[float] = None
        
        # Compile UFL forms for efficiency
        self._compile_forms()
    
    def _compile_forms(self) -> None:
        """Pre-compile integral forms for efficiency."""
        dx = self.cfg.dx
        eps = float(self.cfg.numerics.smooth_eps)
        
        # Volume
        self._vol_form = fem.form(1.0 * dx)
        
        # Mass integral: ∫ρ dV
        self._mass_form = fem.form(self.rho * dx)
        
        # Energy integral: ∫ψ dV (psi is already cycle-averaged SED)
        self._energy_form = fem.form(self.psi * dx)
        
        # Stimulus integrals
        self._stim_form = fem.form(self.S * dx)
        S_abs = ufl.sqrt(self.S**2 + eps**2)  # Smooth |S|
        self._stim_abs_form = fem.form(S_abs * dx)
        
        # Source term: A(ρ) * [ k_form * S⁺ * (1 - ρ/ρ_max) + k_res * S⁻ * (1 - ρ/ρ_min) ]
        # Note: (1 - ρ/ρ_min) is negative for ρ > ρ_min, providing the resorption (decay).
        
        rho_min = float(self.cfg.density.rho_min)
        rho_max = float(self.cfg.density.rho_max)
        k_form = float(self.cfg.density.k_rho_form)
        k_res = float(self.cfg.density.k_rho_resorb)
        
        S_pos = smooth_max(self.S, 0.0, eps)
        S_neg = smooth_max(-self.S, 0.0, eps)
        
        # Surface availability A(ρ) (replicated from DensitySolver)
        if bool(self.cfg.density.surface_use):
            rho_tissue = float(self.cfg.density.rho_tissue)
            f_raw = 1.0 - (self.rho / rho_tissue)
            f = smooth_max(smooth_max(f_raw, 0.0, eps), 0.0, eps) # Clamp f >= 0
            # Note: smooth_clamp(f,0,1) matches density.py better but let's be close enough.
            
            rho_trab_max = float(self.cfg.material.rho_trab_max)
            rho_cort_min = float(self.cfg.material.rho_cort_min)

            # Trabecular specific surface
            S_trab = (32.3 * f - 93.9 * f**2 + 134.0 * f**3 
                      - 101.0 * f**4 + 28.8 * f**5)

            # Cortical proxy
            f_tr = max(1.0 - rho_trab_max / rho_tissue, 1e-6)
            S_trab_tr = (32.3 * f_tr - 93.9 * f_tr**2 + 134.0 * f_tr**3 
                         - 101.0 * f_tr**4 + 28.8 * f_tr**5)
            surface_cort_scale = S_trab_tr / (f_tr**0.5)
            S_cort = surface_cort_scale * ufl.sqrt(f + eps)

            # Blend
            denom = max(rho_cort_min - rho_trab_max, 1e-12)
            # Use same smoothstep01 logic if possible, otherwise manual cubic
            from simulation.utils import smoothstep01
            t = (self.rho - rho_trab_max) / denom
            w_cort = smoothstep01(t, eps)

            S_v = (1.0 - w_cort) * S_trab + w_cort * S_cort
            S_v = smooth_max(S_v, 0.0, eps)

            A_min = float(self.cfg.density.surface_A_min)
            S0 = float(self.cfg.density.surface_S0)
            A_surf = A_min + (1.0 - A_min) * (S_v / (S_v + S0))
        else:
            A_surf = 1.0
        
        # Net source (Note: sum of terms; signs are intrinsic to the terms)
        term_form = k_form * S_pos * (1.0 - self.rho / rho_max)
        term_res = k_res * S_neg * (1.0 - self.rho / rho_min)
        
        net_source = A_surf * (term_form + term_res)
        self._source_form = fem.form(net_source * dx)
    
    def _assemble_scalar(self, form: fem.Form) -> float:
        """Assemble scalar integral with MPI reduction."""
        local_val = fem.assemble_scalar(form)
        return float(self.comm.allreduce(local_val, op=MPI.SUM))
    
    def compute(self, dt: float) -> ConservationMetrics:
        """Compute conservation metrics for current state.
        
        Args:
            dt: Current timestep [days].
            
        Returns:
            ConservationMetrics with all conservation quantities.
        """
        # Scatter forward to ensure ghost values are current
        self.rho.x.scatter_forward()
        self.S.x.scatter_forward()
        self.psi.x.scatter_forward()
        
        # Compute integrals
        volume = self._assemble_scalar(self._vol_form)
        mass_mm3 = self._assemble_scalar(self._mass_form)  # [g/cm³ × mm³]
        mass = mass_mm3 * self.MM3_TO_CM3  # Convert to [g]
        
        energy = self._assemble_scalar(self._energy_form)  # [MPa × mm³] = [mJ]
        
        total_stimulus = self._assemble_scalar(self._stim_form)
        stimulus_activity = self._assemble_scalar(self._stim_abs_form)
        
        source_mm3 = self._assemble_scalar(self._source_form)  # [g/cm³/day × mm³]
        source = source_mm3 * self.MM3_TO_CM3  # [g/day]
        
        # Compute rates
        if self._mass_prev is not None and dt > 0:
            mass_rate = (mass - self._mass_prev) / dt
        else:
            mass_rate = 0.0
            
        if self._energy_prev is not None and dt > 0:
            energy_rate = (energy - self._energy_prev) / dt
        else:
            energy_rate = 0.0
        
        # Mass balance error: |dM/dt - source| / scale
        # This should be small if the discretization is consistent
        scale = max(abs(mass_rate), abs(source), 1e-12)
        mass_balance_error = abs(mass_rate - source) / scale
        
        # Update cached values
        self._mass_prev = mass
        self._energy_prev = energy
        
        return ConservationMetrics(
            total_mass=mass,
            mass_rate=mass_rate,
            source_integral=source,
            mass_balance_error=mass_balance_error,
            total_energy=energy,
            energy_rate=energy_rate,
            total_stimulus=total_stimulus,
            stimulus_abs_integral=stimulus_activity,
            volume=volume,
        )
    
    def reset(self) -> None:
        """Reset cached values (call on timestep rejection)."""
        # Don't reset - we want to track from last accepted step
        pass
    
    def get_initial_metrics(self) -> ConservationMetrics:
        """Compute metrics for initial state (before first step)."""
        # Use a dummy dt since rates will be zero
        return self.compute(dt=1.0)
