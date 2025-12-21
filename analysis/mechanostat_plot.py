"""Visualization of mechanostat (stimulus + density evolution).

Reflects the actual StimulusSolver and DensitySolver implementations:

**Stimulus (S):**
- δ = (m - m_ref) / m_ref where m = ψ/ρ (specific SED)
- Lazy-zone gate: g = 1 - exp(-(|δ|/δ₀)²) suppresses small errors
- Drive = S_max · tanh(δ_eff / κ) with saturation
- PDE: τ·∂S/∂t + S = τ·D_S·∇²S + drive

**Density (ρ):**
- S₊ = max(S,0), S₋ = max(-S,0)
- Surface availability A(ρ) from Martin polynomial (trabecular) / sqrt-proxy (cortical)
- Formation: k_form · A · S₊ · (1 - ρ/ρ_max)
- Resorption: k_res · A · S₋ · (1 - ρ/ρ_min)
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import COLORS, smooth_max, smoothstep01, save_manuscript_figure


def set_modern_style():
    """Configure matplotlib for publication-quality look."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.frameon': False,
    })


# Default parameters (from StimulusParams, DensityParams, MaterialParams)
DEFAULT_STIMULUS_PARAMS = {
    'psi_ref': 0.01,           # Reference SED [MPa]
    'rho_ref': 1.0,            # Reference density [g/cm³]
    'stimulus_S_max': 1.0,     # Max stimulus magnitude
    'stimulus_kappa': 0.5,     # Saturation width
    'stimulus_delta0': 0.10,   # Lazy-zone half-width
    'stimulus_tau': 25.0,      # Time constant [days]
    'stimulus_D': 1.0,         # Diffusion coefficient [mm²/day]
}

DEFAULT_DENSITY_PARAMS = {
    'rho_min': 0.1,            # Min density [g/cm³]
    'rho_max': 2.0,            # Max density [g/cm³]
    'k_rho_form': 2e-02,       # Formation rate [g/cm³/day]
    'k_rho_resorb': 2e-02,     # Resorption rate [g/cm³/day]
    'D_rho': 2e-2,             # Diffusion coefficient [mm²/day]
    'rho_tissue': 2.0,         # Mineralized tissue density [g/cm³]
    'rho_trab_max': 1.0,       # Upper trabecular bound
    'rho_cort_min': 1.25,      # Lower cortical bound
    'surface_A_min': 0.02,     # Min surface availability
    'surface_S0': 1.0,         # Reference S_V [1/mm]
}


def compute_delta(psi: np.ndarray, rho: np.ndarray, **kwargs) -> np.ndarray:
    """Compute normalized deviation δ = (m - m_ref) / m_ref where m = ψ/ρ."""
    params = {**DEFAULT_STIMULUS_PARAMS, **kwargs}
    psi_ref = params['psi_ref']
    rho_ref = params['rho_ref']
    
    eps = 1e-6
    rho_safe = smooth_max(rho, 0.1, eps)
    m = psi / rho_safe
    m_ref = psi_ref / rho_ref
    delta = (m - m_ref) / m_ref
    return delta


def compute_lazy_zone_gate(delta: np.ndarray, delta0: float) -> np.ndarray:
    """Compute lazy-zone gate: g = 1 - exp(-(|δ|/δ₀)²)."""
    if delta0 <= 0:
        return np.ones_like(delta)
    eps = 1e-6
    delta_abs = np.sqrt(delta**2 + eps**2)
    gate = 1.0 - np.exp(-((delta_abs / delta0)**2))
    return gate


def compute_stimulus_drive(delta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute mechanostat drive = S_max · tanh(δ_eff / κ)."""
    params = {**DEFAULT_STIMULUS_PARAMS, **kwargs}
    S_max = params['stimulus_S_max']
    kappa = params['stimulus_kappa']
    delta0 = params['stimulus_delta0']
    
    gate = compute_lazy_zone_gate(delta, delta0)
    delta_eff = delta * gate
    drive = S_max * np.tanh(delta_eff / kappa)
    return drive


# =============================================================================
# Density Model
# =============================================================================

def compute_surface_availability(rho: np.ndarray, **kwargs) -> np.ndarray:
    """Compute surface availability A(ρ) from Martin polynomial."""
    params = {**DEFAULT_DENSITY_PARAMS, **kwargs}
    rho_tissue = params['rho_tissue']
    rho_trab_max = params['rho_trab_max']
    rho_cort_min = params['rho_cort_min']
    A_min = params['surface_A_min']
    S0 = params['surface_S0']
    eps = 1e-6
    
    # Porosity fraction
    f = np.clip(1.0 - rho / rho_tissue, 0.0, 1.0)
    
    # Trabecular (Martin) specific surface [1/mm]
    S_trab = 32.3*f - 93.9*f**2 + 134.0*f**3 - 101.0*f**4 + 28.8*f**5
    
    # Cortical proxy: scale to match at transition
    f_tr = max(1.0 - rho_trab_max / rho_tissue, 1e-6)
    S_trab_tr = 32.3*f_tr - 93.9*f_tr**2 + 134.0*f_tr**3 - 101.0*f_tr**4 + 28.8*f_tr**5
    surface_cort_scale = S_trab_tr / (f_tr**0.5)
    S_cort = surface_cort_scale * np.sqrt(f + eps)
    
    # Blend trabecular → cortical
    denom = max(rho_cort_min - rho_trab_max, 1e-12)
    t = (rho - rho_trab_max) / denom
    w_cort = smoothstep01(t)
    
    S_v = (1.0 - w_cort) * S_trab + w_cort * S_cort
    S_v = np.maximum(S_v, 0.0)
    
    # Surface availability
    A_surf = A_min + (1.0 - A_min) * (S_v / (S_v + S0))
    return A_surf, S_v


def compute_density_rate(S: np.ndarray, rho: float, **kwargs) -> tuple:
    """Compute ∂ρ/∂t from stimulus and current density.
    
    Rate = k_form · A · S₊ · (1 - ρ/ρ_max) - k_res · A · S₋ · (1 - ρ/ρ_min)
    """
    params = {**DEFAULT_DENSITY_PARAMS, **kwargs}
    k_form = params['k_rho_form']
    k_res = params['k_rho_resorb']
    rho_min = params['rho_min']
    rho_max = params['rho_max']
    eps = 1e-6
    
    # Get surface availability
    A_surf, _ = compute_surface_availability(np.atleast_1d(rho), **kwargs)
    A = A_surf[0] if np.ndim(rho) == 0 else A_surf
    
    # Smooth positive/negative parts
    S_pos = smooth_max(S, 0.0, eps)
    S_neg = smooth_max(-S, 0.0, eps)
    
    # Formation and resorption rates
    formation = k_form * A * S_pos * (1.0 - rho / rho_max)
    resorption = k_res * A * S_neg * (1.0 - rho / rho_min)
    
    d_rho = formation - resorption
    return d_rho, formation, resorption, S_pos, S_neg


# =============================================================================
# Plot Functions
# =============================================================================

def plot_stimulus_drive(ax):
    """(a) Mechanostat drive vs δ with lazy-zone and saturation."""
    delta = np.linspace(-3.0, 3.0, 500)
    
    # Different delta0 values
    delta0_values = [0.0, 0.1, 0.3]
    colors = [COLORS['grey'], COLORS['blue'], COLORS['cyan']]
    labels = [r'$\delta_0=0$ (no lazy zone)', r'$\delta_0=0.1$ (default)', r'$\delta_0=0.3$']
    
    for d0, c, lbl in zip(delta0_values, colors, labels):
        drive = compute_stimulus_drive(delta, stimulus_delta0=d0)
        ax.plot(delta, drive, color=c, linewidth=2, label=lbl)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    ax.axhline(1.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    ax.axhline(-1.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    
    ax.set_title(r'(a) Mechanostat Drive: $S_{max} \cdot \tanh(\delta_{eff}/\kappa)$', 
                 loc='left', fontweight='bold')
    ax.set_xlabel(r'Normalized deviation $\delta = (m-m_{ref})/m_{ref}$')
    ax.set_ylabel(r'Drive $\rightarrow S$')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=8)
    
    ax.text(2.0, 0.7, 'Formation', ha='center', color=COLORS['teal'], fontweight='bold')
    ax.text(-2.0, -0.7, 'Resorption', ha='center', color=COLORS['red'], fontweight='bold')


def plot_lazy_zone(ax):
    """(b) Lazy-zone gate function."""
    delta = np.linspace(-1.0, 1.0, 500)
    
    delta0_values = [0.05, 0.1, 0.2, 0.5]
    colors = [COLORS['black'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    
    for d0, c in zip(delta0_values, colors):
        gate = compute_lazy_zone_gate(delta, d0)
        ax.plot(delta, gate, color=c, linewidth=2, label=rf'$\delta_0={d0}$')
    
    ax.axhline(1.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    
    ax.set_title(r'(b) Lazy-Zone Gate: $g = 1 - e^{-(\delta/\delta_0)^2}$', 
                 loc='left', fontweight='bold')
    ax.set_xlabel(r'Deviation $\delta$')
    ax.set_ylabel(r'Gate $g$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)


def plot_saturation_kappa(ax):
    """(c) Effect of saturation width κ."""
    delta = np.linspace(-3.0, 3.0, 500)
    
    kappa_values = [0.2, 0.5, 1.0, 2.0]
    colors = [COLORS['black'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    
    for k, c in zip(kappa_values, colors):
        drive = compute_stimulus_drive(delta, stimulus_kappa=k, stimulus_delta0=0.0)
        ax.plot(delta, drive, color=c, linewidth=2, label=rf'$\kappa={k}$')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    
    ax.set_title(r'(c) Saturation Width $\kappa$ Effect', loc='left', fontweight='bold')
    ax.set_xlabel(r'Deviation $\delta$')
    ax.set_ylabel(r'Drive')
    ax.legend(fontsize=8)


def plot_surface_availability(ax):
    """(d) Surface availability A(ρ) from Martin polynomial."""
    rho = np.linspace(0.1, 2.0, 500)
    A_surf, S_v = compute_surface_availability(rho)
    
    ax2 = ax.twinx()
    
    l1, = ax.plot(rho, A_surf, color=COLORS['blue'], linewidth=2.5, label='$A(\\rho)$')
    l2, = ax2.plot(rho, S_v, color=COLORS['orange'], linewidth=2, linestyle='--', 
                   label='$S_V$ [1/mm]')
    
    # Mark transition zone
    rho_trab_max = DEFAULT_DENSITY_PARAMS['rho_trab_max']
    rho_cort_min = DEFAULT_DENSITY_PARAMS['rho_cort_min']
    ax.axvspan(rho_trab_max, rho_cort_min, alpha=0.15, color=COLORS['grey'])
    
    ax.set_title(r'(d) Surface Availability $A(\rho)$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r'Availability $A$', color=COLORS['blue'])
    ax2.set_ylabel(r'Specific surface $S_V$ [1/mm]', color=COLORS['orange'])
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 1.1)
    
    ax.legend([l1, l2], ['$A(\\rho)$', '$S_V$ [1/mm]'], loc='center right', fontsize=8)


def plot_density_rate(ax):
    """(e) Density rate ∂ρ/∂t vs stimulus S for different ρ."""
    S = np.linspace(-1.0, 1.0, 500)
    rho_values = [0.3, 0.7, 1.2, 1.8]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    
    for r, c in zip(rho_values, colors):
        d_rho, _, _, _, _ = compute_density_rate(S, r)
        ax.plot(S, d_rho * 1000, color=c, linewidth=2, label=rf'$\rho={r}$')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    
    ax.set_title(r'(e) Density Rate $\dot{\rho}$ vs $S$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$')
    ax.set_ylabel(r'Rate $\dot{\rho}$ [$\times 10^{-3}$ g/cm³/day]')
    ax.legend(fontsize=8)
    
    ax.text(0.6, 10, 'Formation', ha='center', color=COLORS['teal'], fontweight='bold', fontsize=9)
    ax.text(-0.6, -10, 'Resorption', ha='center', color=COLORS['red'], fontweight='bold', fontsize=9)


def plot_rate_components(ax):
    """(f) Formation vs resorption rate components."""
    S = np.linspace(-1.0, 1.0, 500)
    rho = 0.7
    
    d_rho, formation, resorption, S_pos, S_neg = compute_density_rate(S, rho)
    
    ax.plot(S, formation * 1000, color=COLORS['teal'], linewidth=2, label='Formation')
    ax.plot(S, -resorption * 1000, color=COLORS['red'], linewidth=2, label='Resorption')
    ax.plot(S, d_rho * 1000, color=COLORS['blue'], linewidth=2.5, linestyle='--', label='Net rate')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    
    ax.set_title(rf'(f) Rate Components ($\rho={rho}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$')
    ax.set_ylabel(r'Rate [$\times 10^{-3}$ g/cm³/day]')
    ax.legend(fontsize=8)


def generate_mechanostat_plot():
    """Generate comprehensive 6-panel mechanostat visualization."""
    set_modern_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    plot_stimulus_drive(axes[0, 0])
    plot_lazy_zone(axes[0, 1])
    plot_saturation_kappa(axes[0, 2])
    plot_surface_availability(axes[1, 0])
    plot_density_rate(axes[1, 1])
    plot_rate_components(axes[1, 2])
    
    plt.tight_layout()
    save_manuscript_figure(fig, 'mechanostat_law')


if __name__ == "__main__":
    generate_mechanostat_plot()
