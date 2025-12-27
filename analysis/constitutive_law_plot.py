"""Visualization of density-dependent constitutive law.

Reflects the actual MechanicsSolver implementation:
- E(ρ) = E₀ (ρ/ρ_ref)^k with smooth_max clamping at ρ_min
- Exponent k blends from n_trab → n_cort via cubic smoothstep in [ρ_trab_max, ρ_cort_min]
- Default: E0=7500 MPa, n_trab=2.0, n_cort=1.3, transition ρ∈[1.0, 1.25] g/cm³
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import COLORS, smooth_max, smoothstep01, save_manuscript_figure, apply_style


# Default material parameters (from MaterialParams and DensityParams)
DEFAULT_PARAMS = {
    'E0': 7500.0,           # Reference Young's modulus [MPa]
    'nu0': 0.3,             # Poisson ratio
    'n_trab': 2.0,          # Trabecular bone exponent
    'n_cort': 1.3,          # Cortical bone exponent
    'rho_trab_max': 1.0,    # Upper bound of trabecular zone [g/cm³]
    'rho_cort_min': 1.25,   # Lower bound of cortical zone [g/cm³]
    'rho_min': 0.1,         # Minimum density (clamped) [g/cm³]
    'rho_max': 2.0,         # Maximum density [g/cm³]
    'rho_ref': 1.0,         # Reference density for scaling [g/cm³]
    'smooth_eps': 1e-6,     # Smoothing parameter
}


def calculate_exponent(rho: np.ndarray, **kwargs) -> np.ndarray:
    """Calculate blended exponent k(ρ) that transitions from n_trab to n_cort.
    
    k = n_trab * (1 - w) + n_cort * w
    where w = smoothstep01((ρ - ρ_trab_max) / (ρ_cort_min - ρ_trab_max))
    """
    params = {**DEFAULT_PARAMS, **kwargs}
    
    n_trab = params['n_trab']
    n_cort = params['n_cort']
    rho_trab_max = params['rho_trab_max']
    rho_cort_min = params['rho_cort_min']
    rho_min = params['rho_min']
    smooth_eps = params['smooth_eps']
    
    # Effective density (clamped at rho_min)
    rho_eff = smooth_max(rho, rho_min, smooth_eps)
    
    # Blend weight
    denom = rho_cort_min - rho_trab_max
    t = (rho_eff - rho_trab_max) / denom
    w = smoothstep01(t)
    
    # Blended exponent
    k = n_trab * (1.0 - w) + n_cort * w
    return k


def calculate_E(rho: np.ndarray, **kwargs) -> np.ndarray:
    """Calculate Young's modulus E(ρ) using blended power-law.
    
    E = E₀ (ρ_eff / ρ_ref)^k
    where k blends from n_trab to n_cort in the transition zone.
    """
    params = {**DEFAULT_PARAMS, **kwargs}
    
    E0 = params['E0']
    rho_ref = params['rho_ref']
    rho_min = params['rho_min']
    smooth_eps = params['smooth_eps']
    
    # Effective density
    rho_eff = smooth_max(rho, rho_min, smooth_eps)
    rho_rel = rho_eff / rho_ref
    
    # Blended exponent
    k = calculate_exponent(rho, **kwargs)
    
    # Young's modulus
    E = E0 * (rho_rel ** k)
    return E


def plot_constitutive_law(ax):
    """Plot E(ρ) with blended exponent showing trabecular→cortical transition."""
    rho = np.linspace(0.05, 2.0, 500)
    E = calculate_E(rho)
    
    # Main curve
    ax.plot(rho, E / 1000, color=COLORS['blue'], linewidth=2.5, label='Blended model')
    
    # Pure power laws for comparison
    E0 = DEFAULT_PARAMS['E0']
    rho_ref = DEFAULT_PARAMS['rho_ref']
    n_trab = DEFAULT_PARAMS['n_trab']
    n_cort = DEFAULT_PARAMS['n_cort']
    
    E_trab = E0 * (rho / rho_ref) ** n_trab
    E_cort = E0 * (rho / rho_ref) ** n_cort
    
    ax.plot(rho, E_trab / 1000, color=COLORS['orange'], linestyle='--', alpha=0.7, 
            label=f'Trabecular ($n={n_trab}$)')
    ax.plot(rho, E_cort / 1000, color=COLORS['teal'], linestyle='--', alpha=0.7,
            label=f'Cortical ($n={n_cort}$)')
    
    # Mark transition zone
    rho_trab_max = DEFAULT_PARAMS['rho_trab_max']
    rho_cort_min = DEFAULT_PARAMS['rho_cort_min']
    ax.axvspan(rho_trab_max, rho_cort_min, alpha=0.15, color=COLORS['grey'],
               label='Transition zone')
    
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r"Young's Modulus $E$ [GPa]")
    ax.set_title(r'(a) Stiffness-Density: $E = E_0 (\rho/\rho_{ref})^{k(\rho)}$', 
                 loc='left', fontweight='bold')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)


def plot_exponent_blending(ax):
    """Show how exponent k transitions from n_trab to n_cort."""
    rho = np.linspace(0.05, 2.0, 500)
    k = calculate_exponent(rho)
    
    ax.plot(rho, k, color=COLORS['blue'], linewidth=2.5)
    
    # Reference lines
    n_trab = DEFAULT_PARAMS['n_trab']
    n_cort = DEFAULT_PARAMS['n_cort']
    rho_trab_max = DEFAULT_PARAMS['rho_trab_max']
    rho_cort_min = DEFAULT_PARAMS['rho_cort_min']
    
    ax.axhline(n_trab, color=COLORS['orange'], linestyle='--', alpha=0.7,
               label=f'$n_{{trab}}={n_trab}$')
    ax.axhline(n_cort, color=COLORS['teal'], linestyle='--', alpha=0.7,
               label=f'$n_{{cort}}={n_cort}$')
    ax.axvspan(rho_trab_max, rho_cort_min, alpha=0.15, color=COLORS['grey'])
    
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r'Exponent $k$')
    ax.set_title(r'(b) Smoothstep Exponent Blending', loc='left', fontweight='bold')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(1.0, 2.5)
    ax.legend(fontsize=8)


def plot_exponent_comparison(ax):
    """Compare different n_trab and n_cort combinations."""
    rho = np.linspace(0.05, 2.0, 500)
    
    configs = [
        {'n_trab': 2.0, 'n_cort': 1.3, 'label': 'Default (2.0→1.3)', 'color': COLORS['blue']},
        {'n_trab': 2.5, 'n_cort': 1.5, 'label': 'Higher (2.5→1.5)', 'color': COLORS['red']},
        {'n_trab': 1.5, 'n_cort': 1.0, 'label': 'Lower (1.5→1.0)', 'color': COLORS['teal']},
        {'n_trab': 2.0, 'n_cort': 2.0, 'label': 'Constant (2.0→2.0)', 'color': COLORS['grey']},
    ]
    
    for cfg in configs:
        E = calculate_E(rho, n_trab=cfg['n_trab'], n_cort=cfg['n_cort'])
        ax.plot(rho, E / 1000, color=cfg['color'], linewidth=2, label=cfg['label'])
    
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r"Young's Modulus $E$ [GPa]")
    ax.set_title(r'(c) Effect of Exponent Choice', loc='left', fontweight='bold')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)


def plot_transition_zone(ax):
    """Show effect of different transition zone widths."""
    rho = np.linspace(0.05, 2.0, 500)
    
    configs = [
        {'rho_trab_max': 0.8, 'rho_cort_min': 1.4, 'label': 'Wide (0.8–1.4)', 'color': COLORS['cyan']},
        {'rho_trab_max': 1.0, 'rho_cort_min': 1.25, 'label': 'Default (1.0–1.25)', 'color': COLORS['blue']},
        {'rho_trab_max': 1.05, 'rho_cort_min': 1.15, 'label': 'Narrow (1.05–1.15)', 'color': COLORS['red']},
    ]
    
    for cfg in configs:
        E = calculate_E(rho, rho_trab_max=cfg['rho_trab_max'], rho_cort_min=cfg['rho_cort_min'])
        ax.plot(rho, E / 1000, color=cfg['color'], linewidth=2, label=cfg['label'])
    
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r"Young's Modulus $E$ [GPa]")
    ax.set_title(r'(d) Transition Zone Width Effect', loc='left', fontweight='bold')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)


def plot_smooth_clamping(ax):
    """Show smooth_max clamping behavior at ρ_min."""
    rho = np.linspace(-0.1, 0.5, 500)
    rho_min = DEFAULT_PARAMS['rho_min']
    
    # Different smoothing parameters
    eps_values = [1e-6, 1e-3, 1e-2, 5e-2]
    colors = [COLORS['black'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    
    for eps, c in zip(eps_values, colors):
        rho_eff = smooth_max(rho, rho_min, eps)
        ax.plot(rho, rho_eff, color=c, linewidth=2, label=rf'$\epsilon={eps}$')
    
    # Reference: hard clamp
    rho_hard = np.maximum(rho, rho_min)
    ax.plot(rho, rho_hard, color=COLORS['grey'], linestyle='--', alpha=0.7, label='Hard clamp')
    
    ax.axhline(rho_min, color=COLORS['grey'], linestyle=':', alpha=0.5)
    ax.axvline(rho_min, color=COLORS['grey'], linestyle=':', alpha=0.5)
    
    ax.set_xlabel(r'Input $\rho$ [g/cm³]')
    ax.set_ylabel(r'Effective $\rho_{eff}$ [g/cm³]')
    ax.set_title(r'(e) Smooth Clamping at $\rho_{min}$', loc='left', fontweight='bold')
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8, loc='lower right')


def plot_log_log_view(ax):
    """Log-log plot showing power-law behavior."""
    rho = np.linspace(0.1, 2.0, 500)
    E = calculate_E(rho)
    
    ax.loglog(rho, E, color=COLORS['blue'], linewidth=2.5, label='Blended model')
    
    # Pure power laws
    E0 = DEFAULT_PARAMS['E0']
    rho_ref = DEFAULT_PARAMS['rho_ref']
    n_trab = DEFAULT_PARAMS['n_trab']
    n_cort = DEFAULT_PARAMS['n_cort']
    
    E_trab = E0 * (rho / rho_ref) ** n_trab
    E_cort = E0 * (rho / rho_ref) ** n_cort
    
    ax.loglog(rho, E_trab, color=COLORS['orange'], linestyle='--', alpha=0.7,
              label=f'Slope $n={n_trab}$')
    ax.loglog(rho, E_cort, color=COLORS['teal'], linestyle='--', alpha=0.7,
              label=f'Slope $n={n_cort}$')
    
    # Mark transition zone
    rho_trab_max = DEFAULT_PARAMS['rho_trab_max']
    rho_cort_min = DEFAULT_PARAMS['rho_cort_min']
    ax.axvspan(rho_trab_max, rho_cort_min, alpha=0.15, color=COLORS['grey'])
    
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r"Young's Modulus $E$ [MPa]")
    ax.set_title(r'(f) Log-Log View (Power-Law Slopes)', loc='left', fontweight='bold')
    ax.legend(fontsize=8)


def generate_constitutive_plot():
    """Generate comprehensive 6-panel constitutive law visualization."""
    apply_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    plot_constitutive_law(axes[0, 0])
    plot_exponent_blending(axes[0, 1])
    plot_exponent_comparison(axes[0, 2])
    plot_transition_zone(axes[1, 0])
    plot_smooth_clamping(axes[1, 1])
    plot_log_log_view(axes[1, 2])
    
    plt.tight_layout()
    save_manuscript_figure(fig, 'constitutive_law')


if __name__ == "__main__":
    generate_constitutive_plot()
