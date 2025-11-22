import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
OUTPUT_DIR = "/mnt/pracovni/Active_projects/GACR_BoneMorphologyModeling/remodeller/results/constitutive_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_modern_style():
    """Configure matplotlib for a modern, publication-quality look."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.color': '#333333',
        'grid.linestyle': '--',
        
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.frameon': False,
    })

# Color palette
COLORS = {
    'blue': '#0077BB',
    'cyan': '#33BBEE',
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
    'black': '#000000'
}

def smooth_max(x, xmin, eps):
    """Numpy implementation of smooth_max."""
    dx = x - xmin
    return xmin + 0.5 * (dx + np.sqrt(dx * dx + eps * eps))

def calculate_constitutive(rho_array, **kwargs):
    """
    Calculate E and n for a range of density values based on the constitutive law.
    
    Parameters:
    -----------
    rho_array : np.ndarray
        Array of density values.
    kwargs : dict
        Override default parameters.
        
    Returns:
    --------
    E : np.ndarray
        Young's modulus [MPa].
    n : np.ndarray
        Exponent n(rho).
    """
    # Default parameters from Config
    params = {
        'E0': 15e3,
        'n_trab': 2.0,
        'n_cort': 1.2,
        'rho_trab_max': 0.6,
        'rho_cort_min': 0.9,
        'rho_min': 0.1,
        'smooth_eps': 5e-7
    }
    params.update(kwargs)
    
    E0 = params['E0']
    n_trab = params['n_trab']
    n_cort = params['n_cort']
    rho_trab_max = params['rho_trab_max']
    rho_cort_min = params['rho_cort_min']
    rho_min = params['rho_min']
    smooth_eps = params['smooth_eps']

    # Apply smooth_max to density (clamping at rho_min)
    rho_eff = smooth_max(rho_array, rho_min, smooth_eps)
    
    # Calculate transition factor w
    # s_raw = (rho - rho1) / (rho2 - rho1)
    if rho_cort_min <= rho_trab_max:
        # Fallback if thresholds are invalid/crossed
        n_eff = np.full_like(rho_array, n_trab)
    else:
        s_raw = (rho_eff - rho_trab_max) / (rho_cort_min - rho_trab_max)
        
        # Clamp s to [0, 1] smoothly
        s0 = smooth_max(s_raw, 0.0, smooth_eps)
        s1 = 1.0 - smooth_max(1.0 - s0, 0.0, smooth_eps)
        
        # Cubic smoothstep
        w = 3.0 * s1**2 - 2.0 * s1**3
        
        # Interpolate exponent
        n_eff = (1.0 - w) * n_trab + w * n_cort

    # Calculate E
    E = E0 * (rho_eff ** n_eff)
    
    return E, n_eff

def plot_base_constitutive(ax1, ax2):
    """Plot the baseline constitutive law (E and n vs rho)."""
    rho = np.linspace(0, 1.2, 1000)
    E, n = calculate_constitutive(rho)
    
    # Plot E(rho)
    ax1.plot(rho, E, color=COLORS['blue'], linewidth=2.5, label='Model')
    ax1.set_xlabel(r'Density $\rho$ [-]')
    ax1.set_ylabel(r"Young's Modulus $E$ [MPa]")
    ax1.set_title(r'(a) Stiffness-Density Relationship $E(\rho)$', loc='left', fontweight='bold')
    
    # Add reference lines for pure power laws
    E_n2 = 15e3 * rho**2
    E_n3 = 15e3 * rho**3
    ax1.plot(rho, E_n2, color=COLORS['grey'], linestyle='--', alpha=0.6, label=r'$E \propto \rho^2$')
    ax1.plot(rho, E_n3, color=COLORS['grey'], linestyle=':', alpha=0.6, label=r'$E \propto \rho^3$')
    ax1.legend()
    
    # Plot n(rho)
    ax2.plot(rho, n, color=COLORS['red'], linewidth=2.5)
    ax2.set_xlabel(r'Density $\rho$ [-]')
    ax2.set_ylabel(r'Exponent $n(\rho)$ [-]')
    ax2.set_title(r'(b) Variable Exponent $n(\rho)$', loc='left', fontweight='bold')
    
    # Annotate regions
    params = {'rho_trab_max': 0.6, 'rho_cort_min': 0.9}
    ax2.axvline(params['rho_trab_max'], color=COLORS['black'], linestyle=':', alpha=0.4)
    ax2.axvline(params['rho_cort_min'], color=COLORS['black'], linestyle=':', alpha=0.4)
    
    # Add shaded regions for clarity
    ax2.axvspan(0, params['rho_trab_max'], color=COLORS['cyan'], alpha=0.05)
    ax2.axvspan(params['rho_cort_min'], 1.2, color=COLORS['orange'], alpha=0.05)
    
    ax2.text(params['rho_trab_max']/2, 1.8, 'Trabecular\n(n=2.0)', ha='center', fontsize=9, color='#444')
    ax2.text((params['rho_trab_max']+params['rho_cort_min'])/2, 1.6, 'Transition', ha='center', fontsize=9, color='#444')
    ax2.text((params['rho_cort_min']+1.2)/2, 1.3, 'Cortical\n(n=1.2)', ha='center', fontsize=9, color='#444')

def plot_transition_effects(ax1, ax2):
    """Analyze effect of changing transition thresholds."""
    rho = np.linspace(0, 1.1, 1000)
    
    # Case 1: Varying rho_trab_max (start of transition)
    thresholds = [0.4, 0.6, 0.8]
    colors = [COLORS['teal'], COLORS['blue'], COLORS['magenta']]
    
    for th, c in zip(thresholds, colors):
        E, n = calculate_constitutive(rho, rho_trab_max=th, rho_cort_min=0.9)
        ax1.plot(rho, n, color=c, linewidth=2, label=f'Start={th}')
        
    ax1.set_title(r'(c) Effect of Transition Start ($\rho_{trab}^{max}$)', loc='left', fontweight='bold')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel(r'$n(\rho)$')
    ax1.legend(fontsize=8)
    ax1.axvline(0.9, color=COLORS['black'], linestyle=':', alpha=0.4, label='End (0.9)')

    # Case 2: Varying rho_cort_min (end of transition)
    thresholds_end = [0.7, 0.9, 1.0]
    
    for th, c in zip(thresholds_end, colors):
        E, n = calculate_constitutive(rho, rho_trab_max=0.5, rho_cort_min=th)
        ax2.plot(rho, n, color=c, linewidth=2, label=f'End={th}')
        
    ax2.set_title(r'(d) Effect of Transition End ($\rho_{cort}^{min}$)', loc='left', fontweight='bold')
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel(r'$n(\rho)$')
    ax2.legend(fontsize=8)
    ax2.axvline(0.5, color=COLORS['black'], linestyle=':', alpha=0.4, label='Start (0.5)')

def plot_exponent_effects(ax):
    """Analyze effect of changing trabecular/cortical exponents."""
    rho = np.linspace(0, 1.1, 1000)
    
    # Baseline
    E_base, _ = calculate_constitutive(rho)
    ax.plot(rho, E_base, color=COLORS['black'], linewidth=2.5, label='Baseline')
    
    # High cortical stiffness (n_cort = 1.0, linear)
    E_lin, _ = calculate_constitutive(rho, n_cort=1.0)
    ax.plot(rho, E_lin, color=COLORS['blue'], linestyle='--', linewidth=2, label=r'$n_{cort}=1.0$')
    
    # Low cortical stiffness (n_cort = 2.0, same as trabecular)
    E_quad, _ = calculate_constitutive(rho, n_cort=2.0)
    ax.plot(rho, E_quad, color=COLORS['red'], linestyle='--', linewidth=2, label=r'$n_{cort}=2.0$')
    
    # Higher trabecular exponent (n_trab = 3.0)
    E_cubic, _ = calculate_constitutive(rho, n_trab=3.0)
    ax.plot(rho, E_cubic, color=COLORS['teal'], linestyle='--', linewidth=2, label=r'$n_{trab}=3.0$')

    ax.set_title(r'(e) Effect of Exponents on Stiffness $E(\rho)$', loc='left', fontweight='bold')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$E$ [MPa]')
    ax.legend(fontsize=8)

def plot_smoothstep_detail(ax):
    """Zoom in on the smoothstep function itself."""
    x = np.linspace(-0.2, 1.2, 500)
    
    # Cubic smoothstep: 3x^2 - 2x^3
    # Clamped to [0,1]
    x_clamped = np.clip(x, 0, 1)
    w = 3*x_clamped**2 - 2*x_clamped**3
    
    ax.plot(x, w, color=COLORS['blue'], linewidth=2.5, label=r'$w(s) = 3s^2 - 2s^3$')
    ax.plot(x, x_clamped, color=COLORS['grey'], linestyle=':', linewidth=2, label='Linear')
    
    ax.set_title(r'(f) Cubic Smoothstep Function', loc='left', fontweight='bold')
    ax.set_xlabel(r'Normalized Coordinate $s$')
    ax.set_ylabel(r'Weight $w$')
    ax.legend(fontsize=8)

def generate_combined_plot():
    set_modern_style()
    # Landscape layout with 3 columns
    # A4 landscape width is 11.69 inches. Height adjusted for "smaller plots".
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 6.5))
    
    # Row 1: (a) Base E, (b) Base n, (c) Trans Start
    # Row 2: (d) Trans End, (e) Exponents, (f) Smoothstep
    
    plot_base_constitutive(axes[0, 0], axes[0, 1])
    plot_transition_effects(axes[0, 2], axes[1, 0])
    plot_exponent_effects(axes[1, 1])
    plot_smoothstep_detail(axes[1, 2])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'constitutive_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_combined_plot()
