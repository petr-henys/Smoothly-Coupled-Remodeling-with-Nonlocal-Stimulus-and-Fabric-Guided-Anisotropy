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
    Calculate E for a range of density values based on the power-law constitutive law.
    
    E = E0 * (rho / rho_ref)^n
    
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
    """
    # Default parameters from Config
    params = {
        'E0': 6500.0,
        'n': 1.5,
        'rho_ref': 1.0,
        'rho_min': 0.1,
        'smooth_eps': 1e-6
    }
    params.update(kwargs)
    
    E0 = params['E0']
    n = params['n']
    rho_ref = params['rho_ref']
    rho_min = params['rho_min']
    smooth_eps = params['smooth_eps']

    # Apply smooth_max to density (clamping at rho_min)
    rho_eff = smooth_max(rho_array, rho_min, smooth_eps)
    
    # Calculate E = E0 * (rho / rho_ref)^n
    E = E0 * (rho_eff / rho_ref) ** n
    
    return E

def plot_base_constitutive(ax):
    """Plot the baseline constitutive law E(rho)."""
    rho = np.linspace(0.05, 2.0, 1000)
    E = calculate_constitutive(rho)
    
    # Plot E(rho)
    ax.plot(rho, E, color=COLORS['blue'], linewidth=2.5, label='Model')
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r"Young's Modulus $E$ [MPa]")
    ax.set_title(r'Stiffness-Density Relationship $E = E_0 (\rho/\rho_{ref})^n$', loc='left', fontweight='bold')
    
    # Add reference lines for pure power laws
    params = {'E0': 6500.0, 'n': 1.5, 'rho_ref': 1.0}
    E_n1 = params['E0'] * (rho / params['rho_ref'])**1.0
    E_n2 = params['E0'] * (rho / params['rho_ref'])**2.0
    ax.plot(rho, E_n1, color=COLORS['grey'], linestyle='--', alpha=0.6, label=r'$n=1$')
    ax.plot(rho, E_n2, color=COLORS['grey'], linestyle=':', alpha=0.6, label=r'$n=2$')
    ax.legend()


def plot_exponent_effects(ax):
    """Analyze effect of changing exponent n."""
    rho = np.linspace(0.05, 2.0, 1000)
    
    # Different exponents
    exponents = [1.0, 1.5, 2.0, 3.0]
    colors = [COLORS['teal'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    
    for n, c in zip(exponents, colors):
        E = calculate_constitutive(rho, n=n)
        ax.plot(rho, E, color=c, linewidth=2, label=f'n={n}')

    ax.set_title(r'Effect of Exponent $n$ on Stiffness', loc='left', fontweight='bold')
    ax.set_xlabel(r'Density $\rho$ [g/cm³]')
    ax.set_ylabel(r'$E$ [MPa]')
    ax.legend(fontsize=8)


def generate_combined_plot():
    set_modern_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    plot_base_constitutive(axes[0])
    plot_exponent_effects(axes[1])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'constitutive_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_combined_plot()
