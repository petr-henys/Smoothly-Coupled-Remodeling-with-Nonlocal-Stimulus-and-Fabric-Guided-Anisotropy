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

def smooth_abs(x, eps=1e-6):
    return np.sqrt(x*x + eps*eps)

def calculate_density_rate(S_array, rho_val, **kwargs):
    """
    Calculate remodeling rate d_rho/dt for a range of stimulus values.
    Using the Linear Thermodynamic Driver:
    rate = k_rho * [S+ * (rho_max - rho) + S- * (rho_min - rho)]
    
    Parameters:
    -----------
    S_array : np.ndarray
        Array of stimulus values.
    rho_val : float
        Current density value.
    kwargs : dict
        Override default parameters.
        
    Returns:
    --------
    d_rho : np.ndarray
        Remodeling rate [1/day].
    S_plus : np.ndarray
        Positive part of stimulus (Formation driver).
    S_minus : np.ndarray
        Negative part of stimulus (Resorption driver).
    """
    # Default parameters from Config
    params = {
        'k_rho': 0.01,
        'rho_min': 0.1,
        'rho_max': 1.0,
        'smooth_eps': 1e-6
    }
    params.update(kwargs)
    
    k_rho = params['k_rho']
    rho_min = params['rho_min']
    rho_max = params['rho_max']
    eps = params['smooth_eps']

    # Smooth positive/negative parts
    # S+ = (S + sqrt(S^2 + eps^2)) / 2
    S_plus = (S_array + np.sqrt(S_array**2 + eps**2)) / 2.0
    
    # S- = (-S + sqrt(S^2 + eps^2)) / 2  (Note: S- is positive magnitude of negative S)
    # Actually in code: S_minus = smooth_plus(-S)
    S_minus = (-S_array + np.sqrt(S_array**2 + eps**2)) / 2.0

    # Rate equation
    # rate = k_rho * (S_plus * (rho_max - rho) + S_minus * (rho_min - rho))
    d_rho = k_rho * (S_plus * (rho_max - rho_val) + S_minus * (rho_min - rho_val))
    
    return d_rho, S_plus, S_minus

def plot_density_rate_curve(ax):
    """(a) Density Rate Curve: d_rho/dt vs S."""
    S = np.linspace(-1.0, 1.0, 1000)
    rho = 0.5
    d_rho, _, _ = calculate_density_rate(S, rho)
    
    ax.plot(S, d_rho, color=COLORS['blue'], linewidth=2.5, label=r'$\rho=0.5$')
    
    ax.text(0.6, 0.002, 'Formation', ha='center', color=COLORS['teal'], fontweight='bold')
    ax.text(-0.6, -0.002, 'Resorption', ha='center', color=COLORS['red'], fontweight='bold')
    
    ax.set_title(r'(a) Remodeling Rate ($\dot{\rho}$ vs $S$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$ [1/day]')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

def plot_driver_components(ax):
    """(b) Components: S+, S-."""
    S = np.linspace(-1.0, 1.0, 1000)
    rho = 0.5
    _, S_plus, S_minus = calculate_density_rate(S, rho)
    
    ax.plot(S, S_plus, color=COLORS['teal'], label=r'$S^+$ (Formation)')
    ax.plot(S, S_minus, color=COLORS['red'], label=r'$S^-$ (Resorption)')
    
    ax.set_title(r'(b) Driver Components', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Magnitude [-]')
    ax.legend(fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

def plot_density_effect(ax):
    """(c) Effect of Current Density."""
    S = np.linspace(-1.0, 1.0, 1000)
    rhos = [0.2, 0.5, 0.8]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['magenta']]
    
    for r, c in zip(rhos, colors):
        d_rho, _, _ = calculate_density_rate(S, r)
        ax.plot(S, d_rho, color=c, linewidth=2, label=r'$\rho=' + str(r) + '$')
        
    ax.set_title(r'(c) Effect of Current Density $\rho$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

def plot_rate_sensitivity(ax):
    """(d) Rate Constant Sensitivity (k_rho)."""
    S = np.linspace(0.0, 1.0, 1000) # Focus on formation side
    rho = 0.5
    ks = [0.005, 0.01, 0.02]
    colors = [COLORS['teal'], COLORS['blue'], COLORS['magenta']]
    
    for k, c in zip(ks, colors):
        d_rho, _, _ = calculate_density_rate(S, rho, k_rho=k)
        ax.plot(S, d_rho, color=c, label=f'$k_{{\\rho}}={k}$')
        
    ax.set_title(r'(d) Rate Sensitivity ($k_{\rho}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)

def plot_bounds_sensitivity(ax):
    """(e) Bounds Sensitivity (rho_max)."""
    S = np.linspace(0.0, 1.0, 1000)
    rho = 0.5
    rmaxs = [0.8, 1.0, 1.2]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for rm, c in zip(rmaxs, colors):
        d_rho, _, _ = calculate_density_rate(S, rho, rho_max=rm)
        ax.plot(S, d_rho, color=c, label=f'$\\rho_{{max}}={rm}$')
        
    ax.set_title(r'(e) Bounds Sensitivity ($\rho_{max}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)

def plot_equilibrium_map(ax):
    """(f) Equilibrium Density Map (rho_eq vs S)."""
    # For linear driver, rho_eq is not a single function of S, 
    # but we can look at where rate = 0.
    # rate = k * [S+ (rmax - rho) + S- (rmin - rho)] = 0
    # S+ rmax + S- rmin = rho (S+ + S-)
    # rho_eq = (S+ rmax + S- rmin) / (S+ + S-)
    # If S > 0: S+ = S, S- = 0 -> rho_eq = rmax
    # If S < 0: S+ = 0, S- = -S -> rho_eq = rmin
    # So it's a step function (bang-bang target), but the rate is proportional to S.
    
    S = np.linspace(-1.0, 1.0, 1000)
    # Add small epsilon to avoid 0/0 at S=0
    S_plus = (S + np.abs(S))/2 + 1e-9
    S_minus = (-S + np.abs(S))/2 + 1e-9
    
    rho_max = 1.0
    rho_min = 0.1
    
    rho_target = (S_plus * rho_max + S_minus * rho_min) / (S_plus + S_minus)
    
    ax.plot(S, rho_target, color=COLORS['black'], linewidth=2, label=r'Target $\rho$')
    
    ax.set_title(r'(f) Target Density Attractor', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Target Density $\rho_{target}$')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.9, r'$\rightarrow \rho_{max}$', ha='center', color=COLORS['teal'])
    ax.text(-0.5, 0.2, r'$\rightarrow \rho_{min}$', ha='center', color=COLORS['red'])

def generate_mechanostat_plot():
    set_modern_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 6.5))
    
    plot_density_rate_curve(axes[0, 0])
    plot_driver_components(axes[0, 1])
    plot_density_effect(axes[0, 2])
    plot_rate_sensitivity(axes[1, 0])
    plot_bounds_sensitivity(axes[1, 1])
    plot_equilibrium_map(axes[1, 2])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'density_evolution_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_mechanostat_plot()
