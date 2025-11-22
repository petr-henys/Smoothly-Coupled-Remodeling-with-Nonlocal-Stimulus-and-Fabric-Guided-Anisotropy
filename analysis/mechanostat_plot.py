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

def calculate_mechanostat(S_array, rho_val, **kwargs):
    """
    Calculate remodeling rate d_rho/dt for a range of stimulus values.
    
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
    lam_eff : np.ndarray
        Effective rate coefficient.
    rho_eq : np.ndarray
        Target density.
    """
    # Default parameters from Config
    params = {
        'S_form_th': 0.2,
        'S_resorb_th': -0.2,
        'k_step': 6.0,
        'lambda_form': 0.1,
        'lambda_resorb': 0.1,
        'S_lazy': 0.25,
        'rho_min': 0.1,
        'rho_max': 1.0,
        'smooth_eps': 1e-6
    }
    params.update(kwargs)
    
    S_form = params['S_form_th']
    S_resorb = params['S_resorb_th']
    k_step = params['k_step']
    lam_form = params['lambda_form']
    lam_resorb = params['lambda_resorb']
    S_lazy = params['S_lazy']
    rho_min = params['rho_min']
    rho_max = params['rho_max']
    eps = params['smooth_eps']

    # 1. Signal saturation factor f(S)
    Sabs = smooth_abs(S_array, eps)
    fS = Sabs / (Sabs + S_lazy) if S_lazy > 0.0 else np.ones_like(Sabs)

    # 2. Smooth Heaviside functions
    # H_form = 1 / (1 + exp(-k * (S - S_form)))
    H_form = 1.0 / (1.0 + np.exp(-k_step * (S_array - S_form)))
    
    # H_resorb = 1 / (1 + exp(-k * (S_resorb - S)))  -> Note direction!
    H_resorb = 1.0 / (1.0 + np.exp(-k_step * (S_resorb - S_array)))

    # 3. Effective rate lambda_eff
    lam_eff = fS * (lam_form * H_form + lam_resorb * H_resorb)

    # 4. Equilibrium density rho_eq
    # If formation -> rho_max
    # If resorption -> rho_min
    # If lazy -> rho_val (current density)
    # We use the partition of unity: H_form + H_resorb + H_lazy = 1 (approx)
    # Actually the code uses: rho_eq = rho_max*Hf + rho_min*Hr + (1 - Hf - Hr)*rho
    rho_eq = rho_max * H_form + rho_min * H_resorb + (1.0 - H_form - H_resorb) * rho_val

    # 5. Rate d_rho/dt = lam_eff * (rho_eq - rho)
    d_rho = lam_eff * (rho_eq - rho_val)
    
    return d_rho, lam_eff, rho_eq, H_form, H_resorb, fS

def plot_mechanostat_curve(ax):
    """(a) Standard Mechanostat Curve: d_rho/dt vs S."""
    S = np.linspace(-1.0, 1.0, 1000)
    rho = 0.5
    d_rho, _, _, _, _, _ = calculate_mechanostat(S, rho)
    
    ax.plot(S, d_rho, color=COLORS['blue'], linewidth=2.5, label=r'$\rho=0.5$')
    
    # Annotate zones
    ax.axvspan(-0.2, 0.2, color=COLORS['grey'], alpha=0.1, label='Lazy Zone')
    ax.axvline(0.2, color=COLORS['black'], linestyle=':', alpha=0.4)
    ax.axvline(-0.2, color=COLORS['black'], linestyle=':', alpha=0.4)
    
    ax.text(0.6, 0.02, 'Formation', ha='center', color=COLORS['teal'], fontweight='bold')
    ax.text(-0.6, -0.02, 'Resorption', ha='center', color=COLORS['red'], fontweight='bold')
    ax.text(0.0, 0.0, 'Lazy', ha='center', va='center', color=COLORS['grey'])
    
    ax.set_title(r'(a) Mechanostat Curve ($\dot{\rho}$ vs $S$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$ [1/day]')
    ax.axhline(0, color='black', linewidth=0.5)

def plot_rate_components(ax):
    """(b) Components: H_form, H_resorb, f(S)."""
    S = np.linspace(-1.0, 1.0, 1000)
    rho = 0.5
    _, _, _, H_form, H_resorb, fS = calculate_mechanostat(S, rho)
    
    ax.plot(S, H_form, color=COLORS['teal'], label=r'$H_{form}$')
    ax.plot(S, H_resorb, color=COLORS['red'], label=r'$H_{resorb}$')
    ax.plot(S, fS, color=COLORS['grey'], linestyle='--', label=r'$f(S)$ (Saturation)')
    
    ax.set_title(r'(b) Activation Functions', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Activation [-]')
    ax.legend(fontsize=8)

def plot_density_effect(ax):
    """(c) Effect of Current Density."""
    S = np.linspace(-1.0, 1.0, 1000)
    rhos = [0.2, 0.5, 0.8]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['magenta']]
    
    for r, c in zip(rhos, colors):
        d_rho, _, _, _, _, _ = calculate_mechanostat(S, r)
        ax.plot(S, d_rho, color=c, linewidth=2, label=r'$\rho=' + str(r) + '$')
        
    ax.set_title(r'(c) Effect of Current Density $\rho$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)

def plot_threshold_sensitivity(ax):
    """(d) Threshold Sensitivity."""
    S = np.linspace(0.0, 1.0, 1000) # Focus on formation side
    rho = 0.5
    thresholds = [0.1, 0.2, 0.4]
    colors = [COLORS['teal'], COLORS['blue'], COLORS['magenta']]
    
    for th, c in zip(thresholds, colors):
        d_rho, _, _, _, _, _ = calculate_mechanostat(S, rho, S_form_th=th)
        ax.plot(S, d_rho, color=c, label=f'$S_{{form}}={th}$')
        
    ax.set_title(r'(d) Threshold Sensitivity ($S_{form}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)

def plot_steepness_sensitivity(ax):
    """(e) Steepness Sensitivity (k_step)."""
    S = np.linspace(0.0, 0.6, 1000) # Zoom in on transition
    rho = 0.5
    ks = [2.0, 6.0, 20.0]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for k, c in zip(ks, colors):
        d_rho, _, _, _, _, _ = calculate_mechanostat(S, rho, k_step=k)
        ax.plot(S, d_rho, color=c, label=f'$k_{{step}}={k}$')
        
    ax.set_title(r'(e) Transition Steepness ($k_{step}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)
    ax.axvline(0.2, color='grey', linestyle=':', label=r'$S_{form}=0.2$')

def plot_saturation_sensitivity(ax):
    """(f) Saturation Sensitivity (S_lazy)."""
    S = np.linspace(0.0, 2.0, 1000)
    rho = 0.5
    # S_lazy controls how fast f(S) approaches 1
    # f(S) = S / (S + S_lazy)
    slazies = [0.05, 0.25, 1.0]
    colors = [COLORS['red'], COLORS['blue'], COLORS['grey']]
    
    for sl, c in zip(slazies, colors):
        d_rho, _, _, _, _, _ = calculate_mechanostat(S, rho, S_lazy=sl)
        ax.plot(S, d_rho, color=c, label=f'$S_{{lazy}}={sl}$')
        
    ax.set_title(r'(f) Signal Saturation ($S_{lazy}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Stimulus $S$ [-]')
    ax.set_ylabel(r'Rate $\dot{\rho}$')
    ax.legend(fontsize=8)

def generate_mechanostat_plot():
    set_modern_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 6.5))
    
    plot_mechanostat_curve(axes[0, 0])
    plot_rate_components(axes[0, 1])
    plot_density_effect(axes[0, 2])
    plot_threshold_sensitivity(axes[1, 0])
    plot_steepness_sensitivity(axes[1, 1])
    plot_saturation_sensitivity(axes[1, 2])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mechanostat_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_mechanostat_plot()
