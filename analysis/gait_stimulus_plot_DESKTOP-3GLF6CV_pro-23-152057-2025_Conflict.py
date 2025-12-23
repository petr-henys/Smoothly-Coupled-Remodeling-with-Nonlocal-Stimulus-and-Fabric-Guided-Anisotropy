"""Visualization of GaitDriver multi-load mechanics and power-mean SED aggregation.

Reflects the actual GaitDriver implementation:
- Solves multiple discrete loading cases (heel strike, mid-stance, toe-off, etc.)
- Each case has day_cycles weight representing frequency per day
- Power-mean SED: ψ = (Σᵢ wᵢ · ψᵢ^p / Σᵢ wᵢ)^(1/p) where wᵢ = day_cycles_i
- p = stimulus_power_p (default 4.0): p=1 → mean, p>1 → peak-biased
- Also computes Q̄ = weighted(σσᵀ) for fabric evolution

SED per case: ψ_i = 0.5 · σ : ε (anisotropic stress from fabric tensor L)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
OUTPUT_DIR = "/mnt/pracovni/Active_projects/GACR_BoneMorphologyModeling/remodeller/results/constitutive_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_modern_style():
    """Configure matplotlib for publication-quality look."""
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

# Default parameters (from StimulusParams)
DEFAULT_PARAMS = {
    'stimulus_power_p': 4.0,  # Power-mean exponent
    'psi_ref': 0.01,          # Reference SED [MPa]
}

# Example loading cases (simplified from scenarios.py)
# Each case: (name, day_cycles, force_magnitude_relative)
EXAMPLE_LOADING_CASES = [
    ('heel_strike', 1000, 1.2),    # Impact phase, high frequency
    ('mid_stance', 2000, 1.5),     # Peak single-leg support, highest frequency
    ('toe_off', 1500, 1.0),        # Propulsion phase
    ('stair_climb', 200, 1.8),     # High load, low frequency
]


def power_mean(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    """Compute weighted power-mean: (Σᵢ wᵢ · vᵢ^p / Σᵢ wᵢ)^(1/p)."""
    if p == 1.0:
        return np.average(values, weights=weights)
    weighted_sum = np.sum(weights * (values ** p))
    total_weight = np.sum(weights)
    return (weighted_sum / total_weight) ** (1.0 / p)


def sed_from_force(force_rel: float, psi_ref: float = 0.01) -> float:
    """Estimate SED from relative force magnitude.
    
    SED ∝ σ:ε ∝ F² (linear elasticity assumption).
    """
    return psi_ref * (force_rel ** 2)


def plot_loading_cases(ax):
    """(a) Discrete loading cases with day_cycles and force magnitude."""
    names = [c[0] for c in EXAMPLE_LOADING_CASES]
    cycles = np.array([c[1] for c in EXAMPLE_LOADING_CASES])
    forces = np.array([c[2] for c in EXAMPLE_LOADING_CASES])
    
    x = np.arange(len(names))
    width = 0.35
    
    # Normalize for visualization
    cycles_norm = cycles / cycles.max()
    
    bars1 = ax.bar(x - width/2, forces, width, color=COLORS['blue'], 
                   label='Force magnitude [rel.]', alpha=0.8)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, cycles, width, color=COLORS['orange'],
                    label='day_cycles', alpha=0.8)
    
    ax.set_xlabel('Loading Case')
    ax.set_ylabel('Force [relative]', color=COLORS['blue'])
    ax2.set_ylabel('Cycles per day', color=COLORS['orange'])
    ax.set_title('(a) Discrete Loading Cases', loc='left', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylim(0, 2.5)
    
    # Combined legend
    ax.legend([bars1, bars2], ['Force [rel.]', 'day_cycles'], 
              loc='upper right', fontsize=8)


def plot_sed_per_case(ax):
    """(b) SED per loading case (ψᵢ = 0.5·σ:ε ∝ F²)."""
    names = [c[0] for c in EXAMPLE_LOADING_CASES]
    forces = np.array([c[2] for c in EXAMPLE_LOADING_CASES])
    sed_values = np.array([sed_from_force(f) for f in forces])
    
    x = np.arange(len(names))
    colors_list = [COLORS['cyan'], COLORS['blue'], COLORS['teal'], COLORS['orange']]
    
    bars = ax.bar(x, sed_values * 1000, color=colors_list, alpha=0.8)
    
    ax.axhline(DEFAULT_PARAMS['psi_ref'] * 1000, color=COLORS['grey'], 
               linestyle='--', label=r'$\psi_{ref}$')
    
    ax.set_xlabel('Loading Case')
    ax.set_ylabel(r'SED $\psi_i$ [$\times 10^{-3}$ MPa]')
    ax.set_title(r'(b) SED per Case: $\psi_i = 0.5\,\sigma:\varepsilon$', 
                 loc='left', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.legend(fontsize=8)


def plot_power_mean_effect(ax):
    """(c) Effect of power-mean exponent p on aggregated SED."""
    forces = np.array([c[2] for c in EXAMPLE_LOADING_CASES])
    weights = np.array([c[1] for c in EXAMPLE_LOADING_CASES])
    sed_values = np.array([sed_from_force(f) for f in forces])
    
    p_values = np.linspace(1.0, 8.0, 100)
    aggregated = [power_mean(sed_values, weights, p) for p in p_values]
    
    ax.plot(p_values, np.array(aggregated) * 1000, color=COLORS['blue'], linewidth=2.5)
    
    # Mark default p=4
    p_default = DEFAULT_PARAMS['stimulus_power_p']
    psi_default = power_mean(sed_values, weights, p_default)
    ax.scatter([p_default], [psi_default * 1000], color=COLORS['red'], s=80, zorder=5,
               label=f'Default $p={p_default}$')
    
    # Mark simple mean (p=1)
    psi_mean = power_mean(sed_values, weights, 1.0)
    ax.axhline(psi_mean * 1000, color=COLORS['grey'], linestyle='--', alpha=0.7,
               label=f'Weighted mean ($p=1$)')
    
    # Mark maximum SED
    ax.axhline(sed_values.max() * 1000, color=COLORS['orange'], linestyle=':', alpha=0.7,
               label=r'Max $\psi_i$')
    
    ax.set_xlabel(r'Power-mean exponent $p$')
    ax.set_ylabel(r'Aggregated SED $\bar{\psi}$ [$\times 10^{-3}$ MPa]')
    ax.set_title(r'(c) Power-Mean: $\bar{\psi} = (\sum w_i \psi_i^p / \sum w_i)^{1/p}$', 
                 loc='left', fontweight='bold')
    ax.set_xlim(1, 8)
    ax.legend(fontsize=8, loc='lower right')
    ax.text(6.5, psi_mean * 1000 * 1.15, 'peak-biased', ha='center', 
            color=COLORS['blue'], fontsize=9)


def plot_contribution_weights(ax):
    """(d) Effective contribution of each case at different p values."""
    forces = np.array([c[2] for c in EXAMPLE_LOADING_CASES])
    weights = np.array([c[1] for c in EXAMPLE_LOADING_CASES])
    sed_values = np.array([sed_from_force(f) for f in forces])
    names = [c[0].replace('_', '\n') for c in EXAMPLE_LOADING_CASES]
    
    p_values = [1.0, 2.0, 4.0, 8.0]
    x = np.arange(len(names))
    width = 0.2
    
    for i, p in enumerate(p_values):
        # Effective weight = w_i * psi_i^p / sum(w_j * psi_j^p)
        powered = weights * (sed_values ** p)
        eff_weights = powered / powered.sum()
        offset = (i - 1.5) * width
        ax.bar(x + offset, eff_weights * 100, width, label=f'$p={p}$', alpha=0.8)
    
    ax.set_xlabel('Loading Case')
    ax.set_ylabel('Effective contribution [%]')
    ax.set_title('(d) Case Contribution vs Exponent', loc='left', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend(fontsize=8, loc='upper left')


def plot_qbar_concept(ax):
    """(e) Conceptual: Q̄ tensor from stress outer products."""
    # Schematic showing Q = σσᵀ aggregation for fabric
    
    # Represent stress directions as vectors in 2D for visualization
    angles = [np.pi/6, np.pi/3, np.pi/4, np.pi/2.5]  # Different loading directions
    magnitudes = [c[2] for c in EXAMPLE_LOADING_CASES]
    weights = [c[1] for c in EXAMPLE_LOADING_CASES]
    names = [c[0] for c in EXAMPLE_LOADING_CASES]
    
    colors_list = [COLORS['cyan'], COLORS['blue'], COLORS['teal'], COLORS['orange']]
    
    # Draw stress vectors
    for i, (ang, mag, w, name) in enumerate(zip(angles, magnitudes, weights, names)):
        dx = np.cos(ang) * mag * 0.3
        dy = np.sin(ang) * mag * 0.3
        linewidth = 1 + (w / max(weights)) * 3  # Thicker for higher frequency
        ax.arrow(0.5, 0.5, dx, dy, head_width=0.05, head_length=0.03,
                 fc=colors_list[i], ec=colors_list[i], linewidth=linewidth)
        ax.text(0.5 + dx * 1.3, 0.5 + dy * 1.3, name.replace('_', '\n'), 
                fontsize=7, ha='center', va='center', color=colors_list[i])
    
    # Draw resultant (weighted average direction)
    avg_x = sum(np.cos(a) * m * w for a, m, w in zip(angles, magnitudes, weights)) / sum(weights)
    avg_y = sum(np.sin(a) * m * w for a, m, w in zip(angles, magnitudes, weights)) / sum(weights)
    ax.arrow(0.5, 0.5, avg_x * 0.5, avg_y * 0.5, head_width=0.06, head_length=0.04,
             fc=COLORS['red'], ec=COLORS['red'], linewidth=3, label=r'$\bar{Q}$ direction')
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_aspect('equal')
    ax.set_title(r'(e) Fabric Tensor $\bar{Q} = \sum w_i (\sigma_i \sigma_i^T)$', 
                 loc='left', fontweight='bold')
    ax.axis('off')
    
    ax.text(0.5, 0.1, '(line width ∝ day_cycles)', ha='center', fontsize=8, 
            color=COLORS['grey'])


def plot_sensitivity_comparison(ax):
    """(f) Compare aggregated SED for different loading scenarios."""
    
    # Scenario 1: Default loading
    default_forces = np.array([c[2] for c in EXAMPLE_LOADING_CASES])
    default_weights = np.array([c[1] for c in EXAMPLE_LOADING_CASES])
    default_sed = np.array([sed_from_force(f) for f in default_forces])
    
    # Scenario 2: More walking (higher mid-stance weight)
    walk_weights = np.array([1000, 4000, 2000, 100])  # More walking, less stairs
    
    # Scenario 3: More intense loading (stair climbing dominant)
    intense_weights = np.array([500, 1000, 500, 1500])  # More stairs
    
    p_values = np.linspace(1.0, 8.0, 50)
    
    scenarios = [
        (default_weights, 'Default mix', COLORS['blue']),
        (walk_weights, 'Walking dominant', COLORS['teal']),
        (intense_weights, 'Stair climbing dominant', COLORS['orange']),
    ]
    
    for weights, label, color in scenarios:
        aggregated = [power_mean(default_sed, weights, p) for p in p_values]
        ax.plot(p_values, np.array(aggregated) * 1000, color=color, 
                linewidth=2, label=label)
    
    ax.axvline(4.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    ax.text(4.1, ax.get_ylim()[1] * 0.9, '$p=4$\n(default)', fontsize=8, 
            color=COLORS['grey'])
    
    ax.set_xlabel(r'Power-mean exponent $p$')
    ax.set_ylabel(r'Aggregated SED [$\times 10^{-3}$ MPa]')
    ax.set_title('(f) Activity Pattern Sensitivity', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')


def generate_gait_analysis_plot():
    """Generate comprehensive 6-panel gait driver visualization."""
    set_modern_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    plot_loading_cases(axes[0, 0])
    plot_sed_per_case(axes[0, 1])
    plot_power_mean_effect(axes[0, 2])
    plot_contribution_weights(axes[1, 0])
    plot_qbar_concept(axes[1, 1])
    plot_sensitivity_comparison(axes[1, 2])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'gait_driver_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate_gait_analysis_plot()
