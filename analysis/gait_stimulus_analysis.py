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

def synthetic_hip_force(t_percent):
    """
    Generate a synthetic 'M-shaped' hip contact force curve.
    t_percent: 0 to 100
    Returns: Force normalized to Body Weight (BW)
    """
    t = t_percent / 100.0
    
    # Double bump profile (Heel strike ~15%, Toe-off ~50%)
    # Using Gaussian mixture
    
    # Peak 1
    f1 = 2.5 * np.exp(-(t - 0.15)**2 / (2 * 0.08**2))
    # Peak 2
    f2 = 2.8 * np.exp(-(t - 0.50)**2 / (2 * 0.08**2))
    # Swing phase baseline
    base = 0.2 * (1.0 - np.exp(-(t - 0.7)**2 / (2 * 0.15**2)))
    
    return f1 + f2 + base

def run_analysis():
    set_modern_style()
    
    t = np.linspace(0, 100, 200)
    force_bw = synthetic_hip_force(t)
    
    # Reference stress/load
    # Assume sigma is proportional to Force.
    # Let's say sigma_ref corresponds to 1.0 BW (standing) or some other value.
    # In the driver, psi = (sigma / sigma_ref)^m.
    # Let's assume sigma_ref corresponds to a force F_ref.
    F_ref = 1.0 # 1 BW
    
    exponents = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    
    # --- Panel A: Gait Cycle Load ---
    ax = axes[0]
    ax.plot(t, force_bw, color=COLORS['black'], linewidth=2, label='Hip Contact Force')
    ax.fill_between(t, 0, force_bw, color=COLORS['grey'], alpha=0.1)
    ax.axhline(F_ref, color=COLORS['grey'], linestyle='--', label='Reference Load ($F_{ref}$)')
    
    ax.set_xlabel('Gait Cycle [%]')
    ax.set_ylabel('Force [BW]')
    ax.set_title('A: Gait Loading Profile', loc='left', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.5)
    ax.legend(loc='upper right')
    
    # --- Panel B: Stimulus Kernel (Instantaneous) ---
    ax = axes[1]
    
    for m in exponents:
        # Instantaneous contribution: (F/F_ref)^m
        # We normalize this to show the relative weighting of peaks
        kernel = (force_bw / F_ref) ** m
        
        # Normalize max to 1 for comparison of shape? 
        # Or keep absolute to show magnitude explosion?
        # Let's keep absolute but log scale might be needed if m is high.
        # Actually, let's plot the raw value to show how peaks are amplified.
        
        label = f'$m={m}$'
        if m == 1:
            color = COLORS['blue']
        elif m == 2:
            color = COLORS['teal']
        elif m == 4:
            color = COLORS['orange']
        else:
            color = COLORS['red']
            
        ax.plot(t, kernel, color=color, linewidth=2, label=label)
        
    ax.set_xlabel('Gait Cycle [%]')
    ax.set_ylabel(r'Instantaneous Drive $(\sigma/\sigma_{ref})^m$')
    ax.set_title('B: Stimulus Amplification', loc='left', fontweight='bold')
    ax.set_xlim(0, 100)
    # ax.set_yscale('log')
    ax.legend(loc='upper right')
    
    # --- Panel C: Accumulated Stimulus vs Exponent ---
    ax = axes[2]
    
    # Calculate cycle average for each m
    # psi_day = N_cyc * mean( (F/F_ref)^m )
    # Let's just plot the mean factor: mean( (F/F_ref)^m )
    
    m_values = np.linspace(1, 10, 50)
    psi_factors = []
    
    # Compare with a constant load of the same average magnitude
    avg_force = np.mean(force_bw)
    psi_factors_const = []
    
    for m in m_values:
        # Dynamic load
        factor = np.mean((force_bw / F_ref) ** m)
        psi_factors.append(factor)
        
        # Constant load (same average force)
        factor_const = (avg_force / F_ref) ** m
        psi_factors_const.append(factor_const)
        
    ax.plot(m_values, psi_factors, color=COLORS['black'], linewidth=2, label='Dynamic Gait Load')
    ax.plot(m_values, psi_factors_const, color=COLORS['grey'], linestyle='--', label=f'Constant Load (Avg={avg_force:.1f} BW)')
    
    ax.set_xlabel('Stimulus Exponent $m$')
    ax.set_ylabel(r'Cycle Factor $\langle (\sigma/\sigma_{ref})^m \rangle$')
    ax.set_title('C: Effect of Exponent', loc='left', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'gait_stimulus_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated {output_path}")

if __name__ == "__main__":
    run_analysis()
