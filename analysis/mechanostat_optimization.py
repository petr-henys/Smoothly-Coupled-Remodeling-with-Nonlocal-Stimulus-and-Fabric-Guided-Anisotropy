import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields
import os
import sys
from pathlib import Path

# Output directory
OUTPUT_DIR = "/mnt/pracovni/Active_projects/GACR_BoneMorphologyModeling/remodeller/results/constitutive_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from simulation.config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("Warning: Could not import simulation.config. Using hardcoded defaults.")
    @dataclass
    class Config:
        E0: float = 15000.0
        n_trab: float = 2.0
        n_cort: float = 1.2
        rho_trab_max: float = 0.6
        rho_cort_min: float = 0.9
        rho_min: float = 0.01
        rho_max: float = 1.0
        psi_ref: float = 20.0
        n_power: float = 2.0
        S_lazy: float = 0.25
        S_form_th: float = 0.2
        S_resorb_th: float = -0.2
        k_step: float = 6.0
        lambda_form: float = 0.1
        lambda_resorb: float = 0.1
        cS: float = 1.0
        tauS: float = 1.0
        rS_gain: float = 1.0

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

def get_stiffness(rho, cfg):
    """Calculate stiffness E(rho) based on simulation logic."""
    rho_eff = max(rho, cfg.rho_min)
    
    if cfg.rho_cort_min <= cfg.rho_trab_max:
        n_eff = cfg.n_trab
    else:
        s_raw = (rho_eff - cfg.rho_trab_max) / (cfg.rho_cort_min - cfg.rho_trab_max)
        s0 = max(0.0, min(1.0, s_raw))
        w = 3.0 * s0**2 - 2.0 * s0**3
        n_eff = (1.0 - w) * cfg.n_trab + w * cfg.n_cort
    
    return cfg.E0 * (rho_eff ** n_eff)

def get_stimulus_driver(rho, strain, cfg):
    """
    Calculate the mechanical driver psi(rho, strain).
    Based on simulation/drivers.py: psi = (sigma_vm / psi_ref)^m
    """
    E = get_stiffness(rho, cfg)
    sigma = E * strain # Uniaxial assumption
    
    # Driver: Stress-based
    psi = (sigma / cfg.psi_ref) ** cfg.n_power
    return psi

def get_steady_state_stimulus(psi, cfg):
    """
    Calculate steady-state stimulus S from driver psi.
    Based on StimulusSolver: tauS * S = rS_gain * (psi - 1.0)
    """
    # Avoid division by zero if tauS is 0 (unlikely)
    tau = cfg.tauS if cfg.tauS > 1e-9 else 1.0
    return (cfg.rS_gain / tau) * (psi - 1.0)

def get_remodeling_rate(S, rho, cfg):
    """Calculate drho/dt based on DensitySolver logic."""
    # Lazy zone gating
    S_abs = abs(S)
    S_lazy = cfg.S_lazy
    f_gate = S_abs / (S_abs + S_lazy) if S_abs > 0 else 0.0
    
    # Sigmoids
    arg_form = -cfg.k_step * (S - cfg.S_form_th)
    arg_resorb = -cfg.k_step * (cfg.S_resorb_th - S)
    
    H_form = 1.0 / (1.0 + np.exp(np.clip(arg_form, -50, 50)))
    H_resorb = 1.0 / (1.0 + np.exp(np.clip(arg_resorb, -50, 50)))
    
    # Target density
    rho_eq = cfg.rho_max * H_form + cfg.rho_min * H_resorb + (1 - H_form - H_resorb) * rho
    
    # Rate
    rate = f_gate * (cfg.lambda_form * H_form + cfg.lambda_resorb * H_resorb)
    
    drho = rate * (rho_eq - rho)
    return drho

def run_analysis():
    set_modern_style()
    cfg = Config() if not HAS_CONFIG else Config
    
    # Setup grid
    rhos = np.linspace(cfg.rho_min, cfg.rho_max, 200)
    strains = [1000e-6, 2000e-6, 3000e-6, 4000e-6] # microstrain
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    
    # --- Panel A: Phase Space (Remodeling Rate) ---
    ax = axes[0]
    
    # 1. Strain Control (Solid)
    for i, eps in enumerate(strains):
        drhos = []
        for rho in rhos:
            psi = get_stimulus_driver(rho, eps, cfg)
            S = get_steady_state_stimulus(psi, cfg)
            drho = get_remodeling_rate(S, rho, cfg)
            drhos.append(drho)
        
        lbl = r'$\varepsilon = {:.0f} \mu\varepsilon$'.format(eps*1e6) if i in [0, len(strains)-1] else None
        ax.plot(rhos, drhos, linestyle='-', linewidth=1.5, label=lbl)
        
    # 2. Stress Control (Dashed)
    # Compare with the 2000 and 4000 microstrain curves at rho=0.5
    rho_ref = 0.5
    E_ref = get_stiffness(rho_ref, cfg)
    strains_stress = [strains[1], strains[3]]
    
    for i, eps in enumerate(strains_stress):
        sigma_fixed = E_ref * eps
        drhos = []
        for rho in rhos:
            # Stress control: sigma is constant
            # psi = (sigma / psi_ref)^m
            psi = (sigma_fixed / cfg.psi_ref) ** cfg.n_power
            S = get_steady_state_stimulus(psi, cfg)
            drho = get_remodeling_rate(S, rho, cfg)
            drhos.append(drho)
            
        lbl = r'$\sigma \approx {:.1f}$ MPa'.format(sigma_fixed) if i == 0 else None
        ax.plot(rhos, drhos, linestyle='--', color='grey', alpha=0.7, linewidth=1.5, label=lbl)

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Density $\rho$ [-]')
    ax.set_ylabel(r'Remodeling Rate $\dot{\rho}$ [1/day]')
    ax.set_title('A: Phase Space', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8)
    
    # --- Panel B: Stability Map (Strain Thresholds) ---
    ax = axes[1]
    
    eps_form = []
    eps_resorb = []
    
    # SED Thresholds (for comparison)
    eps_form_sed = []
    eps_resorb_sed = []
    
    # Invert S(psi) -> psi(S) -> sigma(psi) -> eps(sigma, rho)
    # S = (gain/tau)*(psi - 1) => psi = 1 + S*(tau/gain)
    tau_gain = cfg.tauS / cfg.rS_gain if cfg.rS_gain > 0 else 1.0
    
    psi_form = 1.0 + cfg.S_form_th * tau_gain
    psi_resorb = max(0.0, 1.0 + cfg.S_resorb_th * tau_gain) # psi must be > 0
    
    # Reference Energy for SED normalization (match Stress driver at rho=0.5)
    # psi_stress = (sigma/ref)^m
    # psi_sed = U / U_ref
    # At rho=0.5, let U_ref be such that psi_sed = psi_stress for the same sigma.
    # Actually, we just want the shape. Let's assume psi_sed = (U / U0).
    # Threshold: U = U_form. U = 0.5 * E * eps^2.
    # eps = sqrt(2 * U_form / E).
    # We need to calibrate U_form to match the stress driver's threshold at rho=0.5 for visual comparison.
    
    E_mid = get_stiffness(0.5, cfg)
    sigma_form_mid = cfg.psi_ref * (psi_form ** (1.0/cfg.n_power))
    U_form_mid = 0.5 * (sigma_form_mid**2) / E_mid
    
    sigma_resorb_mid = cfg.psi_ref * (psi_resorb ** (1.0/cfg.n_power))
    U_resorb_mid = 0.5 * (sigma_resorb_mid**2) / E_mid
    
    for rho in rhos:
        E = get_stiffness(rho, cfg)
        
        # 1. Stress Driver Thresholds (Current Model)
        # psi = (sigma/ref)^m => sigma = ref * psi^(1/m)
        sigma_form = cfg.psi_ref * (psi_form ** (1.0/cfg.n_power))
        sigma_resorb = cfg.psi_ref * (psi_resorb ** (1.0/cfg.n_power))
        
        # eps = sigma / E
        eps_form.append((sigma_form / E) * 1e6)
        eps_resorb.append((sigma_resorb / E) * 1e6)
        
        # 2. SED Driver Thresholds (Hypothetical)
        # U = U_form_mid (calibrated at rho=0.5)
        # eps = sqrt(2 * U / E)
        eps_form_sed.append(np.sqrt(2 * U_form_mid / E) * 1e6)
        eps_resorb_sed.append(np.sqrt(2 * U_resorb_mid / E) * 1e6)
        
    # Plot Stress Driver Regions (Filled)
    ax.plot(rhos, eps_form, color=COLORS['teal'], linestyle='-', linewidth=2, label='Stress Drv. Form.')
    ax.plot(rhos, eps_resorb, color=COLORS['red'], linestyle='-', linewidth=2, label='Stress Drv. Res.')
    ax.fill_between(rhos, eps_resorb, eps_form, color=COLORS['grey'], alpha=0.2, label='Lazy Zone (Stress)')
    
    # Plot SED Driver Thresholds (Dashed lines)
    ax.plot(rhos, eps_form_sed, color=COLORS['teal'], linestyle=':', linewidth=2, label='SED Drv. Form.')
    ax.plot(rhos, eps_resorb_sed, color=COLORS['red'], linestyle=':', linewidth=2, label='SED Drv. Res.')
    
    # Trajectories
    y_traj = 2500
    # Strain Ctrl (Horizontal)
    ax.arrow(0.2, y_traj, 0.4, 0, head_width=150, head_length=0.05, fc='k', ec='k', label='Strain Ctrl')
    
    # Stress Ctrl (Curve 1/E)
    rho_start = 0.2
    E_start = get_stiffness(rho_start, cfg)
    sigma_traj = E_start * (y_traj * 1e-6)
    
    traj_stress = []
    rhos_traj = np.linspace(0.2, 0.8, 50)
    for r in rhos_traj:
        E_r = get_stiffness(r, cfg)
        traj_stress.append((sigma_traj / E_r) * 1e6)
        
    ax.plot(rhos_traj, traj_stress, 'k--', label='Stress Ctrl')
    ax.arrow(rhos_traj[-2], traj_stress[-2], rhos_traj[-1]-rhos_traj[-2], traj_stress[-1]-traj_stress[-2], 
             head_width=0, head_length=0, fc='k', ec='k')
    
    ax.set_xlabel(r'Density $\rho$ [-]')
    ax.set_ylabel(r'Strain [$\mu\varepsilon$]')
    ax.set_title('B: Stability Map', loc='left', fontweight='bold')
    ax.set_ylim(0, 5000)
    ax.legend(loc='upper right', fontsize=8)
    
    # --- Panel C: Feedback Stability (Stimulus vs Density) ---
    ax = axes[2]
    
    fixed_strain = 2000e-6
    S_strain_ctrl = []
    S_stress_ctrl = []
    S_sed_ctrl = []
    
    # Match stress at rho=0.5
    E_mid = get_stiffness(0.5, cfg)
    sigma_fixed = E_mid * fixed_strain
    psi_stress_mid = (sigma_fixed / cfg.psi_ref) ** cfg.n_power
    
    for rho in rhos:
        # 1. Stress Driver under Strain Control (Fixed Strain)
        # psi = (E*eps / ref)^m  -> Increases with rho (Unstable)
        psi_strain = get_stimulus_driver(rho, fixed_strain, cfg)
        S_strain_ctrl.append(get_steady_state_stimulus(psi_strain, cfg))
        
        # 2. Stress Driver under Stress Control (Fixed Stress)
        # psi = (sigma / ref)^m -> Constant (Neutral)
        psi_stress = (sigma_fixed / cfg.psi_ref) ** cfg.n_power
        S_stress_ctrl.append(get_steady_state_stimulus(psi_stress, cfg))
        
        # 3. SED Driver under Stress Control (Fixed Stress)
        # psi_sed ~ sigma^2 / E. 
        # We normalize it to match the stress driver at rho=0.5 for comparison.
        # psi_sed(rho) = psi_mid * (E_mid / E(rho))
        E = get_stiffness(rho, cfg)
        psi_sed = psi_stress_mid * (E_mid / E)
        S_sed_ctrl.append(get_steady_state_stimulus(psi_sed, cfg))
        
    ax.plot(rhos, S_strain_ctrl, color=COLORS['red'], linestyle='-', label='Stress Driver (Strain Ctrl)')
    ax.plot(rhos, S_stress_ctrl, color=COLORS['blue'], linestyle='--', label='Stress Driver (Force Ctrl)')
    ax.plot(rhos, S_sed_ctrl, color=COLORS['teal'], linestyle='-.', label='SED Driver (Force Ctrl)')
    
    ax.axhline(cfg.S_form_th, color=COLORS['teal'], linestyle=':', label='Form. Th.')
    ax.axhline(cfg.S_resorb_th, color=COLORS['red'], linestyle=':', label='Res. Th.')
    
    ax.set_xlabel(r'Density $\rho$ [-]')
    ax.set_ylabel(r'Stimulus $S$ [-]')
    ax.set_title('C: Feedback Stability', loc='left', fontweight='bold')
    
    # Add explanatory text about the physics
    ax.text(0.05, 0.20, "Stress Drv (Strain Ctrl): Unstable\nStress Drv (Force Ctrl): Neutral\nSED Drv (Force Ctrl): Stable", 
            transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'mechanostat_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated {output_path}")

if __name__ == "__main__":
    run_analysis()
