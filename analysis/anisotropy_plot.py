"""Visualization of log-fabric anisotropy model.

Reflects the actual FabricSolver implementation:
- Log-fabric tensor L with tr(L)=0 for volume-preserving fabric
- Fabric eigenvalues a_hat_i = exp(l_i - mean_l) from L
- Stiffness: E_i = E_iso * a_hat_i^pE, G_ij = G_iso * (a_hat_i*a_hat_j)^(0.5*pG)
- Target L_target derived from Q̄ = weighted(σσᵀ) via geometric normalization
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import COLORS, save_manuscript_figure, apply_style


# Default fabric parameters (from FabricParams)
DEFAULT_PARAMS = {
    'E0': 7500.0,       # Reference Young's modulus [MPa]
    'nu0': 0.3,         # Poisson ratio
    'pE': 1.0,          # Axial stiffness exponent (stiff_pE)
    'pG': 1.0,          # Shear stiffness exponent (stiff_pG)
    'gammaF': 1.0,      # Fabric eigenvalue exponent
    'm_min': 0.2,       # Min eigenvalue ratio bound
    'm_max': 5.0,       # Max eigenvalue ratio bound
}


def get_log_fabric_tensor(type_str: str) -> np.ndarray:
    """Return a 3D log-fabric tensor L with tr(L)=0.
    
    The log-fabric L stores ln(m_i) where m_i are fabric eigenvalues.
    For volume preservation, tr(L) = ln(m1) + ln(m2) + ln(m3) = 0.
    """
    if type_str == 'isotropic':
        # m1 = m2 = m3 = 1 -> L = 0
        return np.zeros((3, 3))
    elif type_str == 'uniaxial':
        # Strong alignment in Z: m3 >> m1 = m2
        # m1 = m2 = 0.5, m3 = 4.0 -> ln product = 0 (trace constraint)
        # Actually: m1*m2*m3 = 1 for normalized fabric
        # Let m1 = m2 = r, m3 = 1/r^2 with r = 0.5 -> m3 = 4
        r = 0.5
        l1 = l2 = np.log(r)
        l3 = np.log(1.0 / (r * r))
        return np.diag([l1, l2, l3])
    elif type_str == 'biaxial':
        # Two strong directions: m1 = m2 > m3
        # m1 = m2 = 2, m3 = 0.25 -> product = 1
        l1 = l2 = np.log(2.0)
        l3 = np.log(0.25)
        return np.diag([l1, l2, l3])
    elif type_str == 'rotated':
        # Uniaxial rotated 45° in XZ plane
        r = 0.5
        l1 = np.log(r)
        l2 = np.log(r)
        l3 = np.log(1.0 / (r * r))
        D = np.diag([l1, l2, l3])
        # Rotate around Y axis by 45°
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return R @ D @ R.T
    else:
        return np.zeros((3, 3))


def compute_fabric_eigenvalues(L: np.ndarray) -> tuple:
    """Compute fabric eigenvalues a_hat from log-fabric L.
    
    a_hat_i = exp(l_i - mean_l) ensures geometric mean = 1.
    """
    L_sym = 0.5 * (L + L.T)
    eigenvals = np.linalg.eigvalsh(L_sym)
    # Sort descending
    eigenvals = np.sort(eigenvals)[::-1]
    
    mean_l = np.mean(eigenvals)
    a_hat = np.exp(eigenvals - mean_l)
    return a_hat, eigenvals


def calculate_directional_stiffness(theta: float, phi: float, L: np.ndarray,
                                     E0: float = 7500.0, nu0: float = 0.3,
                                     pE: float = 1.0, pG: float = 1.0) -> float:
    """Calculate directional Young's modulus E(n) from log-fabric L.
    
    Uses the actual MechanicsSolver constitutive model:
    - a_hat_i = exp(l_i - mean_l)
    - E_i = E_iso * a_hat_i^pE
    - G_ij = G_iso * (a_hat_i * a_hat_j)^(0.5*pG)
    
    For Young's modulus in direction n (3D orthotropic compliance):
    1/E(n) = sum_i n_i^4/E_i + sum_{i<j} (1/G_ij - 2*nu0/sqrt(Ei*Ej)) * n_i^2*n_j^2
    
    Args:
        theta: Polar angle from Z axis [rad]
        phi: Azimuthal angle in XY plane [rad]
        L: 3x3 log-fabric tensor
        E0: Reference Young's modulus [MPa]
        nu0: Poisson ratio
        pE: Axial stiffness exponent
        pG: Shear stiffness exponent
    """
    # Fabric eigenvalues and eigenvectors
    L_sym = 0.5 * (L + L.T)
    eigenvals, eigenvecs = np.linalg.eigh(L_sym)
    # Sort descending by eigenvalue
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # a_hat = exp(l - mean_l)
    mean_l = np.mean(eigenvals)
    a_hat = np.exp(eigenvals - mean_l)
    
    # Principal stiffnesses
    E_iso = E0  # For visualization, use E0 directly (rho=1)
    G_iso = E_iso / (2.0 * (1.0 + nu0))
    
    E1 = E_iso * (a_hat[0] ** pE)
    E2 = E_iso * (a_hat[1] ** pE)
    E3 = E_iso * (a_hat[2] ** pE)
    
    G12 = G_iso * ((a_hat[0] * a_hat[1]) ** (0.5 * pG))
    G23 = G_iso * ((a_hat[1] * a_hat[2]) ** (0.5 * pG))
    G31 = G_iso * ((a_hat[2] * a_hat[0]) ** (0.5 * pG))
    
    # Direction vector in global frame
    n_global = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Transform to principal frame
    n_local = eigenvecs.T @ n_global
    n1, n2, n3 = n_local[0], n_local[1], n_local[2]
    
    # Compliance in direction n (orthotropic material)
    # 1/E(n) = n1^4/E1 + n2^4/E2 + n3^4/E3
    #        + (1/G23 - 2*nu/sqrt(E2*E3)) * n2^2*n3^2
    #        + (1/G31 - 2*nu/sqrt(E3*E1)) * n3^2*n1^2
    #        + (1/G12 - 2*nu/sqrt(E1*E2)) * n1^2*n2^2
    
    inv_E = (n1**4 / E1 + n2**4 / E2 + n3**4 / E3
             + (1.0/G23 - 2.0*nu0/np.sqrt(E2*E3)) * n2**2 * n3**2
             + (1.0/G31 - 2.0*nu0/np.sqrt(E3*E1)) * n3**2 * n1**2
             + (1.0/G12 - 2.0*nu0/np.sqrt(E1*E2)) * n1**2 * n2**2)
    
    return 1.0 / inv_E


def plot_stiffness_polar(ax, fabric_types: dict):
    """Polar plot of directional stiffness E(θ) in XZ plane (phi=0)."""
    theta = np.linspace(0, 2*np.pi, 360)
    phi = 0.0  # XZ plane
    
    for ftype, color in fabric_types.items():
        L = get_log_fabric_tensor(ftype)
        E_vals = [calculate_directional_stiffness(t, phi, L) for t in theta]
        ax.plot(theta, np.array(E_vals)/1000, color=color, linewidth=2, label=ftype.capitalize())
        
    ax.set_title(r'(a) Directional Stiffness $E(\theta)$ [GPa]', loc='left', fontweight='bold', pad=20)
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)


def plot_ahat_polar(ax, fabric_types: dict):
    """Polar plot of fabric eigenvalue magnitude in direction θ."""
    theta = np.linspace(0, 2*np.pi, 360)
    
    for ftype, color in fabric_types.items():
        L = get_log_fabric_tensor(ftype)
        a_hat, eigenvals = compute_fabric_eigenvalues(L)
        
        # a(θ) = n·M·n where M = diag(a_hat) in principal frame
        L_sym = 0.5 * (L + L.T)
        _, eigenvecs = np.linalg.eigh(L_sym)
        idx = np.argsort(np.linalg.eigvalsh(L_sym))[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        a_vals = []
        for t in theta:
            # Direction in XZ plane
            n_global = np.array([np.sin(t), 0, np.cos(t)])
            n_local = eigenvecs.T @ n_global
            a_n = np.sum(a_hat * n_local**2)
            a_vals.append(a_n)
        
        ax.plot(theta, a_vals, color=color, linewidth=2, label=ftype.capitalize())
        
    ax.set_title(r'(b) Fabric Factor $\hat{a}(\theta)$', loc='left', fontweight='bold', pad=20)
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3)


def plot_stiffness_pE_sensitivity(ax):
    """Stiffness vs angle for different pE exponents."""
    theta = np.linspace(0, np.pi, 180)
    phi = 0.0
    L = get_log_fabric_tensor('uniaxial')
    
    pEs = [0.5, 1.0, 1.5, 2.0]
    colors = [COLORS['grey'], COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for pE, c in zip(pEs, colors):
        vals = [calculate_directional_stiffness(t, phi, L, pE=pE) for t in theta]
        ax.plot(np.degrees(theta), np.array(vals)/1000, color=c, label=rf'$p_E={pE}$')
        
    ax.set_title(r'(c) Stiffness Exponent $p_E$ Effect', loc='left', fontweight='bold')
    ax.set_xlabel(r'Angle $\theta$ from principal axis [deg]')
    ax.set_ylabel(r'Stiffness $E$ [GPa]')
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)


def plot_ahat_scaling(ax):
    """Visualize a_hat = exp(l) scaling with log-eigenvalue l."""
    l_vals = np.linspace(-2, 2, 100)  # l = ln(m)
    a_hat = np.exp(l_vals)
    
    ax.plot(l_vals, a_hat, color=COLORS['blue'], linewidth=2, label=r'$\hat{a} = e^l$')
    ax.axhline(1.0, color=COLORS['grey'], linestyle='--', alpha=0.5, label='Isotropic')
    ax.axvline(0.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    
    # Mark typical bounds
    m_min, m_max = 0.2, 5.0
    ax.axhline(m_min, color=COLORS['red'], linestyle=':', alpha=0.7, label=f'$m_{{min}}={m_min}$')
    ax.axhline(m_max, color=COLORS['red'], linestyle=':', alpha=0.7, label=f'$m_{{max}}={m_max}$')
    
    ax.set_title(r'(d) Log-Fabric Scaling $\hat{a} = e^l$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Log-eigenvalue $l = \ln(m)$')
    ax.set_ylabel(r'Fabric factor $\hat{a}$')
    ax.set_ylim(0, 6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_stiffness_scaling(ax):
    """Visualize E_i = E_iso * a_hat^pE for different pE."""
    a_hat = np.linspace(0.2, 5.0, 100)
    pEs = [0.5, 1.0, 2.0]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for pE, c in zip(pEs, colors):
        E_factor = a_hat ** pE
        ax.plot(a_hat, E_factor, color=c, label=rf'$p_E={pE}$')
    
    ax.axhline(1.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    ax.axvline(1.0, color=COLORS['grey'], linestyle='--', alpha=0.5)
        
    ax.set_title(r'(e) Stiffness Scaling $E_i/E_{iso} = \hat{a}^{p_E}$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Fabric factor $\hat{a}$')
    ax.set_ylabel(r'Stiffness ratio $E_i/E_{iso}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_log_fabric_space(ax):
    """Log-fabric eigenvalue space with tr(L)=0 constraint."""
    # For 3D: l1 + l2 + l3 = 0 defines a plane
    # Project onto 2D: plot l1 vs l2 (l3 = -l1 - l2)
    
    l1 = np.linspace(-2, 2, 100)
    l2_line = -l1  # When l3 = 0: l1 + l2 = 0
    
    # Feasible region: ordered eigenvalues l1 >= l2 >= l3
    # With l3 = -l1 - l2, need l2 >= -l1 - l2 => l2 >= -l1/2
    # And l1 >= l2
    
    ax.fill_between(l1, -l1/2, l1, where=(l1 >= -l1/2), 
                    color=COLORS['teal'], alpha=0.15, label='Feasible (ordered)')
    
    # Zero trace line (isotropic subspace)
    ax.axhline(0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    ax.axvline(0, color=COLORS['grey'], linestyle='--', alpha=0.5)
    
    # Example points
    ax.scatter([0], [0], color=COLORS['grey'], s=80, zorder=5, label='Isotropic')
    
    # Uniaxial: l1 = l2 = ln(0.5), l3 = ln(4) ≈ 1.39
    r = 0.5
    l_uni = np.log(r)
    ax.scatter([l_uni], [l_uni], color=COLORS['blue'], s=80, zorder=5, label='Uniaxial (aligned)')
    
    # Biaxial: l1 = l2 = ln(2), l3 = ln(0.25)
    l_bi = np.log(2.0)
    ax.scatter([l_bi], [l_bi], color=COLORS['magenta'], s=80, zorder=5, label='Biaxial (planar)')
    
    ax.set_title(r'(f) Log-Fabric Space ($l_1 + l_2 + l_3 = 0$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'$l_1 = \ln(m_1)$')
    ax.set_ylabel(r'$l_2 = \ln(m_2)$')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def generate_anisotropy_plot():
    apply_style()
    
    # Layout: 2 rows x 3 columns
    # Row 1: (a) Stiffness Polar, (b) Diffusion Polar, (c) p Sensitivity
    # Row 2: (d) Beta Sensitivity, (e) Eigenvalue Scaling, (f) Eigenvalues
    
    fig = plt.figure(figsize=(11.69, 7.0))
    
    # Row 1
    ax1 = plt.subplot(231, projection='polar')
    ax2 = plt.subplot(232, projection='polar')
    ax3 = plt.subplot(233)
    
    # Row 2
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)
    
    fabric_types = {
        'isotropic': COLORS['grey'],
        'uniaxial': COLORS['blue'],
        'rotated': COLORS['magenta']
    }
    
    plot_stiffness_polar(ax1, fabric_types)
    plot_ahat_polar(ax2, fabric_types)
    plot_stiffness_pE_sensitivity(ax3)
    plot_ahat_scaling(ax4)
    plot_stiffness_scaling(ax5)
    plot_log_fabric_space(ax6)
    
    plt.tight_layout()
    save_manuscript_figure(fig, 'anisotropy_law')


if __name__ == "__main__":
    generate_anisotropy_plot()
