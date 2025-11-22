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

def get_fabric_tensor(type_str):
    """Return a 2D fabric tensor for visualization."""
    if type_str == 'isotropic':
        # In 2D, trace=1 -> 0.5, 0.5
        return np.diag([0.5, 0.5])
    elif type_str == 'uniaxial':
        # Highly aligned in X
        return np.diag([0.9, 0.1])
    elif type_str == 'shear':
        # Principal axes rotated 45 deg
        # A = R * diag(0.9, 0.1) * R.T
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([0.9, 0.1])
        return R @ D @ R.T
    else:
        return np.diag([0.5, 0.5])

def calculate_directional_stiffness(theta, A, rho=1.0, E0=15000, k=2.0, p=0.5, nu=0.3):
    """
    Calculate longitudinal stiffness E(theta) using Zysset-Curnier model.
    
    E(n) approx n . C . n
    But Zysset defines E_i along principal axes.
    For an orthotropic material, 1/E(n) = sum (n_i^2 / E_i) + shear terms...
    Actually, let's compute the full stiffness tensor C and project it: E(n) = n . C . n
    """
    # 1. Eigenvalues of A
    w, v = np.linalg.eigh(A)
    # Sort descending
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    
    m1, m2 = w[0], w[1]
    
    # 2. Principal Stiffnesses
    # E_i = E0 * rho^k * m_i^p
    E1 = E0 * (rho**k) * (m1**p)
    E2 = E0 * (rho**k) * (m2**p)
    
    # Shear modulus (approx)
    # G12 = G0 * rho^k * (m1*m2)^(p/2)
    # Assume G0 approx E0 / (2(1+nu))
    G0 = E0 / (2 * (1 + nu))
    G12 = G0 * (rho**k) * ((m1*m2)**(p/2))
    
    # Poisson ratio
    nu12 = nu # Simplified
    nu21 = nu12 * (E2/E1)
    
    # Compliance Matrix S in principal frame
    # [1/E1, -nu21/E2, 0]
    # [-nu12/E1, 1/E2, 0]
    # [0, 0, 1/G12]
    
    S_mat = np.zeros((3,3))
    S_mat[0,0] = 1.0/E1
    S_mat[1,1] = 1.0/E2
    S_mat[0,1] = -nu12/E1
    S_mat[1,0] = -nu12/E1 # Symmetry check: nu12/E1 = nu21/E2
    S_mat[2,2] = 1.0/G12
    
    # Stiffness Matrix C = inv(S)
    C_mat = np.linalg.inv(S_mat)
    
    # 3. Rotate to global frame
    # We need n in principal frame.
    # n_global = [cos(theta), sin(theta)]
    # n_local = V.T @ n_global
    
    n_global = np.array([np.cos(theta), np.sin(theta)])
    n_local = v.T @ n_global
    
    # Voigt notation for strain: eps = [e11, e22, 2e12]
    # n . C . n is not quite right for E(n).
    # Young's modulus E(n) is usually defined as stress/strain in direction n.
    # 1/E(n) = n_local_i^4 / E1 + n_local_j^4 / E2 + ...
    # For 2D orthotropic:
    # 1/E(theta) = (c^4)/E1 + (s^4)/E2 + (1/G12 - 2*nu12/E1)*c^2*s^2
    
    c = n_local[0]
    s = n_local[1]
    
    inv_E = (c**4)/E1 + (s**4)/E2 + (1.0/G12 - 2.0*nu12/E1)*(c**2)*(s**2)
    
    return 1.0/inv_E, m1

def calculate_directional_diffusion(theta, A, beta_par=0.1, beta_perp=0.01):
    """
    Calculate directional diffusion D(theta) = n.B.n
    B = beta_perp*I + (beta_par - beta_perp)*A
    """
    nx = np.cos(theta)
    ny = np.sin(theta)
    nAn = A[0,0]*nx**2 + 2*A[0,1]*nx*ny + A[1,1]*ny**2
    
    D = beta_perp + (beta_par - beta_perp) * nAn
    return D

def plot_stiffness_polar(ax, fabric_types):
    """Polar plot of stiffness."""
    theta = np.linspace(0, 2*np.pi, 360)
    
    for ftype, color in fabric_types.items():
        A = get_fabric_tensor(ftype)
        C_vals = []
        for t in theta:
            val, _ = calculate_directional_stiffness(t, A, p=1.0) # p=1 for strong effect
            C_vals.append(val)
        ax.plot(theta, C_vals, color=color, linewidth=2, label=ftype.capitalize())
        
    ax.set_title(r'(a) Directional Stiffness $E(\theta)$', loc='left', fontweight='bold', pad=20)
    # ax.set_rticks([10000, 20000, 30000])
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

def plot_diffusion_polar(ax, fabric_types):
    """Polar plot of diffusion."""
    theta = np.linspace(0, 2*np.pi, 360)
    
    for ftype, color in fabric_types.items():
        A = get_fabric_tensor(ftype)
        D = calculate_directional_diffusion(theta, A, beta_par=0.1, beta_perp=0.02)
        ax.plot(theta, D, color=color, linewidth=2, label=ftype.capitalize())
        
    ax.set_title(r'(b) Directional Diffusion $D(\theta)$', loc='left', fontweight='bold', pad=20)
    ax.set_rticks([0.02, 0.06, 0.1])
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3)

def plot_stiffness_p_sensitivity(ax):
    """Linear plot of stiffness vs angle for different p (fabric exponent)."""
    theta = np.linspace(0, np.pi, 180) # Half circle sufficient
    A = get_fabric_tensor('uniaxial') # Aligned
    
    ps = [0.2, 0.5, 1.0, 2.0]
    colors = [COLORS['grey'], COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for p, c in zip(ps, colors):
        vals = []
        for t in theta:
            val, _ = calculate_directional_stiffness(t, A, p=p)
            vals.append(val)
        ax.plot(np.degrees(theta), np.array(vals)/1000, color=c, label=r'$p=' + str(p) + '$')
        
    ax.set_title(r'(c) Anisotropy Sensitivity ($p$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel(r'Stiffness $E$ [GPa]')
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)

def plot_diffusion_beta_sensitivity(ax):
    """Linear plot of diffusion vs angle for different beta ratios."""
    theta = np.linspace(0, np.pi, 180)
    A = get_fabric_tensor('uniaxial')
    
    # Fix beta_par = 0.1, vary beta_perp
    beta_perps = [0.01, 0.05, 0.1] # Ratio 10:1, 2:1, 1:1
    colors = [COLORS['red'], COLORS['orange'], COLORS['grey']]
    
    for bp, c in zip(beta_perps, colors):
        D = calculate_directional_diffusion(theta, A, beta_par=0.1, beta_perp=bp)
        ratio = 0.1/bp
        ax.plot(np.degrees(theta), D, color=c, label=r'$\beta_{\perp}=' + str(bp) + r'$ (Ratio ' + f'{ratio:.0f}:1)')
        
    ax.set_title(r'(d) Diffusion Anisotropy ($\beta_{\perp}$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel(r'Diffusivity $D$ [mm$^2$/day]')
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)

def plot_eigenvalue_scaling(ax):
    """Visualize how stiffness scales with eigenvalues for different p."""
    m = np.linspace(0.1, 1.0, 100)
    ps = [0.5, 1.0, 2.0]
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for p, c in zip(ps, colors):
        E_factor = m**p
        ax.plot(m, E_factor, color=c, label=f'$p={p}$')
        
    ax.set_title(r'(e) Stiffness Scaling $m^p$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Eigenvalue $m$ [-]')
    ax.set_ylabel(r'Factor [-]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_projection_concept(ax):
    """Conceptual plot of unit trace projection."""
    # Just a schematic of eigenvalues
    # Trace = L1 + L2 = 1
    L1 = np.linspace(0, 1, 100)
    L2 = 1 - L1
    
    ax.plot(L1, L2, color=COLORS['black'], linestyle='-', label='Unit Trace Line')
    ax.fill_between(L1, 0, L2, color=COLORS['teal'], alpha=0.1, label='Feasible Region (PSD)')
    
    # Points
    ax.scatter([0.5], [0.5], color=COLORS['grey'], s=50, label='Isotropic')
    ax.scatter([0.9], [0.1], color=COLORS['blue'], s=50, label='Uniaxial')
    
    ax.set_title(r'(f) Fabric Eigenvalue Space ($\lambda_1 + \lambda_2 = 1$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'$\lambda_1$')
    ax.set_ylabel(r'$\lambda_2$')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

def generate_anisotropy_plot():
    set_modern_style()
    
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
        'shear': COLORS['magenta']
    }
    
    plot_stiffness_polar(ax1, fabric_types)
    plot_diffusion_polar(ax2, fabric_types)
    plot_stiffness_p_sensitivity(ax3)
    plot_diffusion_beta_sensitivity(ax4)
    plot_eigenvalue_scaling(ax5)
    plot_projection_concept(ax6)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'anisotropy_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_anisotropy_plot()
