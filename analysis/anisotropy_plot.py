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

def calculate_directional_stiffness(theta, A, E0=15000, nu=0.3, xi=1.0):
    """
    Calculate longitudinal stiffness C_nnnn(theta).
    
    C_nnnn = 2mu + lambda + xi * E * (n.A.n)^2
    """
    # Lame parameters
    lmbda = E0 * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E0 / (2 * (1 + nu))
    C_iso = 2 * mu + lmbda
    
    # Direction vector n
    nx = np.cos(theta)
    ny = np.sin(theta)
    
    # n.A.n
    # A = [[Axx, Axy], [Ayx, Ayy]]
    # n.A.n = Axx*nx^2 + 2*Axy*nx*ny + Ayy*ny^2
    nAn = A[0,0]*nx**2 + 2*A[0,1]*nx*ny + A[1,1]*ny**2
    
    # Stiffness
    C = C_iso + (xi * E0) * (nAn**2)
    
    return C, nAn

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
        C, _ = calculate_directional_stiffness(theta, A, xi=2.0) # High xi to exaggerate effect
        ax.plot(theta, C, color=color, linewidth=2, label=ftype.capitalize())
        
    ax.set_title(r'(a) Directional Stiffness $C(\theta)$', loc='left', fontweight='bold', pad=20)
    ax.set_rticks([20000, 30000, 40000])
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

def plot_stiffness_xi_sensitivity(ax):
    """Linear plot of stiffness vs angle for different xi."""
    theta = np.linspace(0, np.pi, 180) # Half circle sufficient
    A = get_fabric_tensor('uniaxial') # Aligned
    
    xis = [0.0, 0.5, 1.0, 2.0]
    colors = [COLORS['grey'], COLORS['cyan'], COLORS['blue'], COLORS['black']]
    
    for xi, c in zip(xis, colors):
        C, _ = calculate_directional_stiffness(theta, A, xi=xi)
        ax.plot(np.degrees(theta), C/1000, color=c, label=r'$\xi=' + str(xi) + '$')
        
    ax.set_title(r'(c) Anisotropy Factor Sensitivity ($\xi$)', loc='left', fontweight='bold')
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel(r'Stiffness $C$ [GPa]')
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

def plot_fabric_reinforcement_term(ax):
    """Visualize the reinforcement term (n.A.n)^2."""
    theta = np.linspace(0, 2*np.pi, 360)
    A_uni = get_fabric_tensor('uniaxial')
    A_iso = get_fabric_tensor('isotropic')
    
    # n.A.n
    nx = np.cos(theta)
    ny = np.sin(theta)
    
    nAn_uni = A_uni[0,0]*nx**2 + 2*A_uni[0,1]*nx*ny + A_uni[1,1]*ny**2
    nAn_iso = A_iso[0,0]*nx**2 + 2*A_iso[0,1]*nx*ny + A_iso[1,1]*ny**2
    
    ax.plot(theta, nAn_uni**2, color=COLORS['blue'], label='Uniaxial Fabric')
    ax.plot(theta, nAn_iso**2, color=COLORS['grey'], linestyle='--', label='Isotropic Fabric')
    
    ax.set_title(r'(e) Reinforcement Factor $(\mathbf{n}\cdot\mathbf{A}\cdot\mathbf{n})^2$', loc='left', fontweight='bold')
    ax.set_xlabel(r'Angle $\theta$ [rad]')
    ax.set_ylabel(r'Factor [-]')
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.legend(fontsize=8)

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
    # Row 1: (a) Stiffness Polar, (b) Diffusion Polar, (c) Xi Sensitivity
    # Row 2: (d) Beta Sensitivity, (e) Reinforcement, (f) Eigenvalues
    
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
    plot_stiffness_xi_sensitivity(ax3)
    plot_diffusion_beta_sensitivity(ax4)
    plot_fabric_reinforcement_term(ax5)
    plot_projection_concept(ax6)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'anisotropy_law.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_anisotropy_plot()
