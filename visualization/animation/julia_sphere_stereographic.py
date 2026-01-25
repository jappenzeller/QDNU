"""
================================================================================
JULIA SET ON SPHERE - STEREOGRAPHIC PROJECTION
================================================================================

Maps Julia set onto sphere using stereographic projection, preserving the
fractal's nodal structure. If Julia has 4-fold symmetry, you see 4 canyons.

ANALOGY:
- Circular pond with reflecting boundary → standing waves with nodes
- Spherical shell → waves wrap around, interfere, create spherical harmonics
- Julia set → "standing wave" pattern in iteration space
- Stereographic projection → preserves topology when mapping plane → sphere

The Riemann sphere interpretation:
- Complex plane z = x + iy maps to unit sphere
- z = 0 → south pole
- z = ∞ → north pole  
- |z| = 1 → equator
- Julia set structure maps naturally to spherical "canyons"

================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import os

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# STEREOGRAPHIC PROJECTION
# =============================================================================

def sphere_to_complex(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stereographic projection: sphere → complex plane.
    
    Projects from north pole (0, 0, 1) onto equatorial plane.
    
    θ = polar angle (0 at north pole, π at south pole)
    φ = azimuthal angle
    
    z = x + iy where:
        x = sin(θ)cos(φ) / (1 - cos(θ))
        y = sin(θ)sin(φ) / (1 - cos(θ))
    
    Simplified: z = cot(θ/2) * e^(iφ)
    """
    # Avoid division by zero at north pole
    theta_safe = np.clip(theta, 1e-6, np.pi - 1e-6)
    
    # Stereographic projection
    r = np.tan(theta_safe / 2)  # Distance from origin in complex plane
    # Note: cot(θ/2) for projection from north pole, tan(θ/2) for south pole
    # Using tan gives: south pole → 0, north pole → ∞
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    return x, y


def complex_to_sphere(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse stereographic projection: complex plane → sphere.
    
    Returns (X, Y, Z) on unit sphere.
    """
    r_sq = x**2 + y**2
    
    # Sphere coordinates
    X = 2 * x / (1 + r_sq)
    Y = 2 * y / (1 + r_sq)
    Z = (r_sq - 1) / (1 + r_sq)  # Note: Z=-1 at origin, Z=+1 at infinity
    
    return X, Y, Z


# =============================================================================
# JULIA SET ON SPHERE
# =============================================================================

def compute_julia_on_sphere(c: complex,
                            resolution: int = 100,
                            max_iter: int = 50,
                            radius: float = 1.0) -> dict:
    """
    Compute Julia set directly on sphere surface using stereographic projection.
    
    For each point on sphere:
    1. Project to complex plane via stereographic projection
    2. Compute Julia iteration count
    3. Use iteration as height/color
    
    Args:
        c: Julia parameter
        resolution: Grid resolution (theta x phi)
        max_iter: Maximum iterations
        radius: Base sphere radius
    
    Returns:
        dict with vertices, faces, colors, iterations, sphere coords
    """
    # Create sphere grid
    # More points near equator where interesting structure often is
    theta = np.linspace(0.05, np.pi - 0.05, resolution)  # Avoid exact poles
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Project to complex plane
    z_real, z_imag = sphere_to_complex(THETA, PHI)
    Z = z_real + 1j * z_imag
    
    # Compute Julia iterations
    iterations = np.zeros_like(THETA, dtype=np.float32)
    mask = np.ones_like(THETA, dtype=bool)
    
    for n in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        mag = np.abs(Z)
        escaped = mask & (mag > 2)
        
        # Smooth iteration count
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth = n + 1 - np.log2(np.log2(mag + 1e-10) + 1e-10)
            smooth = np.nan_to_num(smooth, nan=n, posinf=n, neginf=n)
        
        iterations[escaped] = smooth[escaped]
        mask = mask & ~escaped
    
    iterations[mask] = max_iter
    
    # Normalize iterations
    iterations_norm = iterations / max_iter
    
    # Sphere coordinates with height modulation
    # Low iterations (inside Julia set) = inward (canyon)
    # High iterations (outside) = outward (plateau)
    height_scale = 0.3
    R = radius * (1 - height_scale * (1 - iterations_norm))
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z_coord = R * np.cos(THETA)
    
    # Vertices
    vertices = np.stack([X.flatten(), Y.flatten(), Z_coord.flatten()], axis=1)
    
    # Colors from iterations
    colors = colormap_canyon(iterations_norm.flatten())
    
    # Faces
    ny, nx = THETA.shape
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    return {
        'vertices': vertices.astype(np.float32),
        'faces': np.array(faces, dtype=np.int32),
        'colors': colors,
        'iterations': iterations,
        'iterations_norm': iterations_norm,
        'theta': THETA,
        'phi': PHI,
        'radius_modulated': R
    }


def colormap_canyon(t: np.ndarray) -> np.ndarray:
    """
    Colormap for canyon visualization.
    Low values (canyons) = deep blue/purple
    High values (plateaus) = bright orange/yellow
    """
    t = np.clip(t, 0, 1)
    
    # Deep canyons: dark blue/purple
    # Transition: magenta/red
    # Plateaus: orange/yellow
    
    r = np.clip(0.1 + 0.9 * t ** 0.7, 0, 1)
    g = np.clip(0.05 + 0.6 * t ** 1.2, 0, 1)
    b = np.clip(0.4 + 0.2 * t - 0.4 * t ** 2, 0, 1)
    
    return np.stack([r, g, b], axis=1).astype(np.float32)


# =============================================================================
# SYMMETRY ANALYSIS
# =============================================================================

def analyze_julia_symmetry(c: complex) -> dict:
    """
    Analyze the symmetry of Julia set for given c.
    
    Julia sets have specific symmetries:
    - c real → reflection symmetry about real axis
    - c on imaginary axis → reflection about imaginary axis
    - c = 0 → circular (infinite symmetry)
    - General c → 2-fold rotational symmetry (z → -z)
    
    The "lobes" or "nodes" depend on c's position.
    """
    info = {
        'c': c,
        'c_polar': (abs(c), np.angle(c)),
        'symmetry_order': 2,  # Julia sets always have z → -z symmetry
        'expected_lobes': 'varies',
        'notes': []
    }
    
    if abs(c.imag) < 1e-10:
        info['notes'].append('c is real: reflection symmetry about real axis')
    
    if abs(c.real) < 1e-10:
        info['notes'].append('c is imaginary: reflection symmetry about imaginary axis')
    
    # Estimate lobe count based on c
    # For c near the main cardioid boundary, expect specific patterns
    r = abs(c)
    if r < 0.25:
        info['expected_lobes'] = '~circular, few distinct lobes'
    elif r < 0.5:
        info['expected_lobes'] = '2-4 lobes typical'
    elif r < 0.7:
        info['expected_lobes'] = '4-8 lobes, more structure'
    else:
        info['expected_lobes'] = 'Cantor dust / disconnected'
    
    return info


def find_lobe_directions(iterations: np.ndarray, 
                         theta: np.ndarray, 
                         phi: np.ndarray,
                         threshold: float = 0.5) -> List[Tuple[float, float]]:
    """
    Find the angular directions of Julia set lobes (canyons).
    
    Returns list of (theta, phi) directions where canyons are deepest.
    """
    # Find points with low iteration count (inside Julia set = canyons)
    canyon_mask = iterations < (iterations.max() * threshold)
    
    if not np.any(canyon_mask):
        return []
    
    # Find local minima in iteration count
    from scipy import ndimage
    
    # Smooth slightly to avoid noise
    smoothed = ndimage.gaussian_filter(iterations, sigma=2)
    
    # Find local minima
    local_min = (smoothed == ndimage.minimum_filter(smoothed, size=10))
    canyon_points = local_min & canyon_mask
    
    # Get coordinates of canyon centers
    lobe_coords = []
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    canyon_flat = canyon_points.flatten()
    
    for i, is_canyon in enumerate(canyon_flat):
        if is_canyon:
            lobe_coords.append((theta_flat[i], phi_flat[i]))
    
    return lobe_coords


# =============================================================================
# SPECIFIC JULIA SETS WITH KNOWN STRUCTURE
# =============================================================================

# Julia sets with known beautiful structure
NOTABLE_JULIA_PARAMS = {
    'dendrite': complex(-1.0, 0),           # Dendrite fractal
    'rabbit': complex(-0.122, 0.745),       # Douady rabbit (3 lobes)
    'siegel': complex(-0.391, -0.587),      # Siegel disk
    'airplane': complex(-1.755, 0),         # Airplane
    'san_marco': complex(-0.75, 0),         # San Marco fractal
    'spiral': complex(0.285, 0.01),         # Spiral
    'galaxy': complex(-0.7, 0.27015),       # Galaxy spiral
    'snowflake': complex(-0.1, 0.651),      # Snowflake pattern
    'four_lobe': complex(-0.4, 0.6),        # Clear 4-fold structure
    'six_lobe': complex(0.285, 0.535),      # 6-fold structure
}


def get_julia_for_n_lobes(n_lobes: int) -> complex:
    """
    Get Julia parameter that produces approximately n-fold symmetry.
    
    Note: True n-fold symmetry is rare; these produce visual n-lobe appearance.
    """
    lobe_params = {
        2: complex(-0.1, 0.651),      # 2-fold
        3: complex(-0.122, 0.745),    # Douady rabbit
        4: complex(-0.4, 0.6),        # 4-fold
        5: complex(0.285, 0.535),     # ~5-fold
        6: complex(0.285, 0.01),      # Spiral with 6-ish arms
        8: complex(-0.7, 0.27015),    # Galaxy
    }
    
    if n_lobes in lobe_params:
        return lobe_params[n_lobes]
    else:
        # Generate parameter on the unit circle boundary
        # This often gives interesting multi-lobe structures
        angle = 2 * np.pi / n_lobes
        r = 0.5
        return complex(r * np.cos(angle), r * np.sin(angle))


# =============================================================================
# SPHERICAL HARMONIC COMPARISON
# =============================================================================

def spherical_harmonic_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute real spherical harmonic Y_l^m for comparison.
    
    This shows the analogy: spherical harmonics have fixed nodal patterns,
    Julia sets have fractal nodal patterns.
    """
    from scipy.special import sph_harm
    
    if m > 0:
        Y = np.sqrt(2) * np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        Y = np.sqrt(2) * np.imag(sph_harm(-m, l, phi, theta))
    else:
        Y = np.real(sph_harm(0, l, phi, theta))
    
    return Y


def create_spherical_harmonic_sphere(l: int, m: int, resolution: int = 100) -> dict:
    """
    Create sphere deformed by spherical harmonic for comparison with Julia.
    """
    theta = np.linspace(0.01, np.pi - 0.01, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    Y = spherical_harmonic_real(l, m, THETA, PHI)
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
    
    R = 1.0 + 0.3 * (Y_norm - 0.5)
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    vertices = np.stack([X.flatten(), Y_coord.flatten(), Z.flatten()], axis=1)
    colors = colormap_canyon(Y_norm.flatten())
    
    ny, nx = THETA.shape
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    return {
        'vertices': vertices.astype(np.float32),
        'faces': np.array(faces, dtype=np.int32),
        'colors': colors,
        'Y': Y,
        'Y_norm': Y_norm,
        'l': l,
        'm': m
    }


# =============================================================================
# EXPORT
# =============================================================================

def export_ply(data: dict, filepath: str):
    """Export mesh to PLY with vertex colors."""
    vertices = data['vertices']
    faces = data['faces']
    colors = data['colors']
    
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        
        for i, v in enumerate(vertices):
            c = (np.clip(colors[i], 0, 1) * 255).astype(np.uint8)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported: {filepath}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_julia_sphere(data: dict, title: str = "Julia on Sphere",
                           save_path: Optional[str] = None):
    """Visualize Julia sphere with matplotlib."""
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D sphere view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    verts = data['vertices']
    X = verts[:, 0].reshape(data['theta'].shape)
    Y = verts[:, 1].reshape(data['theta'].shape)
    Z = verts[:, 2].reshape(data['theta'].shape)
    
    colors = data['colors'].reshape(data['theta'].shape[0], data['theta'].shape[1], 3)
    
    ax1.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95, shade=False)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_title(title)
    
    # 2D iteration map (stereographic view)
    ax2 = fig.add_subplot(1, 2, 2)
    
    im = ax2.imshow(data['iterations'], cmap='magma', origin='lower',
                    extent=[0, 2*np.pi, 0, np.pi])
    ax2.set_xlabel('φ (azimuthal)')
    ax2.set_ylabel('θ (polar)')
    ax2.set_title('Iteration Count (θ-φ space)')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    return fig


def visualize_comparison(c: complex, l: int = 4, m: int = 0,
                         resolution: int = 80,
                         save_path: Optional[str] = None):
    """
    Side-by-side comparison of Julia sphere and spherical harmonic.
    
    Shows the analogy between:
    - Spherical harmonic Y_l^m → fixed nodal pattern
    - Julia set → fractal nodal pattern
    """
    if not HAS_MPL:
        return None
    
    # Compute both
    julia_data = compute_julia_on_sphere(c, resolution=resolution)
    sh_data = create_spherical_harmonic_sphere(l, m, resolution=resolution)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Julia sphere
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    verts = julia_data['vertices']
    shape = julia_data['theta'].shape
    X = verts[:, 0].reshape(shape)
    Y = verts[:, 1].reshape(shape)
    Z = verts[:, 2].reshape(shape)
    colors = julia_data['colors'].reshape(shape[0], shape[1], 3)
    
    ax1.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95, shade=False)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_title(f'Julia Set on Sphere\nc = {c}')
    
    # Spherical harmonic
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    verts = sh_data['vertices']
    shape = (resolution, resolution * 2)
    X = verts[:, 0].reshape(shape)
    Y = verts[:, 1].reshape(shape)
    Z = verts[:, 2].reshape(shape)
    colors = sh_data['colors'].reshape(shape[0], shape[1], 3)
    
    ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95, shade=False)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_title(f'Spherical Harmonic Y_{l}^{m}')
    
    plt.suptitle('Comparison: Julia "Standing Wave" vs Spherical Harmonic', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("JULIA SET ON SPHERE - STEREOGRAPHIC PROJECTION")
    print("=" * 70)
    print("""
Mapping Julia set to sphere via stereographic projection:

  Complex plane          Sphere (Riemann)
  -------------          ----------------
  z = 0           ->     South pole
  z = inf         ->     North pole
  |z| = 1         ->     Equator
  Julia boundary  ->     Canyon structure

The Julia set's nodal structure (lobes) becomes visible as
spherical "canyons" - like standing waves on a sphere.
""")
    
    os.makedirs('julia_sphere_output', exist_ok=True)
    
    # Generate several Julia spheres with different lobe counts
    test_params = [
        ('rabbit_3lobe', complex(-0.122, 0.745), "Douady Rabbit (3 lobes)"),
        ('four_lobe', complex(-0.4, 0.6), "4-lobe structure"),
        ('spiral', complex(0.285, 0.01), "Spiral arms"),
        ('galaxy', complex(-0.7, 0.27015), "Galaxy pattern"),
    ]
    
    for name, c, desc in test_params:
        print(f"\n--- {desc} ---")
        print(f"c = {c}")
        
        # Analyze symmetry
        sym = analyze_julia_symmetry(c)
        print(f"Expected lobes: {sym['expected_lobes']}")
        
        # Compute
        data = compute_julia_on_sphere(c, resolution=100, max_iter=50)
        
        # Export
        export_ply(data, f'julia_sphere_output/{name}_sphere.ply')
        
        # Visualize
        fig = visualize_julia_sphere(data, f'Julia Sphere: {desc}',
                                     f'julia_sphere_output/{name}_view.png')
        plt.close(fig)
    
    # Comparison with spherical harmonic
    print("\n--- Comparison with Spherical Harmonic Y_4^0 ---")
    fig = visualize_comparison(
        c=complex(-0.4, 0.6),
        l=4, m=0,
        resolution=80,
        save_path='julia_sphere_output/comparison_julia_vs_sh.png'
    )
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print("  julia_sphere_output/*_sphere.ply  - Sphere meshes for Blender")
    print("  julia_sphere_output/*_view.png    - Visualization images")
    print("  julia_sphere_output/comparison_*  - Julia vs spherical harmonic")
    print("\nThe sphere PLY files show Julia 'canyons' - import to Blender")


if __name__ == '__main__':
    main()
