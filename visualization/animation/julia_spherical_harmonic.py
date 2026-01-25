"""
================================================================================
SPHERICAL HARMONIC JULIA SET VISUALIZATION
================================================================================

Maps Julia set onto sphere using stereographic projection, analogous to how
spherical harmonics represent standing wave patterns on a sphere.

CONCEPT:
  - Circular pond: waves reflect off boundary, create standing wave nodes
  - Spherical shell: waves wrap around, interfere with themselves
  - Spherical harmonics: eigenfunctions = natural standing wave patterns
  - Julia on sphere: fractal structure creates "harmonic-like" nodal patterns

MAPPING:
  1. For each point (θ, φ) on sphere
  2. Stereographic projection → complex z
  3. Compute Julia iteration count at z
  4. Radius = base_r - depth * (normalized iteration)
  
  Low iteration = near Julia set boundary = canyon/valley
  High iteration = inside/outside = plateau

The Julia set's inherent symmetry (z² + c has 2-fold symmetry) creates
spherical harmonic-like patterns with visible lobes/nodes.

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


# ==============================================================================
# STEREOGRAPHIC PROJECTION
# ==============================================================================

def sphere_to_complex_stereographic(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Stereographic projection from sphere to complex plane.
    
    Projects from SOUTH pole (θ=π maps to z=0, θ=0 maps to z=∞).
    
    Formula: z = cot(θ/2) * e^(iφ) = (1 + cos(θ))/sin(θ) * e^(iφ)
    
    Or equivalently using tan from north pole:
    z = tan(θ/2) * e^(iφ)  [south pole → 0, north pole → ∞]
    
    We use the tan version (south pole centered).
    """
    # Avoid singularity at north pole (θ=0)
    theta_safe = np.clip(theta, 0.001, np.pi - 0.001)
    
    r = np.tan(theta_safe / 2)  # Radial distance in complex plane
    z = r * np.exp(1j * phi)
    
    return z


def complex_to_sphere_stereographic(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse stereographic projection from complex plane to sphere.
    
    Returns (theta, phi) spherical coordinates.
    """
    r = np.abs(z)
    phi = np.angle(z)
    
    # theta = 2 * arctan(r)
    theta = 2 * np.arctan(r)
    
    return theta, phi


# ==============================================================================
# JULIA SET ON SPHERE
# ==============================================================================

def compute_julia_on_sphere(c: complex,
                            theta: np.ndarray,
                            phi: np.ndarray,
                            max_iter: int = 100,
                            escape_radius: float = 2.0) -> np.ndarray:
    """
    Compute Julia set mapped onto sphere via stereographic projection.
    
    Args:
        c: Julia parameter
        theta: Polar angle array (0 = north pole, π = south pole)
        phi: Azimuthal angle array (0 to 2π)
        max_iter: Maximum iterations
        escape_radius: Escape threshold
    
    Returns:
        Array of normalized iteration counts (same shape as theta/phi)
    """
    # Project sphere to complex plane
    z = sphere_to_complex_stereographic(theta, phi)
    
    iterations = np.zeros_like(theta, dtype=np.float32)
    mask = np.ones_like(theta, dtype=bool)
    
    for n in range(max_iter):
        z[mask] = z[mask]**2 + c
        mag = np.abs(z)
        
        escaped = mask & (mag > escape_radius)
        
        # Smooth iteration count
        with np.errstate(invalid='ignore', divide='ignore'):
            # Smooth coloring formula
            smooth = n + 1 - np.log(np.log(mag + 1e-10)) / np.log(2)
            smooth = np.nan_to_num(smooth, nan=n, posinf=n, neginf=n)
        
        iterations[escaped] = smooth[escaped]
        mask = mask & ~escaped
    
    # Points that never escaped
    iterations[mask] = max_iter
    
    # Normalize to [0, 1]
    return iterations / max_iter


def compute_julia_boundary_distance(c: complex,
                                    theta: np.ndarray,
                                    phi: np.ndarray,
                                    max_iter: int = 50) -> np.ndarray:
    """
    Compute distance estimate to Julia set boundary.
    
    Uses derivative tracking for smoother results.
    Points near the boundary → small distance → deep canyon.
    """
    z = sphere_to_complex_stereographic(theta, phi)
    dz = np.ones_like(z)  # Derivative starts at 1
    
    for n in range(max_iter):
        # d/dz of z² + c = 2z
        dz = 2 * z * dz
        z = z**2 + c
        
        # Escape check
        if np.all(np.abs(z) > 1000):
            break
    
    # Distance estimate: |z| * log|z| / |dz|
    mag_z = np.abs(z)
    mag_dz = np.abs(dz) + 1e-10
    
    with np.errstate(invalid='ignore', divide='ignore'):
        distance = mag_z * np.log(mag_z + 1e-10) / mag_dz
        distance = np.nan_to_num(distance, nan=0, posinf=1, neginf=0)
    
    # Normalize
    distance = np.clip(distance, 0, 1)
    
    return distance


# ==============================================================================
# SPHERICAL MESH WITH JULIA MODULATION
# ==============================================================================

def create_julia_sphere_mesh(c: complex,
                              resolution: int = 128,
                              base_radius: float = 1.0,
                              canyon_depth: float = 0.3,
                              max_iter: int = 80,
                              use_distance: bool = False) -> dict:
    """
    Create sphere mesh with Julia set as surface modulation.
    
    The Julia set structure appears as "canyons" (valleys) on the sphere,
    similar to how spherical harmonics create nodal patterns.
    
    Args:
        c: Julia parameter (controls number/shape of lobes)
        resolution: Grid resolution
        base_radius: Base sphere radius
        canyon_depth: How deep the canyons go
        max_iter: Julia iterations
        use_distance: If True, use distance estimate (smoother boundaries)
    
    Returns:
        Dict with vertices, faces, colors, normals
    """
    # Create spherical grid
    # Theta: 0 (north pole) to π (south pole)
    # Phi: 0 to 2π (around equator)
    theta = np.linspace(0.01, np.pi - 0.01, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Compute Julia values
    if use_distance:
        julia_vals = compute_julia_boundary_distance(c, THETA, PHI, max_iter)
        # Distance: small = near boundary = deep canyon
        heights = julia_vals  # Already normalized
    else:
        julia_vals = compute_julia_on_sphere(c, THETA, PHI, max_iter)
        # Iterations: low = near boundary = deep canyon
        # Invert so boundary = valley
        heights = julia_vals
    
    # Modulate radius
    # Low julia value (near boundary) → smaller radius (canyon)
    R = base_radius - canyon_depth * (1 - heights)
    
    # Alternative: make boundary the RIDGE (like standing wave antinode)
    # R = base_radius + canyon_depth * (1 - heights)
    
    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    # Flatten to vertex array
    n_verts = resolution * resolution
    vertices = np.zeros((n_verts, 3), dtype=np.float32)
    vertices[:, 0] = X.flatten()
    vertices[:, 1] = Y.flatten()
    vertices[:, 2] = Z.flatten()
    
    # Colors based on Julia values
    colors = colormap_harmonic(heights.flatten())
    
    # Create faces
    faces = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            idx = j * resolution + i
            # Two triangles per quad
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    faces = np.array(faces, dtype=np.int32)
    
    # Compute normals
    normals = compute_vertex_normals(vertices, faces)
    
    return {
        'vertices': vertices,
        'faces': faces,
        'colors': colors,
        'normals': normals,
        'julia_values': julia_vals,
        'radii': R
    }


def colormap_harmonic(t: np.ndarray) -> np.ndarray:
    """
    Colormap designed for spherical harmonic visualization.
    
    Blue/purple for valleys (low), yellow/white for peaks (high).
    Similar to typical spherical harmonic visualizations.
    """
    t = np.clip(t, 0, 1)
    
    # Deep blue → cyan → white → yellow → orange
    r = np.clip(0.2 + 1.5 * t - 0.5 * t**2, 0, 1)
    g = np.clip(0.1 + 0.9 * t, 0, 1)
    b = np.clip(0.8 - 0.6 * t, 0, 1)
    
    return np.stack([r, g, b], axis=1).astype(np.float32)


def colormap_phase(julia_vals: np.ndarray, phase_angle: np.ndarray) -> np.ndarray:
    """
    Color by both iteration count AND phase of final z.
    
    Gives more detail in the fractal structure.
    """
    t = np.clip(julia_vals, 0, 1)
    p = (phase_angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Hue from phase, brightness from iteration
    h = p
    s = 0.8
    v = 0.3 + 0.7 * t
    
    # HSV to RGB
    c = v * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = v - c
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    mask = (h < 1/6)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0
    mask = (h >= 1/6) & (h < 2/6)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0
    mask = (h >= 2/6) & (h < 3/6)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
    mask = (h >= 3/6) & (h < 4/6)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
    mask = (h >= 4/6) & (h < 5/6)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
    mask = (h >= 5/6)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
    
    r += m
    g += m
    b += m
    
    return np.stack([r, g, b], axis=1).astype(np.float32)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals from mesh."""
    normals = np.zeros_like(vertices)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    for i, face in enumerate(faces):
        normals[face[0]] += face_normals[i]
        normals[face[1]] += face_normals[i]
        normals[face[2]] += face_normals[i]
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
    return (normals / norms).astype(np.float32)


# ==============================================================================
# SPECIFIC JULIA PARAMETERS FOR DIFFERENT NODE COUNTS
# ==============================================================================

# Julia parameters that create different nodal structures
JULIA_PRESETS = {
    '2_node': {
        'c': complex(-1.0, 0.0),
        'description': '2-fold symmetry (period-2 bulb)',
    },
    '3_node': {
        'c': complex(-0.12, 0.74),
        'description': '3-fold symmetry (Douady rabbit)',
    },
    '4_node': {
        'c': complex(0.28, 0.008),
        'description': '4-fold near-symmetry',
    },
    '5_node': {
        'c': complex(-0.504, 0.563),
        'description': '5-fold structure',
    },
    'dendrite': {
        'c': complex(0, 1),
        'description': 'Dendrite (i)',
    },
    'spiral': {
        'c': complex(-0.8, 0.156),
        'description': 'Spiral arms',
    },
    'san_marco': {
        'c': complex(-0.75, 0.0),
        'description': 'San Marco (basilica)',
    },
    'siegel_disk': {
        'c': complex(-0.390541, -0.586788),
        'description': 'Siegel disk',
    },
}


def get_julia_for_node_count(n_nodes: int) -> complex:
    """
    Get Julia parameter c that produces approximately n visible nodes.
    
    Based on the period-n bulbs of the Mandelbrot set.
    """
    if n_nodes == 2:
        return complex(-1.0, 0.0)
    elif n_nodes == 3:
        return complex(-0.12, 0.74)  # Douady rabbit
    elif n_nodes == 4:
        return complex(0.28, 0.008)
    elif n_nodes == 5:
        return complex(-0.504, 0.563)
    elif n_nodes == 6:
        return complex(-0.565, 0.492)
    else:
        # General formula for period-n: c near the center of period-n bulb
        # Approximate: c = (1 - (1/n)²) * e^(2πi/n) / 2 - 1/4 (very rough)
        theta = 2 * np.pi / n_nodes
        r = 0.25 + 0.1 * n_nodes
        return complex(r * np.cos(theta) - 0.5, r * np.sin(theta))


# ==============================================================================
# EXPORT
# ==============================================================================

def export_ply(mesh_data: dict, filepath: str):
    """Export mesh to PLY format."""
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    colors = mesh_data['colors']
    normals = mesh_data.get('normals')
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for i in range(len(vertices)):
            v = vertices[i]
            c = (np.clip(colors[i], 0, 1) * 255).astype(np.uint8)
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if normals is not None:
                n = normals[i]
                line += f" {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
            line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported: {filepath}")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_julia_sphere(mesh_data: dict, title: str = "Julia Sphere",
                           save_path: Optional[str] = None):
    """Visualize the Julia sphere mesh."""
    if not HAS_MPL:
        print("matplotlib required")
        return None
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D view
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    vertices = mesh_data['vertices']
    colors = mesh_data['colors']
    
    # Scatter plot of vertices (faster than full mesh)
    # Subsample for performance
    step = max(1, len(vertices) // 5000)
    ax.scatter(vertices[::step, 0], vertices[::step, 1], vertices[::step, 2],
               c=colors[::step], s=1, alpha=0.8)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(colors='gray')
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_comparison(c: complex, resolution: int = 64, save_path: Optional[str] = None):
    """
    Compare 2D Julia set with spherical projection.
    
    Shows the analogy to circular pond → spherical shell.
    """
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 1. 2D Julia (circular pond analogy)
    ax1 = fig.add_subplot(131)
    ax1.set_facecolor('#16213e')
    
    x = np.linspace(-2, 2, resolution * 2)
    y = np.linspace(-2, 2, resolution * 2)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    iterations = np.zeros_like(X)
    mask = np.ones_like(X, dtype=bool)
    
    for n in range(80):
        Z[mask] = Z[mask]**2 + c
        escaped = np.abs(Z) > 2
        iterations[mask & escaped] = n
        mask = mask & ~escaped
    
    iterations[mask] = 80
    
    ax1.imshow(iterations, cmap='magma', origin='lower', extent=[-2, 2, -2, 2])
    circle = plt.Circle((0, 0), 2, fill=False, color='cyan', linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    ax1.set_title('2D Julia Set\n(Circular Pond)', color='white', fontsize=12)
    ax1.set_xlabel('Re(z)', color='gray')
    ax1.set_ylabel('Im(z)', color='gray')
    ax1.tick_params(colors='gray')
    
    # 2. Unwrapped sphere (Mercator-like)
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor('#16213e')
    
    theta = np.linspace(0.01, np.pi - 0.01, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    julia_sphere = compute_julia_on_sphere(c, THETA, PHI, max_iter=80)
    
    ax2.imshow(julia_sphere, cmap='magma', origin='lower',
               extent=[0, 360, 0, 180], aspect='auto')
    ax2.set_title('Julia on Sphere\n(Unwrapped)', color='white', fontsize=12)
    ax2.set_xlabel('φ (longitude)', color='gray')
    ax2.set_ylabel('θ (latitude)', color='gray')
    ax2.tick_params(colors='gray')
    
    # 3. 3D sphere view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#1a1a2e')
    
    mesh = create_julia_sphere_mesh(c, resolution=64, canyon_depth=0.25, max_iter=60)
    vertices = mesh['vertices']
    colors = mesh['colors']
    
    step = max(1, len(vertices) // 3000)
    ax3.scatter(vertices[::step, 0], vertices[::step, 1], vertices[::step, 2],
                c=colors[::step], s=2, alpha=0.9)
    
    ax3.set_xlim([-1.3, 1.3])
    ax3.set_ylim([-1.3, 1.3])
    ax3.set_zlim([-1.3, 1.3])
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_title('Julia Sphere\n(Standing Waves)', color='white', fontsize=12)
    ax3.tick_params(colors='gray', labelsize=7)
    
    plt.suptitle(f'Julia Set c = {c.real:.3f}{c.imag:+.3f}i\n'
                 f'Circular Pond → Spherical Shell Analogy',
                 color='white', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_node_comparison(save_path: Optional[str] = None):
    """
    Show Julia spheres with different node counts.
    
    Analogous to Y_l^m spherical harmonics with different l values.
    """
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    node_counts = [2, 3, 4, 5]
    
    for i, n in enumerate(node_counts):
        c = get_julia_for_node_count(n)
        
        # 2D Julia
        ax1 = fig.add_subplot(2, 4, i + 1)
        ax1.set_facecolor('#16213e')
        
        x = np.linspace(-2, 2, 128)
        y = np.linspace(-2, 2, 128)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        iterations = np.zeros_like(X)
        mask = np.ones_like(X, dtype=bool)
        
        for _ in range(60):
            Z[mask] = Z[mask]**2 + c
            escaped = np.abs(Z) > 2
            iterations[mask & escaped] = _
            mask = mask & ~escaped
        iterations[mask] = 60
        
        ax1.imshow(iterations, cmap='magma', origin='lower', extent=[-2, 2, -2, 2])
        ax1.set_title(f'{n}-node Julia\nc = {c.real:.2f}{c.imag:+.2f}i',
                      color='white', fontsize=10)
        ax1.tick_params(colors='gray', labelsize=7)
        
        # 3D sphere
        ax2 = fig.add_subplot(2, 4, i + 5, projection='3d')
        ax2.set_facecolor('#1a1a2e')
        
        mesh = create_julia_sphere_mesh(c, resolution=48, canyon_depth=0.3, max_iter=50)
        vertices = mesh['vertices']
        colors = mesh['colors']
        
        step = max(1, len(vertices) // 2000)
        ax2.scatter(vertices[::step, 0], vertices[::step, 1], vertices[::step, 2],
                    c=colors[::step], s=2, alpha=0.9)
        
        ax2.set_xlim([-1.3, 1.3])
        ax2.set_ylim([-1.3, 1.3])
        ax2.set_zlim([-1.3, 1.3])
        ax2.set_box_aspect([1, 1, 1])
        ax2.set_title(f'Spherical\n({n} canyons)', color='white', fontsize=10)
        ax2.tick_params(colors='gray', labelsize=6)
    
    plt.suptitle('Julia Sets with Different Node Counts\n'
                 '(Like Spherical Harmonics with different l)',
                 color='white', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ==============================================================================
# INTEGRATION CLASS FOR EXPLORER
# ==============================================================================

class SphericalJulia:
    """
    Spherical Julia for integration with explorer.
    
    Usage:
        sj = SphericalJulia(resolution=64)
        sj.update(c=complex(-0.4, 0.6))
        mesh = sj.get_mesh()
        sj.export('output.ply')
    """
    
    def __init__(self, resolution: int = 64, base_radius: float = 1.0, canyon_depth: float = 0.3):
        self.resolution = resolution
        self.base_radius = base_radius
        self.canyon_depth = canyon_depth
        self.max_iter = 60
        
        self.c = complex(-0.4, 0.6)
        self.mesh_data = None
    
    def update(self, c: complex = None, a: float = None, b: float = None, pn_c: float = None):
        """
        Update Julia parameter.
        
        Can pass c directly or derive from PN parameters (a, b, c).
        """
        if c is not None:
            self.c = c
        elif b is not None:
            # Derive from PN parameters
            a = a if a is not None else 0.3
            pn_c = pn_c if pn_c is not None else 0.3
            real = -0.4 + 0.3 * np.cos(b)
            imag = 0.3 * np.sin(b) + 0.1 * (a - pn_c)
            self.c = complex(real, imag)
        
        self._compute()
    
    def _compute(self):
        """Compute the mesh."""
        self.mesh_data = create_julia_sphere_mesh(
            self.c,
            resolution=self.resolution,
            base_radius=self.base_radius,
            canyon_depth=self.canyon_depth,
            max_iter=self.max_iter
        )
    
    def get_mesh(self) -> dict:
        """Get mesh data."""
        if self.mesh_data is None:
            self._compute()
        return self.mesh_data
    
    def export(self, filepath: str):
        """Export to PLY."""
        if self.mesh_data is None:
            self._compute()
        export_ply(self.mesh_data, filepath)
    
    def set_node_count(self, n: int):
        """Set Julia parameter for approximately n visible nodes."""
        self.c = get_julia_for_node_count(n)
        self._compute()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("SPHERICAL HARMONIC JULIA VISUALIZATION")
    print("=" * 70)
    print("""
CONCEPT:
  - Circular pond: waves reflect -> standing wave nodes
  - Spherical shell: waves wrap around -> spherical harmonics
  - Julia on sphere: fractal structure -> harmonic-like patterns

The Julia set's inherent symmetry creates visible "lobes" or "canyons"
on the sphere surface, analogous to spherical harmonic nodal patterns.
""")
    
    # Create output directory
    os.makedirs('julia_spherical_output', exist_ok=True)
    
    # Show presets
    print("\nJulia presets for different node counts:")
    for name, preset in JULIA_PRESETS.items():
        print(f"  {name}: c = {preset['c']}, {preset['description']}")
    
    # Generate comparison visualization
    print("\nGenerating comparison visualization...")
    c = complex(-0.4, 0.6)  # Nice 4-ish node structure
    fig1 = visualize_comparison(c, resolution=64, 
                                save_path='julia_spherical_output/comparison.png')
    
    # Generate node comparison
    print("\nGenerating node count comparison...")
    fig2 = visualize_node_comparison(save_path='julia_spherical_output/node_comparison.png')
    
    # Export 4-node mesh
    print("\nExporting 4-node sphere mesh...")
    mesh = create_julia_sphere_mesh(
        get_julia_for_node_count(4),
        resolution=128,
        canyon_depth=0.3,
        max_iter=80
    )
    export_ply(mesh, 'julia_spherical_output/julia_sphere_4node.ply')
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print("  julia_spherical_output/comparison.png     - Pond -> Sphere analogy")
    print("  julia_spherical_output/node_comparison.png - Different node counts")
    print("  julia_spherical_output/julia_sphere_4node.ply - 4-node mesh for Blender")
    
    if HAS_MPL:
        plt.show()


if __name__ == '__main__':
    main()
