"""
REFINED SMOOTH JULIA SPHERE
===========================

Tuned for visible nodal structure with smooth transitions.
Like spherical harmonics but with organic Julia complexity.
"""

import numpy as np
from scipy import ndimage
import os

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def sphere_to_complex(theta, phi):
    """Stereographic projection."""
    theta_safe = np.clip(theta, 1e-6, np.pi - 1e-6)
    r = np.tan(theta_safe / 2)
    return r * np.cos(phi), r * np.sin(phi)


def compute_julia_smooth(c, x, y, max_iter=100):
    """
    Compute Julia with smooth iteration count.
    Returns values normalized to [0, 1].
    """
    Z = x + 1j * y
    
    # Track both iteration count and escape magnitude
    iterations = np.full_like(x, float(max_iter), dtype=np.float64)
    final_mag = np.ones_like(x, dtype=np.float64) * 2
    escaped = np.zeros_like(x, dtype=bool)
    
    for n in range(max_iter):
        mask = ~escaped
        if not np.any(mask):
            break
        
        Z[mask] = Z[mask]**2 + c
        mag = np.abs(Z)
        
        newly_escaped = ~escaped & (mag > 2)
        
        # Smooth iteration: n + 1 - log2(log2|z|)
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth = n + 1 - np.log2(np.log2(mag[newly_escaped] + 1e-10) + 1e-10)
            smooth = np.nan_to_num(smooth, nan=n, posinf=n, neginf=n)
        
        iterations[newly_escaped] = smooth
        final_mag[newly_escaped] = mag[newly_escaped]
        escaped = escaped | newly_escaped
    
    # Normalize
    return iterations / max_iter


def create_julia_sphere_smooth(c, resolution=120, max_iter=60,
                                smoothing_sigma=3.0,
                                canyon_depth=0.35,
                                invert=False):
    """
    Create smooth Julia sphere with visible nodal structure.
    
    Key parameters:
    - smoothing_sigma: Higher = smoother (2-5 good range)
    - canyon_depth: How much the surface deforms (0.2-0.5)
    - invert: If True, Julia boundary = ridge; If False, Julia = canyon
    """
    # Sphere grid
    theta = np.linspace(0.02, np.pi - 0.02, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Project to complex plane
    z_x, z_y = sphere_to_complex(THETA, PHI)
    
    # Compute Julia
    julia = compute_julia_smooth(c, z_x, z_y, max_iter)
    
    # Apply smoothing
    julia_smooth = ndimage.gaussian_filter(julia, sigma=smoothing_sigma)
    
    # Normalize to [0, 1]
    julia_smooth = (julia_smooth - julia_smooth.min()) / (julia_smooth.max() - julia_smooth.min() + 1e-10)
    
    # Apply soft curve for more gradual transitions
    # sqrt gives nice smooth gradient
    height = np.sqrt(julia_smooth)
    
    # Invert if requested (Julia boundary becomes ridge instead of canyon)
    if invert:
        height = 1 - height
    
    # Compute radius
    R = 1.0 - canyon_depth * (1 - height)
    
    # To Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Colors - smooth gradient
    colors = colormap_organic(height.flatten())
    
    # Faces
    ny, nx = THETA.shape
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    # Normals
    normals = compute_normals(vertices, np.array(faces))
    
    return {
        'vertices': vertices.astype(np.float32),
        'faces': np.array(faces, dtype=np.int32),
        'colors': colors,
        'normals': normals,
        'height': height,
        'julia_raw': julia,
        'julia_smooth': julia_smooth,
        'theta': THETA,
        'phi': PHI,
        'R': R,
        'c': c
    }


def colormap_organic(t):
    """
    Organic colormap - smooth gradient from deep to bright.
    
    Deep regions: warm brown/orange (like canyon shadows)
    High regions: bright gold/cream (like sunlit peaks)
    """
    t = np.clip(t, 0, 1)
    
    # Warm palette
    r = 0.3 + 0.65 * t ** 0.8
    g = 0.15 + 0.55 * t ** 1.0
    b = 0.1 + 0.3 * t ** 1.3
    
    return np.clip(np.stack([r, g, b], axis=1), 0, 1).astype(np.float32)


def colormap_harmonic(t):
    """
    Classic spherical harmonic style - blue negative, orange positive.
    """
    t = np.clip(t, 0, 1)
    
    # Below 0.5 = blue, above 0.5 = orange
    blue_weight = np.clip(1 - t * 2, 0, 1)
    orange_weight = np.clip(t * 2 - 1, 0, 1)
    
    r = 0.2 + 0.7 * orange_weight + 0.1 * (1 - blue_weight)
    g = 0.2 + 0.4 * orange_weight + 0.2 * (1 - blue_weight)
    b = 0.3 + 0.6 * blue_weight
    
    return np.clip(np.stack([r, g, b], axis=1), 0, 1).astype(np.float32)


def compute_normals(vertices, faces):
    """Compute vertex normals."""
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


def export_ply(data, filepath):
    """Export PLY with colors and normals."""
    verts = data['vertices']
    faces = data['faces']
    colors = data['colors']
    normals = data.get('normals')
    
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if normals is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        
        for i, v in enumerate(verts):
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


def visualize_sphere(data, title='', save_path=None):
    """Visualize the sphere."""
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 3D sphere - view 1
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_facecolor('#1a1a2e')
    
    verts = data['vertices']
    shape = data['theta'].shape
    X = verts[:, 0].reshape(shape)
    Y = verts[:, 1].reshape(shape)
    Z = verts[:, 2].reshape(shape)
    colors = data['colors'].reshape(shape[0], shape[1], 3)
    
    ax1.plot_surface(X, Y, Z, facecolors=colors, alpha=0.98, shade=True,
                     lightsource=plt.matplotlib.colors.LightSource(45, 30))
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=20, azim=45)
    ax1.set_title(title if title else 'Front View', color='white')
    ax1.axis('off')
    
    # 3D sphere - view 2 (rotated)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_facecolor('#1a1a2e')
    ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.98, shade=True,
                     lightsource=plt.matplotlib.colors.LightSource(-45, 30))
    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=20, azim=135)
    ax2.set_title('Side View', color='white')
    ax2.axis('off')
    
    # Height map
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor('#16213e')
    im = ax3.imshow(data['height'], cmap='magma', origin='lower',
                    extent=[0, 360, 0, 180], aspect='auto')
    ax3.set_xlabel('φ (longitude)', color='gray')
    ax3.set_ylabel('θ (latitude)', color='gray')
    ax3.set_title('Height Map', color='white')
    ax3.tick_params(colors='gray')
    plt.colorbar(im, ax=ax3, label='Height')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_comparison_figure(save_path=None):
    """Create comparison of different smoothing/depth settings."""
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    c = complex(-0.4, 0.6)
    
    configs = [
        {'smoothing_sigma': 1.5, 'canyon_depth': 0.3, 'invert': False},
        {'smoothing_sigma': 3.0, 'canyon_depth': 0.3, 'invert': False},
        {'smoothing_sigma': 5.0, 'canyon_depth': 0.3, 'invert': False},
        {'smoothing_sigma': 3.0, 'canyon_depth': 0.2, 'invert': False},
        {'smoothing_sigma': 3.0, 'canyon_depth': 0.4, 'invert': False},
        {'smoothing_sigma': 3.0, 'canyon_depth': 0.3, 'invert': True},
    ]
    
    titles = [
        'σ=1.5 (less smooth)',
        'σ=3.0 (balanced)',
        'σ=5.0 (very smooth)',
        'depth=0.2 (subtle)',
        'depth=0.4 (deep)',
        'Inverted (ridges)',
    ]
    
    for i, (cfg, title) in enumerate(zip(configs, titles)):
        data = create_julia_sphere_smooth(c, resolution=80, **cfg)
        
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.set_facecolor('#1a1a2e')
        
        verts = data['vertices']
        shape = data['theta'].shape
        X = verts[:, 0].reshape(shape)
        Y = verts[:, 1].reshape(shape)
        Z = verts[:, 2].reshape(shape)
        colors = data['colors'].reshape(shape[0], shape[1], 3)
        
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.98, shade=True)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title, color='white', fontsize=11)
        ax.axis('off')
    
    plt.suptitle(f'Julia Sphere Tuning: c = {c}', color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_lobe_gallery(save_path=None):
    """Create gallery of different Julia sets showing different lobe counts."""
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    julias = [
        (complex(-0.75, 0), 'San Marco\n(2 lobes)'),
        (complex(-0.122, 0.745), 'Douady Rabbit\n(3 lobes)'),
        (complex(-0.4, 0.6), 'Four-Lobe'),
        (complex(0.285, 0.01), 'Spiral'),
    ]
    
    for i, (c, title) in enumerate(julias):
        data = create_julia_sphere_smooth(
            c, resolution=100, 
            smoothing_sigma=3.0, 
            canyon_depth=0.35,
            invert=False
        )
        
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax.set_facecolor('#1a1a2e')
        
        verts = data['vertices']
        shape = data['theta'].shape
        X = verts[:, 0].reshape(shape)
        Y = verts[:, 1].reshape(shape)
        Z = verts[:, 2].reshape(shape)
        colors = data['colors'].reshape(shape[0], shape[1], 3)
        
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.98, shade=True)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=45)
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
        
        # Height map below
        ax2 = fig.add_subplot(2, 4, i + 5)
        ax2.set_facecolor('#16213e')
        ax2.imshow(data['height'], cmap='magma', origin='lower', aspect='auto')
        ax2.set_title(f'c = {c}', color='gray', fontsize=9)
        ax2.axis('off')
    
    plt.suptitle('Julia Spheres: Different Nodal Patterns', color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("REFINED SMOOTH JULIA SPHERE")
    print("=" * 60)
    
    output_dir = 'refined_julia_spheres'
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparison of parameters
    print("\nCreating parameter comparison...")
    fig1 = create_comparison_figure(f'{output_dir}/parameter_comparison.png')
    plt.close(fig1)
    
    # Gallery of different Julia sets
    print("\nCreating lobe gallery...")
    fig2 = create_lobe_gallery(f'{output_dir}/lobe_gallery.png')
    plt.close(fig2)
    
    # Create best version
    print("\nCreating optimized sphere...")
    
    # Balanced settings
    c = complex(-0.4, 0.6)
    data = create_julia_sphere_smooth(
        c,
        resolution=150,
        smoothing_sigma=3.5,
        canyon_depth=0.35,
        invert=False
    )
    
    export_ply(data, f'{output_dir}/julia_sphere_optimized.ply')
    fig3 = visualize_sphere(data, f'Smooth Julia Sphere: c = {c}',
                            f'{output_dir}/optimized_view.png')
    plt.close(fig3)
    
    # Inverted version (ridges instead of canyons)
    print("\nCreating inverted (ridge) version...")
    data_inv = create_julia_sphere_smooth(
        c,
        resolution=150,
        smoothing_sigma=3.5,
        canyon_depth=0.35,
        invert=True
    )
    
    export_ply(data_inv, f'{output_dir}/julia_sphere_ridges.ply')
    fig4 = visualize_sphere(data_inv, f'Julia Sphere (Ridges): c = {c}',
                            f'{output_dir}/ridges_view.png')
    plt.close(fig4)
    
    # Rabbit
    print("\nCreating Douady Rabbit...")
    data_rabbit = create_julia_sphere_smooth(
        complex(-0.122, 0.745),
        resolution=150,
        smoothing_sigma=3.0,
        canyon_depth=0.35
    )
    export_ply(data_rabbit, f'{output_dir}/rabbit_sphere.ply')
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print(f"  {output_dir}/parameter_comparison.png - Tuning options")
    print(f"  {output_dir}/lobe_gallery.png - Different Julia patterns")
    print(f"  {output_dir}/julia_sphere_optimized.ply - Best balanced mesh")
    print(f"  {output_dir}/julia_sphere_ridges.ply - Inverted (ridges)")
    print(f"  {output_dir}/rabbit_sphere.ply - Douady Rabbit")


if __name__ == '__main__':
    main()
