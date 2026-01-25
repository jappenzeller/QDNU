"""
================================================================================
QDNU JULIA VISUALIZATION PIPELINE - SINGLE FILE
================================================================================

Complete visualization chain for quantum PN neuron:

    Bloch E ──┐
              ├── (a,b,c) ──→ Julia c ──→ 2D Julia ──→ Height Map ──→ Spherical
    Bloch I ──┘                  │            │             │
                                 ▼            ▼             ▼
                            julia_2d     julia_canyon   julia_sphere

FEATURES:
- Two Bloch spheres (E/I qubits) with state vectors
- 2D Julia set projection (flat fractal plane)  
- 3D Canyon (Julia iterations → height = terrain)
- Spherical harmonic mapping (Julia wrapped on sphere)
- Vertex coloring based on iterations + Bloch correlation
- PLY/OBJ export for Blender
- 8-keyframe animation support

USAGE:
    # Basic
    from julia_vis_complete import JuliaVisualizer
    vis = JuliaVisualizer(resolution=128)
    vis.update(a=0.3, b=np.pi/4, c=0.3)
    vis.export_all('output/')
    
    # From explorer - add export button callback
    def on_export():
        vis.update(slider_a.val, slider_b.val, slider_c.val)
        vis.export_all('exports/')

DEPENDENCIES:
    pip install numpy scipy matplotlib

Optional for mesh export:
    pip install scikit-image

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import os

# Optional imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found - visualization disabled")


# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES
# ==============================================================================

@dataclass
class PNState:
    """
    PN Neuron state parameters.
    
    a: Excitatory amplitude [0, 1]
    b: Phase [0, 2π] 
    c: Inhibitory amplitude [0, 1]
    """
    a: float = 0.3
    b: float = 0.0
    c: float = 0.3
    
    @property
    def b_degrees(self) -> float:
        return np.degrees(self.b)
    
    def to_julia_c(self) -> complex:
        """Map PN state to complex Julia parameter."""
        real = -0.4 + 0.3 * np.cos(self.b)
        imag = 0.3 * np.sin(self.b) + 0.1 * (self.a - self.c)
        return complex(real, imag)
    
    def to_quaternion_c(self) -> Tuple[float, float, float, float]:
        """Map PN state to quaternion Julia parameter (w, x, y, z)."""
        return (
            -0.2 + 0.3 * np.cos(self.b),
            0.3 * np.sin(self.b) * (self.a - 0.5),
            0.5 * np.sin(self.b / 2),
            0.1 * np.cos(self.b) * (self.c - 0.5)
        )
    
    def to_bloch_E(self) -> Tuple[float, float, float]:
        """Bloch vector for excitatory qubit (x, y, z)."""
        theta = np.pi * (1 - self.a)
        phi = self.b
        return (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )
    
    def to_bloch_I(self) -> Tuple[float, float, float]:
        """Bloch vector for inhibitory qubit (x, y, z)."""
        theta = np.pi * (1 - self.c)
        phi = self.b + np.pi / 4
        return (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )
    
    def concurrence(self) -> float:
        """Approximate entanglement measure."""
        return 0.5 * abs(np.sin(self.b)) * (1 - abs(self.a - self.c))
    
    @classmethod
    def from_keyframe(cls, index: int, a: float = 0.3, c: float = 0.3) -> 'PNState':
        """Create state from keyframe index (0-7)."""
        b = (2 * np.pi * index) / 8
        return cls(a=a, b=b, c=c)


@dataclass
class Mesh:
    """Triangle mesh with optional attributes."""
    vertices: np.ndarray      # (N, 3) float32
    faces: np.ndarray         # (M, 3) int32
    normals: Optional[np.ndarray] = None   # (N, 3)
    colors: Optional[np.ndarray] = None    # (N, 3) RGB float [0,1]
    uvs: Optional[np.ndarray] = None       # (N, 2)
    
    @property
    def n_verts(self) -> int:
        return len(self.vertices)
    
    @property
    def n_faces(self) -> int:
        return len(self.faces)
    
    def compute_normals(self):
        """Compute vertex normals from faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10
        face_normals = face_normals / norms
        
        vertex_normals = np.zeros_like(self.vertices)
        for i, face in enumerate(self.faces):
            vertex_normals[face[0]] += face_normals[i]
            vertex_normals[face[1]] += face_normals[i]
            vertex_normals[face[2]] += face_normals[i]
        
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-10
        self.normals = vertex_normals / norms
        return self.normals


# ==============================================================================
# SECTION 2: COLORMAPS
# ==============================================================================

def colormap_plasma(t: np.ndarray) -> np.ndarray:
    """Attempt to reproduce plasma colormap. t in [0,1], returns (N,3) RGB."""
    t = np.clip(t, 0, 1)
    
    # Approximate plasma: purple → pink → orange → yellow
    r = np.clip(0.05 + 1.2 * t + 0.3 * np.sin(t * np.pi), 0, 1)
    g = np.clip(0.02 + 0.9 * t ** 1.5, 0, 1)
    b = np.clip(0.53 - 0.5 * t + 0.3 * np.sin((1 - t) * np.pi), 0, 1)
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def colormap_magma(t: np.ndarray) -> np.ndarray:
    """Approximate magma colormap."""
    t = np.clip(t, 0, 1)
    
    r = np.clip(t ** 0.5, 0, 1)
    g = np.clip(0.3 * t ** 1.5, 0, 1)
    b = np.clip(0.4 + 0.4 * t - 0.3 * t ** 2, 0, 1)
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def colormap_inferno(t: np.ndarray) -> np.ndarray:
    """Approximate inferno colormap."""
    t = np.clip(t, 0, 1)
    
    r = np.clip(1.5 * t ** 0.8, 0, 1)
    g = np.clip(1.2 * (t - 0.3) ** 1.2 * (t > 0.3), 0, 1)
    b = np.clip(0.5 * (1 - t) * t * 4, 0, 1)
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def apply_colormap(values: np.ndarray, cmap: str = 'plasma') -> np.ndarray:
    """Apply colormap to normalized values."""
    if cmap == 'magma':
        return colormap_magma(values)
    elif cmap == 'inferno':
        return colormap_inferno(values)
    else:
        return colormap_plasma(values)


# ==============================================================================
# SECTION 3: JULIA SET COMPUTATION
# ==============================================================================

def compute_julia_2d(c: complex, 
                     resolution: int = 256,
                     bounds: float = 2.0,
                     max_iter: int = 100) -> np.ndarray:
    """
    Compute 2D Julia set with smooth iteration counts.
    
    Args:
        c: Julia parameter (complex number)
        resolution: Grid resolution
        bounds: Spatial extent [-bounds, bounds]
        max_iter: Maximum iterations
    
    Returns:
        2D array of normalized iteration counts [0, 1]
    """
    x = np.linspace(-bounds, bounds, resolution)
    y = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    iterations = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)
    
    for n in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        mag = np.abs(Z)
        
        escaped = mask & (mag > 2)
        
        # Smooth iteration count
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth = n + 1 - np.log2(np.log2(mag + 1e-10) + 1e-10)
            smooth = np.nan_to_num(smooth, nan=n, posinf=n, neginf=n)
        
        iterations[escaped] = smooth[escaped]
        mask = mask & ~escaped
    
    iterations[mask] = max_iter
    
    # Normalize to [0, 1]
    return iterations / max_iter


def compute_julia_quaternion(cw: float, cx: float, cy: float, cz: float,
                              resolution: int = 128,
                              bounds: float = 1.5,
                              max_iter: int = 50,
                              slice_w: float = 0.0) -> np.ndarray:
    """
    Compute 2D slice of 4D quaternion Julia set.
    
    The slice fixes w=slice_w and varies (x, y, z=0).
    """
    lin = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(lin, lin)
    
    # Initialize quaternion: (w, x, y, z)
    qw = np.full_like(X, slice_w)
    qx = X.copy()
    qy = Y.copy()
    qz = np.zeros_like(X)
    
    iterations = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)
    
    for n in range(max_iter):
        # Quaternion square: q² = (w² - |v|², 2wv)
        qw_new = qw*qw - qx*qx - qy*qy - qz*qz
        qx_new = 2*qw*qx
        qy_new = 2*qw*qy
        qz_new = 2*qw*qz
        
        # Add c
        qw = qw_new + cw
        qx = qx_new + cx
        qy = qy_new + cy
        qz = qz_new + cz
        
        # Magnitude
        mag = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        escaped = mask & (mag > 2)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth = n + 1 - np.log2(np.log2(mag + 1e-10) + 1e-10)
            smooth = np.nan_to_num(smooth, nan=n, posinf=n, neginf=n)
        
        iterations[escaped] = smooth[escaped]
        mask = mask & ~escaped
    
    iterations[mask] = max_iter
    return iterations / max_iter


# ==============================================================================
# SECTION 4: MESH GENERATION
# ==============================================================================

def create_plane_mesh(iterations: np.ndarray,
                      bounds: float = 2.0,
                      z_offset: float = 0.0,
                      colormap: str = 'plasma') -> Mesh:
    """
    Create flat 2D Julia plane mesh.
    
    Args:
        iterations: 2D array of normalized iteration counts
        bounds: XY extent
        z_offset: Z position of plane
        colormap: Color scheme
    
    Returns:
        Mesh with vertices colored by iteration count
    """
    ny, nx = iterations.shape
    
    # Vertex grid
    x = np.linspace(-bounds, bounds, nx)
    y = np.linspace(-bounds, bounds, ny)
    X, Y = np.meshgrid(x, y)
    
    vertices = np.zeros((ny * nx, 3), dtype=np.float32)
    vertices[:, 0] = X.flatten()
    vertices[:, 1] = Y.flatten()
    vertices[:, 2] = z_offset
    
    # Colors
    colors = apply_colormap(iterations.flatten(), colormap)
    
    # UVs
    u, v = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    uvs = np.stack([u.flatten(), v.flatten()], axis=1).astype(np.float32)
    
    # Faces
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    faces = np.array(faces, dtype=np.int32)
    
    mesh = Mesh(vertices=vertices, faces=faces, colors=colors, uvs=uvs)
    mesh.compute_normals()
    return mesh


def create_canyon_mesh(iterations: np.ndarray,
                       bounds: float = 2.0,
                       height_scale: float = 1.0,
                       invert: bool = True,
                       orientation: str = 'xy',
                       colormap: str = 'inferno') -> Mesh:
    """
    Create 3D canyon mesh from Julia iterations.
    
    Height = iteration count (or inverted so boundary = ridges).
    
    Args:
        iterations: 2D array of normalized iteration counts
        bounds: XY extent
        height_scale: Z scale factor
        invert: If True, low iterations = high elevation
        orientation: 'xy' = horizontal terrain, 'xz' = vertical wall
        colormap: Color scheme
    """
    ny, nx = iterations.shape
    
    # Height values
    if invert:
        heights = (1 - iterations) * height_scale
    else:
        heights = iterations * height_scale
    
    # Create vertex grid
    lin = np.linspace(-bounds, bounds, nx)
    
    if orientation == 'xz':
        # Vertical wall (Julia on XZ plane, height goes into Y)
        X, Z = np.meshgrid(lin, lin)
        Y = heights
        
        vertices = np.zeros((ny * nx, 3), dtype=np.float32)
        vertices[:, 0] = X.flatten()
        vertices[:, 1] = Y.flatten()
        vertices[:, 2] = Z.flatten()
    else:
        # Horizontal terrain (Julia on XY plane, height is Z)
        X, Y = np.meshgrid(lin, lin)
        Z = heights
        
        vertices = np.zeros((ny * nx, 3), dtype=np.float32)
        vertices[:, 0] = X.flatten()
        vertices[:, 1] = Y.flatten()
        vertices[:, 2] = Z.flatten()
    
    # Colors
    colors = apply_colormap(iterations.flatten(), colormap)
    
    # Faces
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    faces = np.array(faces, dtype=np.int32)
    
    mesh = Mesh(vertices=vertices, faces=faces, colors=colors)
    mesh.compute_normals()
    return mesh


def create_sphere_mesh(iterations: np.ndarray,
                       base_radius: float = 1.0,
                       height_scale: float = 0.3,
                       center: Tuple[float, float, float] = (0, 0, 0),
                       colormap: str = 'plasma') -> Mesh:
    """
    Map Julia iterations onto sphere surface.
    
    Radius = base_radius + (1 - iterations) * height_scale
    
    Args:
        iterations: 2D array (rows=theta, cols=phi)
        base_radius: Base sphere radius
        height_scale: How much iterations affect radius
        center: Sphere center position
        colormap: Color scheme
    """
    ny, nx = iterations.shape
    
    # Spherical coordinates
    theta = np.linspace(0.01, np.pi - 0.01, ny)  # Avoid poles
    phi = np.linspace(0, 2 * np.pi, nx)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Radius modulated by iterations (low iterations = outward)
    R = base_radius + (1 - iterations) * height_scale
    
    # Spherical to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI) + center[0]
    Y = R * np.sin(THETA) * np.sin(PHI) + center[1]
    Z = R * np.cos(THETA) + center[2]
    
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    
    # Colors
    colors = apply_colormap(iterations.flatten(), colormap)
    
    # UVs
    u = PHI / (2 * np.pi)
    v = THETA / np.pi
    uvs = np.stack([u.flatten(), v.flatten()], axis=1).astype(np.float32)
    
    # Faces
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    faces = np.array(faces, dtype=np.int32)
    
    mesh = Mesh(vertices=vertices, faces=faces, colors=colors, uvs=uvs)
    mesh.compute_normals()
    return mesh


def create_bloch_sphere_mesh(resolution: int = 32,
                              radius: float = 1.0,
                              center: Tuple[float, float, float] = (0, 0, 0)) -> Mesh:
    """Create wireframe-style Bloch sphere mesh."""
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    X = radius * np.sin(THETA) * np.cos(PHI) + center[0]
    Y = radius * np.sin(THETA) * np.sin(PHI) + center[1]
    Z = radius * np.cos(THETA) + center[2]
    
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    
    # Light gray color
    colors = np.full((len(vertices), 3), 0.3, dtype=np.float32)
    
    faces = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            idx = j * resolution + i
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    faces = np.array(faces, dtype=np.int32)
    
    return Mesh(vertices=vertices, faces=faces, colors=colors)


# ==============================================================================
# SECTION 5: MESH EXPORT
# ==============================================================================

def export_ply(mesh: Mesh, filepath: str):
    """Export mesh to PLY format with vertex colors."""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {mesh.n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if mesh.normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        
        if mesh.colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write(f"element face {mesh.n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for i in range(mesh.n_verts):
            v = mesh.vertices[i]
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            
            if mesh.normals is not None:
                n = mesh.normals[i]
                line += f" {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
            
            if mesh.colors is not None:
                c = (mesh.colors[i] * 255).astype(np.uint8)
                line += f" {c[0]} {c[1]} {c[2]}"
            
            f.write(line + "\n")
        
        # Faces
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported: {filepath}")


def export_obj(mesh: Mesh, filepath: str):
    """Export mesh to OBJ format."""
    with open(filepath, 'w') as f:
        f.write(f"# Julia Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces\n\n")
        
        # Vertices (with colors as extension)
        for i, v in enumerate(mesh.vertices):
            line = f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if mesh.colors is not None:
                c = mesh.colors[i]
                line += f" {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            f.write(line + "\n")
        
        # Normals
        if mesh.normals is not None:
            f.write("\n")
            for n in mesh.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # Faces
        f.write("\n")
        for face in mesh.faces:
            if mesh.normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Exported: {filepath}")


# ==============================================================================
# SECTION 6: MAIN VISUALIZER CLASS
# ==============================================================================

class JuliaVisualizer:
    """
    Complete Julia visualization pipeline.
    
    Generates correlated visualizations from PN parameters:
    - 2D Julia plane
    - 3D Canyon (height-mapped terrain)
    - Spherical projection
    - Bloch sphere positions
    
    Usage:
        vis = JuliaVisualizer(resolution=128)
        vis.update(a=0.3, b=np.pi/4, c=0.3)
        vis.export_all('output/')
    """
    
    def __init__(self, resolution: int = 128, bounds: float = 2.0, max_iter: int = 100):
        """
        Initialize visualizer.
        
        Args:
            resolution: Grid resolution for Julia computation
            bounds: Spatial extent
            max_iter: Maximum Julia iterations
        """
        self.resolution = resolution
        self.bounds = bounds
        self.max_iter = max_iter
        
        # Current state
        self.state = PNState()
        
        # Computed data
        self.julia_2d = None
        self.plane_mesh = None
        self.canyon_mesh = None
        self.sphere_mesh = None
        
        # Bloch trace history
        self.trace_E: List[Tuple[float, float, float]] = []
        self.trace_I: List[Tuple[float, float, float]] = []
        self.max_trace = 10
    
    def update(self, a: float, b: float, c: float):
        """
        Update state and recompute all visualizations.
        
        Args:
            a: Excitatory parameter [0, 1]
            b: Phase [0, 2π]
            c: Inhibitory parameter [0, 1]
        """
        self.state = PNState(a=a, b=b, c=c)
        self._compute_all()
        self._update_traces()
    
    def update_from_keyframe(self, keyframe: int):
        """Update from keyframe index (0-7)."""
        b = (2 * np.pi * keyframe) / 8
        self.update(self.state.a, b, self.state.c)
    
    def _compute_all(self):
        """Recompute all visualization components."""
        # Get Julia parameter
        julia_c = self.state.to_julia_c()
        
        # Compute 2D Julia
        self.julia_2d = compute_julia_2d(
            julia_c, 
            resolution=self.resolution,
            bounds=self.bounds,
            max_iter=self.max_iter
        )
        
        # Create meshes
        self.plane_mesh = create_plane_mesh(
            self.julia_2d, 
            bounds=self.bounds,
            z_offset=-2,
            colormap='plasma'
        )
        
        self.canyon_mesh = create_canyon_mesh(
            self.julia_2d,
            bounds=self.bounds,
            height_scale=1.5,
            invert=True,
            orientation='xz',
            colormap='inferno'
        )
        
        self.sphere_mesh = create_sphere_mesh(
            self.julia_2d,
            base_radius=1.0,
            height_scale=0.4,
            center=(0, 4, 0),
            colormap='plasma'
        )
    
    def _update_traces(self):
        """Update Bloch sphere trace history."""
        self.trace_E.append(self.state.to_bloch_E())
        self.trace_I.append(self.state.to_bloch_I())
        
        if len(self.trace_E) > self.max_trace:
            self.trace_E.pop(0)
        if len(self.trace_I) > self.max_trace:
            self.trace_I.pop(0)
    
    def clear_traces(self):
        """Clear Bloch sphere traces."""
        self.trace_E = []
        self.trace_I = []
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            'a': self.state.a,
            'b': self.state.b,
            'b_deg': self.state.b_degrees,
            'c': self.state.c,
            'julia_c': self.state.to_julia_c(),
            'quaternion_c': self.state.to_quaternion_c(),
            'bloch_E': self.state.to_bloch_E(),
            'bloch_I': self.state.to_bloch_I(),
            'concurrence': self.state.concurrence(),
            'trace_E': list(self.trace_E),
            'trace_I': list(self.trace_I),
        }
    
    def export_all(self, output_dir: str = '.', prefix: str = 'julia', fmt: str = 'ply'):
        """
        Export all meshes.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
            fmt: 'ply' or 'obj'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        export_fn = export_ply if fmt == 'ply' else export_obj
        ext = fmt
        
        if self.plane_mesh:
            export_fn(self.plane_mesh, os.path.join(output_dir, f"{prefix}_plane.{ext}"))
        
        if self.canyon_mesh:
            export_fn(self.canyon_mesh, os.path.join(output_dir, f"{prefix}_canyon.{ext}"))
        
        if self.sphere_mesh:
            export_fn(self.sphere_mesh, os.path.join(output_dir, f"{prefix}_sphere.{ext}"))
        
        # Export 2D image if matplotlib available
        if HAS_MPL and self.julia_2d is not None:
            img_path = os.path.join(output_dir, f"{prefix}_2d.png")
            colors = apply_colormap(self.julia_2d, 'plasma')
            plt.imsave(img_path, colors)
            print(f"Exported: {img_path}")
        
        # Export state info
        info_path = os.path.join(output_dir, f"{prefix}_state.txt")
        with open(info_path, 'w') as f:
            info = self.get_state_info()
            f.write(f"a = {info['a']}\n")
            f.write(f"b = {info['b']} ({info['b_deg']:.1f} deg)\n")
            f.write(f"c = {info['c']}\n")
            f.write(f"Julia c = {info['julia_c']}\n")
            f.write(f"Bloch E = {info['bloch_E']}\n")
            f.write(f"Bloch I = {info['bloch_I']}\n")
            f.write(f"Concurrence = {info['concurrence']:.4f}\n")
        print(f"Exported: {info_path}")
    
    def export_animation(self, output_dir: str = 'animation', 
                         n_frames: int = 60,
                         fmt: str = 'ply'):
        """
        Export animation sequence.
        
        Args:
            output_dir: Output directory
            n_frames: Number of frames
            fmt: 'ply' or 'obj'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {n_frames} animation frames...")
        
        for i in range(n_frames):
            b = (2 * np.pi * i) / n_frames
            self.update(self.state.a, b, self.state.c)
            
            prefix = f"frame_{i:04d}"
            
            export_fn = export_ply if fmt == 'ply' else export_obj
            ext = fmt
            
            export_fn(self.canyon_mesh, os.path.join(output_dir, f"{prefix}_canyon.{ext}"))
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_frames}")
        
        print(f"Animation exported to: {output_dir}/")


# ==============================================================================
# SECTION 7: MATPLOTLIB VISUALIZATION (for testing)
# ==============================================================================

def plot_pipeline_overview(vis: JuliaVisualizer, save_path: Optional[str] = None):
    """Create overview plot of all visualization components."""
    if not HAS_MPL:
        print("matplotlib required for plotting")
        return None
    
    fig = plt.figure(figsize=(16, 10))
    info = vis.get_state_info()
    
    # 1. 2D Julia
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(vis.julia_2d, cmap='magma', origin='lower', 
               extent=[-vis.bounds, vis.bounds, -vis.bounds, vis.bounds])
    ax1.set_title(f'2D Julia\nc = {info["julia_c"]:.3f}')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    
    # 2. Height map
    ax2 = fig.add_subplot(2, 3, 2)
    heights = 1 - vis.julia_2d  # Inverted
    im = ax2.imshow(heights, cmap='terrain', origin='lower')
    ax2.set_title('Height Map')
    plt.colorbar(im, ax=ax2)
    
    # 3. Canyon 3D
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    step = max(1, vis.resolution // 32)
    X, Y = np.meshgrid(
        np.linspace(-vis.bounds, vis.bounds, vis.resolution)[::step],
        np.linspace(-vis.bounds, vis.bounds, vis.resolution)[::step]
    )
    Z = heights[::step, ::step] * 1.5
    ax3.plot_surface(X, Y, Z, cmap='inferno', alpha=0.8)
    ax3.set_title('Canyon')
    
    # 4. Bloch E
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    _plot_bloch_sphere(ax4, info['bloch_E'], 'Bloch E', 'orange')
    
    # 5. Bloch I
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    _plot_bloch_sphere(ax5, info['bloch_I'], 'Bloch I', 'cyan')
    
    # 6. Sphere projection
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    _plot_sphere_projection(ax6, vis)
    
    fig.suptitle(f'PN State: a={info["a"]:.2f}, b={info["b_deg"]:.1f}°, c={info["c"]:.2f}\n'
                 f'Concurrence: {info["concurrence"]:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def _plot_bloch_sphere(ax, bloch_vec, title, color):
    """Helper to plot Bloch sphere."""
    # Wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, alpha=0.1, color='gray')
    
    # Axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'r-', alpha=0.3)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'g-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'b-', alpha=0.3)
    
    # State vector
    ax.quiver(0, 0, 0, bloch_vec[0], bloch_vec[1], bloch_vec[2],
              color=color, arrow_length_ratio=0.1, linewidth=2)
    ax.scatter([bloch_vec[0]], [bloch_vec[1]], [bloch_vec[2]], 
               c=color, s=80, edgecolors='k')
    
    ax.set_title(title)
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1, 1, 1])


def _plot_sphere_projection(ax, vis):
    """Helper to plot spherical projection."""
    step = max(1, vis.resolution // 24)
    
    ny, nx = vis.julia_2d.shape
    theta = np.linspace(0.1, np.pi - 0.1, ny)[::step]
    phi = np.linspace(0, 2 * np.pi, nx)[::step]
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    julia_sub = vis.julia_2d[::step, ::step]
    R = 1.0 + (1 - julia_sub) * 0.3
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.magma(julia_sub), alpha=0.9)
    ax.set_title('Spherical Projection')
    ax.set_box_aspect([1, 1, 1])


# ==============================================================================
# SECTION 8: KEYFRAME TABLE
# ==============================================================================

def print_keyframe_table():
    """Print 8-keyframe reference table."""
    print("\n" + "=" * 70)
    print("8-KEYFRAME PHASE CYCLE")
    print("=" * 70)
    print(f"{'KF':<4} {'b (rad)':<10} {'b (deg)':<10} {'Julia c':<20} {'Description'}")
    print("-" * 70)
    
    descriptions = [
        "Phase origin",
        "Rising phase",
        "Quarter cycle",
        "Approaching inversion",
        "Half cycle",
        "Descending phase",
        "Three-quarter cycle",
        "Return to origin"
    ]
    
    for i in range(8):
        state = PNState.from_keyframe(i)
        c = state.to_julia_c()
        print(f"{i:<4} {state.b:<10.4f} {state.b_degrees:<10.1f} {str(c):<20} {descriptions[i]}")
    
    print("=" * 70)


# ==============================================================================
# SECTION 9: MAIN / CLI
# ==============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("QDNU JULIA VISUALIZATION PIPELINE")
    print("=" * 70)
    
    # Show keyframe table
    print_keyframe_table()
    
    # Create visualizer
    vis = JuliaVisualizer(resolution=128, max_iter=100)
    
    # Test with keyframe 2 (90°)
    print("\nComputing visualization for keyframe 2 (90°)...")
    vis.update_from_keyframe(2)
    
    # Print state
    info = vis.get_state_info()
    print(f"\nState:")
    print(f"  a = {info['a']:.3f}")
    print(f"  b = {info['b']:.4f} ({info['b_deg']:.1f}°)")
    print(f"  c = {info['c']:.3f}")
    print(f"  Julia c = {info['julia_c']}")
    print(f"  Bloch E = ({info['bloch_E'][0]:.3f}, {info['bloch_E'][1]:.3f}, {info['bloch_E'][2]:.3f})")
    print(f"  Bloch I = ({info['bloch_I'][0]:.3f}, {info['bloch_I'][1]:.3f}, {info['bloch_I'][2]:.3f})")
    print(f"  Concurrence = {info['concurrence']:.4f}")
    
    # Export
    print("\nExporting meshes...")
    vis.export_all('julia_output', prefix='julia_kf2')
    
    # Plot if matplotlib available
    if HAS_MPL:
        print("\nGenerating overview plot...")
        fig = plot_pipeline_overview(vis, 'julia_output/julia_overview.png')
        plt.show()
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print("  julia_output/julia_kf2_plane.ply   - 2D Julia plane")
    print("  julia_output/julia_kf2_canyon.ply  - 3D height-mapped canyon")
    print("  julia_output/julia_kf2_sphere.ply  - Spherical projection")
    print("  julia_output/julia_kf2_2d.png      - 2D Julia image")
    print("  julia_output/julia_kf2_state.txt   - State parameters")
    print("\nImport PLY files into Blender for rendering")


if __name__ == '__main__':
    main()
