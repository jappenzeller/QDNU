# QDNU Julia Visualization Pipeline - Integration Prompt

## Overview

Integrate the Julia visualization pipeline into the existing A-Gate Quantum Neuron Explorer. The visualization chain creates correlated outputs from PN parameters:

```
Bloch E ──┐
          ├── (a,b,c) ──→ Julia c ──→ 2D Julia ──→ Height Map ──→ Spherical
Bloch I ──┘                              │             │              │
                                         ▼             ▼              ▼
                                   flat plane     3D canyon     sphere surface
```

## Requirements

Add to existing explorer:
1. 2D Julia plane displayed perpendicular to current Julia fingerprint view
2. 3D Canyon mesh - Julia iterations extruded into height (terrain/relief effect)
3. Spherical harmonic projection - Julia wrapped onto sphere
4. Export button to save all meshes as PLY files for Blender

## Integration Code

Add this class to your explorer or import from separate module:

```python
import numpy as np
import os

class JuliaVisualizer:
    """
    Julia visualization pipeline for A-Gate explorer.
    
    Usage:
        vis = JuliaVisualizer(resolution=128)
        vis.update(a, b, c)  # Call when sliders change
        vis.export_all('output/')  # Call on export button
    """
    
    def __init__(self, resolution: int = 128, bounds: float = 2.0, max_iter: int = 100):
        self.resolution = resolution
        self.bounds = bounds
        self.max_iter = max_iter
        self.julia_2d = None
        self.a = 0.3
        self.b = 0.0
        self.c = 0.3
    
    def update(self, a: float, b: float, c: float):
        """Update state and recompute Julia set."""
        self.a, self.b, self.c = a, b, c
        julia_c = self._pn_to_julia_c(a, b, c)
        self.julia_2d = self._compute_julia(julia_c)
    
    def _pn_to_julia_c(self, a: float, b: float, c: float) -> complex:
        """Map PN parameters to Julia constant."""
        real = -0.4 + 0.3 * np.cos(b)
        imag = 0.3 * np.sin(b) + 0.1 * (a - c)
        return complex(real, imag)
    
    def _compute_julia(self, c: complex) -> np.ndarray:
        """Compute 2D Julia set with smooth iteration counts."""
        x = np.linspace(-self.bounds, self.bounds, self.resolution)
        y = np.linspace(-self.bounds, self.bounds, self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        iterations = np.zeros_like(X, dtype=np.float32)
        mask = np.ones_like(X, dtype=bool)
        
        for n in range(self.max_iter):
            Z[mask] = Z[mask]**2 + c
            mag = np.abs(Z)
            escaped = mask & (mag > 2)
            
            with np.errstate(invalid='ignore', divide='ignore'):
                smooth = n + 1 - np.log2(np.log2(mag + 1e-10) + 1e-10)
                smooth = np.nan_to_num(smooth, nan=n)
            
            iterations[escaped] = smooth[escaped]
            mask = mask & ~escaped
        
        iterations[mask] = self.max_iter
        return iterations / self.max_iter
    
    def get_canyon_mesh(self, height_scale: float = 1.5, invert: bool = True):
        """Get 3D canyon mesh data. Returns (vertices, faces, colors)."""
        if self.julia_2d is None:
            return None, None, None
        
        ny, nx = self.julia_2d.shape
        heights = (1 - self.julia_2d) * height_scale if invert else self.julia_2d * height_scale
        
        x = np.linspace(-self.bounds, self.bounds, nx)
        y = np.linspace(-self.bounds, self.bounds, ny)
        X, Y = np.meshgrid(x, y)
        
        vertices = np.stack([X.flatten(), Y.flatten(), heights.flatten()], axis=1)
        colors = self._colormap(self.julia_2d.flatten())
        
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx = j * nx + i
                faces.append([idx, idx + 1, idx + nx])
                faces.append([idx + 1, idx + nx + 1, idx + nx])
        
        return vertices, np.array(faces), colors
    
    def get_sphere_mesh(self, radius: float = 1.0, height_scale: float = 0.3):
        """Get spherical projection mesh. Returns (vertices, faces, colors)."""
        if self.julia_2d is None:
            return None, None, None
        
        ny, nx = self.julia_2d.shape
        theta = np.linspace(0.01, np.pi - 0.01, ny)
        phi = np.linspace(0, 2 * np.pi, nx)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        R = radius + (1 - self.julia_2d) * height_scale
        
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        colors = self._colormap(self.julia_2d.flatten())
        
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx = j * nx + i
                faces.append([idx, idx + 1, idx + nx])
                faces.append([idx + 1, idx + nx + 1, idx + nx])
        
        return vertices, np.array(faces), colors
    
    def _colormap(self, t: np.ndarray) -> np.ndarray:
        """Plasma-like colormap. t in [0,1], returns RGB in [0,1]."""
        t = np.clip(t, 0, 1)
        r = np.clip(0.05 + 1.2 * t + 0.3 * np.sin(t * np.pi), 0, 1)
        g = np.clip(0.02 + 0.9 * t ** 1.5, 0, 1)
        b = np.clip(0.53 - 0.5 * t + 0.3 * np.sin((1 - t) * np.pi), 0, 1)
        return np.stack([r, g, b], axis=1).astype(np.float32)
    
    def export_ply(self, vertices, faces, colors, filepath):
        """Export mesh to PLY format."""
        with open(filepath, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            
            for i, v in enumerate(vertices):
                c = (colors[i] * 255).astype(np.uint8)
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"Exported: {filepath}")
    
    def export_all(self, output_dir: str = '.', prefix: str = 'julia'):
        """Export all meshes."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Canyon
        verts, faces, colors = self.get_canyon_mesh()
        if verts is not None:
            self.export_ply(verts, faces, colors, 
                          os.path.join(output_dir, f"{prefix}_canyon.ply"))
        
        # Sphere
        verts, faces, colors = self.get_sphere_mesh()
        if verts is not None:
            self.export_ply(verts, faces, colors,
                          os.path.join(output_dir, f"{prefix}_sphere.ply"))
        
        # 2D image
        if self.julia_2d is not None:
            import matplotlib.pyplot as plt
            colors_2d = self._colormap(self.julia_2d.flatten()).reshape(
                self.resolution, self.resolution, 3)
            plt.imsave(os.path.join(output_dir, f"{prefix}_2d.png"), colors_2d)
            print(f"Exported: {prefix}_2d.png")
        
        # State info
        with open(os.path.join(output_dir, f"{prefix}_state.txt"), 'w') as f:
            f.write(f"a = {self.a}\nb = {self.b} ({np.degrees(self.b):.1f} deg)\nc = {self.c}\n")
            f.write(f"Julia c = {self._pn_to_julia_c(self.a, self.b, self.c)}\n")
        print(f"Exported: {prefix}_state.txt")
```

## Explorer Integration

Add to your existing explorer class:

```python
# In __init__:
self.julia_vis = JuliaVisualizer(resolution=128)

# In slider callback (when a, b, or c changes):
def on_slider_change(self, val):
    # ... existing code ...
    self.julia_vis.update(self.a, self.b, self.c)
    self.update_canyon_display()  # If displaying in matplotlib
    self.update_sphere_display()  # If displaying in matplotlib

# Add export button:
self.btn_export = Button(ax_export, 'Export Meshes')
self.btn_export.on_clicked(self.on_export)

def on_export(self, event):
    self.julia_vis.export_all('julia_exports', prefix=f'julia_b{int(np.degrees(self.b))}')
```

## Display in Matplotlib (Optional)

To show canyon/sphere in the explorer window:

```python
from mpl_toolkits.mplot3d import Axes3D

# Add 3D axes for canyon
self.ax_canyon = self.fig.add_subplot(2, 4, 7, projection='3d')

def update_canyon_display(self):
    self.ax_canyon.clear()
    
    verts, faces, colors = self.julia_vis.get_canyon_mesh()
    if verts is None:
        return
    
    # Subsample for performance
    step = max(1, self.julia_vis.resolution // 32)
    ny = nx = self.julia_vis.resolution
    
    X = verts[:, 0].reshape(ny, nx)[::step, ::step]
    Y = verts[:, 1].reshape(ny, nx)[::step, ::step]
    Z = verts[:, 2].reshape(ny, nx)[::step, ::step]
    C = self.julia_vis.julia_2d[::step, ::step]
    
    self.ax_canyon.plot_surface(X, Y, Z, facecolors=plt.cm.inferno(C), alpha=0.9)
    self.ax_canyon.set_title('Canyon')
    
    self.fig.canvas.draw_idle()
```

## Blender Import

After exporting:

1. File → Import → Stanford (.ply)
2. Select `julia_canyon.ply`
3. Repeat for `julia_sphere.ply`
4. Position canyon perpendicular to sphere to show correlation

The vertex colors are embedded in the PLY files and will import automatically.

## 8-Keyframe Animation

To export animation frames:

```python
def export_animation(self, n_frames=60):
    for i in range(n_frames):
        b = (2 * np.pi * i) / n_frames
        self.julia_vis.update(self.a, b, self.c)
        self.julia_vis.export_all(
            'animation_frames', 
            prefix=f'frame_{i:04d}'
        )
```

## Key Mappings

| PN State | Julia Parameter | Visualization |
|----------|-----------------|---------------|
| a (excitatory) | Affects imaginary part | Bloch E theta angle |
| b (phase) | Rotates c in complex plane | Morphs fractal shape |
| c (inhibitory) | Affects imaginary part | Bloch I theta angle |

The correlation: as b sweeps 0 → 2π, the Julia set morphs through a complete cycle, the canyon terrain shifts, and the spherical surface deforms - all driven by the same underlying parameters.

## Files Generated

| File | Description |
|------|-------------|
| `julia_canyon.ply` | Height-mapped terrain mesh |
| `julia_sphere.ply` | Spherical projection mesh |
| `julia_2d.png` | 2D Julia set image |
| `julia_state.txt` | Parameter values |

## Dependencies

- numpy (required)
- matplotlib (for display and image export)
- No additional dependencies for mesh export
