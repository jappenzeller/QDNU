"""
================================================================================
QDNU JULIA 3D BOUNDARY CROSSING VISUALIZATION
================================================================================

Visualize Julia set with escape-time as Z-height to reveal depth dynamics
during Mandelbrot boundary crossing.

HYPOTHESIS: The depth at which points get trapped may change BEFORE the 2D
boundary visually changes. The tornado/dendrite could be deepening (warning
sign) before black appears in 2D view. This could be the pre-ictal signal.

Visual Features -> Brain State Mapping:
- Shallow, no wells     -> Interictal (healthy)
- Deepening tornados    -> Pre-ictal (warning)
- Wells punch through   -> Ictal (seizure onset)
- Stable deep structure -> Sustained ictal

Run: python -m visualization.interactive.julia_3d_boundary

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Tuple, Optional
import os

# =============================================================================
# STYLING
# =============================================================================

DARK_BG = '#0a0a1a'
PANEL_BG = '#16213e'
ACCENT_CYAN = '#00d4ff'
ACCENT_ORANGE = '#ff9500'
ACCENT_PURPLE = '#a855f7'
ACCENT_GREEN = '#22c55e'
ACCENT_RED = '#ff6b6b'
TEXT_COLOR = '#cccccc'
TEXT_DIM = '#666666'

TAU = 2 * np.pi

# =============================================================================
# JULIA HEIGHTFIELD COMPUTATION
# =============================================================================

def compute_julia_heightfield(c: complex,
                               resolution: int = 400,
                               x_range: Tuple[float, float] = (-1.6, 1.6),
                               y_range: Tuple[float, float] = (-1.2, 1.2),
                               max_iter: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Julia set with escape-time as height.

    Returns:
        X, Y: meshgrid coordinates
        Z: escape-time heightfield (normalized 0-1, with interior = 1.0)
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y_res = int(resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    y = np.linspace(y_range[0], y_range[1], y_res)
    X, Y = np.meshgrid(x, y)
    Z_complex = X + 1j * Y

    # Escape time array
    escape_time = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)

    for i in range(max_iter):
        Z_complex[mask] = Z_complex[mask] ** 2 + c
        escaped = np.abs(Z_complex) > 2
        new_escaped = escaped & mask

        if np.any(new_escaped):
            # Smooth escape time using continuous potential
            log_zn = np.log(np.abs(Z_complex[new_escaped]) + 1e-10)
            nu = np.log(log_zn / np.log(2) + 1e-10) / np.log(2)
            escape_time[new_escaped] = (i + 1 - nu) / max_iter

        mask[escaped] = False
        if not np.any(mask):
            break

    # Interior points (never escaped) get maximum depth
    escape_time[mask] = 1.0

    return X, Y, escape_time


def compute_depth_metrics(heightfield: np.ndarray) -> dict:
    """
    Compute depth metrics from heightfield.

    Returns dict with:
    - max_depth: maximum depth (1.0 = interior)
    - mean_depth: average escape time
    - depth_variance: how spread out depths are
    - interior_fraction: fraction of points in set
    - well_count: approximate number of deep wells
    """
    interior_mask = heightfield >= 0.99
    interior_fraction = np.mean(interior_mask)

    # Depth metrics (using escape time as depth proxy)
    max_depth = np.max(heightfield)
    mean_depth = np.mean(heightfield)
    depth_variance = np.var(heightfield)

    # Count deep wells (connected regions with depth > 0.8)
    from scipy import ndimage
    deep_mask = heightfield > 0.8
    labeled, well_count = ndimage.label(deep_mask)

    return {
        'max_depth': max_depth,
        'mean_depth': mean_depth,
        'depth_variance': depth_variance,
        'interior_fraction': interior_fraction,
        'well_count': well_count,
    }


def is_in_mandelbrot(c: complex, max_iter: int = 100) -> bool:
    """Check if c is in the Mandelbrot set."""
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def distance_to_mandelbrot_boundary(c: complex, samples: int = 36) -> float:
    """
    Estimate distance to nearest Mandelbrot boundary point.

    Returns positive if outside, negative if inside.
    """
    in_set = is_in_mandelbrot(c)

    # Search for boundary by expanding/contracting radius
    for r in np.linspace(0, 0.5, 20):
        for theta in np.linspace(0, TAU, samples):
            test_c = c + r * np.exp(1j * theta)
            if is_in_mandelbrot(test_c) != in_set:
                # Found transition
                return r if not in_set else -r

    return 0.5 if not in_set else -0.5


# =============================================================================
# JULIA C MAPPING (from PN parameters)
# =============================================================================

def pn_to_julia_c(a: float, b: float, c_param: float) -> complex:
    """
    Map PN parameters to Julia c using boundary-crossing mapping.

    Parameters:
        a: excitatory (0-1)
        b: phase as tau fraction (0-1)
        c_param: inhibitory (0-1)
    """
    # Center at cusp between main cardioid and period-2 bulb
    center_real = -0.75
    radius = 0.35

    phase = b * TAU
    real = center_real + radius * np.cos(phase)
    imag = radius * np.sin(phase) + 0.1 * (a - c_param)

    return complex(real, imag)


# =============================================================================
# PLY EXPORT
# =============================================================================

def export_heightfield_ply(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           filepath: str, height_scale: float = 0.5):
    """Export heightfield as PLY mesh for Blender."""
    ny, nx = X.shape

    # Scale Z for better visualization
    Z_scaled = Z * height_scale

    # Create vertices
    vertices = []
    colors = []

    # Colormap: depth gradient (bright surface -> dark depths)
    for j in range(ny):
        for i in range(nx):
            x, y, z = X[j, i], Y[j, i], Z_scaled[j, i]
            vertices.append((x, y, z))

            # Color based on depth
            depth = Z[j, i]
            if depth >= 0.99:
                # Interior - deep purple/black
                r, g, b = 0.1, 0.02, 0.15
            else:
                # Gradient from cyan (surface) to purple (deep)
                t = depth
                r = 0.1 + 0.4 * (1 - t) + 0.3 * t
                g = 0.6 * (1 - t) + 0.1 * t
                b = 0.8 * (1 - t) + 0.6 * t
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

    # Create faces
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])

    # Write PLY
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")

        for i, v in enumerate(vertices):
            c = colors[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Exported: {filepath}")


# =============================================================================
# 3D VIEWER CLASS
# =============================================================================

class Julia3DBoundaryViewer:
    """
    Interactive 3D viewer for Julia set boundary crossing analysis.
    """

    def __init__(self, resolution: int = 200):
        self.resolution = resolution
        self.max_iter = 256

        # PN parameters
        self.a = 0.31
        self.b = 0.0  # tau fraction (0-1)
        self.c_param = 0.32

        # View parameters
        self.elev = 45
        self.azim = -60
        self.height_scale = 0.5

        # Animation
        self.playing = False
        self.sweep_speed = 0.005
        self.anim = None

        # Data
        self.X = None
        self.Y = None
        self.Z = None
        self.metrics = None
        self.metrics_history = []

        # Setup
        self._setup_figure()
        self._compute()
        self._update_all()

    def _setup_figure(self):
        """Create figure with 3D plot and controls."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(DARK_BG)
        plt.suptitle('QDNU Julia 3D Boundary Crossing', color='white',
                     fontsize=14, y=0.98)

        # Main 3D plot
        self.ax_3d = self.fig.add_axes([0.02, 0.25, 0.55, 0.70], projection='3d')
        self.ax_3d.set_facecolor(DARK_BG)

        # 2D top-down view (for comparison)
        self.ax_2d = self.fig.add_axes([0.60, 0.55, 0.38, 0.40])
        self.ax_2d.set_facecolor(DARK_BG)

        # Metrics display
        self.ax_metrics = self.fig.add_axes([0.60, 0.25, 0.38, 0.25])
        self.ax_metrics.set_facecolor(PANEL_BG)

        # Depth history plot
        self.ax_history = self.fig.add_axes([0.02, 0.08, 0.55, 0.12])
        self.ax_history.set_facecolor(PANEL_BG)

        # Controls
        self._setup_controls()

    def _setup_controls(self):
        """Setup control sliders and buttons."""
        # Phase slider (b)
        self.fig.text(0.60, 0.20, 'Phase (b):', color=ACCENT_GREEN, fontsize=10)
        ax_b = self.fig.add_axes([0.68, 0.195, 0.20, 0.02])
        self.slider_b = Slider(ax_b, '', 0, 1, valinit=0.0,
                               color=ACCENT_GREEN, track_color=PANEL_BG)
        self.slider_b.on_changed(self._on_b_change)

        # Height scale slider
        self.fig.text(0.60, 0.16, 'Height:', color=TEXT_DIM, fontsize=10)
        ax_h = self.fig.add_axes([0.68, 0.155, 0.20, 0.02])
        self.slider_h = Slider(ax_h, '', 0.1, 2.0, valinit=0.5,
                               color='#888888', track_color=PANEL_BG)
        self.slider_h.on_changed(self._on_height_change)

        # View angle sliders
        self.fig.text(0.60, 0.12, 'Elevation:', color=TEXT_DIM, fontsize=10)
        ax_elev = self.fig.add_axes([0.68, 0.115, 0.20, 0.02])
        self.slider_elev = Slider(ax_elev, '', 0, 90, valinit=45,
                                  color='#888888', track_color=PANEL_BG)
        self.slider_elev.on_changed(self._on_view_change)

        # Buttons
        btn_y = 0.02
        btn_w = 0.08
        btn_h = 0.03

        ax_play = self.fig.add_axes([0.60, btn_y, btn_w, btn_h])
        ax_stop = self.fig.add_axes([0.69, btn_y, btn_w, btn_h])
        ax_export = self.fig.add_axes([0.78, btn_y, btn_w, btn_h])
        ax_sequence = self.fig.add_axes([0.87, btn_y, btn_w, btn_h])

        self.btn_play = Button(ax_play, 'Sweep', color='#1a4d1a', hovercolor='#2d6a2d')
        self.btn_stop = Button(ax_stop, 'Stop', color='#4d1a1a', hovercolor='#6a2d2d')
        self.btn_export = Button(ax_export, 'Export PLY', color='#3d1a4d', hovercolor='#5a2d6a')
        self.btn_sequence = Button(ax_sequence, 'Sequence', color='#1a3d4d', hovercolor='#2d5a6a')

        for btn in [self.btn_play, self.btn_stop, self.btn_export, self.btn_sequence]:
            btn.label.set_color('white')
            btn.label.set_fontsize(9)

        self.btn_play.on_clicked(self._on_play)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_export.on_clicked(self._on_export)
        self.btn_sequence.on_clicked(self._on_export_sequence)

        # Info text
        self.info_text = self.fig.text(0.02, 0.02, '', color=TEXT_DIM,
                                        fontsize=9, family='monospace')

    def _compute(self):
        """Compute Julia heightfield and metrics."""
        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        self.X, self.Y, self.Z = compute_julia_heightfield(
            julia_c, self.resolution, max_iter=self.max_iter
        )

        try:
            self.metrics = compute_depth_metrics(self.Z)
        except ImportError:
            # scipy not available
            self.metrics = {
                'max_depth': np.max(self.Z),
                'mean_depth': np.mean(self.Z),
                'depth_variance': np.var(self.Z),
                'interior_fraction': np.mean(self.Z >= 0.99),
                'well_count': 0,
            }

        # Track history
        self.metrics_history.append({
            'b': self.b,
            'max_depth': self.metrics['max_depth'],
            'mean_depth': self.metrics['mean_depth'],
            'interior_fraction': self.metrics['interior_fraction'],
        })
        if len(self.metrics_history) > 200:
            self.metrics_history.pop(0)

    def _update_all(self):
        """Update all displays."""
        self._update_3d()
        self._update_2d()
        self._update_metrics()
        self._update_history()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _update_3d(self):
        """Update 3D surface plot."""
        self.ax_3d.clear()
        self.ax_3d.set_facecolor(DARK_BG)

        # Scale height
        Z_scaled = self.Z * self.height_scale

        # Custom colormap: cyan (shallow) -> purple (deep) -> black (interior)
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            (0.0, '#00d4ff'),   # Shallow - cyan
            (0.3, '#4dabf7'),   # Medium - blue
            (0.6, '#a855f7'),   # Deep - purple
            (0.9, '#3d1a4d'),   # Very deep - dark purple
            (1.0, '#0a0a1a'),   # Interior - near black
        ]
        cmap = LinearSegmentedColormap.from_list('depth',
            [(p, c) for p, c in colors])

        # Plot surface
        surf = self.ax_3d.plot_surface(
            self.X, self.Y, Z_scaled,
            facecolors=cmap(self.Z),
            alpha=0.95,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Set view angle
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        # Styling
        self.ax_3d.set_xlabel('Re(z)', color=TEXT_DIM, fontsize=9)
        self.ax_3d.set_ylabel('Im(z)', color=TEXT_DIM, fontsize=9)
        self.ax_3d.set_zlabel('Depth', color=TEXT_DIM, fontsize=9)

        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        in_mandelbrot = is_in_mandelbrot(julia_c)
        status = "CONNECTED" if in_mandelbrot else "DISCONNECTED"
        status_color = ACCENT_GREEN if in_mandelbrot else ACCENT_RED

        self.ax_3d.set_title(
            f'Julia c = {julia_c.real:.3f} + {julia_c.imag:.3f}i  [{status}]',
            color=status_color, fontsize=11
        )

        # Hide panes
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('#333333')
        self.ax_3d.yaxis.pane.set_edgecolor('#333333')
        self.ax_3d.zaxis.pane.set_edgecolor('#333333')
        self.ax_3d.tick_params(colors=TEXT_DIM, labelsize=7)

    def _update_2d(self):
        """Update 2D top-down view."""
        self.ax_2d.clear()
        self.ax_2d.set_facecolor(DARK_BG)

        # Same colormap as 3D
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            (0.0, '#00d4ff'),
            (0.3, '#4dabf7'),
            (0.6, '#a855f7'),
            (0.9, '#3d1a4d'),
            (1.0, '#0a0a1a'),
        ]
        cmap = LinearSegmentedColormap.from_list('depth',
            [(p, c) for p, c in colors])

        self.ax_2d.imshow(self.Z, extent=[-1.6, 1.6, -1.2, 1.2],
                          origin='lower', cmap=cmap, aspect='equal')

        # Mark Julia c parameter
        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        self.ax_2d.plot(julia_c.real, julia_c.imag, 'o', markersize=8,
                        markerfacecolor='none', markeredgecolor='yellow',
                        markeredgewidth=2)

        self.ax_2d.set_title('2D View (top-down)', color=TEXT_COLOR, fontsize=10)
        self.ax_2d.set_xlabel('Re(z)', color=TEXT_DIM, fontsize=8)
        self.ax_2d.set_ylabel('Im(z)', color=TEXT_DIM, fontsize=8)
        self.ax_2d.tick_params(colors=TEXT_DIM, labelsize=7)

    def _update_metrics(self):
        """Update metrics display."""
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor(PANEL_BG)
        self.ax_metrics.axis('off')

        m = self.metrics
        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        in_m = is_in_mandelbrot(julia_c)

        # Metrics text
        lines = [
            f"Max Depth:      {m['max_depth']:.4f}",
            f"Mean Depth:     {m['mean_depth']:.4f}",
            f"Depth Variance: {m['depth_variance']:.6f}",
            f"Interior Frac:  {m['interior_fraction']:.4f}",
            f"Well Count:     {m['well_count']}",
            f"",
            f"In Mandelbrot:  {'YES' if in_m else 'NO'}",
        ]

        text = '\n'.join(lines)
        self.ax_metrics.text(0.05, 0.95, text, transform=self.ax_metrics.transAxes,
                             color=TEXT_COLOR, fontsize=10, family='monospace',
                             verticalalignment='top')

        # Warning indicator
        if m['max_depth'] > 0.95 and not in_m:
            self.ax_metrics.text(0.95, 0.5, 'DEEPENING!', transform=self.ax_metrics.transAxes,
                                 color=ACCENT_ORANGE, fontsize=12, fontweight='bold',
                                 ha='right', va='center')

    def _update_history(self):
        """Update depth history plot."""
        self.ax_history.clear()
        self.ax_history.set_facecolor(PANEL_BG)

        if len(self.metrics_history) > 1:
            b_vals = [h['b'] for h in self.metrics_history]
            max_depths = [h['max_depth'] for h in self.metrics_history]
            mean_depths = [h['mean_depth'] for h in self.metrics_history]
            interior = [h['interior_fraction'] for h in self.metrics_history]

            self.ax_history.plot(b_vals, max_depths, color=ACCENT_RED,
                                 label='Max Depth', linewidth=1.5)
            self.ax_history.plot(b_vals, mean_depths, color=ACCENT_CYAN,
                                 label='Mean Depth', linewidth=1.5)
            self.ax_history.plot(b_vals, interior, color=ACCENT_PURPLE,
                                 label='Interior %', linewidth=1.5, linestyle='--')

            # Mark current position
            self.ax_history.axvline(x=self.b, color='yellow', linewidth=2, alpha=0.7)

            # Mark boundary crossings (approximate)
            for boundary_b in [0.23, 0.37, 0.64, 0.78]:
                self.ax_history.axvline(x=boundary_b, color=ACCENT_PURPLE,
                                        linewidth=1, alpha=0.5, linestyle=':')

        self.ax_history.set_xlim([0, 1])
        self.ax_history.set_ylim([0, 1.05])
        self.ax_history.set_xlabel('Phase b (τ fraction)', color=TEXT_DIM, fontsize=9)
        self.ax_history.set_ylabel('Depth', color=TEXT_DIM, fontsize=9)
        self.ax_history.legend(loc='upper right', fontsize=7,
                               facecolor=PANEL_BG, edgecolor='#333333',
                               labelcolor=TEXT_COLOR)
        self.ax_history.tick_params(colors=TEXT_DIM, labelsize=7)
        self.ax_history.set_title('Depth vs Phase (boundary crossings marked)',
                                  color=TEXT_COLOR, fontsize=9)

    def _update_info(self):
        """Update info text."""
        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        status = "PLAYING" if self.playing else "READY"
        info = (f"a={self.a:.3f}  b={self.b:.3f}τ  c={self.c_param:.3f}  |  "
                f"Julia c = {julia_c.real:.4f} + {julia_c.imag:.4f}i  |  [{status}]")
        self.info_text.set_text(info)

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def _on_b_change(self, val):
        """Handle phase slider change."""
        self.b = val
        self._compute()
        self._update_all()

    def _on_height_change(self, val):
        """Handle height scale change."""
        self.height_scale = val
        self._update_3d()
        self.fig.canvas.draw_idle()

    def _on_view_change(self, val):
        """Handle view angle change."""
        self.elev = val
        self._update_3d()
        self.fig.canvas.draw_idle()

    def _on_play(self, event):
        """Start parameter sweep."""
        if not self.playing:
            self.playing = True
            self.metrics_history.clear()
            self.anim = animation.FuncAnimation(
                self.fig, self._animate_frame,
                interval=100, blit=False
            )
            self.fig.canvas.draw_idle()

    def _on_stop(self, event):
        """Stop sweep."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _animate_frame(self, frame):
        """Animation frame - sweep through boundary."""
        if not self.playing:
            return

        self.b = (self.b + self.sweep_speed) % 1.0
        self.slider_b.set_val(self.b)

    def _on_export(self, event):
        """Export current heightfield as PLY."""
        output_dir = 'julia_3d_exports'
        os.makedirs(output_dir, exist_ok=True)

        julia_c = pn_to_julia_c(self.a, self.b, self.c_param)
        filename = f"julia3d_b{self.b:.3f}_c{julia_c.real:.3f}_{julia_c.imag:.3f}i.ply"
        filepath = os.path.join(output_dir, filename)

        export_heightfield_ply(self.X, self.Y, self.Z, filepath, self.height_scale)
        print(f"Exported: {filepath}")

    def _on_export_sequence(self, event):
        """Export frame sequence for Blender animation."""
        output_dir = 'julia_3d_sequence'
        os.makedirs(output_dir, exist_ok=True)

        print("Exporting boundary crossing sequence...")

        # Sweep through one full cycle
        n_frames = 100
        original_b = self.b

        for i, b in enumerate(np.linspace(0, 1, n_frames, endpoint=False)):
            self.b = b
            self._compute()

            filename = f"frame_{i:04d}.ply"
            filepath = os.path.join(output_dir, filename)
            export_heightfield_ply(self.X, self.Y, self.Z, filepath, self.height_scale)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_frames} frames exported")

        self.b = original_b
        self._compute()
        self._update_all()

        print(f"Sequence exported to {output_dir}/")
        print(f"Import to Blender: File > Import > Stanford (.ply)")

    def run(self):
        """Run the viewer."""
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QDNU JULIA 3D BOUNDARY CROSSING VISUALIZATION")
    print("=" * 70)
    print("""
This tool visualizes Julia sets with escape-time as Z-height to reveal
depth dynamics during Mandelbrot boundary crossing.

HYPOTHESIS: Depth changes may precede visible 2D boundary changes.
- Deepening tornados could be a PRE-ICTAL warning signal
- Wells punching through indicate boundary crossing (ICTAL onset)

Controls:
  - Phase slider: Sweep through boundary crossing
  - Height slider: Adjust 3D exaggeration
  - Elevation slider: Tilt view angle
  - Mouse drag on 3D plot to rotate

  - Sweep: Animate through full phase cycle
  - Export PLY: Save current frame for Blender
  - Sequence: Export 100-frame animation sequence

Boundary Crossings (marked with dotted lines):
  - b ~ 0.23 tau: EXIT Mandelbrot (becomes disconnected)
  - b ~ 0.37 tau: ENTER Mandelbrot (becomes connected)
  - b ~ 0.64 tau: EXIT Mandelbrot
  - b ~ 0.78 tau: ENTER Mandelbrot
""")

    viewer = Julia3DBoundaryViewer(resolution=200)
    viewer.run()


if __name__ == '__main__':
    main()
