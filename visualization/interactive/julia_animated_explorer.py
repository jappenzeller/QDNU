"""
================================================================================
QDNU JULIA VISUALIZATION - ANIMATED EXPLORER (v2)
================================================================================

Clean interface matching agate_parameter_explorer style:
- Dark theme (#0a0a1a background)
- High-detail Julia visualization
- Text input boxes for a, b, c values
- Organized control panel
- Smooth animations

Run: python -m visualization.interactive.julia_animated_explorer

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Tuple, List
import os


# =============================================================================
# STYLING CONSTANTS
# =============================================================================

DARK_BG = '#0a0a1a'
PANEL_BG = '#16213e'
ACCENT_CYAN = '#00d4ff'
ACCENT_ORANGE = '#ff9500'
ACCENT_PURPLE = '#a855f7'
ACCENT_GREEN = '#22c55e'
TEXT_COLOR = '#cccccc'
TEXT_DIM = '#666666'

# Phase notation: τ (tau) = 2π = full circle
TAU = 2 * np.pi
TAU_SYMBOL = 'τ'  # Unicode tau


def phase_to_tau_str(radians: float) -> str:
    """Convert radians to τ fraction string (e.g., '1/4 τ' for π/2)."""
    # Normalize to [0, τ)
    tau_fraction = (radians / TAU) % 1.0

    # Common fractions
    fractions = [
        (0, '0'),
        (1/8, '1/8'),
        (1/4, '1/4'),
        (3/8, '3/8'),
        (1/2, '1/2'),
        (5/8, '5/8'),
        (3/4, '3/4'),
        (7/8, '7/8'),
        (1, '1'),
    ]

    # Find closest fraction
    for frac_val, frac_str in fractions:
        if abs(tau_fraction - frac_val) < 0.02:
            if frac_str == '0':
                return f'0'
            elif frac_str == '1':
                return f'{TAU_SYMBOL}'
            else:
                return f'{frac_str} {TAU_SYMBOL}'

    # Otherwise show decimal
    return f'{tau_fraction:.2f} {TAU_SYMBOL}'


# =============================================================================
# CORE DATA
# =============================================================================

@dataclass
class PNState:
    """PN Neuron state."""
    a: float = 0.31
    b: float = 0.0  # Start at 0 to animate through inflection
    c: float = 0.32  # Nearly balanced |a-c| ≈ 0.01 for connectivity transitions

    @property
    def b_tau(self) -> float:
        """Phase as τ fraction (0-1)."""
        return self.b / TAU

    def to_julia_c(self) -> complex:
        """
        Map to Julia c parameter with BOUNDARY-CROSSING behavior.

        This mapping is centered at the cusp between the main Mandelbrot
        cardioid and the period-2 bulb, with radius large enough to
        cross the boundary. This creates connectivity transitions (flips):

        - b ~ 0.23 tau (84 deg): Julia becomes DISCONNECTED
        - b ~ 0.37 tau (135 deg): Julia becomes CONNECTED
        - b ~ 0.64 tau (229 deg): Julia becomes DISCONNECTED
        - b ~ 0.78 tau (280 deg): Julia becomes CONNECTED

        The E-I balance (a-c) shifts the path vertically.
        """
        # Center at the cusp between main cardioid and period-2 bulb
        center_real = -0.75

        # Radius large enough to cross the Mandelbrot boundary
        radius = 0.35

        real = center_real + radius * np.cos(self.b)
        imag = radius * np.sin(self.b) + 0.1 * (self.a - self.c)

        return complex(real, imag)

    def to_bloch_E(self) -> Tuple[float, float, float]:
        theta = np.pi * (1 - self.a)
        phi = self.b
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))

    def to_bloch_I(self) -> Tuple[float, float, float]:
        theta = np.pi * (1 - self.c)
        phi = self.b + np.pi / 4
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))

    def concurrence(self) -> float:
        return 0.5 * abs(np.sin(self.b)) * (1 - abs(self.a - self.c))

    def sensitivity(self) -> float:
        a_sens = 4 * self.a * (1 - self.a)
        c_sens = 4 * self.c * (1 - self.c)
        b_sens = abs(np.sin(2 * self.b))
        return (a_sens + c_sens + b_sens) / 3


# =============================================================================
# JULIA COMPUTATION (HIGH QUALITY)
# =============================================================================

def compute_julia_2d(c: complex, resolution: int = 400,
                     x_range: Tuple[float, float] = (-1.8, 1.8),
                     y_range: Tuple[float, float] = (-1.2, 1.2),
                     max_iter: int = 300) -> np.ndarray:
    """Compute high-quality Julia set with smooth coloring."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y_res = int(resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    y = np.linspace(y_range[0], y_range[1], y_res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    iterations = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        escaped = np.abs(Z) > 2
        new_escaped = escaped & mask

        if np.any(new_escaped):
            log_zn = np.log(np.abs(Z[new_escaped]) + 1e-10)
            nu = np.log(log_zn / np.log(2) + 1e-10) / np.log(2)
            iterations[new_escaped] = i + 1 - nu

        mask[escaped] = False
        if not np.any(mask):
            break

    iterations[mask] = max_iter
    return iterations


def colormap_julia(iterations: np.ndarray, max_iter: int = 300) -> np.ndarray:
    """Custom colormap matching the explorer style."""
    norm = iterations / max_iter
    interior_mask = iterations >= max_iter - 1

    # Cyclic coloring for fractal bands
    t = np.mod(norm * 12, 1.0)

    # Cyan/purple/yellow palette
    r = 0.1 + 0.5 * (1 + np.sin(2 * np.pi * t + 0.0)) / 2
    g = 0.1 + 0.6 * (1 + np.sin(2 * np.pi * t + 2.1)) / 2
    b = 0.2 + 0.7 * (1 + np.sin(2 * np.pi * t + 4.2)) / 2

    # Boost brightness near boundary
    boundary_factor = np.exp(-3 * (norm - 0.1) ** 2)
    r += 0.4 * boundary_factor
    g += 0.5 * boundary_factor
    b += 0.3 * boundary_factor

    # Interior = dark purple
    r[interior_mask] = 0.08
    g[interior_mask] = 0.02
    b[interior_mask] = 0.12

    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def create_sphere_mesh(iterations: np.ndarray, max_iter: int = 300,
                       resolution: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spherical projection mesh."""
    # Resample iterations to sphere resolution
    from scipy.ndimage import zoom
    ny, nx = iterations.shape
    target_ny, target_nx = resolution, resolution * 2
    zoom_factors = (target_ny / ny, target_nx / nx)
    julia_resampled = zoom(iterations, zoom_factors, order=1)

    theta = np.linspace(0.02, np.pi - 0.02, target_ny)
    phi = np.linspace(0, 2 * np.pi, target_nx)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    norm = julia_resampled / max_iter
    R = 1.0 - 0.25 * (1 - norm)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    colors = colormap_julia(julia_resampled, max_iter)

    return X, Y, Z, colors


# =============================================================================
# MESH EXPORT
# =============================================================================

def export_ply(X, Y, Z, colors, filepath):
    """Export mesh to PLY."""
    ny, nx = X.shape
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors_flat = colors.reshape(-1, 3)

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])

    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")

        for i, v in enumerate(vertices):
            c = (np.clip(colors_flat[i], 0, 1) * 255).astype(np.uint8)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Exported: {filepath}")


# =============================================================================
# CLEAN EXPLORER CLASS
# =============================================================================

class JuliaExplorer:
    """
    Clean Julia explorer with organized interface.
    """

    def __init__(self, resolution: int = 400):
        self.resolution = resolution
        self.max_iter = 300

        # State - nearly balanced |a-c| ≈ 0.01 for connectivity transitions
        self.state = PNState(a=0.31, b=0.0, c=0.32)
        self.julia_iterations = None

        # Animation
        self.playing = False
        self.animation_speed = 0.03
        self.anim = None

        # Traces
        self.trace_E: List[Tuple[float, float, float]] = []
        self.trace_I: List[Tuple[float, float, float]] = []
        self.max_trace = 30
        self.concurrence_history: List[float] = []
        self.sine_history: List[float] = []  # Phase history for sine wave
        self.max_history = 100

        # Setup
        self._setup_figure()
        self._compute_julia()
        self._update_all()

    def _setup_figure(self):
        """Create clean figure layout."""
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor(DARK_BG)
        plt.suptitle('QDNU Julia Explorer', color='white', fontsize=16, y=0.98)

        # Grid layout:
        # Left: Julia 2D (large)
        # Middle: Sphere 3D (bottom), Sine wave (top)
        # Right: Bloch spheres, concurrence, controls

        # Main Julia 2D plot (large, left)
        self.ax_julia = self.fig.add_axes([0.03, 0.35, 0.38, 0.58])
        self.ax_julia.set_facecolor(DARK_BG)

        # Sine wave (middle-top) - shows phase cycling and boundary
        self.ax_sine = self.fig.add_axes([0.03, 0.08, 0.38, 0.22])
        self.ax_sine.set_facecolor(PANEL_BG)

        # Sphere 3D (middle)
        self.ax_sphere = self.fig.add_axes([0.42, 0.35, 0.28, 0.58], projection='3d')
        self.ax_sphere.set_facecolor(DARK_BG)

        # Bloch E (top-right)
        self.ax_bloch_E = self.fig.add_axes([0.72, 0.62, 0.13, 0.32], projection='3d')
        self.ax_bloch_E.set_facecolor(DARK_BG)

        # Bloch I (top-right)
        self.ax_bloch_I = self.fig.add_axes([0.86, 0.62, 0.13, 0.32], projection='3d')
        self.ax_bloch_I.set_facecolor(DARK_BG)

        # Concurrence history (right side)
        self.ax_conc = self.fig.add_axes([0.72, 0.42, 0.27, 0.15])
        self.ax_conc.set_facecolor(PANEL_BG)

        # Control panel area
        self._setup_controls()

    def _setup_controls(self):
        """Setup control panel with text boxes and sliders."""
        # Parameter labels and values - positioned on right side
        label_x = 0.72
        slider_x = 0.78
        text_x = 0.95
        width_slider = 0.14
        width_text = 0.04

        # Title
        self.fig.text(label_x, 0.38, 'PARAMETERS', color=TEXT_COLOR,
                      fontsize=11, fontweight='bold')

        # a parameter
        self.fig.text(label_x, 0.33, 'a (excitatory):', color=ACCENT_ORANGE, fontsize=10)
        ax_slider_a = self.fig.add_axes([slider_x, 0.325, width_slider, 0.02])
        self.slider_a = Slider(ax_slider_a, '', 0, 1, valinit=0.31,
                               color=ACCENT_ORANGE, track_color=PANEL_BG)
        ax_text_a = self.fig.add_axes([text_x, 0.32, width_text, 0.025])
        self.text_a = TextBox(ax_text_a, '', initial='0.310', color=PANEL_BG,
                              hovercolor='#1e3a5f')
        self.text_a.label.set_color(TEXT_COLOR)

        # b parameter (phase in τ fractions)
        self.fig.text(label_x, 0.28, f'b (phase {TAU_SYMBOL}):', color=ACCENT_GREEN, fontsize=10)
        ax_slider_b = self.fig.add_axes([slider_x, 0.275, width_slider, 0.02])
        self.slider_b = Slider(ax_slider_b, '', 0, 1, valinit=0.0,
                               color=ACCENT_GREEN, track_color=PANEL_BG)
        ax_text_b = self.fig.add_axes([text_x, 0.27, width_text, 0.025])
        self.text_b = TextBox(ax_text_b, '', initial='0.000', color=PANEL_BG,
                              hovercolor='#1e3a5f')
        self.text_b.label.set_color(TEXT_COLOR)

        # c parameter - nearly balanced with a for connectivity transitions
        self.fig.text(label_x, 0.23, 'c (inhibitory):', color=ACCENT_CYAN, fontsize=10)
        ax_slider_c = self.fig.add_axes([slider_x, 0.225, width_slider, 0.02])
        self.slider_c = Slider(ax_slider_c, '', 0, 1, valinit=0.32,
                               color=ACCENT_CYAN, track_color=PANEL_BG)
        ax_text_c = self.fig.add_axes([text_x, 0.22, width_text, 0.025])
        self.text_c = TextBox(ax_text_c, '', initial='0.320', color=PANEL_BG,
                              hovercolor='#1e3a5f')
        self.text_c.label.set_color(TEXT_COLOR)

        # Speed
        self.fig.text(label_x, 0.18, 'Speed:', color=TEXT_DIM, fontsize=10)
        ax_slider_speed = self.fig.add_axes([slider_x, 0.175, width_slider, 0.02])
        self.slider_speed = Slider(ax_slider_speed, '', 0.01, 0.1, valinit=0.03,
                                   color='#888888', track_color=PANEL_BG)

        # Connect callbacks
        self.slider_a.on_changed(lambda v: self._on_slider_change('a', v))
        self.slider_b.on_changed(lambda v: self._on_slider_change('b', v))
        self.slider_c.on_changed(lambda v: self._on_slider_change('c', v))
        self.slider_speed.on_changed(lambda v: setattr(self, 'animation_speed', v))

        self.text_a.on_submit(lambda t: self._on_text_submit('a', t))
        self.text_b.on_submit(lambda t: self._on_text_submit('b', t))
        self.text_c.on_submit(lambda t: self._on_text_submit('c', t))

        # Playback buttons
        btn_width = 0.05
        btn_height = 0.03
        btn_y = 0.12
        btn_spacing = 0.055

        ax_play = self.fig.add_axes([label_x, btn_y, btn_width, btn_height])
        ax_pause = self.fig.add_axes([label_x + btn_spacing, btn_y, btn_width, btn_height])
        ax_stop = self.fig.add_axes([label_x + 2 * btn_spacing, btn_y, btn_width, btn_height])
        ax_step = self.fig.add_axes([label_x + 3 * btn_spacing, btn_y, btn_width, btn_height])
        ax_export = self.fig.add_axes([label_x + 4 * btn_spacing, btn_y, 0.06, btn_height])

        self.btn_play = Button(ax_play, 'Play', color='#1a4d1a', hovercolor='#2d6a2d')
        self.btn_pause = Button(ax_pause, 'Pause', color='#4d4d1a', hovercolor='#6a6a2d')
        self.btn_stop = Button(ax_stop, 'Stop', color='#4d1a1a', hovercolor='#6a2d2d')
        self.btn_step = Button(ax_step, 'Step', color='#1a3d4d', hovercolor='#2d5a6a')
        self.btn_export = Button(ax_export, 'Export', color='#3d1a4d', hovercolor='#5a2d6a')

        for btn in [self.btn_play, self.btn_pause, self.btn_stop, self.btn_step, self.btn_export]:
            btn.label.set_color('white')
            btn.label.set_fontsize(9)

        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_step.on_clicked(self._on_step)
        self.btn_export.on_clicked(self._on_export)

        # Phase keyframe buttons (0 to 7/8 τ in 1/8 τ steps)
        self.fig.text(label_x, 0.07, 'Keyframes:', color=TEXT_DIM, fontsize=8)
        self.keyframe_buttons = []
        for i in range(8):
            ax_kf = self.fig.add_axes([label_x + i * 0.034, 0.035, 0.032, 0.025])
            label = f'{i}/8' if i > 0 else '0'
            btn = Button(ax_kf, label, color='#2a2a4a', hovercolor='#4a4a6a')
            btn.label.set_color(TEXT_COLOR)
            btn.label.set_fontsize(7)
            btn.on_clicked(lambda e, idx=i: self._on_keyframe(idx))
            self.keyframe_buttons.append(btn)

        # Info display
        self.info_text = self.fig.text(0.72, 0.01, '', color=TEXT_DIM,
                                        fontsize=8, family='monospace')

    def _compute_julia(self):
        """Compute Julia set."""
        c = self.state.to_julia_c()
        self.julia_iterations = compute_julia_2d(c, self.resolution, max_iter=self.max_iter)

    def _update_traces(self):
        """Update Bloch sphere traces and history."""
        self.trace_E.append(self.state.to_bloch_E())
        self.trace_I.append(self.state.to_bloch_I())

        if len(self.trace_E) > self.max_trace:
            self.trace_E.pop(0)
        if len(self.trace_I) > self.max_trace:
            self.trace_I.pop(0)

        self.concurrence_history.append(self.state.concurrence())
        if len(self.concurrence_history) > self.max_history:
            self.concurrence_history.pop(0)

        self.sine_history.append(self.state.b)
        if len(self.sine_history) > self.max_history:
            self.sine_history.pop(0)

    def _update_all(self):
        """Update all displays."""
        self._update_julia()
        self._update_sine_wave()
        self._update_sphere()
        self._update_bloch_spheres()
        self._update_concurrence()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _update_julia(self):
        """Update Julia 2D display."""
        self.ax_julia.clear()
        self.ax_julia.set_facecolor(DARK_BG)

        colors = colormap_julia(self.julia_iterations, self.max_iter)
        self.ax_julia.imshow(colors, extent=[-1.8, 1.8, -1.2, 1.2],
                             origin='lower', aspect='equal', interpolation='bilinear')

        c = self.state.to_julia_c()
        self.ax_julia.plot(c.real, c.imag, 'o', markersize=10,
                           markerfacecolor='none', markeredgecolor='yellow',
                           markeredgewidth=2)

        self.ax_julia.set_xlabel('Re(z)', color=TEXT_DIM, fontsize=10)
        self.ax_julia.set_ylabel('Im(z)', color=TEXT_DIM, fontsize=10)
        self.ax_julia.set_title(f'Julia Set: c = {c.real:.4f} + {c.imag:.4f}i',
                                color='white', fontsize=12)
        self.ax_julia.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in self.ax_julia.spines.values():
            spine.set_color('#333333')

    def _update_sine_wave(self):
        """Update sine wave showing phase cycling and boundary detection."""
        self.ax_sine.clear()
        self.ax_sine.set_facecolor(PANEL_BG)

        # Full sine wave reference (one complete τ cycle)
        t_ref = np.linspace(0, TAU, 200)
        y_ref = np.sin(t_ref)
        self.ax_sine.plot(t_ref, y_ref, color='#444466', linewidth=1.5, alpha=0.6)
        self.ax_sine.fill_between(t_ref, y_ref, alpha=0.08, color=ACCENT_CYAN)

        # Current phase position
        b = self.state.b % TAU
        y_current = np.sin(b)

        # Mark the inflection regions where Julia connectivity changes
        # Based on boundary-crossing analysis:
        # ~0.23 tau: EXITS Mandelbrot (disconnected)
        # ~0.37 tau: ENTERS Mandelbrot (connected)
        # ~0.64 tau: EXITS Mandelbrot (disconnected)
        # ~0.78 tau: ENTERS Mandelbrot (connected)
        inflection_positions = [
            0.23 * TAU,   # ~84 deg - becomes disconnected
            0.37 * TAU,   # ~135 deg - becomes connected
            0.64 * TAU,   # ~229 deg - becomes disconnected
            0.78 * TAU,   # ~280 deg - becomes connected
        ]
        for i, bp in enumerate(inflection_positions):
            self.ax_sine.axvline(x=bp, color=ACCENT_PURPLE, linewidth=2, alpha=0.7,
                                linestyle='-', label='Inflection' if i == 0 else '')
            self.ax_sine.fill_betweenx([-1.3, 1.3], bp - 0.1, bp + 0.1,
                                       alpha=0.2, color=ACCENT_PURPLE)

        # Current position marker (vertical line + dot)
        self.ax_sine.axvline(x=b, color=ACCENT_GREEN, linewidth=2.5, alpha=0.9)
        self.ax_sine.scatter([b], [y_current], c=ACCENT_GREEN, s=200, zorder=10,
                            edgecolors='white', linewidths=2)

        # Phase history trail
        if len(self.sine_history) > 1:
            hist = np.array(self.sine_history) % TAU
            y_hist = np.sin(hist)
            alphas = np.linspace(0.1, 0.7, len(hist))
            for i in range(len(hist)):
                self.ax_sine.scatter([hist[i]], [y_hist[i]], c=ACCENT_GREEN,
                                    s=20, alpha=alphas[i])

        # Keyframe markers (0 to 7/8 τ)
        for i in range(8):
            kf_b = (TAU * i) / 8
            self.ax_sine.axvline(x=kf_b, color='yellow', linewidth=1, alpha=0.25,
                                linestyle='--')
            label = f'{i}/8' if i > 0 else '0'
            self.ax_sine.text(kf_b, 1.2, label, ha='center', fontsize=7,
                             color='yellow', alpha=0.7)

        # Labels
        self.ax_sine.set_xlim([0, TAU])
        self.ax_sine.set_ylim([-1.4, 1.5])
        self.ax_sine.set_xlabel(f'Phase b ({TAU_SYMBOL})', color=TEXT_DIM, fontsize=10)
        self.ax_sine.set_ylabel('sin(b)', color=TEXT_DIM, fontsize=10)

        phase_str = phase_to_tau_str(b)
        conc = self.state.concurrence()
        title_color = ACCENT_PURPLE if conc > 0.2 else TEXT_COLOR
        self.ax_sine.set_title(f'Phase: {phase_str}  |  Concurrence: {conc:.3f}',
                               color=title_color, fontsize=11)

        # Custom x-ticks in τ notation
        self.ax_sine.set_xticks([0, TAU/4, TAU/2, 3*TAU/4, TAU])
        self.ax_sine.set_xticklabels(['0', f'1/4{TAU_SYMBOL}', f'1/2{TAU_SYMBOL}',
                                      f'3/4{TAU_SYMBOL}', TAU_SYMBOL])
        self.ax_sine.tick_params(colors=TEXT_DIM, labelsize=8)

        for spine in self.ax_sine.spines.values():
            spine.set_color('#333333')

    def _update_sphere(self):
        """Update sphere display."""
        self.ax_sphere.clear()
        self.ax_sphere.set_facecolor(DARK_BG)

        try:
            X, Y, Z, colors = create_sphere_mesh(self.julia_iterations, self.max_iter)
            self.ax_sphere.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95,
                                        shade=True, linewidth=0, antialiased=True)
        except ImportError:
            # scipy not available, skip sphere
            self.ax_sphere.text(0, 0, 0, 'scipy required\nfor sphere', color=TEXT_DIM,
                               ha='center', va='center')

        self.ax_sphere.set_box_aspect([1, 1, 1])
        self.ax_sphere.set_title('Spherical Projection', color='white', fontsize=11)
        self.ax_sphere.axis('off')

    def _update_bloch_spheres(self):
        """Update Bloch sphere displays."""
        for ax, bloch_vec, trace, title, color in [
            (self.ax_bloch_E, self.state.to_bloch_E(), self.trace_E, 'E (Excitatory)', ACCENT_ORANGE),
            (self.ax_bloch_I, self.state.to_bloch_I(), self.trace_I, 'I (Inhibitory)', ACCENT_CYAN)
        ]:
            ax.clear()
            ax.set_facecolor(DARK_BG)

            # Wireframe sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, alpha=0.08, color='white', linewidth=0.5)

            # Axes
            ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='#444444', alpha=0.5, linewidth=0.5)
            ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='#444444', alpha=0.5, linewidth=0.5)
            ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='#444444', alpha=0.5, linewidth=0.5)

            # Trace
            if len(trace) > 1:
                trace_arr = np.array(trace)
                alphas = np.linspace(0.1, 0.7, len(trace))
                for i in range(len(trace) - 1):
                    ax.plot([trace_arr[i, 0], trace_arr[i + 1, 0]],
                            [trace_arr[i, 1], trace_arr[i + 1, 1]],
                            [trace_arr[i, 2], trace_arr[i + 1, 2]],
                            color=color, alpha=alphas[i], linewidth=1.5)

            # State vector
            ax.quiver(0, 0, 0, bloch_vec[0], bloch_vec[1], bloch_vec[2],
                      color=color, arrow_length_ratio=0.15, linewidth=2)
            ax.scatter([bloch_vec[0]], [bloch_vec[1]], [bloch_vec[2]],
                       c=color, s=60, edgecolors='white', linewidths=1)

            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
            ax.set_title(title, color=color, fontsize=10)
            ax.axis('off')

    def _update_concurrence(self):
        """Update concurrence history plot."""
        self.ax_conc.clear()
        self.ax_conc.set_facecolor(PANEL_BG)

        if len(self.concurrence_history) > 1:
            t = np.arange(len(self.concurrence_history))
            self.ax_conc.fill_between(t, self.concurrence_history,
                                       alpha=0.3, color=ACCENT_PURPLE)
            self.ax_conc.plot(t, self.concurrence_history, color=ACCENT_PURPLE, linewidth=1.5)
            self.ax_conc.scatter([len(self.concurrence_history) - 1],
                                  [self.concurrence_history[-1]],
                                  c=ACCENT_PURPLE, s=50, edgecolors='white', zorder=5)

        self.ax_conc.axhline(y=0.25, color='yellow', alpha=0.3, linestyle='--', linewidth=1)
        self.ax_conc.text(2, 0.27, 'High', color='yellow', fontsize=7, alpha=0.5)

        self.ax_conc.set_xlim([0, self.max_history])
        self.ax_conc.set_ylim([0, 0.55])
        self.ax_conc.set_xlabel('Time', color=TEXT_DIM, fontsize=9)
        self.ax_conc.set_ylabel('Concurrence', color=TEXT_DIM, fontsize=9)
        conc = self.state.concurrence()
        self.ax_conc.set_title(f'Entanglement: {conc:.4f}', color=ACCENT_PURPLE, fontsize=10)
        self.ax_conc.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self.ax_conc.spines.values():
            spine.set_color('#333333')

    def _update_info(self):
        """Update info text."""
        c = self.state.to_julia_c()
        status = 'PLAYING' if self.playing else 'PAUSED'
        phase_str = phase_to_tau_str(self.state.b)
        info = (f"Julia c = {c.real:.4f} + {c.imag:.4f}i  |  "
                f"b = {phase_str}  |  "
                f"Sensitivity: {self.state.sensitivity():.4f}  |  [{status}]")
        self.info_text.set_text(info)

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def _on_slider_change(self, param, val):
        """Handle slider change."""
        if param == 'a':
            self.state.a = val
            self.text_a.set_val(f'{val:.3f}')
        elif param == 'b':
            # val is τ fraction (0-1), convert to radians
            self.state.b = val * TAU
            self.text_b.set_val(f'{val:.3f}')
        elif param == 'c':
            self.state.c = val
            self.text_c.set_val(f'{val:.3f}')

        self._compute_julia()
        self._update_traces()
        self._update_all()

    def _on_text_submit(self, param, text):
        """Handle text box submit."""
        try:
            val = float(text)
            if param == 'a':
                val = np.clip(val, 0, 1)
                self.slider_a.set_val(val)
            elif param == 'b':
                # val is τ fraction (0-1)
                val = val % 1.0
                self.slider_b.set_val(val)
            elif param == 'c':
                val = np.clip(val, 0, 1)
                self.slider_c.set_val(val)
        except ValueError:
            pass  # Invalid input, ignore

    def _on_keyframe(self, idx):
        """Jump to keyframe (idx/8 τ)."""
        b_tau = idx / 8.0  # 0, 1/8, 2/8, ... 7/8 τ
        self.slider_b.set_val(b_tau)

    def _apply_preset(self, vals):
        """Apply preset values."""
        a, b, c = vals
        self.slider_a.set_val(a)
        self.slider_b.set_val(b)
        self.slider_c.set_val(c)

    def _on_play(self, event):
        """Start animation."""
        if not self.playing:
            self.playing = True
            self.anim = animation.FuncAnimation(
                self.fig, self._animate_frame,
                interval=50, blit=False
            )
            self.fig.canvas.draw_idle()

    def _on_pause(self, event):
        """Pause animation."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_stop(self, event):
        """Stop and reset."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()

        self.trace_E.clear()
        self.trace_I.clear()
        self.concurrence_history.clear()
        self.sine_history.clear()

        self._apply_preset((0.31, 0.0, 0.32))  # Nearly balanced |a-c| ≈ 0.01 for flip

    def _on_step(self, event):
        """Step one frame."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()

        # Step by animation_speed (in τ fraction units)
        b_tau = (self.state.b / TAU + self.animation_speed / 10) % 1.0
        self.slider_b.set_val(b_tau)

    def _animate_frame(self, frame):
        """Animation frame."""
        if not self.playing:
            return

        # Advance by animation_speed (in τ fraction units)
        b_tau = (self.state.b / TAU + self.animation_speed / 10) % 1.0
        self.slider_b.set_val(b_tau)

    def _on_export(self, event):
        """Export current state."""
        output_dir = 'julia_exports'
        os.makedirs(output_dir, exist_ok=True)

        b_tau = self.state.b / TAU
        prefix = f"julia_a{self.state.a:.2f}_b{b_tau:.3f}tau_c{self.state.c:.2f}"

        # Julia 2D image
        colors = colormap_julia(self.julia_iterations, self.max_iter)
        plt.imsave(os.path.join(output_dir, f"{prefix}_2d.png"), colors)

        # Sphere mesh
        try:
            X, Y, Z, sphere_colors = create_sphere_mesh(self.julia_iterations, self.max_iter)
            export_ply(X, Y, Z, sphere_colors, os.path.join(output_dir, f"{prefix}_sphere.ply"))
        except ImportError:
            print("scipy required for sphere export")

        # State info
        c = self.state.to_julia_c()
        phase_str = phase_to_tau_str(self.state.b)
        with open(os.path.join(output_dir, f"{prefix}_state.txt"), 'w') as f:
            f.write(f"a = {self.state.a}\n")
            f.write(f"b = {phase_str} ({b_tau:.4f} {TAU_SYMBOL})\n")
            f.write(f"c = {self.state.c}\n")
            f.write(f"Julia c = {c}\n")
            f.write(f"Concurrence = {self.state.concurrence()}\n")
            f.write(f"Sensitivity = {self.state.sensitivity()}\n")
            f.write(f"\nNote: {TAU_SYMBOL} (tau) = 2{chr(0x03C0)} = full circle\n")

        print(f"Exported to {output_dir}/{prefix}_*")

    def run(self):
        """Run the explorer."""
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("QDNU JULIA EXPLORER (v2)")
    print("=" * 60)
    print("""
Clean interface with:
  - High-detail Julia visualization (400x resolution)
  - Text input boxes for a, b, c values
  - Organized control panel
  - Smooth animations

Controls:
  - Sliders OR text boxes to set a, b, c values
  - Play/Pause/Stop/Step for animation
  - Presets: Balanced, Ictal, Interictal, Spiral
  - Export: Save current state to julia_exports/
""")

    explorer = JuliaExplorer(resolution=400)
    explorer.run()


if __name__ == '__main__':
    main()
