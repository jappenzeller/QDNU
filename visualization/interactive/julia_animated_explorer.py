"""
================================================================================
QDNU JULIA VISUALIZATION - ANIMATED EXPLORER
================================================================================

Complete explorer with:
- Two Bloch spheres (E, I) with traces
- 2D Julia set
- 3D Canyon (height-mapped)
- Spherical projection
- Sine wave showing phase position
- Pulse indicator (concurrence-driven)
- Playback controls (Play/Pause/Stop/Step)
- Export to PLY for Blender

Run: python julia_animated_explorer.py

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
import time

# =============================================================================
# SECTION 1: CORE DATA
# =============================================================================

@dataclass
class PNState:
    """PN Neuron state."""
    a: float = 0.3
    b: float = 0.0
    c: float = 0.3
    
    @property
    def b_deg(self) -> float:
        return np.degrees(self.b)
    
    def to_julia_c(self) -> complex:
        real = -0.4 + 0.3 * np.cos(self.b)
        imag = 0.3 * np.sin(self.b) + 0.1 * (self.a - self.c)
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


# =============================================================================
# SECTION 2: JULIA COMPUTATION
# =============================================================================

def compute_julia_2d(c: complex, resolution: int = 128, 
                     bounds: float = 2.0, max_iter: int = 80) -> np.ndarray:
    """Compute 2D Julia set."""
    x = np.linspace(-bounds, bounds, resolution)
    y = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    iterations = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)
    
    for n in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        mag = np.abs(Z)
        escaped = mask & (mag > 2)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth = n + 1 - np.log2(np.log2(mag + 1e-10) + 1e-10)
            smooth = np.nan_to_num(smooth, nan=n)
        
        iterations[escaped] = smooth[escaped]
        mask = mask & ~escaped
    
    iterations[mask] = max_iter
    return iterations / max_iter


# =============================================================================
# SECTION 3: MESH GENERATION & EXPORT
# =============================================================================

def create_canyon_mesh(julia_2d: np.ndarray, bounds: float = 2.0, 
                       height_scale: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create canyon mesh from Julia iterations."""
    ny, nx = julia_2d.shape
    heights = (1 - julia_2d) * height_scale
    
    x = np.linspace(-bounds, bounds, nx)
    y = np.linspace(-bounds, bounds, ny)
    X, Y = np.meshgrid(x, y)
    
    vertices = np.stack([X.flatten(), Y.flatten(), heights.flatten()], axis=1)
    colors = colormap_plasma(julia_2d.flatten())
    
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    return vertices.astype(np.float32), np.array(faces, dtype=np.int32), colors


def create_sphere_mesh(julia_2d: np.ndarray, radius: float = 1.0,
                       height_scale: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spherical projection mesh."""
    ny, nx = julia_2d.shape
    theta = np.linspace(0.01, np.pi - 0.01, ny)
    phi = np.linspace(0, 2 * np.pi, nx)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    R = radius + (1 - julia_2d) * height_scale
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors = colormap_plasma(julia_2d.flatten())
    
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    return vertices.astype(np.float32), np.array(faces, dtype=np.int32), colors


def colormap_plasma(t: np.ndarray) -> np.ndarray:
    """Plasma-like colormap."""
    t = np.clip(t, 0, 1)
    r = np.clip(0.05 + 1.2 * t + 0.3 * np.sin(t * np.pi), 0, 1)
    g = np.clip(0.02 + 0.9 * t ** 1.5, 0, 1)
    b = np.clip(0.53 - 0.5 * t + 0.3 * np.sin((1 - t) * np.pi), 0, 1)
    return np.stack([r, g, b], axis=1).astype(np.float32)


def export_ply(vertices, faces, colors, filepath):
    """Export mesh to PLY."""
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
# SECTION 4: ANIMATED EXPLORER CLASS
# =============================================================================

class JuliaAnimatedExplorer:
    """
    Interactive explorer with animation playback.
    
    Features:
    - Bloch spheres with traces
    - Julia 2D, Canyon 3D, Sphere
    - Sine wave phase indicator
    - Pulse (concurrence) indicator
    - Play/Pause/Stop/Step controls
    """
    
    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.bounds = 2.0
        self.max_iter = 60
        
        # State
        self.state = PNState(a=0.3, b=0.0, c=0.3)
        self.julia_2d = None
        
        # Animation
        self.playing = False
        self.animation_speed = 0.05  # Radians per frame
        self.frame_interval = 50  # ms between frames
        self.anim = None
        
        # Traces
        self.trace_E: List[Tuple[float, float, float]] = []
        self.trace_I: List[Tuple[float, float, float]] = []
        self.max_trace = 20
        
        # Sine wave history
        self.sine_history: List[float] = []
        self.max_sine_history = 100
        
        # Pulse history
        self.pulse_history: List[float] = []
        self.max_pulse_history = 50
        
        # Setup figure
        self._setup_figure()
        self._compute_julia()
        self._update_all_displays()
    
    def _setup_figure(self):
        """Create figure with all subplots and controls."""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Layout:
        # Row 1: Bloch E | Bloch I | Julia 2D | Canyon 3D
        # Row 2: Sine Wave | Pulse | Sphere | Controls
        
        # Row 1
        self.ax_bloch_E = self.fig.add_subplot(2, 4, 1, projection='3d')
        self.ax_bloch_I = self.fig.add_subplot(2, 4, 2, projection='3d')
        self.ax_julia_2d = self.fig.add_subplot(2, 4, 3)
        self.ax_canyon = self.fig.add_subplot(2, 4, 4, projection='3d')
        
        # Row 2
        self.ax_sine = self.fig.add_subplot(2, 4, 5)
        self.ax_pulse = self.fig.add_subplot(2, 4, 6)
        self.ax_sphere = self.fig.add_subplot(2, 4, 7, projection='3d')
        self.ax_controls = self.fig.add_subplot(2, 4, 8)
        self.ax_controls.axis('off')
        
        # Style 3D axes
        for ax in [self.ax_bloch_E, self.ax_bloch_I, self.ax_canyon, self.ax_sphere]:
            ax.set_facecolor('#1a1a2e')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        
        # Style 2D axes
        for ax in [self.ax_julia_2d, self.ax_sine, self.ax_pulse]:
            ax.set_facecolor('#16213e')
        
        # Sliders
        ax_slider_a = self.fig.add_axes([0.78, 0.35, 0.15, 0.02])
        ax_slider_b = self.fig.add_axes([0.78, 0.30, 0.15, 0.02])
        ax_slider_c = self.fig.add_axes([0.78, 0.25, 0.15, 0.02])
        ax_slider_speed = self.fig.add_axes([0.78, 0.20, 0.15, 0.02])
        
        self.slider_a = Slider(ax_slider_a, 'a (E)', 0, 1, valinit=0.3, color='orange')
        self.slider_b = Slider(ax_slider_b, 'b (φ)', 0, 2*np.pi, valinit=0, color='green')
        self.slider_c = Slider(ax_slider_c, 'c (I)', 0, 1, valinit=0.3, color='cyan')
        self.slider_speed = Slider(ax_slider_speed, 'Speed', 0.01, 0.2, valinit=0.05, color='white')
        
        self.slider_a.on_changed(self._on_slider_change)
        self.slider_b.on_changed(self._on_slider_change)
        self.slider_c.on_changed(self._on_slider_change)
        self.slider_speed.on_changed(self._on_speed_change)
        
        # Playback buttons
        ax_play = self.fig.add_axes([0.78, 0.12, 0.04, 0.04])
        ax_pause = self.fig.add_axes([0.83, 0.12, 0.04, 0.04])
        ax_stop = self.fig.add_axes([0.88, 0.12, 0.04, 0.04])
        ax_step = self.fig.add_axes([0.78, 0.06, 0.04, 0.04])
        ax_export = self.fig.add_axes([0.83, 0.06, 0.09, 0.04])
        
        self.btn_play = Button(ax_play, '>', color='#2d4a22', hovercolor='#3d6a32')
        self.btn_pause = Button(ax_pause, '||', color='#4a4a22', hovercolor='#6a6a32')
        self.btn_stop = Button(ax_stop, 'X', color='#4a2222', hovercolor='#6a3232')
        self.btn_step = Button(ax_step, '>>', color='#22444a', hovercolor='#32646a')
        self.btn_export = Button(ax_export, 'Export', color='#3a2a5a', hovercolor='#5a4a7a')
        
        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_step.on_clicked(self._on_step)
        self.btn_export.on_clicked(self._on_export)
        
        # Keyframe buttons
        self.keyframe_buttons = []
        for i in range(8):
            ax_kf = self.fig.add_axes([0.78 + i * 0.024, 0.42, 0.022, 0.025])
            btn = Button(ax_kf, str(i), color='#2a2a4a', hovercolor='#4a4a6a')
            btn.on_clicked(lambda event, idx=i: self._on_keyframe(idx))
            self.keyframe_buttons.append(btn)
        
        # Info text
        self.info_text = self.fig.text(0.78, 0.48, '', fontsize=9, color='white',
                                        family='monospace', verticalalignment='top')
        
        plt.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.05, 
                           wspace=0.3, hspace=0.25)
    
    def _compute_julia(self):
        """Compute Julia set from current state."""
        c = self.state.to_julia_c()
        self.julia_2d = compute_julia_2d(c, self.resolution, self.bounds, self.max_iter)
    
    def _update_traces(self):
        """Update Bloch sphere traces."""
        self.trace_E.append(self.state.to_bloch_E())
        self.trace_I.append(self.state.to_bloch_I())
        
        if len(self.trace_E) > self.max_trace:
            self.trace_E.pop(0)
        if len(self.trace_I) > self.max_trace:
            self.trace_I.pop(0)
        
        # Sine wave
        self.sine_history.append(self.state.b)
        if len(self.sine_history) > self.max_sine_history:
            self.sine_history.pop(0)
        
        # Pulse (concurrence)
        self.pulse_history.append(self.state.concurrence())
        if len(self.pulse_history) > self.max_pulse_history:
            self.pulse_history.pop(0)
    
    def _update_all_displays(self):
        """Update all visualization displays."""
        self._update_bloch_spheres()
        self._update_julia_2d()
        self._update_canyon()
        self._update_sphere()
        self._update_sine_wave()
        self._update_pulse()
        self._update_info()
        
        self.fig.canvas.draw_idle()
    
    def _update_bloch_spheres(self):
        """Update Bloch sphere displays."""
        for ax, bloch_vec, trace, title, color in [
            (self.ax_bloch_E, self.state.to_bloch_E(), self.trace_E, 'Bloch E (Excitatory)', 'orange'),
            (self.ax_bloch_I, self.state.to_bloch_I(), self.trace_I, 'Bloch I (Inhibitory)', 'cyan')
        ]:
            ax.clear()
            ax.set_facecolor('#1a1a2e')
            
            # Wireframe sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, alpha=0.1, color='white')
            
            # Axes
            ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'r-', alpha=0.3, linewidth=1)
            ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'g-', alpha=0.3, linewidth=1)
            ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'b-', alpha=0.3, linewidth=1)
            
            # Trace
            if len(trace) > 1:
                trace_arr = np.array(trace)
                alphas = np.linspace(0.1, 0.8, len(trace))
                for i in range(len(trace) - 1):
                    ax.plot([trace_arr[i, 0], trace_arr[i+1, 0]],
                           [trace_arr[i, 1], trace_arr[i+1, 1]],
                           [trace_arr[i, 2], trace_arr[i+1, 2]],
                           color=color, alpha=alphas[i], linewidth=2)
            
            # State vector
            ax.quiver(0, 0, 0, bloch_vec[0], bloch_vec[1], bloch_vec[2],
                     color=color, arrow_length_ratio=0.1, linewidth=3)
            ax.scatter([bloch_vec[0]], [bloch_vec[1]], [bloch_vec[2]],
                      c=color, s=100, edgecolors='white')
            
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_zlim([-1.3, 1.3])
            ax.set_title(title, color='white', fontsize=10)
            ax.tick_params(colors='gray', labelsize=7)
    
    def _update_julia_2d(self):
        """Update 2D Julia display."""
        self.ax_julia_2d.clear()
        self.ax_julia_2d.set_facecolor('#16213e')
        
        self.ax_julia_2d.imshow(self.julia_2d, cmap='magma', origin='lower',
                                extent=[-self.bounds, self.bounds, -self.bounds, self.bounds])
        
        c = self.state.to_julia_c()
        self.ax_julia_2d.set_title(f'Julia 2D: c = {c.real:.2f}{c.imag:+.2f}i', 
                                   color='white', fontsize=10)
        self.ax_julia_2d.tick_params(colors='gray', labelsize=7)
    
    def _update_canyon(self):
        """Update 3D canyon display."""
        self.ax_canyon.clear()
        self.ax_canyon.set_facecolor('#1a1a2e')
        
        step = max(1, self.resolution // 24)
        ny, nx = self.julia_2d.shape
        
        x = np.linspace(-self.bounds, self.bounds, nx)[::step]
        y = np.linspace(-self.bounds, self.bounds, ny)[::step]
        X, Y = np.meshgrid(x, y)
        Z = (1 - self.julia_2d[::step, ::step]) * 1.5
        
        self.ax_canyon.plot_surface(X, Y, Z, cmap='inferno', alpha=0.9)
        self.ax_canyon.set_title('Canyon (Height Map)', color='white', fontsize=10)
        self.ax_canyon.tick_params(colors='gray', labelsize=6)
    
    def _update_sphere(self):
        """Update spherical projection display."""
        self.ax_sphere.clear()
        self.ax_sphere.set_facecolor('#1a1a2e')
        
        step = max(1, self.resolution // 20)
        ny, nx = self.julia_2d.shape
        
        theta = np.linspace(0.1, np.pi - 0.1, ny)[::step]
        phi = np.linspace(0, 2 * np.pi, nx)[::step]
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        julia_sub = self.julia_2d[::step, ::step]
        R = 1.0 + (1 - julia_sub) * 0.3
        
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        self.ax_sphere.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(julia_sub), alpha=0.9)
        self.ax_sphere.set_title('Spherical Projection', color='white', fontsize=10)
        self.ax_sphere.tick_params(colors='gray', labelsize=6)
        self.ax_sphere.set_box_aspect([1, 1, 1])
    
    def _update_sine_wave(self):
        """Update sine wave phase indicator."""
        self.ax_sine.clear()
        self.ax_sine.set_facecolor('#16213e')
        
        # Full sine wave reference
        t_ref = np.linspace(0, 2 * np.pi, 200)
        y_ref = np.sin(t_ref)
        self.ax_sine.plot(t_ref, y_ref, 'w-', alpha=0.3, linewidth=1)
        self.ax_sine.fill_between(t_ref, y_ref, alpha=0.1, color='cyan')
        
        # Current position marker
        b = self.state.b % (2 * np.pi)
        y_current = np.sin(b)
        self.ax_sine.axvline(x=b, color='lime', linewidth=2, alpha=0.8)
        self.ax_sine.scatter([b], [y_current], c='lime', s=150, zorder=5, edgecolors='white')
        
        # History trail
        if len(self.sine_history) > 1:
            hist = np.array(self.sine_history) % (2 * np.pi)
            y_hist = np.sin(hist)
            alphas = np.linspace(0.1, 0.6, len(hist))
            for i in range(len(hist)):
                self.ax_sine.scatter([hist[i]], [y_hist[i]], c='lime', 
                                    s=30, alpha=alphas[i])
        
        # Keyframe markers
        for i in range(8):
            kf_b = (2 * np.pi * i) / 8
            self.ax_sine.axvline(x=kf_b, color='yellow', linewidth=1, alpha=0.3, linestyle='--')
            self.ax_sine.text(kf_b, 1.15, str(i), ha='center', fontsize=8, color='yellow')
        
        self.ax_sine.set_xlim([0, 2 * np.pi])
        self.ax_sine.set_ylim([-1.3, 1.4])
        self.ax_sine.set_xlabel('Phase b (rad)', color='gray', fontsize=9)
        self.ax_sine.set_ylabel('sin(b)', color='gray', fontsize=9)
        self.ax_sine.set_title(f'Phase: {b:.2f} rad ({np.degrees(b):.1f}°)', 
                               color='white', fontsize=10)
        self.ax_sine.tick_params(colors='gray', labelsize=7)
        self.ax_sine.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax_sine.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    def _update_pulse(self):
        """Update pulse/heartbeat indicator."""
        self.ax_pulse.clear()
        self.ax_pulse.set_facecolor('#16213e')
        
        concurrence = self.state.concurrence()
        
        # Pulse history
        if len(self.pulse_history) > 1:
            t = np.arange(len(self.pulse_history))
            self.ax_pulse.fill_between(t, self.pulse_history, alpha=0.3, color='magenta')
            self.ax_pulse.plot(t, self.pulse_history, 'magenta', linewidth=2)
        
        # Current pulse indicator (circle that scales with concurrence)
        pulse_size = 0.3 + concurrence * 0.7
        circle = plt.Circle((len(self.pulse_history), concurrence), pulse_size * 0.15,
                            color='magenta', alpha=0.8)
        self.ax_pulse.add_patch(circle)
        
        # Glow effect
        for i in range(3):
            glow = plt.Circle((len(self.pulse_history), concurrence), 
                             pulse_size * 0.15 * (1.5 + i * 0.5),
                             color='magenta', alpha=0.1 / (i + 1), fill=False, linewidth=2)
            self.ax_pulse.add_patch(glow)
        
        self.ax_pulse.set_xlim([0, self.max_pulse_history])
        self.ax_pulse.set_ylim([0, 1.0])
        self.ax_pulse.set_xlabel('Time', color='gray', fontsize=9)
        self.ax_pulse.set_ylabel('Concurrence', color='gray', fontsize=9)
        self.ax_pulse.set_title(f'Pulse (Entanglement): {concurrence:.3f}', 
                                color='white', fontsize=10)
        self.ax_pulse.tick_params(colors='gray', labelsize=7)
        
        # Threshold lines
        self.ax_pulse.axhline(y=0.5, color='yellow', alpha=0.3, linestyle='--')
        self.ax_pulse.text(2, 0.52, 'High', color='yellow', fontsize=8, alpha=0.5)
    
    def _update_info(self):
        """Update info text."""
        bloch_E = self.state.to_bloch_E()
        bloch_I = self.state.to_bloch_I()
        c = self.state.to_julia_c()
        
        info = f"""State:
  a = {self.state.a:.3f}
  b = {self.state.b:.3f} ({self.state.b_deg:.1f}°)
  c = {self.state.c:.3f}

Julia c = {c.real:.3f}{c.imag:+.3f}i

Bloch E: ({bloch_E[0]:+.2f}, {bloch_E[1]:+.2f}, {bloch_E[2]:+.2f})
Bloch I: ({bloch_I[0]:+.2f}, {bloch_I[1]:+.2f}, {bloch_I[2]:+.2f})

Concurrence: {self.state.concurrence():.4f}

{'[>] PLAYING' if self.playing else '[||] PAUSED'}"""
        
        self.info_text.set_text(info)
    
    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    
    def _on_slider_change(self, val):
        """Handle slider changes."""
        self.state = PNState(
            a=self.slider_a.val,
            b=self.slider_b.val,
            c=self.slider_c.val
        )
        self._compute_julia()
        self._update_traces()
        self._update_all_displays()
    
    def _on_speed_change(self, val):
        """Handle speed slider change."""
        self.animation_speed = val
    
    def _on_keyframe(self, idx):
        """Jump to keyframe."""
        b = (2 * np.pi * idx) / 8
        self.slider_b.set_val(b)
    
    def _on_play(self, event):
        """Start animation."""
        if not self.playing:
            self.playing = True
            self.anim = animation.FuncAnimation(
                self.fig, self._animate_frame,
                interval=self.frame_interval, blit=False
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
        """Stop and reset animation."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()
        
        # Reset to start
        self.slider_b.set_val(0)
        self.trace_E.clear()
        self.trace_I.clear()
        self.sine_history.clear()
        self.pulse_history.clear()
        
        self._update_all_displays()
    
    def _on_step(self, event):
        """Step one frame forward."""
        self.playing = False
        if self.anim:
            self.anim.event_source.stop()
        
        b = (self.state.b + self.animation_speed) % (2 * np.pi)
        self.slider_b.set_val(b)
    
    def _animate_frame(self, frame):
        """Animation frame update."""
        if not self.playing:
            return
        
        # Advance phase
        b = (self.state.b + self.animation_speed) % (2 * np.pi)
        
        # Update slider (triggers full update)
        self.slider_b.set_val(b)
    
    def _on_export(self, event):
        """Export all meshes."""
        output_dir = 'julia_exports'
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = f"julia_b{int(self.state.b_deg):03d}"
        
        # Canyon
        verts, faces, colors = create_canyon_mesh(self.julia_2d, self.bounds)
        export_ply(verts, faces, colors, os.path.join(output_dir, f"{prefix}_canyon.ply"))
        
        # Sphere
        verts, faces, colors = create_sphere_mesh(self.julia_2d)
        export_ply(verts, faces, colors, os.path.join(output_dir, f"{prefix}_sphere.ply"))
        
        # 2D image
        colors_2d = colormap_plasma(self.julia_2d.flatten()).reshape(
            self.resolution, self.resolution, 3)
        plt.imsave(os.path.join(output_dir, f"{prefix}_2d.png"), colors_2d)
        
        # State info
        with open(os.path.join(output_dir, f"{prefix}_state.txt"), 'w') as f:
            f.write(f"a = {self.state.a}\n")
            f.write(f"b = {self.state.b} ({self.state.b_deg:.1f} deg)\n")
            f.write(f"c = {self.state.c}\n")
            f.write(f"Julia c = {self.state.to_julia_c()}\n")
            f.write(f"Concurrence = {self.state.concurrence()}\n")
        
        print(f"Exported to {output_dir}/{prefix}_*")
    
    def run(self):
        """Run the explorer."""
        plt.show()


# =============================================================================
# SECTION 5: MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("QDNU JULIA ANIMATED EXPLORER")
    print("=" * 60)
    print("""
Controls:
  - Sliders: Adjust a, b, c, speed
  - Keyframes 0-7: Jump to preset phases
  - [>] Play: Start animation
  - [||] Pause: Pause animation
  - [X] Stop: Stop and reset
  - [>>] Step: Advance one frame
  - Export: Save meshes to julia_exports/

Visualizations:
  - Bloch E/I: Qubit state vectors with traces
  - Julia 2D: Standard fractal image
  - Canyon: Height-mapped terrain
  - Sphere: Julia on spherical surface
  - Sine Wave: Phase position indicator
  - Pulse: Entanglement (concurrence) meter
""")
    
    explorer = JuliaAnimatedExplorer(resolution=64)
    explorer.run()


if __name__ == '__main__':
    main()
