"""
4D Harmonic Oscillator Visualization for Quantum PN Neuron.

Visualizes the limit cycle / oscillatory manifold under periodic driving.
Under sinusoidal input f(t) = A*sin(wt) + offset, the quantum PN neuron
behaves as a driven 4D harmonic oscillator:

Dimensions:
  x1 = a (Excitatory amplitude)
  x2 = b (Phase accumulation)
  x3 = c (Inhibitory amplitude)
  x4 = Q (Quantum property: concurrence, purity, or fidelity)

The system traces a closed orbit (limit cycle) in this 4D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.signal import find_peaks, correlate
from scipy.fft import fft, fftfreq
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_agate import create_single_channel_agate
from .agate_visualization import extract_visualization_data


@dataclass
class OscillatorConfig:
    """Configuration for harmonic oscillator simulation."""
    lambda_a: float = 0.1
    lambda_c: float = 0.05
    dt: float = 0.001
    duration: float = 20.0
    drive_frequency: float = 0.2
    drive_amplitude: float = 0.4
    drive_offset: float = 0.5


def generate_sine_input(config: OscillatorConfig) -> np.ndarray:
    """Generate sinusoidal driving signal."""
    n_steps = int(config.duration / config.dt)
    t = np.arange(n_steps) * config.dt
    return config.drive_amplitude * np.sin(2 * np.pi * config.drive_frequency * t) + config.drive_offset


def simulate_driven_oscillator(config: OscillatorConfig,
                                compute_quantum: bool = True,
                                record_every: int = 10) -> Dict:
    """
    Simulate PN dynamics under sinusoidal driving.

    Args:
        config: Oscillator configuration
        compute_quantum: Whether to compute quantum properties (slower)
        record_every: Record state every N steps (for memory efficiency)

    Returns:
        Dict with time series: 't', 'f', 'a', 'b', 'c', 'concurrence', etc.
    """
    n_steps = int(config.duration / config.dt)

    # Generate input signal
    t_all = np.arange(n_steps) * config.dt
    f_all = config.drive_amplitude * np.sin(2 * np.pi * config.drive_frequency * t_all) + config.drive_offset

    # Initialize state
    a, b, c = 0.0, 0.0, 0.0

    # History storage
    history = {
        't': [], 'f': [], 'a': [], 'b': [], 'c': [],
        'concurrence': [], 'purity_E': [], 'purity_I': []
    }

    print(f"Simulating {n_steps} steps ({config.duration}s at dt={config.dt})...")

    for i in range(n_steps):
        f_t = abs(f_all[i])  # Rectify (PN dynamics expects positive input)

        # PN dynamics (clamp mode)
        da = (f_t * (1 - a) - config.lambda_a * a) * config.dt
        db = f_t * config.dt
        dc = (f_t * c + config.lambda_c * (1 - c)) * config.dt

        a = np.clip(a + da, 0, 1)
        b = max(0, b + db)  # b is unbounded above
        c = np.clip(c + dc, 0, 1)

        # Record state
        if i % record_every == 0:
            history['t'].append(t_all[i])
            history['f'].append(f_all[i])
            history['a'].append(a)
            history['b'].append(b)
            history['c'].append(c)

            # Compute quantum properties
            if compute_quantum:
                try:
                    circuit = create_single_channel_agate(a, b, c)
                    viz = extract_visualization_data(circuit)
                    history['concurrence'].append(viz['concurrence'])
                    history['purity_E'].append(viz['purity_E'])
                    history['purity_I'].append(viz['purity_I'])
                except Exception as e:
                    history['concurrence'].append(0.0)
                    history['purity_E'].append(1.0)
                    history['purity_I'].append(1.0)
            else:
                history['concurrence'].append(0.0)
                history['purity_E'].append(1.0)
                history['purity_I'].append(1.0)

        if i % 5000 == 0:
            print(f"  Step {i}/{n_steps} ({100*i/n_steps:.1f}%)")

    # Convert to arrays
    for key in history:
        history[key] = np.array(history[key])

    print(f"Recorded {len(history['t'])} samples")
    return history


class HarmonicOscillatorVisualizer:
    """
    Visualize quantum PN neuron as 4D harmonic oscillator.
    """

    def __init__(self, history: Dict):
        """
        Args:
            history: Dict with 't', 'f', 'a', 'b', 'c', 'concurrence', etc.
        """
        self.history = history
        self.n_frames = len(history['t'])

        # Extract arrays
        self.t = np.array(history['t'])
        self.f = np.array(history['f'])
        self.a = np.array(history['a'])
        self.b = np.array(history['b'])
        self.c = np.array(history['c'])
        self.conc = np.array(history.get('concurrence', np.zeros(self.n_frames)))

        # Compute derived quantities
        self.ei_balance = np.abs(self.a - self.c)
        self.phase = np.arctan2(self.a - np.mean(self.a), self.c - np.mean(self.c))

        # Detect input frequency for Poincare section
        self._detect_input_frequency()

    def _detect_input_frequency(self):
        """Detect the driving frequency from input signal."""
        # FFT of input
        n = len(self.f)
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0.01
        freq = fftfreq(n, dt)
        spectrum = np.abs(fft(self.f - np.mean(self.f)))

        # Find dominant frequency
        peak_idx = np.argmax(spectrum[1:n//2]) + 1
        self.driving_frequency = abs(freq[peak_idx]) if peak_idx < len(freq) else 0.1
        self.driving_period = 1.0 / self.driving_frequency if self.driving_frequency > 0 else np.inf

        print(f"Detected driving frequency: {self.driving_frequency:.4f} Hz")
        print(f"Driving period: {self.driving_period:.4f} s")

    # === 3D Trajectory Visualizations ===

    def plot_3d_trajectory(self,
                           x_var: str = 'a',
                           y_var: str = 'c',
                           z_var: str = 'concurrence',
                           color_var: str = 'time',
                           ax: Optional[Axes3D] = None,
                           skip_transient: int = 0) -> Axes3D:
        """
        Plot 3D trajectory with 4th dimension as color.

        Args:
            x_var, y_var, z_var: Variables for axes ('a', 'b', 'c', 'concurrence', 'ei_balance')
            color_var: Variable for color ('time', 'phase', 'f', or any of above)
            skip_transient: Number of initial frames to skip
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
            ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

        # Get data
        var_map = {
            'a': self.a, 'b': self.b, 'c': self.c,
            'concurrence': self.conc, 'ei_balance': self.ei_balance,
            'time': self.t, 'phase': self.phase, 'f': self.f
        }

        x = var_map[x_var][skip_transient:]
        y = var_map[y_var][skip_transient:]
        z = var_map[z_var][skip_transient:]
        colors = var_map[color_var][skip_transient:]

        # Normalize colors
        norm = Normalize(vmin=colors.min(), vmax=colors.max())

        # Plot as colored line segments
        for i in range(len(x) - 1):
            ax.plot3D(x[i:i+2], y[i:i+2], z[i:i+2],
                     color=cm.plasma(norm(colors[i])), linewidth=1, alpha=0.7)

        # Mark start and current end
        ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='s', label='End')

        # Highlight balance points (|a-c| < 0.01)
        balance_mask = self.ei_balance[skip_transient:] < 0.01
        if np.any(balance_mask):
            ax.scatter(x[balance_mask], y[balance_mask], z[balance_mask],
                      color='#00ff00', s=30, alpha=0.5, label='Balance points')

        ax.set_xlabel(x_var, color='white', fontsize=10)
        ax.set_ylabel(y_var, color='white', fontsize=10)
        ax.set_zlabel(z_var, color='white', fontsize=10)
        ax.set_title(f'4D Trajectory: ({x_var}, {y_var}, {z_var}) colored by {color_var}',
                    color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')

        # Add colorbar
        sm = cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_var, shrink=0.6)

        return ax

    def plot_oscillatory_surface(self,
                                  x_var: str = 'a',
                                  y_var: str = 'c',
                                  z_var: str = 'concurrence',
                                  resolution: int = 50,
                                  skip_transient: int = 0,
                                  ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot the oscillatory surface traced by the trajectory.

        Interpolates the trajectory onto a grid to show the 2D manifold
        embedded in 3D space.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
            ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

        var_map = {
            'a': self.a, 'b': self.b, 'c': self.c,
            'concurrence': self.conc, 'ei_balance': self.ei_balance
        }

        x = var_map[x_var][skip_transient:]
        y = var_map[y_var][skip_transient:]
        z = var_map[z_var][skip_transient:]

        # Create grid for surface interpolation
        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate z values onto grid
        try:
            Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

            # Plot surface
            surf = ax.plot_surface(Xi, Yi, Zi, cmap='magma', alpha=0.6,
                                   linewidth=0, antialiased=True)

            # Overlay trajectory
            ax.plot3D(x, y, z, color='white', linewidth=1, alpha=0.8, label='Trajectory')

            plt.colorbar(surf, ax=ax, label=z_var, shrink=0.6)

        except Exception as e:
            print(f"Surface interpolation failed: {e}")
            # Fall back to scatter plot
            ax.scatter(x, y, z, c=z, cmap='magma', s=5, alpha=0.5)
            ax.plot3D(x, y, z, color='white', linewidth=0.5, alpha=0.5)

        ax.set_xlabel(x_var, color='white')
        ax.set_ylabel(y_var, color='white')
        ax.set_zlabel(z_var, color='white')
        ax.set_title(f'Oscillatory Surface: {z_var}({x_var}, {y_var})', color='white')
        ax.tick_params(colors='white')

        return ax

    # === Poincare Section ===

    def compute_poincare_section(self,
                                  phase_offset: float = 0.0,
                                  skip_transient: int = 0) -> Optional[Dict]:
        """
        Compute Poincare section by sampling at fixed phase of input.

        Samples once per driving period when input phase = phase_offset.

        Returns:
            Dict with sampled values at each crossing
        """
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0.01
        samples_per_period = int(self.driving_period / dt)

        if samples_per_period < 2:
            print("Warning: Sampling rate too low for Poincare section")
            return None

        # Sample indices (every period, offset by phase)
        phase_samples = int(phase_offset * samples_per_period / (2 * np.pi))
        indices = np.arange(skip_transient + phase_samples, self.n_frames, samples_per_period)
        indices = indices[indices < self.n_frames]

        if len(indices) == 0:
            return None

        return {
            'indices': indices,
            't': self.t[indices],
            'a': self.a[indices],
            'b': self.b[indices],
            'c': self.c[indices],
            'concurrence': self.conc[indices],
            'ei_balance': self.ei_balance[indices],
            'f': self.f[indices]
        }

    def plot_poincare_section(self,
                               x_var: str = 'a',
                               y_var: str = 'c',
                               color_var: str = 'concurrence',
                               phase_offset: float = 0.0,
                               skip_transient: int = 0,
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D Poincare section.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')

        ax.set_facecolor('#0a0a1a')

        section = self.compute_poincare_section(phase_offset, skip_transient)
        if section is None:
            ax.text(0.5, 0.5, 'Insufficient data for Poincare section',
                   transform=ax.transAxes, ha='center', va='center', color='white')
            return ax

        x = section[x_var]
        y = section[y_var]
        colors = section[color_var]

        scatter = ax.scatter(x, y, c=colors, cmap='plasma', s=50, alpha=0.8)

        # Connect points to show iteration
        ax.plot(x, y, 'w-', alpha=0.3, linewidth=0.5)

        # Mark first and last
        ax.scatter(x[0], y[0], color='green', s=150, marker='o', zorder=10, label='First')
        ax.scatter(x[-1], y[-1], color='red', s=150, marker='s', zorder=10, label='Last')

        # a = c diagonal
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
               max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, '--', color='#00ff00', alpha=0.5, label='a = c')

        ax.set_xlabel(x_var, color='white', fontsize=12)
        ax.set_ylabel(y_var, color='white', fontsize=12)
        ax.set_title(f'Poincare Section (phase={phase_offset:.2f})', color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.set_aspect('equal')

        plt.colorbar(scatter, ax=ax, label=color_var)

        return ax

    def plot_poincare_3d(self,
                          x_var: str = 'a',
                          y_var: str = 'c',
                          z_var: str = 'concurrence',
                          phase_offset: float = 0.0,
                          skip_transient: int = 0,
                          ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D Poincare section.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
            ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

        section = self.compute_poincare_section(phase_offset, skip_transient)
        if section is None:
            return ax

        x = section[x_var]
        y = section[y_var]
        z = section[z_var]

        # Color by iteration number
        colors = np.arange(len(x))

        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.8)
        ax.plot3D(x, y, z, 'w-', alpha=0.3, linewidth=0.5)

        ax.scatter(x[0], y[0], z[0], color='green', s=150, marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=150, marker='s')

        ax.set_xlabel(x_var, color='white')
        ax.set_ylabel(y_var, color='white')
        ax.set_zlabel(z_var, color='white')
        ax.set_title('3D Poincare Section', color='white')
        ax.tick_params(colors='white')

        plt.colorbar(scatter, ax=ax, label='Iteration', shrink=0.6)

        return ax

    # === Lissajous Figures ===

    def plot_lissajous_grid(self,
                            skip_transient: int = 0,
                            figsize: Tuple[int, int] = (14, 14)) -> plt.Figure:
        """
        Plot grid of all 2D projections (Lissajous figures).

        Shows phase relationships between all pairs of variables.
        """
        variables = ['a', 'c', 'concurrence', 'ei_balance']
        var_map = {
            'a': self.a[skip_transient:],
            'c': self.c[skip_transient:],
            'concurrence': self.conc[skip_transient:],
            'ei_balance': self.ei_balance[skip_transient:]
        }

        n = len(variables)
        fig, axes = plt.subplots(n, n, figsize=figsize, facecolor='#0a0a1a')

        t_norm = np.linspace(0, 1, len(var_map['a']))

        for i, var_y in enumerate(variables):
            for j, var_x in enumerate(variables):
                ax = axes[i, j]
                ax.set_facecolor('#0a0a1a')

                if i == j:
                    # Diagonal: histogram
                    ax.hist(var_map[var_x], bins=30, color='#7b68ee', alpha=0.7)
                    ax.set_ylabel('Count', color='white', fontsize=8)
                else:
                    # Off-diagonal: scatter/line plot
                    x = var_map[var_x]
                    y = var_map[var_y]

                    # Plot as fading line
                    for k in range(len(x) - 1):
                        ax.plot(x[k:k+2], y[k:k+2],
                               color=cm.plasma(t_norm[k]), alpha=0.5, linewidth=0.5)

                    # Add a=c line for relevant plots
                    if var_x == 'a' and var_y == 'c':
                        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
                        ax.plot(lim, lim, '--', color='#00ff00', alpha=0.5)

                # Labels on edges only
                if i == n - 1:
                    ax.set_xlabel(var_x, color='white', fontsize=9)
                if j == 0:
                    ax.set_ylabel(var_y, color='white', fontsize=9)

                ax.tick_params(colors='white', labelsize=6)

        fig.suptitle('Lissajous Grid: Phase Relationships', color='white', fontsize=14)
        plt.tight_layout()

        return fig

    # === Limit Cycle Analysis ===

    def analyze_limit_cycle(self, skip_transient: int = 0) -> Dict:
        """
        Analyze the limit cycle structure.

        Returns:
            Dict with limit cycle properties
        """
        a = self.a[skip_transient:]
        c = self.c[skip_transient:]
        conc = self.conc[skip_transient:]

        # Compute "radius" from center of limit cycle
        a_center = np.mean(a)
        c_center = np.mean(c)

        radius = np.sqrt((a - a_center)**2 + (c - c_center)**2)

        # Angle in (a, c) plane
        angle = np.arctan2(c - c_center, a - a_center)

        # Find period by autocorrelation
        a_centered = a - np.mean(a)
        autocorr = correlate(a_centered, a_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find first peak after zero
        peaks, _ = find_peaks(autocorr, height=0)

        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0.01
        period = peaks[0] * dt if len(peaks) > 0 else np.nan

        return {
            'center_a': a_center,
            'center_c': c_center,
            'mean_radius': np.mean(radius),
            'radius_std': np.std(radius),
            'period': period,
            'frequency': 1.0 / period if period > 0 and not np.isnan(period) else np.nan,
            'mean_concurrence': np.mean(conc),
            'concurrence_amplitude': (np.max(conc) - np.min(conc)) / 2,
            'radius': radius,
            'angle': angle
        }

    def plot_limit_cycle_analysis(self, skip_transient: int = 0) -> plt.Figure:
        """
        Comprehensive limit cycle visualization.
        """
        fig = plt.figure(figsize=(16, 12), facecolor='#0a0a1a')

        analysis = self.analyze_limit_cycle(skip_transient)

        # 1. (a, c) phase portrait
        ax1 = fig.add_subplot(2, 3, 1, facecolor='#0a0a1a')
        a = self.a[skip_transient:]
        c = self.c[skip_transient:]
        t_norm = np.linspace(0, 1, len(a))

        for i in range(len(a) - 1):
            ax1.plot(a[i:i+2], c[i:i+2], color=cm.plasma(t_norm[i]), linewidth=1)

        # Mark center
        ax1.scatter(analysis['center_a'], analysis['center_c'],
                   color='white', s=100, marker='+', linewidths=2)

        # a = c line
        ax1.plot([0, 1], [0, 1], '--', color='#00ff00', alpha=0.5)

        ax1.set_xlabel('a', color='white')
        ax1.set_ylabel('c', color='white')
        ax1.set_title('Phase Portrait (a, c)', color='white')
        ax1.tick_params(colors='white')
        ax1.set_aspect('equal')

        # 2. Radius vs angle (polar-ish)
        ax2 = fig.add_subplot(2, 3, 2, facecolor='#0a0a1a')
        ax2.scatter(analysis['angle'], analysis['radius'],
                   c=self.conc[skip_transient:], cmap='plasma', s=5, alpha=0.5)
        ax2.set_xlabel('Angle (rad)', color='white')
        ax2.set_ylabel('Radius from center', color='white')
        ax2.set_title('Limit Cycle Shape', color='white')
        ax2.tick_params(colors='white')

        # 3. Radius over time
        ax3 = fig.add_subplot(2, 3, 3, facecolor='#0a0a1a')
        t = self.t[skip_transient:]
        ax3.plot(t, analysis['radius'], color='#ff6b35', linewidth=0.5)
        ax3.axhline(analysis['mean_radius'], color='white', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Radius', color='white')
        ax3.set_title(f'Radius (mean={analysis["mean_radius"]:.4f})', color='white')
        ax3.tick_params(colors='white')

        # 4. Concurrence vs angle (shows where interference happens)
        ax4 = fig.add_subplot(2, 3, 4, facecolor='#0a0a1a')
        ax4.scatter(analysis['angle'], self.conc[skip_transient:],
                   c=self.ei_balance[skip_transient:], cmap='RdYlGn_r', s=5, alpha=0.5)
        ax4.set_xlabel('Angle (rad)', color='white')
        ax4.set_ylabel('Concurrence', color='white')
        ax4.set_title('Concurrence vs Orbit Position', color='white')
        ax4.tick_params(colors='white')

        # 5. |a-c| vs angle (shows balance points on orbit)
        ax5 = fig.add_subplot(2, 3, 5, facecolor='#0a0a1a')
        ax5.scatter(analysis['angle'], self.ei_balance[skip_transient:],
                   c=self.conc[skip_transient:], cmap='plasma', s=5, alpha=0.5)
        ax5.axhline(0.01, color='#00ff00', linestyle='--', alpha=0.7, label='Balance threshold')
        ax5.set_xlabel('Angle (rad)', color='white')
        ax5.set_ylabel('|a - c|', color='white')
        ax5.set_title('E-I Balance vs Orbit Position', color='white')
        ax5.tick_params(colors='white')
        ax5.legend(facecolor='#1a1a2e', labelcolor='white')

        # 6. Summary stats
        ax6 = fig.add_subplot(2, 3, 6, facecolor='#0a0a1a')
        ax6.axis('off')

        period_str = f"{analysis['period']:.4f}" if not np.isnan(analysis['period']) else "N/A"
        freq_str = f"{analysis['frequency']:.4f}" if not np.isnan(analysis['frequency']) else "N/A"

        summary = f"""
        LIMIT CYCLE ANALYSIS
        ================================

        Center:
          a_center = {analysis['center_a']:.4f}
          c_center = {analysis['center_c']:.4f}

        Shape:
          Mean radius = {analysis['mean_radius']:.4f}
          Radius std  = {analysis['radius_std']:.4f}

        Timing:
          Period = {period_str} s
          Frequency = {freq_str} Hz
          Driving freq = {self.driving_frequency:.4f} Hz

        Quantum:
          Mean concurrence = {analysis['mean_concurrence']:.4f}
          Conc. amplitude = {analysis['concurrence_amplitude']:.4f}
        """

        ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
                fontsize=11, color='white', family='monospace',
                verticalalignment='top')

        plt.tight_layout()
        return fig


# === Convenience Functions ===

def visualize_harmonic_oscillator(history: Dict,
                                   skip_transient_fraction: float = 0.2,
                                   save_dir: Optional[str] = None) -> Tuple[HarmonicOscillatorVisualizer, Dict]:
    """
    Generate comprehensive harmonic oscillator visualizations.

    Args:
        history: Simulation history dict
        skip_transient_fraction: Fraction of initial data to skip
        save_dir: Directory to save figures (optional)

    Returns:
        Tuple of (visualizer, figures dict)
    """
    n = len(history['t'])
    skip = int(n * skip_transient_fraction)

    viz = HarmonicOscillatorVisualizer(history)

    print("\n" + "="*60)
    print("4D HARMONIC OSCILLATOR ANALYSIS")
    print("="*60)

    # Generate figures
    figures = {}

    # 1. 3D trajectory
    fig1 = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
    ax1 = fig1.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    viz.plot_3d_trajectory('a', 'c', 'concurrence', 'time', ax=ax1, skip_transient=skip)
    figures['trajectory_3d'] = fig1

    # 2. Oscillatory surface
    fig2 = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
    ax2 = fig2.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    viz.plot_oscillatory_surface('a', 'c', 'concurrence', skip_transient=skip, ax=ax2)
    figures['surface'] = fig2

    # 3. Poincare section
    fig3, ax3 = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')
    viz.plot_poincare_section('a', 'c', 'concurrence', skip_transient=skip, ax=ax3)
    figures['poincare'] = fig3

    # 4. Lissajous grid
    figures['lissajous'] = viz.plot_lissajous_grid(skip_transient=skip)

    # 5. Limit cycle analysis
    figures['limit_cycle'] = viz.plot_limit_cycle_analysis(skip_transient=skip)

    # Save if requested
    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            fig.savefig(save_path / f'{name}.png', dpi=150, facecolor='#0a0a1a')
            print(f"Saved: {save_path / f'{name}.png'}")

    print(f"\nGenerated {len(figures)} figures")

    return viz, figures


def run_oscillator_demo(duration: float = 20.0,
                        drive_frequency: float = 0.2,
                        save_dir: Optional[str] = None):
    """
    Run a complete harmonic oscillator demonstration.

    Args:
        duration: Simulation duration in seconds
        drive_frequency: Driving frequency in Hz
        save_dir: Directory to save figures
    """
    config = OscillatorConfig(
        duration=duration,
        drive_frequency=drive_frequency,
        drive_amplitude=0.4,
        drive_offset=0.5,
        lambda_a=0.1,
        lambda_c=0.05,
        dt=0.001
    )

    print("="*60)
    print("4D HARMONIC OSCILLATOR DEMO")
    print("="*60)
    print(f"Duration: {duration}s")
    print(f"Driving: {drive_frequency} Hz, A={config.drive_amplitude}, offset={config.drive_offset}")
    print(f"PN params: lambda_a={config.lambda_a}, lambda_c={config.lambda_c}")
    print()

    # Simulate
    history = simulate_driven_oscillator(config, compute_quantum=True, record_every=10)

    # Visualize
    viz, figures = visualize_harmonic_oscillator(history, save_dir=save_dir)

    return config, history, viz, figures


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')  # Interactive backend

    config, history, viz, figures = run_oscillator_demo(
        duration=20.0,
        drive_frequency=0.2,
        save_dir='figures/harmonic_oscillator'
    )

    plt.show()
