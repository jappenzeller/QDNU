"""
Phase Space Analysis for Quantum PN Neuron Dynamics.

Provides dynamical systems analysis including:
- 2D/3D phase portraits with vector fields
- Fixed point analysis and stability classification
- Nullclines and trajectories
- Quantum observable mapping
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Dict, List
from pathlib import Path

import sys
# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class DynamicsConfig:
    """Configuration for PN dynamics analysis."""
    lambda_a: float = 0.1      # Excitatory decay rate
    lambda_c: float = 0.05     # Inhibitory growth rate
    f_constant: float = 0.5    # Constant input (for autonomous analysis)
    dt: float = 0.001          # Integration timestep

    # Bounds
    a_bounds: Tuple[float, float] = (0.0, 1.0)
    b_bounds: Tuple[float, float] = (0.0, 2*np.pi)
    c_bounds: Tuple[float, float] = (0.0, 1.0)


def compute_derivatives(state: np.ndarray, config: DynamicsConfig,
                        f_t: Optional[float] = None) -> np.ndarray:
    """
    Compute (da/dt, db/dt, dc/dt).

    Returns:
        [da/dt, db/dt, dc/dt] array
    """
    a, b, c = state
    f = f_t if f_t is not None else config.f_constant

    da = -config.lambda_a * a + f * (1 - a)
    db = f * (1 - b)
    dc = config.lambda_c * c + f * (1 - c)

    return np.array([da, db, dc])


def compute_jacobian(state: np.ndarray, config: DynamicsConfig,
                     f_t: Optional[float] = None) -> np.ndarray:
    """Compute Jacobian matrix at given state."""
    f = f_t if f_t is not None else config.f_constant

    J = np.array([
        [-(config.lambda_a + f), 0, 0],
        [0, -f, 0],
        [0, 0, config.lambda_c - f]
    ])
    return J


def integrate_rk4(initial: np.ndarray, config: DynamicsConfig,
                  n_steps: int, f_func: Optional[Callable[[float], float]] = None,
                  record_every: int = 1) -> np.ndarray:
    """
    Integrate trajectory using RK4.

    Returns:
        (n_recorded, 3) trajectory array
    """
    n_recorded = (n_steps + record_every - 1) // record_every
    trajectory = np.zeros((n_recorded, 3))

    state = initial.copy()
    dt = config.dt

    record_idx = 0
    for i in range(n_steps):
        t = i * dt
        f_t = f_func(t) if f_func else config.f_constant

        # RK4 step
        k1 = compute_derivatives(state, config, f_t)
        k2 = compute_derivatives(state + 0.5*dt*k1, config, f_t)
        k3 = compute_derivatives(state + 0.5*dt*k2, config, f_t)
        k4 = compute_derivatives(state + dt*k3, config, f_t)

        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Apply bounds
        state[0] = np.clip(state[0], *config.a_bounds)
        state[1] = np.clip(state[1], *config.b_bounds)
        state[2] = np.clip(state[2], *config.c_bounds)

        # Record
        if i % record_every == 0:
            trajectory[record_idx] = state
            record_idx += 1

    return trajectory[:record_idx]


def find_fixed_points(config: DynamicsConfig) -> List[Dict]:
    """
    Find fixed points analytically.

    Returns list of dicts: {'point', 'eigenvalues', 'stability', 'type'}
    """
    f = config.f_constant
    lambda_a = config.lambda_a
    lambda_c = config.lambda_c

    fixed_points = []

    if f == 0:
        fixed_points.append({
            'point': np.array([0.0, 0.0, 0.0]),
            'eigenvalues': np.array([-lambda_a, 0, lambda_c]),
            'stability': 'saddle',
            'type': 'origin (f=0)'
        })
    else:
        a_star = f / (lambda_a + f)
        b_star = 1.0

        if f > lambda_c:
            c_star = f / (f - lambda_c)
            c_star = min(c_star, 1.0)
        else:
            c_star = 1.0

        point = np.array([a_star, b_star, c_star])
        J = compute_jacobian(point, config)
        eigenvalues = np.linalg.eigvals(J)

        real_parts = np.real(eigenvalues)
        if all(real_parts < 0):
            stability = 'stable'
        elif all(real_parts > 0):
            stability = 'unstable'
        else:
            stability = 'saddle'

        fixed_points.append({
            'point': point,
            'eigenvalues': eigenvalues,
            'stability': stability,
            'type': 'interior' if c_star < 1.0 else 'boundary (c=1)'
        })

    return fixed_points


def compute_vector_field_2d(config: DynamicsConfig, plane: str = 'ac',
                            fixed_coord: float = 0.5,
                            resolution: int = 20) -> Tuple[np.ndarray, ...]:
    """
    Compute 2D vector field for phase portrait.

    Args:
        plane: 'ac', 'ab', or 'bc'
        fixed_coord: Value of third coordinate
        resolution: Grid resolution

    Returns:
        X, Y, U, V arrays for quiver plot
    """
    ranges = {
        'ac': ((0.01, 0.99), (0.01, 0.99)),
        'ab': ((0.01, 0.99), (0.01, 2*np.pi - 0.01)),
        'bc': ((0.01, 2*np.pi - 0.01), (0.01, 0.99)),
    }

    x_range, y_range = ranges[plane]
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    idx_map = {
        'ac': (0, 2, 1),
        'ab': (0, 1, 2),
        'bc': (1, 2, 0),
    }
    x_idx, y_idx, fixed_idx = idx_map[plane]

    for i in range(resolution):
        for j in range(resolution):
            state = np.zeros(3)
            state[x_idx] = X[i, j]
            state[y_idx] = Y[i, j]
            state[fixed_idx] = fixed_coord

            deriv = compute_derivatives(state, config)
            U[i, j] = deriv[x_idx]
            V[i, j] = deriv[y_idx]

    return X, Y, U, V


def compute_nullclines(config: DynamicsConfig, plane: str = 'ac') -> Dict:
    """Compute nullclines (where da/dt=0, dc/dt=0, etc)."""
    f = config.f_constant
    lambda_a = config.lambda_a
    lambda_c = config.lambda_c

    nullclines = {}

    if plane == 'ac':
        a_null = f / (lambda_a + f) if (lambda_a + f) > 0 else 0
        nullclines['da_dt_zero'] = {
            'type': 'vertical',
            'value': a_null,
            'label': 'da/dt = 0'
        }

        if f > lambda_c:
            c_null = f / (f - lambda_c)
            if c_null <= 1:
                nullclines['dc_dt_zero'] = {
                    'type': 'horizontal',
                    'value': c_null,
                    'label': 'dc/dt = 0'
                }

    return nullclines


def plot_phase_portrait_2d(config: DynamicsConfig, plane: str = 'ac',
                           fixed_coord: float = 0.5, n_trajectories: int = 16,
                           t_max: float = 20.0, ax: Optional[plt.Axes] = None,
                           show_nullclines: bool = True,
                           show_fixed_points: bool = True) -> plt.Axes:
    """Generate 2D phase portrait with dark theme."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')

    ax.set_facecolor('#0a0a1a')

    labels = {
        'ac': ('a (Excitatory)', 'c (Inhibitory)', (0, 1), (0, 1)),
        'ab': ('a (Excitatory)', 'b (Phase)', (0, 1), (0, 2*np.pi)),
        'bc': ('b (Phase)', 'c (Inhibitory)', (0, 2*np.pi), (0, 1)),
    }
    x_label, y_label, x_lim, y_lim = labels[plane]
    idx_map = {'ac': (0, 2), 'ab': (0, 1), 'bc': (1, 2)}
    x_idx, y_idx = idx_map[plane]

    # Vector field
    X, Y, U, V = compute_vector_field_2d(config, plane, fixed_coord, resolution=20)
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    ax.quiver(X, Y, U/magnitude, V/magnitude, magnitude,
              cmap='plasma', alpha=0.6, scale=25)

    # Nullclines
    if show_nullclines:
        nullclines = compute_nullclines(config, plane)
        colors = {'da_dt_zero': '#ff6b35', 'dc_dt_zero': '#7b68ee'}
        for name, nc in nullclines.items():
            if nc['type'] == 'vertical':
                ax.axvline(nc['value'], color=colors.get(name, 'white'),
                          linestyle='--', linewidth=2, label=nc['label'])
            else:
                ax.axhline(nc['value'], color=colors.get(name, 'white'),
                          linestyle='--', linewidth=2, label=nc['label'])

    # Sample trajectories
    np.random.seed(42)
    n_steps = int(t_max / config.dt)
    fixed_idx = {'ac': 1, 'ab': 2, 'bc': 0}[plane]

    for _ in range(n_trajectories):
        initial = np.zeros(3)
        initial[x_idx] = np.random.uniform(*x_lim)
        initial[y_idx] = np.random.uniform(*y_lim)
        initial[fixed_idx] = fixed_coord

        traj = integrate_rk4(initial, config, n_steps)

        # Plot with time coloring
        colors_arr = np.linspace(0, 1, len(traj))
        for i in range(len(traj) - 1):
            ax.plot(traj[i:i+2, x_idx], traj[i:i+2, y_idx],
                   color=plt.cm.viridis(colors_arr[i]), alpha=0.6, linewidth=0.5)

        ax.scatter(traj[0, x_idx], traj[0, y_idx], c='green', s=20, zorder=5)
        ax.scatter(traj[-1, x_idx], traj[-1, y_idx], c='red', s=20, zorder=5)

    # Fixed points
    if show_fixed_points:
        fps = find_fixed_points(config)
        for fp in fps:
            color = '#00ff00' if fp['stability'] == 'stable' else '#ff0000'
            marker = 'o' if fp['stability'] == 'stable' else 'x'
            ax.scatter(fp['point'][x_idx], fp['point'][y_idx],
                      c=color, s=200, marker=marker, edgecolors='white',
                      linewidths=2, zorder=10)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_label, color='white', fontsize=12)
    ax.set_ylabel(y_label, color='white', fontsize=12)
    ax.set_title(f'Phase Portrait ({plane} plane)', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', loc='upper right')

    return ax


def plot_phase_portrait_3d(config: DynamicsConfig, n_trajectories: int = 10,
                           t_max: float = 20.0, ax: Optional[Axes3D] = None) -> Axes3D:
    """Generate 3D phase portrait in full (a, b, c) space."""
    if ax is None:
        fig = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

    np.random.seed(42)
    n_steps = int(t_max / config.dt)

    for _ in range(n_trajectories):
        initial = np.array([
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 2*np.pi - 0.1),
            np.random.uniform(0.1, 0.9)
        ])

        traj = integrate_rk4(initial, config, n_steps)

        colors_arr = np.linspace(0, 1, len(traj))
        for i in range(len(traj) - 1):
            ax.plot3D(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
                     color=plt.cm.viridis(colors_arr[i]), alpha=0.6, linewidth=0.5)

        ax.scatter(*initial, c='green', s=50)
        ax.scatter(*traj[-1], c='red', s=50)

    # Fixed points
    fps = find_fixed_points(config)
    for fp in fps:
        color = '#00ff00' if fp['stability'] == 'stable' else '#ff0000'
        ax.scatter(*fp['point'], c=color, s=200, marker='*', edgecolors='white')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2*np.pi])
    ax.set_zlim([0, 1])
    ax.set_xlabel('a (E)', color='white')
    ax.set_ylabel('b (phi)', color='white')
    ax.set_zlabel('c (I)', color='white')
    ax.set_title('3D Phase Space', color='white')
    ax.tick_params(colors='white')

    return ax


# === Quantum Observable Mapping ===

def classical_to_quantum_angles(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Map classical (a,b,c) to quantum gate parameters."""
    rx_angle = 2 * np.pi * a
    phase_angle = b
    ry_angle = 2 * np.pi * c
    return rx_angle, phase_angle, ry_angle


def compute_bloch_coords_from_abc(a: float, b: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate Bloch sphere coordinates for both qubits from (a,b,c).

    Returns:
        (bloch_q0, bloch_q1) each as [x, y, z]
    """
    theta0 = np.pi * a
    phi0 = b
    bloch_q0 = np.array([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0)
    ])

    theta1 = np.pi * c
    phi1 = b + np.pi/2
    bloch_q1 = np.array([
        np.sin(theta1) * np.cos(phi1),
        np.sin(theta1) * np.sin(phi1),
        np.cos(theta1)
    ])

    return bloch_q0, bloch_q1


def estimate_concurrence(a: float, b: float, c: float) -> float:
    """Estimate entanglement (concurrence) from classical parameters."""
    coupling_strength = np.sin(np.pi/4)
    superposition = 4 * a * (1 - a) * c * (1 - c)
    coherence = np.abs(np.cos(b - np.pi/4))
    concurrence = coupling_strength * np.sqrt(superposition) * coherence
    return np.clip(concurrence, 0, 1)


def trajectory_to_quantum_observables(trajectory: np.ndarray,
                                       config: DynamicsConfig) -> Dict:
    """Convert classical trajectory to quantum observables."""
    n_frames = len(trajectory)

    bloch_q0 = np.zeros((n_frames, 3))
    bloch_q1 = np.zeros((n_frames, 3))
    concurrence = np.zeros(n_frames)
    purity_q0 = np.zeros(n_frames)
    purity_q1 = np.zeros(n_frames)

    for i, (a, b, c) in enumerate(trajectory):
        b0, b1 = compute_bloch_coords_from_abc(a, b, c)
        bloch_q0[i] = b0
        bloch_q1[i] = b1
        concurrence[i] = estimate_concurrence(a, b, c)
        purity_q0[i] = (1 + np.linalg.norm(b0)**2) / 2
        purity_q1[i] = (1 + np.linalg.norm(b1)**2) / 2

    return {
        'time': np.arange(n_frames) * config.dt,
        'bloch_q0': bloch_q0,
        'bloch_q1': bloch_q1,
        'concurrence': concurrence,
        'purity_q0': purity_q0,
        'purity_q1': purity_q1,
        'classical': trajectory
    }


def plot_quantum_observables(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """Plot quantum observables derived from classical trajectory."""
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a1a')

    t = results['time']
    traj = results['classical']

    # 1. Classical parameters
    ax1 = fig.add_subplot(2, 3, 1, facecolor='#0a0a1a')
    ax1.plot(t, traj[:, 0], color='#ff6b35', label='a (E)', linewidth=1.5)
    ax1.plot(t, traj[:, 2], color='#7b68ee', label='c (I)', linewidth=1.5)
    ax1.set_xlabel('Time (s)', color='white')
    ax1.set_ylabel('Value', color='white')
    ax1.set_title('Classical Parameters', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')

    # 2. Concurrence (entanglement)
    ax2 = fig.add_subplot(2, 3, 2, facecolor='#0a0a1a')
    ax2.fill_between(t, 0, results['concurrence'], color='#ff00ff', alpha=0.5)
    ax2.plot(t, results['concurrence'], color='#ff00ff', linewidth=1)
    ax2.set_xlabel('Time (s)', color='white')
    ax2.set_ylabel('Concurrence', color='white')
    ax2.set_title('Entanglement', color='white')
    ax2.tick_params(colors='white')
    ax2.set_ylim([0, 1])

    # 3. Purity
    ax3 = fig.add_subplot(2, 3, 3, facecolor='#0a0a1a')
    ax3.plot(t, results['purity_q0'], color='#ff6b35', label='Q0 (E)', linewidth=1.5)
    ax3.plot(t, results['purity_q1'], color='#7b68ee', label='Q1 (I)', linewidth=1.5)
    ax3.set_xlabel('Time (s)', color='white')
    ax3.set_ylabel('Purity', color='white')
    ax3.set_title('State Purity', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a2e', labelcolor='white')

    # 4. Bloch sphere Q0
    ax4 = fig.add_subplot(2, 3, 4, projection='3d', facecolor='#0a0a1a')
    bloch0 = results['bloch_q0']
    colors = np.linspace(0, 1, len(bloch0))
    ax4.scatter(bloch0[:, 0], bloch0[:, 1], bloch0[:, 2],
               c=colors, cmap='plasma', s=1, alpha=0.5)

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_wireframe(x, y, z, color='white', alpha=0.1)

    ax4.set_xlabel('X', color='white')
    ax4.set_ylabel('Y', color='white')
    ax4.set_zlabel('Z', color='white')
    ax4.set_title('Bloch Sphere Q0 (E)', color='white')

    # 5. Bloch sphere Q1
    ax5 = fig.add_subplot(2, 3, 5, projection='3d', facecolor='#0a0a1a')
    bloch1 = results['bloch_q1']
    ax5.scatter(bloch1[:, 0], bloch1[:, 1], bloch1[:, 2],
               c=colors, cmap='viridis', s=1, alpha=0.5)
    ax5.plot_wireframe(x, y, z, color='white', alpha=0.1)

    ax5.set_xlabel('X', color='white')
    ax5.set_ylabel('Y', color='white')
    ax5.set_zlabel('Z', color='white')
    ax5.set_title('Bloch Sphere Q1 (I)', color='white')

    # 6. Phase space with concurrence coloring
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#0a0a1a')
    scatter = ax6.scatter(traj[:, 0], traj[:, 2],
                         c=results['concurrence'], cmap='magma',
                         s=1, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Concurrence', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax6.set_xlabel('a (E)', color='white')
    ax6.set_ylabel('c (I)', color='white')
    ax6.set_title('Phase Space (colored by entanglement)', color='white')
    ax6.tick_params(colors='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#0a0a1a', edgecolor='none')
        print(f"Saved: {save_path}")

    return fig


def run_phase_analysis(lambda_a: float = 0.1, lambda_c: float = 0.05,
                       f_constant: float = 0.5, t_max: float = 20.0,
                       save_dir: Optional[str] = None):
    """Run complete phase space analysis and generate figures."""
    config = DynamicsConfig(lambda_a=lambda_a, lambda_c=lambda_c,
                           f_constant=f_constant)

    print(f"Phase Analysis: lambda_a={lambda_a}, lambda_c={lambda_c}, f={f_constant}")

    # Find fixed points
    fps = find_fixed_points(config)
    print(f"\nFixed Points ({len(fps)}):")
    for fp in fps:
        print(f"  {fp['type']}: {fp['point']}, stability={fp['stability']}")
        print(f"    eigenvalues: {fp['eigenvalues']}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 2D phase portrait
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')
        plot_phase_portrait_2d(config, plane='ac', ax=ax)
        plt.savefig(save_path / 'phase_portrait_ac.png', dpi=200,
                   facecolor='#0a0a1a', edgecolor='none')
        plt.close()
        print(f"Saved: {save_path / 'phase_portrait_ac.png'}")

        # 3D phase portrait
        fig = plt.figure(figsize=(12, 10), facecolor='#0a0a1a')
        ax3d = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
        plot_phase_portrait_3d(config, ax=ax3d)
        plt.savefig(save_path / 'phase_portrait_3d.png', dpi=200,
                   facecolor='#0a0a1a', edgecolor='none')
        plt.close()
        print(f"Saved: {save_path / 'phase_portrait_3d.png'}")

        # Quantum observables
        initial = np.array([0.1, 0.1, 0.1])
        n_steps = int(t_max / config.dt)
        traj = integrate_rk4(initial, config, n_steps)
        results = trajectory_to_quantum_observables(traj, config)
        plot_quantum_observables(results, save_path=str(save_path / 'quantum_observables.png'))
        plt.close()

    return config, fps


if __name__ == '__main__':
    # Run analysis with default parameters
    save_dir = Path(__file__).parent.parent / 'figures' / 'phase_analysis'
    run_phase_analysis(save_dir=str(save_dir))
    plt.show()
