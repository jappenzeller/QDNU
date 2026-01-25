"""
Limit Cycle Explorer for PN Dynamics + 3D/4D Julia Set Visualization

Goal: Find stable oscillatory parameter trajectories (limit cycles) in the 
PN dynamical system that can drive smooth Julia set animations.

For 4D visualization:
- Julia set parameter c = f(a, b, c_pn) from PN dynamics
- As (a, b, c_pn) traces a limit cycle, c traces a closed loop
- 3D Julia slices + time = 4D animated structure

Author: QDNU Project
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: PN DYNAMICS WITH LIMIT CYCLE SEARCH
# =============================================================================

class PNOscillator:
    """
    PN dynamics based on FitzHugh-Nagumo model - GUARANTEED limit cycles.
    
    FitzHugh-Nagumo is a simplified Hodgkin-Huxley model:
    - Fast variable (v/a): Excitatory membrane potential
    - Slow variable (w/c): Inhibitory recovery variable
    - Add phase (b) driven by oscillation
    
    Equations:
    da/dt = a - a³/3 - c + I_ext       (fast, cubic nonlinearity)
    dc/dt = ε(a + α - β*c)             (slow recovery, ε << 1)
    db/dt = ω + κ*a                    (phase accumulates with excitation)
    
    The cubic nonlinearity + timescale separation guarantees a limit cycle
    for appropriate parameter choices.
    """
    
    def __init__(self, epsilon=0.08, alpha=0.7, beta=0.8, I_ext=0.5, 
                 omega=0.3, kappa=0.15):
        """
        Parameters for FitzHugh-Nagumo oscillator.
        
        Args:
            epsilon: Timescale separation (smaller = more separation)
            alpha: Recovery variable offset
            beta: Recovery variable coupling
            I_ext: External current (tonic drive) - controls oscillation
            omega: Base phase velocity
            kappa: Phase modulation by excitation
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.I_ext = I_ext
        self.omega = omega
        self.kappa = kappa
    
    def derivatives(self, state, t, f_t=None):
        """Compute derivatives for ODE integration."""
        a, b, c = state
        
        I = self.I_ext if f_t is None else f_t
        
        # FitzHugh-Nagumo dynamics
        da = a - (a**3) / 3 - c + I
        dc = self.epsilon * (a + self.alpha - self.beta * c)
        db = self.omega + self.kappa * a
        
        return [da, db, dc]
    
    def integrate(self, t_span, initial_state=None, n_points=10000):
        """
        Integrate dynamics and return trajectory.
        
        For FHN: a ∈ [-2, 2], c ∈ [-1, 1] typically
        
        Returns:
            dict with 't', 'a', 'b', 'c' arrays
        """
        if initial_state is None:
            initial_state = [0.0, 0.0, 0.0]  # Start at origin
        
        t = np.linspace(0, t_span, n_points)
        solution = odeint(self.derivatives, initial_state, t)
        
        # Wrap b to [0, 2π]
        b_wrapped = np.mod(solution[:, 1], 2 * np.pi)
        
        return {
            't': t,
            'a': solution[:, 0],
            'b': b_wrapped,
            'c': solution[:, 2],
            'b_unwrapped': solution[:, 1]
        }
    
    def find_limit_cycle(self, t_span=200, transient=100, n_points=20000):
        """
        Find limit cycle by integrating past transients.
        Uses FFT-based period detection instead of unreliable peak finding.
        
        Returns:
            dict with limit cycle data or None if no stable cycle found
        """
        traj = self.integrate(t_span, n_points=n_points)
        
        # Skip transient
        transient_idx = int(len(traj['t']) * transient / t_span)
        
        a_steady = traj['a'][transient_idx:]
        c_steady = traj['c'][transient_idx:]
        t_steady = traj['t'][transient_idx:]
        b_steady = traj['b'][transient_idx:]
        
        # Check for oscillation via variance
        a_var = np.var(a_steady)
        c_var = np.var(c_steady)
        
        if a_var < 0.01 or c_var < 0.001:
            return None  # Too little variation = no oscillation
        
        # Find period via FFT
        dt = traj['t'][1] - traj['t'][0]
        n = len(a_steady)
        
        # Remove mean and compute FFT
        a_centered = a_steady - np.mean(a_steady)
        yf = np.abs(np.fft.fft(a_centered))[:n//2]
        xf = np.fft.fftfreq(n, dt)[:n//2]
        
        # Find dominant frequency (skip DC)
        peak_idx = np.argmax(yf[1:]) + 1
        peak_freq = xf[peak_idx]
        
        if peak_freq < 1e-6:
            return None
        
        period = 1.0 / peak_freq
        period_samples = int(period / dt)
        
        # Extract approximately 2 cycles for smooth looping
        n_cycles = 2
        cycle_samples = min(period_samples * n_cycles, len(a_steady) - 1)
        
        return {
            'period': period,
            'a': a_steady[:cycle_samples],
            'b': b_steady[:cycle_samples],
            'c': c_steady[:cycle_samples],
            't': t_steady[:cycle_samples] - t_steady[0],
            'amplitude_a': np.max(a_steady) - np.min(a_steady),
            'amplitude_c': np.max(c_steady) - np.min(c_steady),
            'mean_a': np.mean(a_steady),
            'mean_c': np.mean(c_steady),
            'frequency': peak_freq
        }


def parameter_sweep_for_oscillations():
    """
    Sweep parameter space to find regimes with stable limit cycles.
    
    Returns:
        List of (params, cycle_info) tuples for oscillatory regimes
    """
    results = []
    
    # Parameter ranges to explore
    w_EI_range = np.linspace(0.2, 0.8, 5)
    w_IE_range = np.linspace(-0.6, -0.1, 5)
    omega_range = np.linspace(0.2, 1.0, 5)
    
    print("Sweeping parameter space for limit cycles...")
    
    for w_EI in w_EI_range:
        for w_IE in w_IE_range:
            for omega in omega_range:
                osc = PNOscillator(w_EI=w_EI, w_IE=w_IE, omega=omega)
                cycle = osc.find_limit_cycle()
                
                if cycle is not None:
                    # Quality metric: amplitude and stability
                    quality = cycle['amplitude_a'] * cycle['amplitude_c']
                    
                    results.append({
                        'w_EI': w_EI,
                        'w_IE': w_IE,
                        'omega': omega,
                        'period': cycle['period'],
                        'quality': quality,
                        'cycle': cycle
                    })
    
    # Sort by quality
    results.sort(key=lambda x: x['quality'], reverse=True)
    
    print(f"Found {len(results)} oscillatory regimes")
    return results


# =============================================================================
# PART 2: JULIA SET IN 3D (Quaternion Julia Set)
# =============================================================================

def quaternion_julia_3d(c_real, c_i, c_j, c_k, resolution=64, max_iter=20, 
                        bound=1.5, slice_w=0.0):
    """
    Generate 3D Julia set using quaternions.
    
    The quaternion Julia set is defined by:
    q_{n+1} = q_n^2 + c
    
    where q and c are quaternions (4D), and we take a 3D slice.
    
    Args:
        c_real, c_i, c_j, c_k: Quaternion parameter c
        resolution: Grid resolution per axis
        max_iter: Maximum iterations
        bound: Space bounds [-bound, bound]^3
        slice_w: W-coordinate for 3D slice of 4D set
    
    Returns:
        3D array of iteration counts (for isosurface extraction)
    """
    # Create 3D grid (x, y, z) with fixed w = slice_w
    x = np.linspace(-bound, bound, resolution)
    y = np.linspace(-bound, bound, resolution)
    z = np.linspace(-bound, bound, resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    W = np.full_like(X, slice_w)
    
    # Initialize quaternion components
    qr = X.copy()
    qi = Y.copy()
    qj = Z.copy()
    qk = W.copy()
    
    # Iteration count array
    iterations = np.zeros_like(X)
    mask = np.ones_like(X, dtype=bool)
    
    for n in range(max_iter):
        if not np.any(mask):
            break
        
        # Quaternion multiplication: q^2
        # (a + bi + cj + dk)^2 = 
        # a^2 - b^2 - c^2 - d^2 + 2ab*i + 2ac*j + 2ad*k
        qr_new = qr**2 - qi**2 - qj**2 - qk**2 + c_real
        qi_new = 2*qr*qi + c_i
        qj_new = 2*qr*qj + c_j
        qk_new = 2*qr*qk + c_k
        
        qr, qi, qj, qk = qr_new, qi_new, qj_new, qk_new
        
        # Check escape
        magnitude = np.sqrt(qr**2 + qi**2 + qj**2 + qk**2)
        escaped = mask & (magnitude > 2)
        iterations[escaped] = n
        mask = mask & ~escaped
    
    # Points that never escaped get max_iter
    iterations[mask] = max_iter
    
    return iterations


def map_pn_to_quaternion(a, b, c_pn):
    """
    Map PN/FHN parameters (a, b, c) to quaternion Julia parameter.
    
    FHN ranges: a ∈ [-2, 2], c ∈ [-1, 1], b ∈ [0, 2π]
    
    Mapping strategy:
    - c_real: Scaled excitatory (a)
    - c_i: E-I interaction term
    - c_j: Phase-modulated inhibitory
    - c_k: Cross term
    
    Returns:
        (c_real, c_i, c_j, c_k) quaternion components
    """
    # Scale to interesting Julia parameter region [-0.8, 0.8]
    # FHN a is typically in [-2, 2], so scale by 0.2
    # FHN c is typically in [-0.5, 1.5], so scale appropriately
    
    c_real = 0.2 * a                      # Direct excitatory mapping
    c_i = 0.15 * (a - c_pn) * np.cos(b)   # E-I difference with phase
    c_j = 0.3 * c_pn * np.sin(b)          # Inhibitory with phase  
    c_k = 0.1 * a * c_pn                  # Interaction term
    
    return c_real, c_i, c_j, c_k


# =============================================================================
# PART 3: ANIMATION FRAME GENERATION
# =============================================================================

def generate_julia_animation_frames(cycle, n_frames=60, resolution=48):
    """
    Generate frames for 4D Julia animation from limit cycle.
    
    Each frame is a 3D Julia set with parameters derived from 
    the PN limit cycle at that time point.
    
    Args:
        cycle: Limit cycle dict from PNOscillator.find_limit_cycle()
        n_frames: Number of animation frames
        resolution: 3D grid resolution
    
    Returns:
        List of (iteration_volume, params) tuples
    """
    frames = []
    
    # Sample limit cycle at n_frames points
    cycle_indices = np.linspace(0, len(cycle['a']) - 1, n_frames, dtype=int)
    
    print(f"Generating {n_frames} Julia set frames at resolution {resolution}...")
    
    for i, idx in enumerate(cycle_indices):
        a = cycle['a'][idx]
        b = cycle['b'][idx]
        c_pn = cycle['c'][idx]
        
        # Map to quaternion
        c_real, c_i, c_j, c_k = map_pn_to_quaternion(a, b, c_pn)
        
        # Generate 3D Julia
        julia_vol = quaternion_julia_3d(
            c_real, c_i, c_j, c_k,
            resolution=resolution,
            max_iter=15
        )
        
        frames.append({
            'volume': julia_vol,
            'a': a, 'b': b, 'c': c_pn,
            'c_quat': (c_real, c_i, c_j, c_k),
            'frame': i
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{n_frames}")
    
    return frames


# =============================================================================
# PART 4: VISUALIZATION
# =============================================================================

def plot_limit_cycle_3d(cycle, title="PN Limit Cycle"):
    """Plot limit cycle trajectory in (a, b, c) space."""
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(cycle['a'], cycle['b'], cycle['c'], 'b-', linewidth=1)
    ax1.scatter(cycle['a'][0], cycle['b'][0], cycle['c'][0], 
                c='g', s=100, marker='o', label='Start')
    ax1.set_xlabel('a (Excitatory)')
    ax1.set_ylabel('b (Phase)')
    ax1.set_zlabel('c (Inhibitory)')
    ax1.set_title(f'{title}\nPeriod: {cycle["period"]:.2f}')
    ax1.legend()
    
    # Time series
    ax2 = fig.add_subplot(122)
    ax2.plot(cycle['t'], cycle['a'], 'r-', label='a (E)', linewidth=2)
    ax2.plot(cycle['t'], cycle['c'], 'b-', label='c (I)', linewidth=2)
    ax2.plot(cycle['t'], cycle['b'] / (2*np.pi), 'g--', label='b/2π', linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_julia_slice(julia_vol, slice_idx=None, title="Julia Set Slice"):
    """Plot 2D slice through 3D Julia volume."""
    if slice_idx is None:
        slice_idx = julia_vol.shape[2] // 2
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(julia_vol[:, :, slice_idx], cmap='hot', origin='lower')
    ax.set_title(f'{title} (z-slice {slice_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Iterations')
    return fig


def plot_julia_isosurface(julia_vol, threshold=None, title="3D Julia Set"):
    """
    Plot isosurface of Julia set using marching cubes approximation.
    Uses matplotlib's plot_surface with sampled points.
    """
    if threshold is None:
        threshold = np.max(julia_vol) * 0.5
    
    # Find surface points (where iterations ~ threshold)
    surface_mask = np.abs(julia_vol - threshold) < 1.5
    
    # Get coordinates
    x, y, z = np.where(surface_mask)
    
    if len(x) == 0:
        print("No surface points found at this threshold")
        return None
    
    # Subsample for performance
    max_points = 5000
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by iteration value
    colors = julia_vol[surface_mask]
    if len(colors) > max_points:
        colors = colors[idx]
    
    scatter = ax.scatter(x, y, z, c=colors, cmap='plasma', alpha=0.6, s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig


def create_animation_summary(frames, output_path='julia_animation_summary.png'):
    """Create summary grid of animation frames."""
    n_show = min(16, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        frame = frames[idx]
        slice_z = frame['volume'].shape[2] // 2
        
        axes[i].imshow(frame['volume'][:, :, slice_z], cmap='inferno', origin='lower')
        axes[i].set_title(f"t={idx}: a={frame['a']:.2f}, c={frame['c']:.2f}")
        axes[i].axis('off')
    
    plt.suptitle('Julia Set Animation Frames (z-slice)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved animation summary to {output_path}")
    return fig


# =============================================================================
# PART 5: STABLE OSCILLATOR CONFIGURATIONS
# =============================================================================

# Pre-tuned configurations that produce good limit cycles
# Based on FitzHugh-Nagumo model
STABLE_OSCILLATOR_CONFIGS = {
    'default': {
        'epsilon': 0.08,
        'alpha': 0.7,
        'beta': 0.8,
        'I_ext': 0.5,
        'omega': 0.3,
        'kappa': 0.15,
        'description': 'Standard FHN oscillator'
    },
    'fast': {
        'epsilon': 0.15,
        'alpha': 0.7,
        'beta': 0.8,
        'I_ext': 0.6,
        'omega': 0.5,
        'kappa': 0.2,
        'description': 'Fast oscillation for quick animation'
    },
    'slow': {
        'epsilon': 0.04,
        'alpha': 0.7,
        'beta': 0.8,
        'I_ext': 0.4,
        'omega': 0.2,
        'kappa': 0.1,
        'description': 'Slow oscillation for smooth morphing'
    },
    'large_amplitude': {
        'epsilon': 0.06,
        'alpha': 0.8,
        'beta': 0.7,
        'I_ext': 0.7,
        'omega': 0.35,
        'kappa': 0.25,
        'description': 'Large amplitude excursions'
    },
    'spiking': {
        'epsilon': 0.02,
        'alpha': 0.7,
        'beta': 0.8,
        'I_ext': 0.35,
        'omega': 0.4,
        'kappa': 0.3,
        'description': 'Sharp spikes (relaxation oscillator)'
    }
}


def get_stable_oscillator(config_name='default'):
    """Get a pre-tuned oscillator configuration."""
    if config_name not in STABLE_OSCILLATOR_CONFIGS:
        print(f"Available configs: {list(STABLE_OSCILLATOR_CONFIGS.keys())}")
        raise ValueError(f"Unknown config: {config_name}")
    
    config = STABLE_OSCILLATOR_CONFIGS[config_name]
    print(f"Using config '{config_name}': {config['description']}")
    
    return PNOscillator(
        epsilon=config['epsilon'],
        alpha=config['alpha'],
        beta=config['beta'],
        I_ext=config['I_ext'],
        omega=config['omega'],
        kappa=config['kappa']
    )


# =============================================================================
# PART 6: BLENDER EXPORT
# =============================================================================

def export_for_blender(frames, output_path='julia_frames.npz'):
    """
    Export animation frames in format suitable for Blender import.
    
    Saves:
    - Voxel data for each frame
    - PN parameters for each frame
    - Quaternion c values for each frame
    """
    volumes = np.array([f['volume'] for f in frames])
    params = np.array([[f['a'], f['b'], f['c']] for f in frames])
    quats = np.array([f['c_quat'] for f in frames])
    
    np.savez_compressed(
        output_path,
        volumes=volumes,
        params=params,
        quaternions=quats,
        n_frames=len(frames)
    )
    
    print(f"Exported {len(frames)} frames to {output_path}")
    print(f"  Volume shape: {volumes.shape}")
    print(f"  Load with: data = np.load('{output_path}')")
    
    return output_path


def export_cycle_for_blender(cycle, output_path='limit_cycle.npz'):
    """Export limit cycle trajectory for Blender path animation."""
    np.savez_compressed(
        output_path,
        a=cycle['a'],
        b=cycle['b'],
        c=cycle['c'],
        t=cycle['t'],
        period=cycle['period']
    )
    
    print(f"Exported limit cycle to {output_path}")
    print(f"  {len(cycle['a'])} points, period={cycle['period']:.3f}")


# =============================================================================
# MAIN EXPLORATION
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LIMIT CYCLE + 3D JULIA SET EXPLORER")
    print("=" * 60)
    
    # 1. Test stable oscillator configurations
    print("\n--- Testing Stable Oscillator Configs ---")
    
    best_config = None
    best_quality = 0
    
    for name, config in STABLE_OSCILLATOR_CONFIGS.items():
        osc = get_stable_oscillator(name)
        cycle = osc.find_limit_cycle(t_span=150, transient=100)
        
        if cycle:
            quality = cycle['amplitude_a'] * cycle['amplitude_c']
            print(f"  {name}: period={cycle['period']:.2f}, "
                  f"amp_a={cycle['amplitude_a']:.3f}, "
                  f"amp_c={cycle['amplitude_c']:.3f}, "
                  f"quality={quality:.4f}")
            
            if quality > best_quality:
                best_quality = quality
                best_config = name
        else:
            print(f"  {name}: No stable limit cycle found")
    
    print(f"\nBest config: {best_config} (quality={best_quality:.4f})")
    
    # 2. Generate visualization for best config
    print("\n--- Generating Visualizations ---")
    
    # Use default if no oscillations found (FHN should always work)
    if best_config is None:
        best_config = 'default'
        print(f"No oscillations detected, using default config")
    
    osc = get_stable_oscillator(best_config)
    cycle = osc.find_limit_cycle(t_span=200, transient=150, n_points=30000)
    
    if cycle:
        # Plot limit cycle
        fig_cycle = plot_limit_cycle_3d(cycle, f"PN Limit Cycle ({best_config})")
        fig_cycle.savefig('limit_cycle_trajectory.png', dpi=150, bbox_inches='tight')
        print("Saved: limit_cycle_trajectory.png")
        
        # Generate Julia animation frames (low res for testing)
        frames = generate_julia_animation_frames(cycle, n_frames=24, resolution=32)
        
        # Create animation summary
        fig_anim = create_animation_summary(frames, 'julia_animation_summary.png')
        
        # Plot one 3D Julia
        fig_julia = plot_julia_isosurface(frames[0]['volume'], 
                                          title=f"3D Julia at t=0")
        if fig_julia:
            fig_julia.savefig('julia_3d_sample.png', dpi=150, bbox_inches='tight')
            print("Saved: julia_3d_sample.png")
        
        # Export for Blender
        export_for_blender(frames, 'julia_frames.npz')
        export_cycle_for_blender(cycle, 'limit_cycle.npz')
        
        plt.show()
    else:
        print("ERROR: Could not find stable limit cycle!")
    
    print("\n--- Done ---")
