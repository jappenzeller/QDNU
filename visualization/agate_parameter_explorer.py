"""
================================================================================
A-GATE PARAMETER SPACE EXPLORER
================================================================================

Visualization tied to practical quantum computing questions:

1. EEG → (a, b, c) → Where do ictal vs interictal states land?
2. Circuit entanglement → Where is concurrence maximized?
3. Fidelity landscape → Where do we get best discrimination?
4. Julia structure → Does fractal boundary correlate with quantum sensitivity?

The sphere visualizations now represent:
- Height = quantum metric (concurrence, fidelity, sensitivity)
- Color = EEG state classification probability
- Trajectory = how EEG evolves through parameter space

================================================================================
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import os

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# A-GATE QUANTUM METRICS
# =============================================================================

@dataclass
class AGateState:
    """
    A-Gate quantum state from PN parameters.
    
    The A-Gate circuit:
    - 2 qubits (E = excitatory, I = inhibitory)
    - Parameters (a, b, c) control rotation angles
    - Entanglement via controlled operations
    """
    a: float  # Excitatory amplitude [0, 1]
    b: float  # Phase [0, 2π]
    c: float  # Inhibitory amplitude [0, 1]
    
    def theta_E(self) -> float:
        """Rotation angle for excitatory qubit."""
        return np.pi * (1 - self.a)
    
    def theta_I(self) -> float:
        """Rotation angle for inhibitory qubit."""
        return np.pi * (1 - self.c)
    
    def phi_E(self) -> float:
        """Phase for excitatory qubit."""
        return self.b
    
    def phi_I(self) -> float:
        """Phase for inhibitory qubit (offset by π/4 for asymmetry)."""
        return self.b + np.pi / 4
    
    def bloch_E(self) -> Tuple[float, float, float]:
        """Bloch vector for E qubit."""
        theta, phi = self.theta_E(), self.phi_E()
        return (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )
    
    def bloch_I(self) -> Tuple[float, float, float]:
        """Bloch vector for I qubit."""
        theta, phi = self.theta_I(), self.phi_I()
        return (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )
    
    def concurrence(self) -> float:
        """
        Entanglement measure for the 2-qubit state.
        
        Simplified model: concurrence depends on:
        - Phase b (coupling strength)
        - Balance of a and c (E-I balance)
        
        Max entanglement when b = π/2 or 3π/2 and a ≈ c
        """
        phase_factor = np.abs(np.sin(self.b))
        balance_factor = 1 - np.abs(self.a - self.c)
        return 0.5 * phase_factor * balance_factor
    
    def purity(self) -> float:
        """
        State purity (1 = pure, 0.5 = maximally mixed for 2 qubits).
        
        In our model, purity relates to how "clean" the EEG signal is.
        """
        # Simplified: purity decreases when parameters are extreme
        a_factor = 1 - 0.3 * (2 * self.a - 1) ** 2
        c_factor = 1 - 0.3 * (2 * self.c - 1) ** 2
        return 0.5 + 0.5 * a_factor * c_factor
    
    def sensitivity(self) -> float:
        """
        Parameter sensitivity - how much small changes affect the state.
        
        High sensitivity = small EEG changes → large state changes
        This is where discrimination power lives.
        """
        # Sensitivity peaks at intermediate values, zero at extremes
        a_sens = 4 * self.a * (1 - self.a)  # Peaks at 0.5
        c_sens = 4 * self.c * (1 - self.c)
        b_sens = np.abs(np.sin(2 * self.b))  # Peaks at π/4, 3π/4, etc.
        
        return (a_sens + c_sens + b_sens) / 3
    
    def to_julia_c(self) -> complex:
        """Map to Julia parameter for visualization."""
        real = -0.4 + 0.3 * np.cos(self.b)
        imag = 0.3 * np.sin(self.b) + 0.1 * (self.a - self.c)
        return complex(real, imag)


def compute_fidelity_to_target(state: AGateState, target: str = 'bell') -> float:
    """
    Compute fidelity to a target state.
    
    For seizure detection, we might want:
    - 'bell': Maximally entangled (high concurrence)
    - 'separable': Product state (low concurrence)
    - 'ictal_template': Learned ictal signature
    """
    if target == 'bell':
        # Fidelity to Bell state ≈ concurrence for our model
        return state.concurrence()
    elif target == 'separable':
        # Fidelity to |00⟩
        return (1 + state.bloch_E()[2]) / 2 * (1 + state.bloch_I()[2]) / 2
    else:
        return 0.5


# =============================================================================
# EEG → PARAMETER MAPPING
# =============================================================================

def eeg_to_parameters(eeg_features: Dict[str, float]) -> AGateState:
    """
    Map EEG features to A-Gate parameters.
    
    Example features (from preprocessing):
    - 'power_alpha': Alpha band power
    - 'power_beta': Beta band power
    - 'phase_sync': Phase synchronization
    - 'entropy': Signal entropy
    
    This is where the Kaggle preprocessing connects to the circuit.
    """
    # Default mapping (to be tuned based on EEG data)
    a = np.clip(eeg_features.get('excitatory_index', 0.5), 0, 1)
    c = np.clip(eeg_features.get('inhibitory_index', 0.5), 0, 1)
    
    # Phase from synchronization or dominant frequency phase
    phase_sync = eeg_features.get('phase_sync', 0.5)
    b = phase_sync * 2 * np.pi
    
    return AGateState(a=a, b=b, c=c)


def simulate_ictal_interictal_distribution(n_samples: int = 100) -> Tuple[List[AGateState], List[AGateState]]:
    """
    Simulate typical ictal vs interictal parameter distributions.
    
    Based on EEG characteristics:
    - Ictal: High synchronization, altered E-I balance, specific phase patterns
    - Interictal: More variable, closer to "normal" baseline
    """
    np.random.seed(42)
    
    # Interictal: centered, moderate variability
    interictal = []
    for _ in range(n_samples):
        a = np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9)
        b = np.random.uniform(0, 2 * np.pi)
        c = np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9)
        interictal.append(AGateState(a=a, b=b, c=c))
    
    # Ictal: shifted distribution, tighter phase clustering
    ictal = []
    for _ in range(n_samples):
        # Ictal often shows increased excitation
        a = np.clip(np.random.normal(0.7, 0.1), 0.3, 0.95)
        # Phase clustering around specific values
        b = np.random.vonmises(np.pi/2, 2.0) % (2 * np.pi)  # Clustered around π/2
        # Altered inhibition
        c = np.clip(np.random.normal(0.4, 0.12), 0.1, 0.8)
        ictal.append(AGateState(a=a, b=b, c=c))
    
    return ictal, interictal


# =============================================================================
# PARAMETER SPACE METRICS
# =============================================================================

def compute_metric_field(metric: str, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a quantum metric across the (a, b, c) parameter space.
    
    For visualization, we slice at c = 0.5 and show (a, b) plane.
    
    Returns:
        A, B grids and metric values for both c=0.3 and c=0.7
    """
    a_range = np.linspace(0.05, 0.95, resolution)
    b_range = np.linspace(0, 2 * np.pi, resolution)
    A, B = np.meshgrid(a_range, b_range, indexing='ij')
    
    metric_low_c = np.zeros_like(A)
    metric_high_c = np.zeros_like(A)
    
    for i in range(resolution):
        for j in range(resolution):
            state_low = AGateState(a=A[i, j], b=B[i, j], c=0.3)
            state_high = AGateState(a=A[i, j], b=B[i, j], c=0.7)
            
            if metric == 'concurrence':
                metric_low_c[i, j] = state_low.concurrence()
                metric_high_c[i, j] = state_high.concurrence()
            elif metric == 'sensitivity':
                metric_low_c[i, j] = state_low.sensitivity()
                metric_high_c[i, j] = state_high.sensitivity()
            elif metric == 'purity':
                metric_low_c[i, j] = state_low.purity()
                metric_high_c[i, j] = state_high.purity()
            elif metric == 'fidelity_bell':
                metric_low_c[i, j] = compute_fidelity_to_target(state_low, 'bell')
                metric_high_c[i, j] = compute_fidelity_to_target(state_high, 'bell')
    
    return A, B, metric_low_c, metric_high_c


def compute_discrimination_field(ictal_states: List[AGateState], 
                                  interictal_states: List[AGateState],
                                  resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute discrimination potential across parameter space.
    
    At each point, compute how well that region separates ictal from interictal
    based on distance to the nearest samples of each class.
    """
    a_range = np.linspace(0.05, 0.95, resolution)
    b_range = np.linspace(0, 2 * np.pi, resolution)
    A, B = np.meshgrid(a_range, b_range, indexing='ij')
    
    # Compute density of each class
    ictal_density = np.zeros_like(A)
    interictal_density = np.zeros_like(A)
    
    sigma = 0.1  # Kernel width
    
    for state in ictal_states:
        dist_a = (A - state.a) ** 2
        dist_b_raw = np.abs(B - state.b)
        dist_b = np.minimum(dist_b_raw, 2 * np.pi - dist_b_raw) ** 2  # Circular
        ictal_density += np.exp(-(dist_a + dist_b) / (2 * sigma ** 2))
    
    for state in interictal_states:
        dist_a = (A - state.a) ** 2
        dist_b_raw = np.abs(B - state.b)
        dist_b = np.minimum(dist_b_raw, 2 * np.pi - dist_b_raw) ** 2
        interictal_density += np.exp(-(dist_a + dist_b) / (2 * sigma ** 2))
    
    # Normalize
    ictal_density /= len(ictal_states)
    interictal_density /= len(interictal_states)
    
    # Discrimination = difference in densities
    # Positive = more ictal, Negative = more interictal
    discrimination = ictal_density - interictal_density
    
    return A, B, discrimination


# =============================================================================
# SPHERE VISUALIZATION WITH QUANTUM METRICS
# =============================================================================

def create_quantum_metric_sphere(metric: str = 'concurrence',
                                  resolution: int = 100,
                                  smoothing: float = 2.0) -> dict:
    """
    Create sphere where height/color represents a quantum metric.
    
    The sphere maps (θ, φ) to (a, b) with c interpolated or fixed.
    
    This answers: "Where in parameter space is entanglement/sensitivity highest?"
    """
    theta = np.linspace(0.02, np.pi - 0.02, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Map sphere coords to A-Gate parameters
    # θ → a (pole to pole = 0 to 1)
    # φ → b (around equator = 0 to 2π)
    # c = function of θ (vary with latitude)
    
    A_param = THETA / np.pi  # [0, 1]
    B_param = PHI  # [0, 2π]
    C_param = 0.3 + 0.4 * np.sin(THETA)  # Varies with latitude
    
    # Compute metric at each point
    metric_values = np.zeros_like(THETA)
    
    for i in range(resolution):
        for j in range(resolution * 2):
            state = AGateState(a=A_param[i, j], b=B_param[i, j], c=C_param[i, j])
            
            if metric == 'concurrence':
                metric_values[i, j] = state.concurrence()
            elif metric == 'sensitivity':
                metric_values[i, j] = state.sensitivity()
            elif metric == 'purity':
                metric_values[i, j] = state.purity()
    
    # Smooth
    if smoothing > 0:
        metric_values = ndimage.gaussian_filter(metric_values, sigma=smoothing)
    
    # Normalize
    metric_norm = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-10)
    
    # Sphere with metric as height
    canyon_depth = 0.3
    R = 1.0 - canyon_depth * (1 - metric_norm)
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors = colormap_quantum(metric_norm.flatten())
    
    # Faces
    ny, nx = THETA.shape
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    return {
        'vertices': vertices.astype(np.float32),
        'faces': np.array(faces, dtype=np.int32),
        'colors': colors,
        'metric_values': metric_values,
        'metric_norm': metric_norm,
        'theta': THETA,
        'phi': PHI,
        'A_param': A_param,
        'B_param': B_param,
        'C_param': C_param,
        'metric': metric
    }


def colormap_quantum(t):
    """Colormap for quantum metrics - blue (low) to red (high)."""
    t = np.clip(t, 0, 1)
    
    # Blue → purple → red → orange
    r = np.clip(0.2 + 0.8 * t, 0, 1)
    g = np.clip(0.1 + 0.3 * np.sin(t * np.pi), 0, 1)
    b = np.clip(0.8 - 0.7 * t, 0, 1)
    
    return np.stack([r, g, b], axis=1).astype(np.float32)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_parameter_space(save_path: str = None):
    """
    Visualize A-Gate parameter space with quantum metrics and EEG distributions.
    """
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Simulate EEG distributions
    ictal, interictal = simulate_ictal_interictal_distribution(200)
    
    # 1. Concurrence field with EEG samples
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#16213e')
    
    A, B, conc_low, conc_high = compute_metric_field('concurrence', 60)
    im1 = ax1.contourf(B, A, conc_low, levels=20, cmap='plasma')
    
    # Plot EEG samples
    ictal_a = [s.a for s in ictal]
    ictal_b = [s.b for s in ictal]
    inter_a = [s.a for s in interictal]
    inter_b = [s.b for s in interictal]
    
    ax1.scatter(ictal_b, ictal_a, c='red', s=10, alpha=0.5, label='Ictal')
    ax1.scatter(inter_b, inter_a, c='cyan', s=10, alpha=0.5, label='Interictal')
    
    ax1.set_xlabel('b (phase)', color='gray')
    ax1.set_ylabel('a (excitatory)', color='gray')
    ax1.set_title('Concurrence (c=0.3)\n+ EEG Distributions', color='white')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.tick_params(colors='gray')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Sensitivity field
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#16213e')
    
    A, B, sens_low, sens_high = compute_metric_field('sensitivity', 60)
    im2 = ax2.contourf(B, A, sens_low, levels=20, cmap='viridis')
    
    ax2.scatter(ictal_b, ictal_a, c='red', s=10, alpha=0.5)
    ax2.scatter(inter_b, inter_a, c='cyan', s=10, alpha=0.5)
    
    ax2.set_xlabel('b (phase)', color='gray')
    ax2.set_ylabel('a (excitatory)', color='gray')
    ax2.set_title('Sensitivity\n(discrimination potential)', color='white')
    ax2.tick_params(colors='gray')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Discrimination field
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#16213e')
    
    A, B, discrim = compute_discrimination_field(ictal, interictal, 60)
    im3 = ax3.contourf(B, A, discrim, levels=20, cmap='RdBu_r')
    
    ax3.set_xlabel('b (phase)', color='gray')
    ax3.set_ylabel('a (excitatory)', color='gray')
    ax3.set_title('Discrimination\n(red=ictal, blue=interictal)', color='white')
    ax3.tick_params(colors='gray')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Concurrence sphere
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_facecolor('#1a1a2e')
    
    sphere_data = create_quantum_metric_sphere('concurrence', resolution=60, smoothing=2.0)
    verts = sphere_data['vertices']
    shape = sphere_data['theta'].shape
    X = verts[:, 0].reshape(shape)
    Y = verts[:, 1].reshape(shape)
    Z = verts[:, 2].reshape(shape)
    colors = sphere_data['colors'].reshape(shape[0], shape[1], 3)
    
    ax4.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95, shade=True)
    ax4.set_box_aspect([1, 1, 1])
    ax4.set_title('Concurrence Sphere\n(height = entanglement)', color='white')
    ax4.axis('off')
    
    # 5. Sensitivity sphere
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.set_facecolor('#1a1a2e')
    
    sphere_data = create_quantum_metric_sphere('sensitivity', resolution=60, smoothing=2.0)
    verts = sphere_data['vertices']
    X = verts[:, 0].reshape(shape)
    Y = verts[:, 1].reshape(shape)
    Z = verts[:, 2].reshape(shape)
    colors = sphere_data['colors'].reshape(shape[0], shape[1], 3)
    
    ax5.plot_surface(X, Y, Z, facecolors=colors, alpha=0.95, shade=True)
    ax5.set_box_aspect([1, 1, 1])
    ax5.set_title('Sensitivity Sphere\n(height = param sensitivity)', color='white')
    ax5.axis('off')
    
    # 6. Metric comparison
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#16213e')
    
    # Compare metrics for ictal vs interictal
    ictal_conc = [s.concurrence() for s in ictal]
    inter_conc = [s.concurrence() for s in interictal]
    ictal_sens = [s.sensitivity() for s in ictal]
    inter_sens = [s.sensitivity() for s in interictal]
    
    ax6.scatter(ictal_sens, ictal_conc, c='red', alpha=0.5, label='Ictal', s=20)
    ax6.scatter(inter_sens, inter_conc, c='cyan', alpha=0.5, label='Interictal', s=20)
    
    ax6.set_xlabel('Sensitivity', color='gray')
    ax6.set_ylabel('Concurrence', color='gray')
    ax6.set_title('Quantum Metrics\nIctal vs Interictal', color='white')
    ax6.legend()
    ax6.tick_params(colors='gray')
    
    plt.suptitle('A-Gate Parameter Space: Where Do Seizures Live?', 
                 color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_circuit_insight(save_path: str = None):
    """
    Focus on the quantum circuit behavior - what the A-Gate actually does.
    """
    if not HAS_MPL:
        return None
    
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Show how parameters affect the circuit
    b_values = np.linspace(0, 2 * np.pi, 100)
    
    # Fixed a, c - vary b
    conc_balanced = [AGateState(0.5, b, 0.5).concurrence() for b in b_values]
    conc_exc = [AGateState(0.7, b, 0.3).concurrence() for b in b_values]
    conc_inh = [AGateState(0.3, b, 0.7).concurrence() for b in b_values]
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#16213e')
    ax1.plot(np.degrees(b_values), conc_balanced, 'g-', label='Balanced (a=c=0.5)', linewidth=2)
    ax1.plot(np.degrees(b_values), conc_exc, 'r-', label='Excitatory (a=0.7, c=0.3)', linewidth=2)
    ax1.plot(np.degrees(b_values), conc_inh, 'b-', label='Inhibitory (a=0.3, c=0.7)', linewidth=2)
    ax1.set_xlabel('Phase b (degrees)', color='gray')
    ax1.set_ylabel('Concurrence', color='gray')
    ax1.set_title('Entanglement vs Phase\n(E-I Balance Effect)', color='white')
    ax1.legend()
    ax1.tick_params(colors='gray')
    ax1.axhline(y=0.15, color='yellow', linestyle='--', alpha=0.5, label='Target threshold')
    
    # Bloch sphere trajectories
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('#1a1a2e')
    
    # Trace as b varies
    for a, c, color, label in [(0.5, 0.5, 'green', 'Balanced'), 
                                (0.7, 0.3, 'red', 'Excitatory'),
                                (0.3, 0.7, 'blue', 'Inhibitory')]:
        bloch_E = [AGateState(a, b, c).bloch_E() for b in b_values]
        x = [b[0] for b in bloch_E]
        y = [b[1] for b in bloch_E]
        z = [b[2] for b in bloch_E]
        ax2.plot(x, y, z, color=color, linewidth=2, label=label)
    
    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_wireframe(xs, ys, zs, alpha=0.1, color='white')
    
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_title('Bloch Trajectories\nas Phase Cycles', color='white')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.axis('off')
    
    # Sensitivity landscape
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#16213e')
    
    a_vals = np.linspace(0.1, 0.9, 50)
    c_vals = np.linspace(0.1, 0.9, 50)
    A, C = np.meshgrid(a_vals, c_vals)
    
    # Compute max concurrence over all b for each (a, c)
    max_conc = np.zeros_like(A)
    for i in range(len(a_vals)):
        for j in range(len(c_vals)):
            concs = [AGateState(A[i,j], b, C[i,j]).concurrence() for b in np.linspace(0, 2*np.pi, 20)]
            max_conc[i, j] = max(concs)
    
    im3 = ax3.contourf(A, C, max_conc, levels=20, cmap='plasma')
    ax3.plot([0, 1], [0, 1], 'w--', alpha=0.5, label='a = c (balanced)')
    ax3.set_xlabel('a (excitatory)', color='gray')
    ax3.set_ylabel('c (inhibitory)', color='gray')
    ax3.set_title('Max Concurrence\n(over all phases)', color='white')
    ax3.tick_params(colors='gray')
    ax3.legend()
    plt.colorbar(im3, ax=ax3)
    
    # Key insight panel
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#16213e')
    ax4.axis('off')
    
    insight_text = """
    A-GATE QUANTUM INSIGHTS
    ─────────────────────────────────────
    
    1. ENTANGLEMENT PEAKS
       • Maximum at b = π/2, 3π/2 (90°, 270°)
       • Requires balanced E-I (a ≈ c)
       • This is where seizure/non-seizure 
         discrimination should be strongest
    
    2. E-I BALANCE EFFECT
       • Imbalanced a ≠ c reduces max concurrence
       • Ictal: typically a > c (excitation dominant)
       • This shifts WHERE peak entanglement occurs
    
    3. PHASE SENSITIVITY
       • Near b = 0, π: low entanglement, low sensitivity
       • Near b = π/2: high entanglement, high sensitivity
       • Seizure onset may correlate with phase shifts
    
    4. DISCRIMINATION STRATEGY
       • Tune circuit to amplify natural differences
       • Ictal → different (a, b, c) region
       • Use concurrence as classification feature
    """
    
    ax4.text(0.05, 0.95, insight_text, transform=ax4.transAxes,
             fontsize=10, color='white', family='monospace',
             verticalalignment='top')
    
    plt.suptitle('A-Gate Circuit: Quantum Behavior Analysis', 
                 color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# EXPORT
# =============================================================================

def export_ply(data: dict, filepath: str):
    """Export sphere mesh."""
    verts = data['vertices']
    faces = data['faces']
    colors = data['colors']
    
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        
        for i, v in enumerate(verts):
            c = (np.clip(colors[i], 0, 1) * 255).astype(np.uint8)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("A-GATE PARAMETER SPACE EXPLORER")
    print("=" * 60)
    print("""
    Connecting visualization to quantum computing questions:
    
    1. Where do ictal/interictal EEG states land in parameter space?
    2. Where is entanglement (concurrence) maximized?
    3. Where is parameter sensitivity highest?
    4. How can we tune the circuit for better discrimination?
    """)
    
    output_dir = 'agate_parameter_space'
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter space visualization
    print("\nGenerating parameter space analysis...")
    fig1 = visualize_parameter_space(f'{output_dir}/parameter_space.png')
    plt.close(fig1)
    
    # Circuit insight
    print("\nGenerating circuit insight analysis...")
    fig2 = visualize_circuit_insight(f'{output_dir}/circuit_insight.png')
    plt.close(fig2)
    
    # Export spheres
    print("\nExporting quantum metric spheres...")
    
    for metric in ['concurrence', 'sensitivity']:
        data = create_quantum_metric_sphere(metric, resolution=100, smoothing=2.5)
        export_ply(data, f'{output_dir}/{metric}_sphere.ply')
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print(f"  {output_dir}/parameter_space.png - EEG distribution + metrics")
    print(f"  {output_dir}/circuit_insight.png - Circuit behavior analysis")
    print(f"  {output_dir}/concurrence_sphere.ply - Entanglement landscape")
    print(f"  {output_dir}/sensitivity_sphere.ply - Sensitivity landscape")


if __name__ == '__main__':
    main()
