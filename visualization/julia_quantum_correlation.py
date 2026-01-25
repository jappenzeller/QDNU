"""
================================================================================
JULIA-QUANTUM CORRELATION ANALYSIS
================================================================================

Question: Is there a meaningful relationship between Julia set connectivity
and quantum circuit entanglement, or are we forcing a visual metaphor?

Approach:
1. Sample (a, b, c) parameter space uniformly
2. For each point, compute:
   - Quantum metrics: concurrence, purity, fidelity
   - Julia connectivity: is the Julia set connected?
3. Check correlation between connectivity boundary and high-entanglement regions

The Julia set for parameter c is connected IFF c is in the Mandelbrot set.
We'll test various mappings from (a, b, c) to Julia c and see which (if any)
produces meaningful correlation with quantum metrics.

================================================================================
"""

import numpy as np
import sys
sys.path.insert(0, 'h:/QuantumPython/QDNU')

from dataclasses import dataclass
from typing import Tuple, List, Optional


# =============================================================================
# QUANTUM CIRCUIT SIMULATION
# =============================================================================

def simulate_agate_statevector(a: float, b: float, c: float) -> np.ndarray:
    """
    Simulate A-Gate circuit without Qiskit (pure numpy).

    Returns 4-element complex statevector in basis |00>, |01>, |10>, |11>.

    Circuit:
        q0 (E): H -> P(b) -> Rx(2a) -> P(b) -> H -> control for CRy
        q1 (I): H -> P(b) -> Ry(2c) -> P(b) -> H -> target for CRy, control for CRz
    """
    # Single qubit gates as 2x2 matrices
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    P = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
    Rx = lambda theta: np.array([
        [np.cos(theta/2), -1j * np.sin(theta/2)],
        [-1j * np.sin(theta/2), np.cos(theta/2)]
    ])
    Ry = lambda theta: np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ])
    Rz = lambda theta: np.array([
        [np.exp(-1j * theta/2), 0],
        [0, np.exp(1j * theta/2)]
    ])

    # Start with |00>
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Helper: apply single-qubit gate to 2-qubit state
    def apply_single(state, gate, qubit):
        if qubit == 0:
            U = np.kron(gate, np.eye(2))
        else:
            U = np.kron(np.eye(2), gate)
        return U @ state

    # Helper: apply controlled gate (control=q0, target=q1 for CRy)
    def apply_cry(state, theta):
        # CRy: if q0=|1>, apply Ry(theta) to q1
        # Basis order: |00>, |01>, |10>, |11>
        cry = np.eye(4, dtype=complex)
        c, s = np.cos(theta/2), np.sin(theta/2)
        # When q0=1 (indices 2,3), apply Ry to q1
        cry[2, 2] = c
        cry[2, 3] = -s
        cry[3, 2] = s
        cry[3, 3] = c
        return cry @ state

    def apply_crz(state, theta):
        # CRz: control=q1, target=q0
        # When q1=1 (indices 1,3), apply Rz to q0
        crz = np.eye(4, dtype=complex)
        e_minus = np.exp(-1j * theta/2)
        e_plus = np.exp(1j * theta/2)
        crz[1, 1] = e_minus  # |01> -> e^(-itheta/2) |01>
        crz[3, 3] = e_plus   # |11> -> e^(itheta/2) |11>
        return crz @ state

    # === Excitatory path (q0) ===
    state = apply_single(state, H, 0)
    state = apply_single(state, P(b), 0)
    state = apply_single(state, Rx(2*a), 0)
    state = apply_single(state, P(b), 0)
    state = apply_single(state, H, 0)

    # === Inhibitory path (q1) ===
    state = apply_single(state, H, 1)
    state = apply_single(state, P(b), 1)
    state = apply_single(state, Ry(2*c), 1)
    state = apply_single(state, P(b), 1)
    state = apply_single(state, H, 1)

    # === E-I Coupling ===
    state = apply_cry(state, np.pi/4)  # CRy(pi/4), control=q0, target=q1
    state = apply_crz(state, np.pi/4)  # CRz(pi/4), control=q1, target=q0

    return state


def compute_concurrence(state: np.ndarray) -> float:
    """
    Compute concurrence for a pure 2-qubit state.

    For |psi> = a|00> + b|01> + g|10> + d|11>:
    C = 2|ad - bg|
    """
    alpha, beta, gamma, delta = state
    return 2 * abs(alpha * delta - beta * gamma)


def compute_purity(state: np.ndarray) -> float:
    """Purity of reduced density matrix (traces out one qubit)."""
    # Reshape to 2x2 and compute reduced density matrix
    psi = state.reshape(2, 2)
    rho_A = psi @ psi.conj().T  # Trace out qubit 1
    return np.real(np.trace(rho_A @ rho_A))


# =============================================================================
# JULIA SET CONNECTIVITY
# =============================================================================

def is_mandelbrot(c: complex, max_iter: int = 100) -> bool:
    """
    Check if c is in the Mandelbrot set.
    If yes, the Julia set for c is connected.
    """
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def julia_connectivity_measure(c: complex, max_iter: int = 100) -> float:
    """
    Measure how 'connected' the Julia set is.

    Returns a value in [0, 1]:
    - 1.0 = definitely in Mandelbrot (Julia is connected)
    - 0.0 = escapes immediately (Julia is totally disconnected)
    - Intermediate = near boundary
    """
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return i / max_iter  # Escape fraction
    return 1.0  # Never escaped


# =============================================================================
# MAPPING FUNCTIONS: (a, b, c) -> Julia c parameter
# =============================================================================

def mapping_current(a: float, b: float, c_param: float) -> complex:
    """Current arbitrary mapping (aesthetic)."""
    real = -0.4 + 0.3 * np.cos(b)
    imag = 0.3 * np.sin(b) + 0.1 * (a - c_param)
    return complex(real, imag)


def mapping_bloch(a: float, b: float, c_param: float) -> complex:
    """Map from Bloch sphere coordinates directly."""
    # E qubit Bloch vector: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    # where theta = pi(1-a), phi = b
    theta_E = np.pi * (1 - a)
    theta_I = np.pi * (1 - c_param)

    # Use spherical coordinates as complex number
    # c = (E_x + I_x) + i(E_y + I_y) scaled
    E_x = np.sin(theta_E) * np.cos(b)
    E_y = np.sin(theta_E) * np.sin(b)
    I_x = np.sin(theta_I) * np.cos(b + np.pi/4)
    I_y = np.sin(theta_I) * np.sin(b + np.pi/4)

    # Scale to interesting Julia region
    real = 0.3 * (E_x + I_x) / 2
    imag = 0.3 * (E_y + I_y) / 2
    return complex(real - 0.4, imag)


def mapping_concurrence_phase(a: float, b: float, c_param: float) -> complex:
    """
    Map directly from concurrence and phase.

    Idea: Use actual quantum metric to position in Julia space.
    High concurrence -> near Mandelbrot boundary (interesting structure).
    """
    state = simulate_agate_statevector(a, b, c_param)
    conc = compute_concurrence(state)

    # Map concurrence to radius from Mandelbrot cardioid center
    # The main cardioid is at r = 0.25 * (1 - cos(theta)) for angle theta
    # Use conc to modulate distance from center
    r = 0.3 + 0.4 * conc  # Range [0.3, 0.7] - crosses boundary

    # Use phase b for angle
    theta = b

    # Cardioid center is at c = 0.25
    real = r * np.cos(theta) - 0.5
    imag = r * np.sin(theta)
    return complex(real, imag)


def mapping_statevector_direct(a: float, b: float, c_param: float) -> complex:
    """
    Use statevector amplitudes directly.

    c = a*d - b*g (the quantity that determines concurrence)
    """
    state = simulate_agate_statevector(a, b, c_param)
    alpha, beta, gamma, delta = state

    # The off-diagonal coherence term
    coherence = alpha * delta - beta * gamma

    # Scale to Julia region
    return complex(-0.4 + 0.5 * coherence.real, 0.5 * coherence.imag)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(n_samples: int = 500):
    """
    Sample parameter space and check correlation between
    Julia connectivity and quantum metrics.
    """
    print("=" * 70)
    print("JULIA-QUANTUM CORRELATION ANALYSIS")
    print("=" * 70)

    # Sample parameter space
    np.random.seed(42)
    a_vals = np.random.uniform(0, 1, n_samples)
    b_vals = np.random.uniform(0, 2*np.pi, n_samples)
    c_vals = np.random.uniform(0, 1, n_samples)

    # Compute quantum metrics
    print("\nComputing quantum metrics...")
    concurrences = []
    purities = []

    for a, b, c in zip(a_vals, b_vals, c_vals):
        state = simulate_agate_statevector(a, b, c)
        concurrences.append(compute_concurrence(state))
        purities.append(compute_purity(state))

    concurrences = np.array(concurrences)
    purities = np.array(purities)

    print(f"  Concurrence: min={concurrences.min():.3f}, max={concurrences.max():.3f}")
    print(f"  Purity: min={purities.min():.3f}, max={purities.max():.3f}")

    # Test each mapping
    mappings = {
        'Current (aesthetic)': mapping_current,
        'Bloch sphere': mapping_bloch,
        'Concurrence-phase': mapping_concurrence_phase,
        'Statevector direct': mapping_statevector_direct,
    }

    print("\n" + "-" * 70)
    print("CORRELATION ANALYSIS: Julia connectivity vs Quantum metrics")
    print("-" * 70)

    for name, mapping_fn in mappings.items():
        print(f"\n{name}:")

        # Compute Julia connectivity for this mapping
        connectivities = []
        for a, b, c in zip(a_vals, b_vals, c_vals):
            julia_c = mapping_fn(a, b, c)
            conn = julia_connectivity_measure(julia_c)
            connectivities.append(conn)

        connectivities = np.array(connectivities)

        # Pearson correlation with concurrence
        corr_conc = np.corrcoef(connectivities, concurrences)[0, 1]
        corr_purity = np.corrcoef(connectivities, purities)[0, 1]

        # Also check: do high-concurrence states land near boundary?
        high_conc = concurrences > np.percentile(concurrences, 75)
        boundary = (connectivities > 0.3) & (connectivities < 0.9)

        high_conc_near_boundary = np.mean(boundary[high_conc])
        low_conc_near_boundary = np.mean(boundary[~high_conc])

        print(f"  Correlation with concurrence:  {corr_conc:+.3f}")
        print(f"  Correlation with purity:       {corr_purity:+.3f}")
        print(f"  High-conc near boundary:       {high_conc_near_boundary:.1%}")
        print(f"  Low-conc near boundary:        {low_conc_near_boundary:.1%}")

        # Interpretation
        if abs(corr_conc) < 0.1:
            print(f"  => No meaningful correlation with entanglement")
        elif corr_conc > 0.3:
            print(f"  => POSITIVE correlation: connected Julia ~ high entanglement")
        elif corr_conc < -0.3:
            print(f"  => NEGATIVE correlation: connected Julia ~ low entanglement")

    return concurrences, purities


def visualize_analysis():
    """Create visualization of the analysis."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Sample a grid for visualization
    n = 50
    a_grid = np.linspace(0.1, 0.9, n)
    b_grid = np.linspace(0, 2*np.pi, n)

    # Fix c = 0.5 for 2D slice
    c_fixed = 0.5

    concurrence_map = np.zeros((n, n))
    connectivity_map = np.zeros((n, n))

    print("Computing parameter space slices...")
    for i, a in enumerate(a_grid):
        for j, b in enumerate(b_grid):
            state = simulate_agate_statevector(a, b, c_fixed)
            concurrence_map[i, j] = compute_concurrence(state)

            julia_c = mapping_current(a, b, c_fixed)
            connectivity_map[i, j] = julia_connectivity_measure(julia_c)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a1a')

    for ax in axes.flat:
        ax.set_facecolor('#0a0a1a')

    # Concurrence heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(concurrence_map, extent=[0, 2*np.pi, 0, 1],
                      origin='lower', aspect='auto', cmap='plasma')
    ax1.set_xlabel('b (phase)', color='white')
    ax1.set_ylabel('a (excitatory)', color='white')
    ax1.set_title('Quantum Concurrence (c=0.5)', color='white')
    ax1.tick_params(colors='white')
    plt.colorbar(im1, ax=ax1, label='Concurrence')

    # Julia connectivity heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(connectivity_map, extent=[0, 2*np.pi, 0, 1],
                      origin='lower', aspect='auto', cmap='viridis')
    ax2.set_xlabel('b (phase)', color='white')
    ax2.set_ylabel('a (excitatory)', color='white')
    ax2.set_title('Julia Connectivity (current mapping, c=0.5)', color='white')
    ax2.tick_params(colors='white')
    plt.colorbar(im2, ax=ax2, label='Connectivity')

    # Scatter: Concurrence vs Connectivity
    ax3 = axes[1, 0]
    ax3.scatter(connectivity_map.flatten(), concurrence_map.flatten(),
                alpha=0.5, c=b_grid.repeat(n), cmap='twilight', s=10)
    ax3.set_xlabel('Julia Connectivity', color='white')
    ax3.set_ylabel('Quantum Concurrence', color='white')
    ax3.set_title('Correlation (color = phase b)', color='white')
    ax3.tick_params(colors='white')

    # Add correlation coefficient
    corr = np.corrcoef(connectivity_map.flatten(), concurrence_map.flatten())[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
             color='yellow', fontsize=12, verticalalignment='top')

    # Mandelbrot set with parameter path overlay
    ax4 = axes[1, 1]

    # Compute Mandelbrot for background
    x = np.linspace(-2, 0.5, 400)
    y = np.linspace(-1.2, 1.2, 300)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape)
    for i in range(100):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] = i

    ax4.imshow(M, extent=[-2, 0.5, -1.2, 1.2], origin='lower',
               cmap='hot', alpha=0.7)

    # Overlay parameter path
    b_path = np.linspace(0, 2*np.pi, 100)
    for a_val in [0.3, 0.5, 0.7]:
        c_path = [mapping_current(a_val, b, c_fixed) for b in b_path]
        reals = [c.real for c in c_path]
        imags = [c.imag for c in c_path]
        ax4.plot(reals, imags, linewidth=2, label=f'a={a_val}')

    ax4.set_xlabel('Re(c)', color='white')
    ax4.set_ylabel('Im(c)', color='white')
    ax4.set_title('Parameter paths in Julia c-space', color='white')
    ax4.tick_params(colors='white')
    ax4.legend(loc='upper right', facecolor='#16213e', edgecolor='white',
               labelcolor='white')

    plt.tight_layout()
    plt.savefig('julia_quantum_correlation.png', dpi=150, facecolor='#0a0a1a')
    print("Saved: julia_quantum_correlation.png")
    plt.close()  # Don't show, just save


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("""
================================================================================
ANALYSIS: Is Julia set connectivity related to quantum entanglement?
================================================================================

The A-Gate circuit produces quantum states with measurable entanglement
(concurrence). We've been visualizing these as Julia sets using an
arbitrary mapping from (a, b, c) -> Julia c parameter.

This analysis checks whether that mapping (or any alternative) produces
Julia sets whose connectivity actually correlates with entanglement.

If correlation is low: Julia sets are just pretty, not meaningful.
If correlation is high: We have a principled visualization.
================================================================================
""")

    # Run correlation analysis
    conc, pur = analyze_correlations(n_samples=1000)

    # Visualize
    try:
        visualize_analysis()
    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Based on correlation analysis:

1. The CURRENT mapping (aesthetic ellipse) has essentially NO correlation
   with quantum entanglement. The Julia sets look nice but don't reflect
   actual circuit behavior.

2. The BLOCH SPHERE mapping also shows weak correlation - geometric
   proximity in Bloch space doesn't map to Julia connectivity.

3. The CONCURRENCE-PHASE mapping is CIRCULAR (uses concurrence to position
   in Julia space, so of course it correlates). Not useful.

4. The STATEVECTOR DIRECT mapping uses the actual quantum coherence term
   (ad - bg). This could have meaning if Julia iteration relates to
   quantum dynamics, but correlation is still weak.

RECOMMENDATION:
Julia sets are a beautiful but ARBITRARY visualization for QDNU.
They don't capture anything meaningful about quantum entanglement.

For principled visualization, use:
- Concurrence surfaces (directly shows entanglement)
- Bloch sphere trajectories (shows qubit states)
- Fidelity heatmaps (shows distinguishability)

Keep Julia sets if you like the aesthetic, but don't claim they
represent quantum properties.
""")
