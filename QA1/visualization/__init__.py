"""
Quantum A-Gate Visualization Module for QDNU.

Provides real-time visualization of quantum neuron activation including:
- Dual Bloch sphere visualization (E/I qubits)
- Entanglement measurement (concurrence)
- Julia set fractal fingerprints
- Parameter evolution tracking
- Phase space analysis and dynamical systems tools
- 4D harmonic oscillator visualization for limit cycles

Usage:
    # Core visualization
    from visualization.agate_visualization import extract_visualization_data
    from visualization.fractal_generator import generate_fractal_from_state

    # Interactive explorer (launches matplotlib window)
    from visualization.interactive_explorer import interactive_agate_explorer

    # EEG visualization
    from visualization.eeg_visualizer import EEGQuantumVisualizer

    # Phase space analysis
    from visualization.phase_analysis import run_phase_analysis, DynamicsConfig

    # 4D harmonic oscillator (limit cycles under periodic driving)
    from visualization.harmonic_oscillator import run_oscillator_demo, HarmonicOscillatorVisualizer
"""

# Core visualization functions (always available)
from .agate_visualization import (
    extract_visualization_data,
    get_bloch_coords,
    get_purity,
    bloch_to_spherical,
    spherical_to_bloch,
    bloch_to_unity_coords,
    draw_bloch_sphere,
    draw_entanglement_indicator,
    draw_probabilities,
)

from .fractal_generator import (
    julia_set,
    generate_fractal_from_state,
    fractal_to_image,
    statevector_to_julia_param,
    save_fractal,
    generate_fractal_sequence,
)

from .phase_analysis import (
    DynamicsConfig,
    compute_derivatives,
    compute_jacobian,
    integrate_rk4,
    find_fixed_points,
    compute_vector_field_2d,
    compute_nullclines,
    plot_phase_portrait_2d,
    plot_phase_portrait_3d,
    trajectory_to_quantum_observables,
    plot_quantum_observables,
    run_phase_analysis,
)

from .harmonic_oscillator import (
    OscillatorConfig,
    HarmonicOscillatorVisualizer,
    simulate_driven_oscillator,
    visualize_harmonic_oscillator,
    run_oscillator_demo,
)

__all__ = [
    # Core visualization
    'extract_visualization_data',
    'get_bloch_coords',
    'get_purity',
    'bloch_to_spherical',
    'spherical_to_bloch',
    'bloch_to_unity_coords',
    'draw_bloch_sphere',
    'draw_entanglement_indicator',
    'draw_probabilities',
    # Fractals
    'julia_set',
    'generate_fractal_from_state',
    'fractal_to_image',
    'statevector_to_julia_param',
    'save_fractal',
    'generate_fractal_sequence',
    # Phase analysis
    'DynamicsConfig',
    'compute_derivatives',
    'compute_jacobian',
    'integrate_rk4',
    'find_fixed_points',
    'compute_vector_field_2d',
    'compute_nullclines',
    'plot_phase_portrait_2d',
    'plot_phase_portrait_3d',
    'trajectory_to_quantum_observables',
    'plot_quantum_observables',
    'run_phase_analysis',
    # Harmonic oscillator
    'OscillatorConfig',
    'HarmonicOscillatorVisualizer',
    'simulate_driven_oscillator',
    'visualize_harmonic_oscillator',
    'run_oscillator_demo',
]
