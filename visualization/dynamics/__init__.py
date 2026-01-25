"""
Dynamical Systems Visualization

Phase space analysis, harmonic oscillators, and limit cycles.
"""

from .phase_analysis import (
    DynamicsConfig,
    compute_derivatives,
    integrate_rk4,
    find_fixed_points,
    plot_phase_portrait_2d,
    plot_phase_portrait_3d,
    run_phase_analysis,
)

from .harmonic_oscillator import (
    OscillatorConfig,
    HarmonicOscillatorVisualizer,
    simulate_driven_oscillator,
    run_oscillator_demo,
)

__all__ = [
    # Phase analysis
    'DynamicsConfig',
    'compute_derivatives',
    'integrate_rk4',
    'find_fixed_points',
    'plot_phase_portrait_2d',
    'plot_phase_portrait_3d',
    'run_phase_analysis',
    # Harmonic oscillator
    'OscillatorConfig',
    'HarmonicOscillatorVisualizer',
    'simulate_driven_oscillator',
    'run_oscillator_demo',
]
