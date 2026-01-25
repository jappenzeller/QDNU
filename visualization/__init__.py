"""
QDNU Visualization Module

Organized by visualization type:
- quantum/: Bloch spheres, fractals, entanglement
- dynamics/: Phase space, harmonic oscillators, limit cycles
- animation/: Animated visualizations
- interactive/: Real-time parameter explorers
"""

from .quantum import (
    extract_visualization_data,
    draw_bloch_sphere,
    generate_fractal_from_state,
)

from .dynamics import (
    DynamicsConfig,
    OscillatorConfig,
    HarmonicOscillatorVisualizer,
    simulate_driven_oscillator,
    run_oscillator_demo,
)

__all__ = [
    # Quantum visualizations
    'extract_visualization_data',
    'draw_bloch_sphere',
    'generate_fractal_from_state',
    # Dynamics visualizations
    'DynamicsConfig',
    'OscillatorConfig',
    'HarmonicOscillatorVisualizer',
    'simulate_driven_oscillator',
    'run_oscillator_demo',
]
