"""
Interactive Visualization

Real-time parameter exploration tools.
"""

from .interactive_explorer import (
    interactive_agate_explorer,
    animate_parameter_sweep,
    visualize_eeg_activation,
)

from .julia_animated_explorer import JuliaExplorer
from .julia_3d_boundary import Julia3DBoundaryViewer

__all__ = [
    'interactive_agate_explorer',
    'animate_parameter_sweep',
    'visualize_eeg_activation',
    'JuliaExplorer',
    'Julia3DBoundaryViewer',
]
