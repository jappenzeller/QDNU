"""
Animation Module

Animated visualizations including limit cycle Julia set explorer.
"""

from .limit_cycle_julia_explorer import (
    PNOscillator,
    quaternion_julia_3d,
    map_pn_to_quaternion,
    generate_julia_animation_frames,
)

__all__ = [
    'PNOscillator',
    'quaternion_julia_3d',
    'map_pn_to_quaternion',
    'generate_julia_animation_frames',
]
