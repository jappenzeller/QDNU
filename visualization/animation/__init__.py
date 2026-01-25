"""
Animation Module

Animated visualizations including limit cycle Julia set explorer
and 3D Julia surface generator for Blender export.
"""

from .limit_cycle_julia_explorer import (
    PNOscillator,
    quaternion_julia_3d,
    map_pn_to_quaternion,
    generate_julia_animation_frames,
)

from .julia_surface_generator import (
    # Data structures
    Mesh,
    PNState,
    QuaternionC,
    # Dynamics
    PNOscillatorFHN,
    KeyframeSystem,
    # Julia computation
    PNToQuaternionMapper,
    QuaternionJulia,
    IsosurfaceExtractor,
    # Export
    MeshExporter,
    # High-level
    JuliaSurfaceGenerator,
    # Visualization
    preview_volume_slices,
    preview_mesh,
)

__all__ = [
    # Limit cycle explorer
    'PNOscillator',
    'quaternion_julia_3d',
    'map_pn_to_quaternion',
    'generate_julia_animation_frames',
    # Julia surface generator
    'Mesh',
    'PNState',
    'QuaternionC',
    'PNOscillatorFHN',
    'KeyframeSystem',
    'PNToQuaternionMapper',
    'QuaternionJulia',
    'IsosurfaceExtractor',
    'MeshExporter',
    'JuliaSurfaceGenerator',
    'preview_volume_slices',
    'preview_mesh',
]
