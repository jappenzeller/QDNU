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

from .julia_vis_complete import (
    JuliaVisualizer,
    compute_julia_2d,
    compute_julia_quaternion,
    create_plane_mesh,
    create_canyon_mesh,
    create_sphere_mesh,
    export_ply,
    export_obj,
    plot_pipeline_overview,
    print_keyframe_table,
    apply_colormap,
)

from .julia_sphere_stereographic import (
    compute_julia_on_sphere as compute_julia_stereographic,
    sphere_to_complex,
    complex_to_sphere,
    analyze_julia_symmetry,
    NOTABLE_JULIA_PARAMS,
    get_julia_for_n_lobes,
    create_spherical_harmonic_sphere,
)

from .julia_spherical_harmonic import (
    SphericalJulia,
    create_julia_sphere_mesh,
    get_julia_for_node_count,
    JULIA_PRESETS,
)

from .julia_sphere_refined import (
    create_julia_sphere_smooth,
    compute_julia_smooth,
    colormap_organic,
    colormap_harmonic,
    visualize_sphere as visualize_julia_sphere,
    create_comparison_figure,
    create_lobe_gallery,
    export_ply as export_refined_ply,
)

__all__ = [
    # Limit cycle explorer
    'PNOscillator',
    'quaternion_julia_3d',
    'map_pn_to_quaternion',
    'generate_julia_animation_frames',
    # Julia surface generator (quaternion/isosurface)
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
    # Julia visualizer (2D/canyon/sphere)
    'JuliaVisualizer',
    'compute_julia_2d',
    'compute_julia_quaternion',
    'create_plane_mesh',
    'create_canyon_mesh',
    'create_sphere_mesh',
    'export_ply',
    'export_obj',
    'plot_pipeline_overview',
    'print_keyframe_table',
    'apply_colormap',
    # Spherical Julia (stereographic projection)
    'compute_julia_stereographic',
    'sphere_to_complex',
    'complex_to_sphere',
    'analyze_julia_symmetry',
    'NOTABLE_JULIA_PARAMS',
    'get_julia_for_n_lobes',
    'create_spherical_harmonic_sphere',
    # Spherical Julia (harmonic analogy)
    'SphericalJulia',
    'create_julia_sphere_mesh',
    'get_julia_for_node_count',
    'JULIA_PRESETS',
    # Refined smooth sphere (tunable smoothing/depth)
    'create_julia_sphere_smooth',
    'compute_julia_smooth',
    'colormap_organic',
    'colormap_harmonic',
    'visualize_julia_sphere',
    'create_comparison_figure',
    'create_lobe_gallery',
    'export_refined_ply',
]
