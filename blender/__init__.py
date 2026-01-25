"""
Blender Integration Module

Scripts and tools for Julia visualization in Blender.

Usage:
    # Generate live mesh (from VS Code terminal)
    python -m blender.generate_live --preset rabbit
    python -m blender.generate_live --metric concurrence

    # In Blender: Open blender/live_reload.py in Text Editor and Run Script
    # This creates a "Julia" panel in the 3D View sidebar (N key)
    # Features:
    #   - Reload Julia: Import single PLY file
    #   - Auto Reload: Watch file for changes
    #   - Import Sequence: Import folder of PLYs as animation

Modules:
    generate_live   - CLI for generating PLY meshes
    live_reload     - Blender addon for live reload panel
    import_sequence - Import PLY sequence as animation
"""
