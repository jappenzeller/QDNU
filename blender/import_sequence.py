"""
BLENDER: Import Julia Mesh Sequence
===================================

Import a folder of PLY meshes and set up frame-by-frame animation.

Usage in Blender:
    import bpy
    import sys
    sys.path.insert(0, r"H:\QuantumPython\QDNU")
    from blender.import_sequence import import_and_animate
    import_and_animate(r"H:\QuantumPython\QDNU\research\explorer_meshes")

Or use the "Import Sequence" button in the Julia panel (3D View > Sidebar > Julia)
"""

import bpy
import os
from pathlib import Path

# Default paths
DEFAULT_MESH_FOLDER = Path(__file__).parent.parent / "research" / "explorer_meshes"


def clear_julia_meshes():
    """Remove existing julia meshes and default cube."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith('julia_') or obj.name == 'Cube':
            obj.select_set(True)
    bpy.ops.object.delete()


def import_ply_sequence(folder: str):
    """Import all PLY files from folder, sorted by name."""
    folder = Path(folder)
    files = sorted(folder.glob("*.ply"))

    if not files:
        print(f"ERROR: No .ply files found in {folder}")
        return []

    print(f"Found {len(files)} PLY files")

    objects = []
    for i, filepath in enumerate(files):
        bpy.ops.wm.ply_import(filepath=str(filepath))

        obj = bpy.context.active_object
        obj.name = f"julia_{i:03d}"
        objects.append(obj)
        print(f"  Imported: {filepath.name} -> {obj.name}")

    return objects


def setup_visibility_animation(objects, frames_per_mesh: int = 3):
    """Animate visibility so meshes swap each frame."""
    n = len(objects)
    total_frames = n * frames_per_mesh

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    for i, obj in enumerate(objects):
        show_start = i * frames_per_mesh + 1
        show_end = show_start + frames_per_mesh - 1

        # Start hidden
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert('hide_viewport', frame=1)
        obj.keyframe_insert('hide_render', frame=1)

        # Show during its frames
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert('hide_viewport', frame=show_start)
        obj.keyframe_insert('hide_render', frame=show_start)

        # Hide after
        if show_end < total_frames:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert('hide_viewport', frame=show_end + 1)
            obj.keyframe_insert('hide_render', frame=show_end + 1)

    # Constant interpolation (snap, not fade)
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kf in fc.keyframe_points:
                    kf.interpolation = 'CONSTANT'

    print(f"Animation: {total_frames} frames ({n} meshes x {frames_per_mesh} frames each)")
    return total_frames


def create_julia_material():
    """Create cyan/purple material for Julia meshes."""
    mat = bpy.data.materials.new(name="JuliaMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (200, 0)
    mix.inputs['Fac'].default_value = 0.3

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 100)
    principled.inputs['Base Color'].default_value = (0.2, 0.8, 0.9, 1)  # Cyan
    principled.inputs['Metallic'].default_value = 0.3
    principled.inputs['Roughness'].default_value = 0.4

    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, -100)
    emission.inputs['Color'].default_value = (0.8, 0.3, 0.9, 1)  # Purple
    emission.inputs['Strength'].default_value = 2.0

    links.new(principled.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    return mat


def apply_material(objects, material):
    """Apply material to all objects."""
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    print("Applied material to all meshes")


def add_three_point_lighting():
    """Add key, fill, and rim lights."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 200

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 100

    # Rim light
    bpy.ops.object.light_add(type='POINT', location=(0, 3, 2))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 150
    rim.data.color = (0.7, 0.5, 1.0)  # Purple tint

    print("Added 3-point lighting")


def setup_camera():
    """Position camera to view the mesh at origin."""
    cam = bpy.data.objects.get('Camera')
    if cam:
        cam.location = (4, -4, 3)
        cam.rotation_euler = (1.1, 0, 0.8)
    print("Camera positioned")


def setup_render_settings():
    """Configure render for video output."""
    scene = bpy.context.scene

    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 24

    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'

    print("Render: Eevee, 1080p, 24fps, H264")


def import_and_animate(folder: str = None, frames_per_mesh: int = 3,
                       add_material: bool = True, add_lighting: bool = True):
    """
    Main function: import PLY sequence and set up animation.

    Args:
        folder: Path to folder containing PLY files
        frames_per_mesh: How many frames each mesh is visible
        add_material: Apply Julia material
        add_lighting: Add 3-point lighting

    Returns:
        List of imported objects
    """
    if folder is None:
        folder = str(DEFAULT_MESH_FOLDER)

    print("\n" + "=" * 50)
    print("JULIA MESH SEQUENCE IMPORTER")
    print("=" * 50 + "\n")

    if not os.path.exists(folder):
        print(f"ERROR: Folder not found: {folder}")
        return []

    clear_julia_meshes()

    objects = import_ply_sequence(folder)
    if not objects:
        return []

    total_frames = setup_visibility_animation(objects, frames_per_mesh)

    if add_material:
        mat = create_julia_material()
        apply_material(objects, mat)

    if add_lighting:
        add_three_point_lighting()

    setup_camera()
    setup_render_settings()

    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"\nTimeline: {total_frames} frames")
    print("Press SPACE to play animation")
    print("Ctrl+F12 to render video")

    return objects


# For running directly in Blender's text editor
if __name__ == "__main__":
    import_and_animate()
