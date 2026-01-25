"""
BLENDER SCRIPT: Julia Surface + Bloch Spheres + Parameter Control

Features:
- Julia surface mesh sequence
- Two Bloch spheres (E and I qubits)
- State vector points on spheres
- Trace of past 10 positions
- Custom panel with a, b, c sliders

Run in Blender Scripting workspace.
"""

import bpy
import bmesh
import os
import math
import numpy as np
from mathutils import Vector
from bpy.props import FloatProperty, IntProperty

# =============================================================================
# CONFIGURATION
# =============================================================================

MESH_FOLDER = r"C:\path\to\your\keyframes"  # EDIT THIS

FRAMES_PER_MESH = 3
TRACE_LENGTH = 10  # Number of past positions to show

# Bloch sphere positions
BLOCH_E_POS = Vector((-3, 0, 0))  # Excitatory qubit sphere
BLOCH_I_POS = Vector((3, 0, 0))   # Inhibitory qubit sphere
BLOCH_RADIUS = 1.0

# =============================================================================
# QUANTUM STATE CALCULATION (simplified for visualization)
# =============================================================================

def pn_to_bloch_vectors(a, b, c):
    """
    Convert PN parameters (a, b, c) to Bloch sphere coordinates.

    This is a simplified visualization mapping.
    Real values come from quantum circuit simulation.

    Args:
        a: Excitatory [0, 1]
        b: Phase [0, 2pi]
        c: Inhibitory [0, 1]

    Returns:
        (bloch_E, bloch_I) - each is (x, y, z) on unit sphere
    """
    # Excitatory qubit (q0) - affected by a and b
    theta_E = math.pi * (1 - a)  # a=0 -> north pole, a=1 -> south pole
    phi_E = b

    x_E = math.sin(theta_E) * math.cos(phi_E)
    y_E = math.sin(theta_E) * math.sin(phi_E)
    z_E = math.cos(theta_E)

    # Inhibitory qubit (q1) - affected by c and b
    theta_I = math.pi * (1 - c)
    phi_I = b + math.pi / 4  # Slight phase offset

    x_I = math.sin(theta_I) * math.cos(phi_I)
    y_I = math.sin(theta_I) * math.sin(phi_I)
    z_I = math.cos(theta_I)

    return (x_E, y_E, z_E), (x_I, y_I, z_I)


def compute_concurrence(a, b, c):
    """Approximate concurrence (entanglement) from parameters."""
    # Simplified: entanglement peaks when a and c differ
    return 0.5 * abs(math.sin(b)) * (1 - abs(a - c))


# =============================================================================
# BLOCH SPHERE CREATION
# =============================================================================

def create_bloch_sphere(name, location, radius=1.0):
    """Create a wireframe Bloch sphere with axes."""

    # Main sphere (wireframe)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=32,
        ring_count=16,
        location=location
    )
    sphere = bpy.context.active_object
    sphere.name = f"{name}_sphere"

    # Wireframe material
    mat = bpy.data.materials.new(name=f"{name}_wire_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.1
    mat.blend_method = 'BLEND'
    sphere.data.materials.append(mat)

    # Make wireframe
    mod = sphere.modifiers.new(name="Wireframe", type='WIREFRAME')
    mod.thickness = 0.01

    # Create axes
    axis_length = radius * 1.3
    axis_colors = {
        'X': (1, 0, 0, 1),  # Red
        'Y': (0, 0, 1, 1),  # Blue
        'Z': (0, 1, 0, 1),  # Green (UP = |0>)
    }

    axes = []
    for axis, color in axis_colors.items():
        # Create cylinder for axis
        if axis == 'X':
            rot = (0, math.pi/2, 0)
        elif axis == 'Y':
            rot = (math.pi/2, 0, 0)
        else:
            rot = (0, 0, 0)

        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length * 2,
            location=location,
            rotation=rot
        )
        ax = bpy.context.active_object
        ax.name = f"{name}_axis_{axis}"

        # Color material
        ax_mat = bpy.data.materials.new(name=f"{name}_axis_{axis}_mat")
        ax_mat.use_nodes = True
        ax_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
        ax.data.materials.append(ax_mat)
        axes.append(ax)

    # Add axis labels - Z is green and UP = |0>
    labels = [
        ('|0>', (0, 0, radius * 1.4), (0, 1, 0, 1)),   # Green, top
        ('|1>', (0, 0, -radius * 1.4), (0, 1, 0, 1)),  # Green, bottom
        ('|+>', (radius * 1.4, 0, 0), (1, 0, 0, 1)),   # Red
        ('|->', (-radius * 1.4, 0, 0), (1, 0, 0, 1)),  # Red
        ('|+i>', (0, radius * 1.4, 0), (0, 0, 1, 1)),  # Blue
        ('|-i>', (0, -radius * 1.4, 0), (0, 0, 1, 1)), # Blue
    ]

    for label_text, offset, color in labels:
        bpy.ops.object.text_add(location=(
            location.x + offset[0],
            location.y + offset[1],
            location.z + offset[2]
        ))
        txt = bpy.context.active_object
        txt.data.body = label_text
        txt.data.size = 0.2
        txt.name = f"{name}_label_{label_text}"

        # Color the label
        txt_mat = bpy.data.materials.new(name=f"{name}_label_{label_text}_mat")
        txt_mat.use_nodes = True
        txt_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
        txt.data.materials.append(txt_mat)

    return sphere


def create_state_point(name, location, color=(1, 0.5, 0, 1)):
    """Create a point (small sphere) to represent quantum state on Bloch sphere."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.08,
        segments=16,
        ring_count=8,
        location=location
    )
    point = bpy.context.active_object
    point.name = name

    # Emissive material
    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 5.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    point.data.materials.append(mat)
    return point


def create_trace_curve(name, color=(1, 0.5, 0, 1)):
    """Create a curve to show trace of past positions."""
    # Create curve data
    curve_data = bpy.data.curves.new(name=f"{name}_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.015

    # Create spline
    spline = curve_data.splines.new('POLY')
    spline.points.add(TRACE_LENGTH - 1)  # Total = TRACE_LENGTH points

    # Create object
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Material
    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs["Base Color"].default_value = color
    nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.7
    mat.blend_method = 'BLEND'
    curve_obj.data.materials.append(mat)

    return curve_obj


# =============================================================================
# CUSTOM PROPERTIES AND UI PANEL
# =============================================================================

def register_properties():
    """Register custom properties for a, b, c parameters."""
    bpy.types.Scene.pn_a = FloatProperty(
        name="a (Excitatory)",
        description="Excitatory parameter",
        default=0.3,
        min=0.0,
        max=1.0,
        update=update_visualization
    )

    bpy.types.Scene.pn_b = FloatProperty(
        name="b (Phase)",
        description="Phase parameter (radians)",
        default=0.0,
        min=0.0,
        max=6.283185,  # 2pi
        update=update_visualization
    )

    bpy.types.Scene.pn_c = FloatProperty(
        name="c (Inhibitory)",
        description="Inhibitory parameter",
        default=0.3,
        min=0.0,
        max=1.0,
        update=update_visualization
    )

    bpy.types.Scene.pn_b_degrees = FloatProperty(
        name="b (degrees)",
        description="Phase in degrees",
        default=0.0,
        min=0.0,
        max=360.0,
        update=update_b_from_degrees
    )


def update_b_from_degrees(self, context):
    """Update b (radians) when degrees slider changes."""
    context.scene.pn_b = math.radians(context.scene.pn_b_degrees)


def update_visualization(self, context):
    """Called when a, b, or c changes - updates Bloch spheres and traces."""
    scene = context.scene
    a = scene.pn_a
    b = scene.pn_b
    c = scene.pn_c

    # Update degrees display
    if abs(math.degrees(b) - scene.pn_b_degrees) > 0.1:
        scene.pn_b_degrees = math.degrees(b)

    # Calculate Bloch vectors
    bloch_E, bloch_I = pn_to_bloch_vectors(a, b, c)

    # Update state points
    point_E = bpy.data.objects.get("state_point_E")
    point_I = bpy.data.objects.get("state_point_I")

    if point_E:
        point_E.location = BLOCH_E_POS + Vector(bloch_E) * BLOCH_RADIUS

    if point_I:
        point_I.location = BLOCH_I_POS + Vector(bloch_I) * BLOCH_RADIUS

    # Update traces
    update_trace("trace_E", point_E.location if point_E else BLOCH_E_POS)
    update_trace("trace_I", point_I.location if point_I else BLOCH_I_POS)


# Storage for trace history
trace_history_E = []
trace_history_I = []

def update_trace(trace_name, new_pos):
    """Update trace curve with new position."""
    global trace_history_E, trace_history_I

    if trace_name == "trace_E":
        history = trace_history_E
    else:
        history = trace_history_I

    # Add new position
    history.append(Vector(new_pos))

    # Trim to max length
    while len(history) > TRACE_LENGTH:
        history.pop(0)

    # Update curve
    curve_obj = bpy.data.objects.get(trace_name)
    if curve_obj and len(history) > 1:
        spline = curve_obj.data.splines[0]

        # Ensure enough points
        while len(spline.points) < len(history):
            spline.points.add(1)

        # Update point positions
        for i, pos in enumerate(history):
            spline.points[i].co = (pos.x, pos.y, pos.z, 1)


class PN_PT_ControlPanel(bpy.types.Panel):
    """Panel for PN Neuron parameter control."""
    bl_label = "PN Neuron Control"
    bl_idname = "PN_PT_control_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PN Neuron'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Parameters
        box = layout.box()
        box.label(text="Parameters", icon='SETTINGS')

        row = box.row()
        row.prop(scene, "pn_a", slider=True)

        row = box.row()
        row.prop(scene, "pn_b_degrees", slider=True, text="b (Phase deg)")

        row = box.row()
        row.prop(scene, "pn_c", slider=True)

        # Keyframe buttons
        box2 = layout.box()
        box2.label(text="Keyframes", icon='KEYFRAME')

        row = box2.row(align=True)
        for i in range(8):
            op = row.operator("pn.set_keyframe", text=str(i))
            op.keyframe_index = i

        # Info display
        box3 = layout.box()
        box3.label(text="State Info", icon='INFO')

        a, b, c = scene.pn_a, scene.pn_b, scene.pn_c
        bloch_E, bloch_I = pn_to_bloch_vectors(a, b, c)
        concurrence = compute_concurrence(a, b, c)

        box3.label(text=f"Bloch E: ({bloch_E[0]:.2f}, {bloch_E[1]:.2f}, {bloch_E[2]:.2f})")
        box3.label(text=f"Bloch I: ({bloch_I[0]:.2f}, {bloch_I[1]:.2f}, {bloch_I[2]:.2f})")
        box3.label(text=f"Concurrence: {concurrence:.3f}")


class PN_OT_SetKeyframe(bpy.types.Operator):
    """Set parameters to a specific keyframe."""
    bl_idname = "pn.set_keyframe"
    bl_label = "Set Keyframe"

    keyframe_index: IntProperty()

    def execute(self, context):
        # 8 keyframes evenly distributed
        b = (2 * math.pi * self.keyframe_index) / 8
        context.scene.pn_b = b
        context.scene.pn_b_degrees = math.degrees(b)
        return {'FINISHED'}


class PN_OT_ResetTrace(bpy.types.Operator):
    """Clear trace history."""
    bl_idname = "pn.reset_trace"
    bl_label = "Reset Traces"

    def execute(self, context):
        global trace_history_E, trace_history_I
        trace_history_E = []
        trace_history_I = []
        return {'FINISHED'}


# =============================================================================
# JULIA MESH SEQUENCE (from previous script)
# =============================================================================

def import_ply_sequence(folder):
    """Import all PLY files from folder."""
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    files = sorted([f for f in os.listdir(folder) if f.endswith('.ply')])

    if not files:
        print(f"No .ply files found in {folder}")
        return []

    print(f"Found {len(files)} PLY files")

    objects = []
    for i, filename in enumerate(files):
        filepath = os.path.join(folder, filename)
        bpy.ops.import_mesh.ply(filepath=filepath)

        obj = bpy.context.active_object
        obj.name = f"julia_{i:03d}"
        obj.location = (0, 0, 0)  # Center
        objects.append(obj)
        print(f"  Imported: {filename}")

    return objects


def setup_mesh_animation(objects, frames_per_mesh=3):
    """Animate mesh sequence visibility."""
    n = len(objects)
    if n == 0:
        return 0

    total_frames = n * frames_per_mesh

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    for i, obj in enumerate(objects):
        show_start = i * frames_per_mesh + 1
        show_end = show_start + frames_per_mesh - 1

        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert('hide_viewport', frame=1)
        obj.keyframe_insert('hide_render', frame=1)

        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert('hide_viewport', frame=show_start)
        obj.keyframe_insert('hide_render', frame=show_start)

        if show_end < total_frames:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert('hide_viewport', frame=show_end + 1)
            obj.keyframe_insert('hide_render', frame=show_end + 1)

    # Constant interpolation
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kf in fc.keyframe_points:
                    kf.interpolation = 'CONSTANT'

    return total_frames


def add_julia_material(objects):
    """Add material to Julia meshes."""
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
    principled.inputs['Base Color'].default_value = (0.2, 0.8, 0.9, 1)
    principled.inputs['Metallic'].default_value = 0.3
    principled.inputs['Roughness'].default_value = 0.4

    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, -100)
    emission.inputs['Color'].default_value = (0.8, 0.3, 0.9, 1)
    emission.inputs['Strength'].default_value = 2.0

    links.new(principled.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


# =============================================================================
# SCENE SETUP
# =============================================================================

def clear_scene():
    """Remove existing objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def add_lighting():
    """Add scene lighting."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 5))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 300

    # Fill
    bpy.ops.object.light_add(type='AREA', location=(-5, -3, 3))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 150

    # Rim
    bpy.ops.object.light_add(type='POINT', location=(0, 5, 2))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 200
    rim.data.color = (0.7, 0.5, 1.0)


def setup_camera():
    """Add and position camera."""
    bpy.ops.object.camera_add(location=(0, -10, 3))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(80), 0, 0)

    bpy.context.scene.camera = cam


def setup_render():
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 24
    scene.eevee.use_bloom = True


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    PN_PT_ControlPanel,
    PN_OT_SetKeyframe,
    PN_OT_ResetTrace,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.pn_a
    del bpy.types.Scene.pn_b
    del bpy.types.Scene.pn_c
    del bpy.types.Scene.pn_b_degrees


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("JULIA + BLOCH SPHERE VISUALIZATION")
    print("="*60 + "\n")

    # Clear and setup
    clear_scene()

    # Register UI
    register()

    # Create Bloch spheres
    print("Creating Bloch spheres...")
    create_bloch_sphere("bloch_E", BLOCH_E_POS, BLOCH_RADIUS)
    create_bloch_sphere("bloch_I", BLOCH_I_POS, BLOCH_RADIUS)

    # Create state points
    print("Creating state points...")
    create_state_point("state_point_E", BLOCH_E_POS, color=(1, 0.5, 0, 1))  # Orange
    create_state_point("state_point_I", BLOCH_I_POS, color=(0, 0.8, 1, 1))  # Cyan

    # Create traces
    print("Creating trace curves...")
    create_trace_curve("trace_E", color=(1, 0.5, 0, 1))
    create_trace_curve("trace_I", color=(0, 0.8, 1, 1))

    # Import Julia meshes (if folder exists)
    print("\nImporting Julia meshes...")
    if os.path.exists(MESH_FOLDER):
        julia_objects = import_ply_sequence(MESH_FOLDER)
        if julia_objects:
            setup_mesh_animation(julia_objects, FRAMES_PER_MESH)
            add_julia_material(julia_objects)
    else:
        print(f"Mesh folder not found: {MESH_FOLDER}")
        print("Skipping Julia mesh import - edit MESH_FOLDER path")

    # Scene setup
    print("\nSetting up scene...")
    add_lighting()
    setup_camera()
    setup_render()

    # Initialize visualization
    bpy.context.scene.pn_a = 0.3
    bpy.context.scene.pn_b = 0.0
    bpy.context.scene.pn_c = 0.3

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nControls:")
    print("  - Open sidebar (N key) -> 'PN Neuron' tab")
    print("  - Use sliders to adjust a, b, c")
    print("  - Click keyframe buttons (0-7) for preset phases")
    print("  - Press Z -> 'Rendered' to see materials")
    print("  - Spacebar to play animation")


if __name__ == "__main__":
    main()
