"""
BLENDER LIVE RELOAD SCRIPT
==========================

Paste this into Blender's Text Editor and run it.
Creates a panel in the 3D View sidebar (N key) with reload controls.

Workflow:
1. In VS Code: python -m visualization.blender.generate_live
2. In Blender: Click "Reload Julia" button (or enable auto-reload)
"""

import bpy
import os
from pathlib import Path

# Default path - update this to match your project
LIVE_PLY_PATH = r"H:\QuantumPython\QDNU\blender\live_julia.ply"

class JuliaReloadProperties(bpy.types.PropertyGroup):
    ply_path: bpy.props.StringProperty(
        name="PLY Path",
        default=LIVE_PLY_PATH,
        subtype='FILE_PATH'
    )
    auto_reload: bpy.props.BoolProperty(
        name="Auto Reload",
        default=False,
        description="Automatically reload when file changes"
    )
    last_mtime: bpy.props.FloatProperty(default=0.0)
    target_object: bpy.props.StringProperty(
        name="Target Object",
        default="JuliaSphere",
        description="Object to replace mesh data"
    )
    # Sequence import properties
    sequence_folder: bpy.props.StringProperty(
        name="Sequence Folder",
        default=r"H:\QuantumPython\QDNU\research\explorer_meshes",
        subtype='DIR_PATH',
        description="Folder containing PLY sequence"
    )
    frames_per_mesh: bpy.props.IntProperty(
        name="Frames/Mesh",
        default=3,
        min=1,
        max=30,
        description="How many frames each mesh is visible"
    )


class JULIA_OT_reload(bpy.types.Operator):
    """Reload Julia mesh from PLY file"""
    bl_idname = "julia.reload"
    bl_label = "Reload Julia"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.julia_reload
        ply_path = bpy.path.abspath(props.ply_path)

        if not os.path.exists(ply_path):
            self.report({'ERROR'}, f"File not found: {ply_path}")
            return {'CANCELLED'}

        # Update last modified time
        props.last_mtime = os.path.getmtime(ply_path)

        # Import PLY
        bpy.ops.wm.ply_import(filepath=ply_path)

        # Get the newly imported object (should be selected)
        new_obj = context.active_object
        if not new_obj:
            self.report({'ERROR'}, "Import failed")
            return {'CANCELLED'}

        # Find or create target object
        target_name = props.target_object
        target = bpy.data.objects.get(target_name)

        if target and target.type == 'MESH':
            # Replace mesh data
            old_mesh = target.data
            target.data = new_obj.data
            new_obj.data = None

            # Copy transforms
            new_obj.data = target.data.copy()
            target.data.name = target_name

            # Remove the imported object
            bpy.data.objects.remove(new_obj)

            # Clean up old mesh if orphaned
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

            self.report({'INFO'}, f"Updated {target_name}")
        else:
            # Rename new object to target name
            new_obj.name = target_name
            new_obj.data.name = target_name
            self.report({'INFO'}, f"Created {target_name}")

        return {'FINISHED'}


class JULIA_OT_set_material(bpy.types.Operator):
    """Apply emission material to Julia object"""
    bl_idname = "julia.set_material"
    bl_label = "Apply Material"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.julia_reload
        obj = bpy.data.objects.get(props.target_object)

        if not obj:
            self.report({'ERROR'}, f"Object not found: {props.target_object}")
            return {'CANCELLED'}

        # Create or get material
        mat_name = "JuliaMaterial"
        mat = bpy.data.materials.get(mat_name)

        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear default nodes
            nodes.clear()

            # Create nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (300, 0)

            principled = nodes.new('ShaderNodeBsdfPrincipled')
            principled.location = (0, 0)

            # Vertex color node
            vcol = nodes.new('ShaderNodeVertexColor')
            vcol.location = (-300, 0)
            vcol.layer_name = "Col"  # PLY imports vertex colors as "Col"

            # Connect
            links.new(vcol.outputs['Color'], principled.inputs['Base Color'])
            links.new(vcol.outputs['Color'], principled.inputs['Emission Color'])
            principled.inputs['Emission Strength'].default_value = 0.3
            links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        # Assign material
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        self.report({'INFO'}, "Material applied")
        return {'FINISHED'}


class JULIA_OT_import_sequence(bpy.types.Operator):
    """Import PLY sequence and set up animation"""
    bl_idname = "julia.import_sequence"
    bl_label = "Import Sequence"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.julia_reload
        folder = bpy.path.abspath(props.sequence_folder)

        if not os.path.exists(folder):
            self.report({'ERROR'}, f"Folder not found: {folder}")
            return {'CANCELLED'}

        # Import using the sequence module
        try:
            import sys
            qdnu_root = str(Path(__file__).parent.parent)
            if qdnu_root not in sys.path:
                sys.path.insert(0, qdnu_root)

            from blender.import_sequence import import_and_animate
            objects = import_and_animate(
                folder=folder,
                frames_per_mesh=props.frames_per_mesh,
                add_material=True,
                add_lighting=True
            )

            if objects:
                self.report({'INFO'}, f"Imported {len(objects)} meshes")
            else:
                self.report({'WARNING'}, "No meshes imported")

        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}


class JULIA_PT_panel(bpy.types.Panel):
    """Julia Live Reload Panel"""
    bl_label = "Julia Live Reload"
    bl_idname = "JULIA_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Julia'

    def draw(self, context):
        layout = self.layout
        props = context.scene.julia_reload

        # File path
        layout.prop(props, "ply_path")
        layout.prop(props, "target_object")

        # Reload button
        row = layout.row(align=True)
        row.scale_y = 2.0
        row.operator("julia.reload", icon='FILE_REFRESH')

        # Auto reload toggle
        layout.prop(props, "auto_reload", icon='TIME')

        # Material button
        layout.separator()
        layout.operator("julia.set_material", icon='MATERIAL')

        # Status
        if os.path.exists(bpy.path.abspath(props.ply_path)):
            mtime = os.path.getmtime(bpy.path.abspath(props.ply_path))
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime)
            layout.label(text=f"Last modified: {dt.strftime('%H:%M:%S')}")
        else:
            layout.label(text="File not found", icon='ERROR')

        # Sequence Import section
        layout.separator()
        box = layout.box()
        box.label(text="Sequence Animation", icon='SEQUENCE')
        box.prop(props, "sequence_folder")
        box.prop(props, "frames_per_mesh")
        row = box.row()
        row.scale_y = 1.5
        row.operator("julia.import_sequence", icon='IMPORT')


# File watcher timer
def check_file_changed():
    """Timer callback to check for file changes"""
    try:
        context = bpy.context
        props = context.scene.julia_reload

        if not props.auto_reload:
            return 1.0  # Check every second

        ply_path = bpy.path.abspath(props.ply_path)
        if os.path.exists(ply_path):
            mtime = os.path.getmtime(ply_path)
            if mtime > props.last_mtime and props.last_mtime > 0:
                # File changed - reload
                bpy.ops.julia.reload()
                print(f"Auto-reloaded: {ply_path}")
            props.last_mtime = mtime
    except:
        pass

    return 1.0  # Check every second


# Registration
classes = [
    JuliaReloadProperties,
    JULIA_OT_reload,
    JULIA_OT_set_material,
    JULIA_OT_import_sequence,
    JULIA_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.julia_reload = bpy.props.PointerProperty(type=JuliaReloadProperties)
    bpy.app.timers.register(check_file_changed, persistent=True)

def unregister():
    if bpy.app.timers.is_registered(check_file_changed):
        bpy.app.timers.unregister(check_file_changed)
    del bpy.types.Scene.julia_reload
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
