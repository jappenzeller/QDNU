"""
BLENDER STARTUP SCRIPT
======================

Run this ONCE to set up auto-registration. After that, the Julia panel
will appear automatically every time you open this .blend file.

In Blender:
1. Switch to Scripting workspace
2. Open this file (startup.py)
3. Run Script (Alt+P)
4. Save the .blend file

The script registers a load handler that auto-runs on file open.
"""

import bpy
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================

def get_qdnu_paths():
    """Get QDNU project paths."""
    # Try to get path from current file location
    if bpy.data.filepath:
        blender_dir = Path(bpy.data.filepath).parent
    else:
        # Fallback to hardcoded path
        blender_dir = Path(r"H:\QuantumPython\QDNU\blender")

    qdnu_root = blender_dir.parent
    return blender_dir, qdnu_root


def setup_python_path():
    """Add QDNU to Python path."""
    blender_dir, qdnu_root = get_qdnu_paths()

    if str(qdnu_root) not in sys.path:
        sys.path.insert(0, str(qdnu_root))
        print(f"Added to path: {qdnu_root}")

    return blender_dir, qdnu_root


def register_julia_panel():
    """Register the Julia Live Reload panel."""
    blender_dir, qdnu_root = setup_python_path()

    live_reload_path = blender_dir / "live_reload.py"
    if live_reload_path.exists():
        exec(compile(open(str(live_reload_path)).read(), str(live_reload_path), 'exec'))
        print("Julia Live Reload panel registered!")
        print("Find it in: 3D View > Sidebar (N) > Julia tab")
    else:
        print(f"Warning: live_reload.py not found at {live_reload_path}")

    print(f"\nQDNU Project Root: {qdnu_root}")
    print(f"Blender Directory: {blender_dir}")


# =============================================================================
# AUTO-LOAD HANDLER
# =============================================================================

def load_handler(dummy):
    """Called when .blend file is loaded."""
    print("\n" + "="*50)
    print("QDNU: Auto-registering Julia panel...")
    print("="*50)
    register_julia_panel()


def register_load_handler():
    """Register the load handler (persists in .blend file)."""
    # Remove existing handler if present
    for handler in bpy.app.handlers.load_post:
        if handler.__name__ == 'load_handler':
            bpy.app.handlers.load_post.remove(handler)

    # Add our handler
    bpy.app.handlers.load_post.append(load_handler)
    print("Load handler registered - Julia panel will auto-load on file open")


# =============================================================================
# RUN ON SCRIPT EXECUTION
# =============================================================================

# Register immediately
register_julia_panel()

# Also register the load handler for future opens
register_load_handler()

print("\n" + "="*50)
print("SETUP COMPLETE!")
print("="*50)
print("Now save the .blend file (Ctrl+S)")
print("Next time you open it, Julia panel will auto-register.")
