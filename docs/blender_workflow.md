# Blender Workflow for Julia Surface Animation

## Overview

Two approaches for animating the Julia surface in Blender:

1. **Mesh Sequence** - Import multiple PLY files as animation frames (best for looping)
2. **Shape Keys** - Morph between keyframe meshes (smoother but more setup)

For your 8-keyframe loop, I recommend **Mesh Sequence** with the Stop Motion OBJ addon.

---

## Method 1: Mesh Sequence (Recommended)

### Step 1: Install Stop Motion OBJ Addon

1. Download from: https://github.com/neverhood311/Stop-motion-OBJ
2. In Blender: Edit -> Preferences -> Add-ons -> Install
3. Select the downloaded .zip file
4. Enable "Import-Export: Stop motion OBJ"

### Step 2: Prepare Your Files

Export 8 keyframes from your explorer (or more for smoother animation):
```
julia_kf00.ply
julia_kf01.ply
julia_kf02.ply
julia_kf03.ply
julia_kf04.ply
julia_kf05.ply
julia_kf06.ply
julia_kf07.ply
```

**Important**: Files must be numbered sequentially.

### Step 3: Import Mesh Sequence

1. File -> Import -> Mesh Sequence
2. Navigate to your folder
3. Select the FIRST file (julia_kf00.ply)
4. Click "Import Mesh Sequence"

The addon creates an object that swaps meshes each frame.

### Step 4: Configure Animation

In the sidebar (N panel) -> Mesh Sequence tab:

- **Frame Mode**: "Repeat" for looping
- **Start Frame**: 1
- **Playback Speed**: Adjust to taste (1.0 = one mesh per frame)

For 8 keyframes looping over 24 frames (1 second at 24fps):
- Speed = 8/24 = 0.333

Or for 8 keyframes over 48 frames (2 seconds):
- Speed = 8/48 = 0.167

### Step 5: Set Up Scene

```
Timeline:
- Start Frame: 1
- End Frame: 24 (or 48 for slower)

Output:
- Resolution: 1920x1080 (or your preference)
- Frame Rate: 24 fps
- Output Format: FFmpeg video -> MP4
```

---

## Method 2: Manual Import (No Addon)

If you don't want to use the addon:

### Step 1: Import First Mesh

1. File -> Import -> Stanford (.ply)
2. Select julia_kf00.ply
3. Position/scale as needed

### Step 2: Create Shape Keys

1. Select the mesh
2. Object Data Properties -> Shape Keys
3. Click "+" to add Basis key

### Step 3: Import Other Meshes as Shape Keys

For each keyframe (kf01 through kf07):

1. File -> Import -> Stanford (.ply)
2. Select both meshes (original + imported)
3. Select original LAST (active)
4. Object -> Join as Shapes

**Note**: This only works if all meshes have the SAME vertex count and topology. The marching cubes output may vary, so this method can be tricky.

### Step 4: Animate Shape Keys

1. On frame 1: Set kf00 to 1.0, others to 0.0
2. On frame 3: Set kf01 to 1.0, others to 0.0
3. Continue for each keyframe...
4. Right-click values -> Insert Keyframe

---

## Materials & Lighting

### Basic Material Setup

1. Select mesh -> Material Properties -> New
2. Use Principled BSDF:
   - Base Color: Your choice (cyan/purple matches your explorer)
   - Metallic: 0.3
   - Roughness: 0.4
   - Subsurface: 0.1 (for organic look)

### Emission Glow (like your explorer)

1. Add Emission shader
2. Mix with Principled BSDF using Mix Shader
3. Emission color: Purple/magenta
4. Emission strength: 2-5

Node setup:
```
[Principled BSDF] --+
                    +-- [Mix Shader] --> [Material Output]
[Emission] ---------+
     ^
Mix Factor: 0.3
```

### Lighting

Simple 3-point setup:
1. Key light: Area light, front-right, white
2. Fill light: Area light, front-left, 50% intensity
3. Rim light: Point light behind, colored (purple/blue)

Or use HDRI:
1. World Properties -> Surface -> Environment Texture
2. Use an HDRI from polyhaven.com

---

## Camera Animation (Optional)

For orbiting camera:

1. Add Empty at origin (Shift+A -> Empty -> Plain Axes)
2. Select Camera, then Empty
3. Ctrl+P -> Parent to Empty
4. Select Empty
5. Insert keyframe on Rotation Z at frame 1 (I -> Rotation)
6. Go to final frame
7. Rotate Empty 360 deg on Z
8. Insert keyframe

---

## Render Settings

### For Preview

```
Render Engine: Eevee
Samples: 64
Output: 1280x720
```

### For Final

```
Render Engine: Cycles
Samples: 128-256
Output: 1920x1080 or 4K
Denoising: ON
```

### Output as Loop-Ready Video

```
Output Properties:
- File Format: FFmpeg Video
- Container: MPEG-4
- Codec: H.264
- Quality: High or Perceptually Lossless

Frame Range:
- Start: 1
- End: 24 (or 48)
```

---

## Quick Start Checklist

```
[ ] Export 8 keyframes from explorer (julia_kf00.ply - julia_kf07.ply)
[ ] Install Stop Motion OBJ addon
[ ] Import mesh sequence (select first file)
[ ] Set playback to "Repeat"
[ ] Add material (Principled BSDF + Emission)
[ ] Set up lighting (3-point or HDRI)
[ ] Configure timeline (24 or 48 frames)
[ ] Set output format (MP4)
[ ] Render -> Render Animation (Ctrl+F12)
```

---

## Troubleshooting

**Mesh looks inside-out:**
- Select mesh -> Edit Mode -> Select All -> Mesh -> Normals -> Recalculate Outside

**Animation too fast/slow:**
- Adjust Speed in Mesh Sequence panel
- Or change timeline end frame

**Mesh sequence not looping:**
- Set Frame Mode to "Repeat"
- Ensure end frame matches your loop point

**File sizes too large:**
- Reduce resolution in julia_surface_generator.py
- Use Decimate modifier in Blender

**Gaps between keyframes:**
- Generate more frames (24, 60, or 120 instead of 8)
- Use `gen.generate_smooth_animation('frames/', n_frames=60)`

---

## Python Script for Batch Import (Alternative)

If Stop Motion OBJ doesn't work, here's a Blender Python script:

```python
# Run in Blender's Scripting workspace

import bpy
import os

# Path to your PLY files
folder = "/path/to/your/keyframes/"
files = sorted([f for f in os.listdir(folder) if f.endswith('.ply')])

# Import each as separate object
for i, f in enumerate(files):
    bpy.ops.import_mesh.ply(filepath=os.path.join(folder, f))
    obj = bpy.context.active_object
    obj.name = f"julia_kf{i:02d}"

    # Hide in viewport except on its frame
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert('hide_viewport', frame=1)
    obj.keyframe_insert('hide_render', frame=1)

    # Show on corresponding frame
    frame = i * 3 + 1  # Adjust spacing
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert('hide_viewport', frame=frame)
    obj.keyframe_insert('hide_render', frame=frame)

    # Hide again next frame
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert('hide_viewport', frame=frame+1)
    obj.keyframe_insert('hide_render', frame=frame+1)

print(f"Imported {len(files)} meshes")
```

---

## Next Steps

1. Test with 8 keyframes first
2. Once working, generate 60+ frames for smoother animation
3. Add camera orbit for more dynamic video
4. Export and verify loop is seamless
