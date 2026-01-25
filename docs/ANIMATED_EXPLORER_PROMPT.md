# QDNU Julia Animated Explorer - Integration Prompt

## Overview

Add animated playback with sine wave phase indicator and pulse (entanglement) meter to the A-Gate Quantum Neuron Explorer.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Bloch E    │  Bloch I    │  Julia 2D   │  Canyon 3D                │
│  (trace)    │  (trace)    │  (fractal)  │  (terrain)                │
├─────────────────────────────────────────────────────────────────────┤
│  Sine Wave  │  Pulse      │  Sphere     │  Controls                 │
│  (phase)    │  (concur.)  │  (project)  │  [▶][⏸][⏹][⏭][Export]    │
│             │             │             │  [0][1][2][3][4][5][6][7] │
│             │             │             │  ─── a ─── b ─── c ───    │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components to Add

### 1. Sine Wave Display

Shows current phase position on reference sine wave with:
- Full sine wave (0 to 2π) as reference
- Current position marker (green dot)
- Trail of recent positions
- Keyframe markers (0-7 vertical lines)

```python
def update_sine_wave(self):
    self.ax_sine.clear()
    
    # Reference wave
    t = np.linspace(0, 2 * np.pi, 200)
    self.ax_sine.plot(t, np.sin(t), 'w-', alpha=0.3)
    self.ax_sine.fill_between(t, np.sin(t), alpha=0.1, color='cyan')
    
    # Current position
    b = self.state.b % (2 * np.pi)
    self.ax_sine.axvline(x=b, color='lime', linewidth=2)
    self.ax_sine.scatter([b], [np.sin(b)], c='lime', s=150, zorder=5)
    
    # Keyframe markers
    for i in range(8):
        kf_b = (2 * np.pi * i) / 8
        self.ax_sine.axvline(x=kf_b, color='yellow', alpha=0.3, linestyle='--')
        self.ax_sine.text(kf_b, 1.1, str(i), ha='center', color='yellow', fontsize=8)
    
    self.ax_sine.set_xlim([0, 2 * np.pi])
    self.ax_sine.set_ylim([-1.3, 1.4])
    self.ax_sine.set_title(f'Phase: {np.degrees(b):.1f}°')
```

### 2. Pulse (Concurrence) Display

Heartbeat-style meter showing entanglement level:
- Rolling history graph
- Pulsing circle that scales with concurrence
- Glow effect for high values

```python
def update_pulse(self):
    self.ax_pulse.clear()
    concurrence = self.state.concurrence()  # 0 to ~0.5
    
    # History trail
    self.pulse_history.append(concurrence)
    if len(self.pulse_history) > 50:
        self.pulse_history.pop(0)
    
    t = np.arange(len(self.pulse_history))
    self.ax_pulse.fill_between(t, self.pulse_history, alpha=0.3, color='magenta')
    self.ax_pulse.plot(t, self.pulse_history, 'magenta', linewidth=2)
    
    # Pulse circle
    size = 0.3 + concurrence * 0.7
    circle = plt.Circle((len(self.pulse_history), concurrence), size * 0.15,
                        color='magenta', alpha=0.8)
    self.ax_pulse.add_patch(circle)
    
    # Glow rings
    for i in range(3):
        glow = plt.Circle((len(self.pulse_history), concurrence),
                         size * 0.15 * (1.5 + i * 0.5),
                         color='magenta', alpha=0.1/(i+1), fill=False)
        self.ax_pulse.add_patch(glow)
    
    self.ax_pulse.set_xlim([0, 50])
    self.ax_pulse.set_ylim([0, 1])
    self.ax_pulse.set_title(f'Concurrence: {concurrence:.3f}')
```

### 3. Playback Controls

```python
# Button setup
ax_play = fig.add_axes([0.78, 0.12, 0.04, 0.04])
ax_pause = fig.add_axes([0.83, 0.12, 0.04, 0.04])
ax_stop = fig.add_axes([0.88, 0.12, 0.04, 0.04])
ax_step = fig.add_axes([0.78, 0.06, 0.04, 0.04])

btn_play = Button(ax_play, '▶')
btn_pause = Button(ax_pause, '⏸')
btn_stop = Button(ax_stop, '⏹')
btn_step = Button(ax_step, '⏭')

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_stop.on_clicked(on_stop)
btn_step.on_clicked(on_step)

# Animation state
playing = False
animation_speed = 0.05  # radians per frame
anim = None

def on_play(event):
    global playing, anim
    if not playing:
        playing = True
        anim = animation.FuncAnimation(fig, animate_frame, interval=50, blit=False)
        fig.canvas.draw_idle()

def on_pause(event):
    global playing
    playing = False
    if anim:
        anim.event_source.stop()

def on_stop(event):
    global playing
    playing = False
    if anim:
        anim.event_source.stop()
    # Reset to start
    slider_b.set_val(0)
    clear_traces()

def on_step(event):
    global playing
    playing = False
    if anim:
        anim.event_source.stop()
    b = (state.b + animation_speed) % (2 * np.pi)
    slider_b.set_val(b)

def animate_frame(frame):
    if not playing:
        return
    b = (state.b + animation_speed) % (2 * np.pi)
    slider_b.set_val(b)  # Triggers full update
```

### 4. Keyframe Buttons (0-7)

```python
# Create 8 keyframe buttons
keyframe_buttons = []
for i in range(8):
    ax_kf = fig.add_axes([0.78 + i * 0.024, 0.42, 0.022, 0.025])
    btn = Button(ax_kf, str(i))
    btn.on_clicked(lambda event, idx=i: on_keyframe(idx))
    keyframe_buttons.append(btn)

def on_keyframe(idx):
    b = (2 * np.pi * idx) / 8
    slider_b.set_val(b)
```

### 5. Trace History for Bloch Spheres

```python
# In __init__:
self.trace_E = []  # List of (x, y, z) tuples
self.trace_I = []
self.max_trace = 20

# When state changes:
def update_traces(self):
    self.trace_E.append(self.state.to_bloch_E())
    self.trace_I.append(self.state.to_bloch_I())
    
    if len(self.trace_E) > self.max_trace:
        self.trace_E.pop(0)
    if len(self.trace_I) > self.max_trace:
        self.trace_I.pop(0)

# In Bloch sphere drawing:
def draw_bloch_trace(ax, trace, color):
    if len(trace) > 1:
        trace_arr = np.array(trace)
        alphas = np.linspace(0.1, 0.8, len(trace))
        for i in range(len(trace) - 1):
            ax.plot([trace_arr[i, 0], trace_arr[i+1, 0]],
                   [trace_arr[i, 1], trace_arr[i+1, 1]],
                   [trace_arr[i, 2], trace_arr[i+1, 2]],
                   color=color, alpha=alphas[i], linewidth=2)
```

## State Class Update

```python
@dataclass
class PNState:
    a: float = 0.3
    b: float = 0.0
    c: float = 0.3
    
    @property
    def b_deg(self) -> float:
        return np.degrees(self.b)
    
    def to_julia_c(self) -> complex:
        real = -0.4 + 0.3 * np.cos(self.b)
        imag = 0.3 * np.sin(self.b) + 0.1 * (self.a - self.c)
        return complex(real, imag)
    
    def to_bloch_E(self) -> Tuple[float, float, float]:
        theta = np.pi * (1 - self.a)
        phi = self.b
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))
    
    def to_bloch_I(self) -> Tuple[float, float, float]:
        theta = np.pi * (1 - self.c)
        phi = self.b + np.pi / 4
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))
    
    def concurrence(self) -> float:
        """Entanglement measure: peaks at b=π/2, 3π/2."""
        return 0.5 * abs(np.sin(self.b)) * (1 - abs(self.a - self.c))
```

## Complete Figure Layout

```python
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')

# Row 1: Bloch spheres, Julia 2D, Canyon
ax_bloch_E = fig.add_subplot(2, 4, 1, projection='3d')
ax_bloch_I = fig.add_subplot(2, 4, 2, projection='3d')
ax_julia_2d = fig.add_subplot(2, 4, 3)
ax_canyon = fig.add_subplot(2, 4, 4, projection='3d')

# Row 2: Sine, Pulse, Sphere, Controls
ax_sine = fig.add_subplot(2, 4, 5)
ax_pulse = fig.add_subplot(2, 4, 6)
ax_sphere = fig.add_subplot(2, 4, 7, projection='3d')
ax_controls = fig.add_subplot(2, 4, 8)
ax_controls.axis('off')

# Sliders
ax_a = fig.add_axes([0.78, 0.35, 0.15, 0.02])
ax_b = fig.add_axes([0.78, 0.30, 0.15, 0.02])
ax_c = fig.add_axes([0.78, 0.25, 0.15, 0.02])
ax_speed = fig.add_axes([0.78, 0.20, 0.15, 0.02])

slider_a = Slider(ax_a, 'a', 0, 1, valinit=0.3)
slider_b = Slider(ax_b, 'b', 0, 2*np.pi, valinit=0)
slider_c = Slider(ax_c, 'c', 0, 1, valinit=0.3)
slider_speed = Slider(ax_speed, 'Speed', 0.01, 0.2, valinit=0.05)
```

## Main Update Loop

```python
def update_all(self):
    """Called when any parameter changes."""
    self.compute_julia()
    self.update_traces()
    
    # Update all displays
    self.update_bloch_spheres()
    self.update_julia_2d()
    self.update_canyon()
    self.update_sphere()
    self.update_sine_wave()
    self.update_pulse()
    self.update_info_text()
    
    self.fig.canvas.draw_idle()

def on_slider_change(val):
    state.a = slider_a.val
    state.b = slider_b.val
    state.c = slider_c.val
    update_all()
```

## Export Function

```python
def on_export(event):
    output_dir = 'julia_exports'
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = f"julia_b{int(state.b_deg):03d}"
    
    # Canyon mesh
    verts, faces, colors = create_canyon_mesh(julia_2d, bounds)
    export_ply(verts, faces, colors, f"{output_dir}/{prefix}_canyon.ply")
    
    # Sphere mesh
    verts, faces, colors = create_sphere_mesh(julia_2d)
    export_ply(verts, faces, colors, f"{output_dir}/{prefix}_sphere.ply")
    
    # 2D image
    plt.imsave(f"{output_dir}/{prefix}_2d.png", julia_2d, cmap='magma')
    
    print(f"Exported to {output_dir}/{prefix}_*")
```

## Files

The complete standalone explorer is in `julia_animated_explorer.py`. Run it with:

```bash
python julia_animated_explorer.py
```

## Controls Summary

| Button | Action |
|--------|--------|
| ▶ | Start animation (phase cycles 0→2π) |
| ⏸ | Pause animation |
| ⏹ | Stop and reset to b=0 |
| ⏭ | Step forward one frame |
| 0-7 | Jump to keyframe (0°, 45°, 90°, etc.) |
| Export | Save canyon.ply, sphere.ply, 2d.png |

## Visualization Correlation

As animation plays:
1. **Sine wave** shows phase b marching around circle
2. **Bloch spheres** rotate with traces showing path
3. **Julia 2D** morphs through fractal shapes
4. **Canyon** terrain shifts height profile
5. **Sphere** surface deforms
6. **Pulse** shows entanglement (peaks at 90°, 270°)

All driven by the single phase parameter b, demonstrating the correlation between quantum state, fractal geometry, and entanglement.
