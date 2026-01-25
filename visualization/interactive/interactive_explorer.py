"""
Interactive A-Gate Explorer for QDNU.

Provides real-time visualization of quantum neuron activation with:
- Dual Bloch sphere visualization (Excitatory/Inhibitory qubits)
- Julia set fractal fingerprint
- 2D Julia plane, 3D Canyon mesh, Spherical projection
- Entanglement and purity indicators
- Measurement probability distribution
- Export buttons for mesh generation
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime

import sys
from pathlib import Path
# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdnu import create_single_channel_agate as create_single_channel_circuit
from visualization.quantum.agate_visualization import (
    extract_visualization_data,
    draw_bloch_sphere,
    draw_entanglement_indicator,
    draw_probabilities,
)
from visualization.quantum.fractal_generator import generate_fractal_from_state, statevector_to_julia_param, julia_set
from visualization.animation.julia_vis_complete import JuliaVisualizer, apply_colormap


def interactive_agate_explorer():
    """
    Launch interactive explorer with sliders for a, b, c parameters.

    Controls:
        - a slider: Excitatory amplitude [0, 1]
        - b slider: Phase [0, 2*pi]
        - c slider: Inhibitory amplitude [0, 1]
        - Reset button: Return to initial state
        - Export buttons: Save configs and generate meshes
    """
    # Set up figure with dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0A0A1A')
    fig.suptitle('A-Gate Quantum Neuron Explorer with Julia Visualization',
                 fontsize=14, fontweight='bold', color='white')

    # 3x3 Subplots layout:
    # Row 1: Bloch E, Bloch I, 2D Julia
    # Row 2: Canyon 3D, Sphere 3D, Fractal fingerprint
    # Row 3: Entanglement, Probabilities, Params
    ax_E = fig.add_subplot(331, projection='3d')
    ax_I = fig.add_subplot(332, projection='3d')
    ax_julia2d = fig.add_subplot(333)
    ax_canyon = fig.add_subplot(334, projection='3d')
    ax_sphere = fig.add_subplot(335, projection='3d')
    ax_F = fig.add_subplot(336)
    ax_ent = fig.add_subplot(337)
    ax_prob = fig.add_subplot(338)
    ax_params = fig.add_subplot(339)

    # Set backgrounds
    for ax in [ax_E, ax_I, ax_canyon, ax_sphere]:
        ax.set_facecolor('#0A0A1A')
    for ax in [ax_julia2d, ax_F, ax_ent, ax_prob, ax_params]:
        ax.set_facecolor('#1a1a2e')

    # Initial parameters
    init_a, init_b, init_c = 0.5, 1.0, 0.3

    # Initialize Julia visualizer (lower resolution for real-time)
    julia_vis = JuliaVisualizer(resolution=64, bounds=2.0, max_iter=50)

    def update(val=None):
        a = slider_a.val
        b = slider_b.val
        c = slider_c.val

        # Create circuit and extract data
        circuit = create_single_channel_circuit(a, b, c)
        viz = extract_visualization_data(circuit)

        # Update Julia visualizer
        julia_vis.update(a, b, c)

        # Draw Bloch spheres
        draw_bloch_sphere(ax_E, viz['bloch_E'], color='#FF6B35',
                          label=f'E (q0) r={np.linalg.norm(viz["bloch_E"]):.2f}')
        draw_bloch_sphere(ax_I, viz['bloch_I'], color='#7B68EE',
                          label=f'I (q1) r={np.linalg.norm(viz["bloch_I"]):.2f}')

        # Draw 2D Julia (using quantum statevector for richer patterns)
        ax_julia2d.clear()
        ax_julia2d.set_facecolor('#1a1a2e')
        julia_c_quantum = statevector_to_julia_param(viz['statevector'], method='weighted')
        julia_2d_quantum = julia_set(julia_c_quantum, width=150, height=150, max_iter=80)
        ax_julia2d.imshow(julia_2d_quantum, cmap='magma', origin='lower',
                         extent=[-2, 2, -2, 2])
        ax_julia2d.set_title(f'2D Julia (c={julia_c_quantum.real:.2f}{julia_c_quantum.imag:+.2f}i)', fontsize=10)
        ax_julia2d.set_xlabel('Re(z)')
        ax_julia2d.set_ylabel('Im(z)')

        # Draw 3D Canyon (using quantum-derived Julia)
        ax_canyon.clear()
        ax_canyon.set_facecolor('#0A0A1A')
        julia_norm = julia_2d_quantum / julia_2d_quantum.max() if julia_2d_quantum.max() > 0 else julia_2d_quantum
        step = max(1, 150 // 24)
        ny, nx = julia_norm.shape
        x = np.linspace(-2, 2, nx)[::step]
        y = np.linspace(-2, 2, ny)[::step]
        X, Y = np.meshgrid(x, y)
        Z = (1 - julia_norm[::step, ::step]) * 1.5
        ax_canyon.plot_surface(X, Y, Z, cmap='inferno', alpha=0.9,
                              linewidth=0, antialiased=True)
        ax_canyon.set_title('Canyon (Height Map)', fontsize=10)
        ax_canyon.set_xlabel('X')
        ax_canyon.set_ylabel('Y')
        ax_canyon.set_zlabel('H')
        ax_canyon.view_init(elev=35, azim=45)

        # Draw Spherical projection (using quantum-derived Julia)
        ax_sphere.clear()
        ax_sphere.set_facecolor('#0A0A1A')
        step = max(1, 150 // 20)
        ny, nx = julia_norm.shape
        theta = np.linspace(0.1, np.pi - 0.1, ny)[::step]
        phi = np.linspace(0, 2 * np.pi, nx)[::step]
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        julia_sub = julia_norm[::step, ::step]
        R = 1.0 + (1 - julia_sub) * 0.3
        Xs = R * np.sin(THETA) * np.cos(PHI)
        Ys = R * np.sin(THETA) * np.sin(PHI)
        Zs = R * np.cos(THETA)
        ax_sphere.plot_surface(Xs, Ys, Zs, facecolors=plt.cm.plasma(julia_sub),
                              alpha=0.9, linewidth=0)
        ax_sphere.set_title('Spherical Projection', fontsize=10)
        ax_sphere.set_box_aspect([1, 1, 1])
        ax_sphere.view_init(elev=20, azim=45)

        # Draw fractal fingerprint (original)
        ax_F.clear()
        ax_F.set_facecolor('#1a1a2e')
        fractal = generate_fractal_from_state(viz['statevector'], width=150, height=150, max_iter=60)
        ax_F.imshow(fractal, cmap='magma', origin='lower', aspect='equal')
        ax_F.set_title('Fractal Fingerprint', fontsize=10)
        ax_F.axis('off')

        # Draw entanglement/purity
        draw_entanglement_indicator(ax_ent, viz['concurrence'], viz['purity_E'], viz['purity_I'])
        ax_ent.set_facecolor('#1a1a2e')

        # Draw probabilities
        draw_probabilities(ax_prob, viz['probabilities'])
        ax_prob.set_facecolor('#1a1a2e')

        # Draw parameter values
        ax_params.clear()
        ax_params.set_facecolor('#1a1a2e')
        julia_c = julia_vis.state.to_julia_c()
        param_text = f"""PN Parameters:
  a = {a:.4f}
  b = {b:.4f} ({np.degrees(b):.1f} deg)
  c = {c:.4f}

Julia c = {julia_c.real:.3f}{julia_c.imag:+.3f}i

Bloch E: ({viz['bloch_E'][0]:+.2f}, {viz['bloch_E'][1]:+.2f}, {viz['bloch_E'][2]:+.2f})
Bloch I: ({viz['bloch_I'][0]:+.2f}, {viz['bloch_I'][1]:+.2f}, {viz['bloch_I'][2]:+.2f})

Concurrence: {viz['concurrence']:.4f}
Purity E:    {viz['purity_E']:.4f}
Purity I:    {viz['purity_I']:.4f}"""
        ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       color='white')
        ax_params.axis('off')
        ax_params.set_title('State Info', fontsize=10)

        fig.canvas.draw_idle()

    # Sliders
    slider_ax_a = plt.axes([0.12, 0.05, 0.22, 0.015])
    slider_ax_b = plt.axes([0.42, 0.05, 0.22, 0.015])
    slider_ax_c = plt.axes([0.72, 0.05, 0.22, 0.015])

    slider_a = Slider(slider_ax_a, 'a (E)', 0.0, 1.0, valinit=init_a, color='#FF6B35')
    slider_b = Slider(slider_ax_b, 'b (phi)', 0.0, 2*np.pi, valinit=init_b, color='#00D084')
    slider_c = Slider(slider_ax_c, 'c (I)', 0.0, 1.0, valinit=init_c, color='#7B68EE')

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_c.on_changed(update)

    # Reset button
    reset_ax = plt.axes([0.02, 0.015, 0.06, 0.025])
    reset_btn = Button(reset_ax, 'Reset', color='#2a2a4a', hovercolor='#3a3a5a')

    def reset(event):
        slider_a.reset()
        slider_b.reset()
        slider_c.reset()

    reset_btn.on_clicked(reset)

    # Export button - saves current params for Julia surface generator
    export_ax = plt.axes([0.09, 0.015, 0.08, 0.025])
    export_btn = Button(export_ax, 'Export Config', color='#1a4a2a', hovercolor='#2a5a3a')

    # Store for exported configs
    exported_configs = []

    def export_config(event):
        a = slider_a.val
        b = slider_b.val
        c = slider_c.val

        # Get current visualization data
        circuit = create_single_channel_circuit(a, b, c)
        viz = extract_visualization_data(circuit)

        config = {
            'a': float(a),
            'b': float(b),
            'b_deg': float(np.degrees(b)),
            'c': float(c),
            'concurrence': float(viz['concurrence']),
            'purity_E': float(viz['purity_E']),
            'purity_I': float(viz['purity_I']),
            'timestamp': datetime.now().isoformat()
        }

        exported_configs.append(config)

        # Print Python code ready to use
        print("\n" + "=" * 60)
        print("EXPORTED CONFIG FOR JULIA VISUALIZATION")
        print("=" * 60)
        print(f"\n# Parameters: a={a:.4f}, b={b:.4f} ({np.degrees(b):.1f} deg), c={c:.4f}")
        julia_c = julia_vis.state.to_julia_c()
        print(f"# Julia c = {julia_c}")
        print(f"# Concurrence: {viz['concurrence']:.4f}")
        print()
        print("# Using JuliaVisualizer (canyon/sphere/plane)")
        print("from visualization.animation import JuliaVisualizer")
        print(f"vis = JuliaVisualizer(resolution=128)")
        print(f"vis.update(a={a:.4f}, b={b:.4f}, c={c:.4f})")
        print(f"vis.export_all('julia_output/', prefix='julia_b{int(np.degrees(b))}', fmt='ply')")
        print()

        # Save to JSON file
        export_dir = Path(__file__).parent.parent.parent / 'research'
        export_dir.mkdir(exist_ok=True)
        export_file = export_dir / 'explorer_exports.json'

        # Load existing or create new
        if export_file.exists():
            with open(export_file, 'r') as f:
                all_exports = json.load(f)
        else:
            all_exports = []

        all_exports.append(config)

        with open(export_file, 'w') as f:
            json.dump(all_exports, f, indent=2)

        print(f"Config saved to: {export_file}")
        print(f"Total saved configs: {len(all_exports)}")
        print("=" * 60)

    export_btn.on_clicked(export_config)

    # Export All Meshes button - exports plane, canyon, sphere
    mesh_ax = plt.axes([0.18, 0.015, 0.1, 0.025])
    mesh_btn = Button(mesh_ax, 'Export Meshes', color='#4a1a4a', hovercolor='#5a2a5a')

    def export_meshes(event):
        a = slider_a.val
        b = slider_b.val
        c = slider_c.val

        print(f"\nExporting meshes for a={a:.4f}, b={np.degrees(b):.1f} deg, c={c:.4f}...")

        # Create output directory
        output_dir = Path(__file__).parent.parent.parent / 'research' / 'julia_meshes'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate prefix with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"julia_b{int(np.degrees(b))}_{timestamp}"

        # Use higher resolution for export
        export_vis = JuliaVisualizer(resolution=128, bounds=2.0, max_iter=100)
        export_vis.update(a, b, c)
        export_vis.export_all(str(output_dir), prefix=prefix, fmt='ply')

        print(f"\nExported to: {output_dir}/")
        print(f"  {prefix}_plane.ply   - 2D Julia plane")
        print(f"  {prefix}_canyon.ply  - 3D height-mapped terrain")
        print(f"  {prefix}_sphere.ply  - Spherical projection")
        print(f"  {prefix}_2d.png      - 2D Julia image")
        print(f"  {prefix}_state.txt   - State parameters")

    mesh_btn.on_clicked(export_meshes)

    # Animation export button
    anim_ax = plt.axes([0.29, 0.015, 0.1, 0.025])
    anim_btn = Button(anim_ax, 'Export Anim', color='#1a3a4a', hovercolor='#2a4a5a')

    def export_animation(event):
        a = slider_a.val
        c = slider_c.val
        n_frames = 24  # One full cycle

        print(f"\nExporting {n_frames}-frame animation (a={a:.4f}, c={c:.4f})...")

        output_dir = Path(__file__).parent.parent.parent / 'research' / 'julia_animation'
        output_dir.mkdir(parents=True, exist_ok=True)

        export_vis = JuliaVisualizer(resolution=96, bounds=2.0, max_iter=80)
        export_vis.state.a = a
        export_vis.state.c = c
        export_vis.export_animation(str(output_dir), n_frames=n_frames, fmt='ply')

        print(f"\nAnimation exported to: {output_dir}/")
        print(f"Import into Blender with Stop Motion OBJ addon")

    anim_btn.on_clicked(export_animation)

    # Spherical Julia export button (stereographic projection)
    sphere_ax = plt.axes([0.40, 0.015, 0.1, 0.025])
    sphere_btn = Button(sphere_ax, 'Sphere PLY', color='#2a4a1a', hovercolor='#3a5a2a')

    def export_spherical(event):
        from visualization.animation import SphericalJulia

        a = slider_a.val
        b = slider_b.val
        c = slider_c.val

        print(f"\nExporting spherical Julia (stereographic)...")

        # Create output directory
        output_dir = Path(__file__).parent.parent.parent / 'research' / 'julia_spheres'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create spherical Julia with PN parameters
        sj = SphericalJulia(resolution=128, canyon_depth=0.3)
        sj.update(a=a, b=b, pn_c=c)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = output_dir / f"julia_sphere_b{int(np.degrees(b))}_{timestamp}.ply"

        sj.export(str(filepath))

        print(f"Exported: {filepath}")
        print(f"  Julia c = {sj.c}")
        print(f"  Resolution: 128x128 ({128*128} vertices)")
        print(f"  Import into Blender for spherical harmonic visualization")

    sphere_btn.on_clicked(export_spherical)

    # Initial draw
    update()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09, top=0.94, left=0.04, right=0.98, hspace=0.25, wspace=0.2)
    plt.show()


def animate_parameter_sweep(param: str = 'a', frames: int = 100, save_path: str = None):
    """
    Create animation of parameter sweep.

    Args:
        param: 'a', 'b', or 'c'
        frames: Number of frames
        save_path: Optional path to save animation as GIF
    """
    from matplotlib.animation import FuncAnimation

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0A0A1A')
    fig.suptitle(f'Parameter Sweep: {param}', fontsize=14, fontweight='bold')

    ax_E = fig.add_subplot(131, projection='3d')
    ax_I = fig.add_subplot(132, projection='3d')
    ax_F = fig.add_subplot(133)

    for ax in [ax_E, ax_I]:
        ax.set_facecolor('#0A0A1A')
    ax_F.set_facecolor('#1a1a2e')

    # Parameter sweep values
    if param == 'a':
        values = np.linspace(0, 1, frames)
        fixed = {'b': np.pi/2, 'c': 0.3}
    elif param == 'b':
        values = np.linspace(0, 2*np.pi, frames)
        fixed = {'a': 0.5, 'c': 0.3}
    else:  # c
        values = np.linspace(0, 1, frames)
        fixed = {'a': 0.5, 'b': np.pi/2}

    def animate(i):
        params = fixed.copy()
        params[param] = values[i]

        circuit = create_single_channel_circuit(**params)
        viz = extract_visualization_data(circuit)

        draw_bloch_sphere(ax_E, viz['bloch_E'], color='#FF6B35',
                          label=f'E (q0) ent={viz["concurrence"]:.2f}')
        draw_bloch_sphere(ax_I, viz['bloch_I'], color='#7B68EE',
                          label=f'I (q1)')

        ax_F.clear()
        ax_F.set_facecolor('#1a1a2e')
        fractal = generate_fractal_from_state(viz['statevector'], width=150, height=150)
        ax_F.imshow(fractal, cmap='magma', origin='lower')
        ax_F.set_title(f'{param} = {values[i]:.3f}', fontsize=12)
        ax_F.axis('off')

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to: {save_path}")

    plt.tight_layout()
    plt.show()

    return anim


def visualize_eeg_activation(eeg_signal: np.ndarray, lambda_a: float = 0.1,
                             lambda_c: float = 0.05, dt: float = 0.001,
                             sample_every: int = 10):
    """
    Visualize A-Gate activation through an EEG signal.

    Args:
        eeg_signal: Normalized EEG signal [0, 1]
        lambda_a: Excitatory decay rate
        lambda_c: Inhibitory decay rate
        dt: Timestep
        sample_every: Sample frames every N steps
    """
    from pn_dynamics import PNDynamics

    # Initialize dynamics
    pn = PNDynamics(lambda_a=lambda_a, lambda_c=lambda_c, saturation_mode='symmetric')

    # Collect frames
    frames_data = []
    for i, f_t in enumerate(eeg_signal):
        pn.step(f_t, dt=dt)
        if i % sample_every == 0:
            a, b, c = pn.get_params()
            circuit = create_single_channel_circuit(a, b, c)
            viz = extract_visualization_data(circuit)
            viz['a'] = a
            viz['b'] = b
            viz['c'] = c
            viz['f'] = f_t
            viz['time'] = i * dt
            frames_data.append(viz)

    print(f"Generated {len(frames_data)} visualization frames")

    # Animate
    from matplotlib.animation import FuncAnimation

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#0A0A1A')

    ax_E = fig.add_subplot(231, projection='3d')
    ax_I = fig.add_subplot(232, projection='3d')
    ax_F = fig.add_subplot(233)
    ax_signal = fig.add_subplot(234)
    ax_params = fig.add_subplot(235)
    ax_ent = fig.add_subplot(236)

    # Pre-compute signal plot data
    times = [f['time'] for f in frames_data]
    a_vals = [f['a'] for f in frames_data]
    c_vals = [f['c'] for f in frames_data]
    ent_vals = [f['concurrence'] for f in frames_data]

    def animate(i):
        viz = frames_data[i]

        # Bloch spheres
        draw_bloch_sphere(ax_E, viz['bloch_E'], color='#FF6B35', label='E (q0)')
        draw_bloch_sphere(ax_I, viz['bloch_I'], color='#7B68EE', label='I (q1)')

        # Fractal
        ax_F.clear()
        ax_F.set_facecolor('#1a1a2e')
        fractal = generate_fractal_from_state(viz['statevector'], width=150, height=150, max_iter=60)
        ax_F.imshow(fractal, cmap='magma', origin='lower')
        ax_F.set_title(f't = {viz["time"]:.3f}s', fontsize=11)
        ax_F.axis('off')

        # Signal and parameters
        ax_signal.clear()
        ax_signal.set_facecolor('#1a1a2e')
        ax_signal.plot(times[:i+1], a_vals[:i+1], 'r-', label='a (E)', linewidth=1.5)
        ax_signal.plot(times[:i+1], c_vals[:i+1], 'b-', label='a (I)', linewidth=1.5)
        ax_signal.axvline(viz['time'], color='yellow', linestyle='--', alpha=0.7)
        ax_signal.set_xlim(0, times[-1])
        ax_signal.set_ylim(0, 1)
        ax_signal.set_xlabel('Time (s)')
        ax_signal.set_ylabel('Parameter')
        ax_signal.legend(loc='upper right', fontsize=8)
        ax_signal.set_title('PN Parameters', fontsize=11)

        # Parameter text
        ax_params.clear()
        ax_params.set_facecolor('#1a1a2e')
        text = f"a = {viz['a']:.4f}\nb = {viz['b']:.4f}\nc = {viz['c']:.4f}\n\nConcurrence: {viz['concurrence']:.4f}"
        ax_params.text(0.5, 0.5, text, transform=ax_params.transAxes,
                      fontsize=12, ha='center', va='center', fontfamily='monospace')
        ax_params.axis('off')
        ax_params.set_title('Current State', fontsize=11)

        # Entanglement history
        ax_ent.clear()
        ax_ent.set_facecolor('#1a1a2e')
        ax_ent.fill_between(times[:i+1], ent_vals[:i+1], color='magenta', alpha=0.5)
        ax_ent.plot(times[:i+1], ent_vals[:i+1], 'magenta', linewidth=1.5)
        ax_ent.set_xlim(0, times[-1])
        ax_ent.set_ylim(0, 1)
        ax_ent.set_xlabel('Time (s)')
        ax_ent.set_ylabel('Concurrence')
        ax_ent.set_title('Entanglement', fontsize=11)

        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_data), interval=100, blit=False)
    plt.tight_layout()
    plt.show()

    return anim


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'animate':
            param = sys.argv[2] if len(sys.argv) > 2 else 'a'
            animate_parameter_sweep(param)
        elif sys.argv[1] == 'eeg':
            # Demo with synthetic EEG-like signal
            t = np.linspace(0, 5, 5000)
            signal = 0.5 + 0.3 * np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
            signal = np.clip(signal, 0, 1)
            visualize_eeg_activation(signal, sample_every=50)
    else:
        interactive_agate_explorer()
