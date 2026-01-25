"""
Interactive A-Gate Explorer for QDNU.

Provides real-time visualization of quantum neuron activation with:
- Dual Bloch sphere visualization (Excitatory/Inhibitory qubits)
- Julia set fractal fingerprint
- Entanglement and purity indicators
- Measurement probability distribution
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_agate import create_single_channel_agate as create_single_channel_circuit
from visualization.agate_visualization import (
    extract_visualization_data,
    draw_bloch_sphere,
    draw_entanglement_indicator,
    draw_probabilities,
)
from visualization.fractal_generator import generate_fractal_from_state


def interactive_agate_explorer():
    """
    Launch interactive explorer with sliders for a, b, c parameters.

    Controls:
        - a slider: Excitatory amplitude [0, 1]
        - b slider: Phase [0, 2*pi]
        - c slider: Inhibitory amplitude [0, 1]
        - Reset button: Return to initial state
    """
    # Set up figure with dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0A0A1A')
    fig.suptitle('A-Gate Quantum Neuron Explorer', fontsize=14, fontweight='bold', color='white')

    # Subplots layout: 2 Bloch spheres + fractal on top, properties below
    ax_E = fig.add_subplot(231, projection='3d')
    ax_I = fig.add_subplot(232, projection='3d')
    ax_F = fig.add_subplot(233)
    ax_ent = fig.add_subplot(234)
    ax_prob = fig.add_subplot(235)
    ax_params = fig.add_subplot(236)

    # Set backgrounds
    for ax in [ax_E, ax_I]:
        ax.set_facecolor('#0A0A1A')
    for ax in [ax_F, ax_ent, ax_prob, ax_params]:
        ax.set_facecolor('#1a1a2e')

    # Initial parameters
    init_a, init_b, init_c = 0.5, 1.0, 0.3

    def update(val=None):
        a = slider_a.val
        b = slider_b.val
        c = slider_c.val

        # Create circuit and extract data
        circuit = create_single_channel_circuit(a, b, c)
        viz = extract_visualization_data(circuit)

        # Draw Bloch spheres
        draw_bloch_sphere(ax_E, viz['bloch_E'], color='#FF6B35',
                          label=f'E (q0) r={np.linalg.norm(viz["bloch_E"]):.2f}')
        draw_bloch_sphere(ax_I, viz['bloch_I'], color='#7B68EE',
                          label=f'I (q1) r={np.linalg.norm(viz["bloch_I"]):.2f}')

        # Draw fractal
        ax_F.clear()
        ax_F.set_facecolor('#1a1a2e')
        fractal = generate_fractal_from_state(viz['statevector'], width=200, height=200, max_iter=80)
        ax_F.imshow(fractal, cmap='magma', origin='lower', aspect='equal')
        ax_F.set_title(f'Julia Set Fingerprint', fontsize=11)
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
        param_text = f"""
PN Neuron Parameters:
  a (Excitatory) = {a:.4f}
  b (Phase)      = {b:.4f} ({np.degrees(b):.1f} deg)
  c (Inhibitory) = {c:.4f}

Bloch Vectors:
  E: ({viz['bloch_E'][0]:+.3f}, {viz['bloch_E'][1]:+.3f}, {viz['bloch_E'][2]:+.3f})
  I: ({viz['bloch_I'][0]:+.3f}, {viz['bloch_I'][1]:+.3f}, {viz['bloch_I'][2]:+.3f})

Quantum Properties:
  Concurrence:  {viz['concurrence']:.4f}
  Purity E:     {viz['purity_E']:.4f}
  Purity I:     {viz['purity_I']:.4f}
  Global Phase: {viz['global_phase']:.4f}
"""
        ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       color='white')
        ax_params.axis('off')
        ax_params.set_title('State Info', fontsize=11)

        fig.canvas.draw_idle()

    # Sliders
    slider_ax_a = plt.axes([0.15, 0.06, 0.2, 0.02])
    slider_ax_b = plt.axes([0.45, 0.06, 0.2, 0.02])
    slider_ax_c = plt.axes([0.75, 0.06, 0.2, 0.02])

    slider_a = Slider(slider_ax_a, 'a (E)', 0.0, 1.0, valinit=init_a, color='#FF6B35')
    slider_b = Slider(slider_ax_b, 'b (phi)', 0.0, 2*np.pi, valinit=init_b, color='#00D084')
    slider_c = Slider(slider_ax_c, 'c (I)', 0.0, 1.0, valinit=init_c, color='#7B68EE')

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_c.on_changed(update)

    # Reset button
    reset_ax = plt.axes([0.02, 0.02, 0.08, 0.03])
    reset_btn = Button(reset_ax, 'Reset', color='#2a2a4a', hovercolor='#3a3a5a')

    def reset(event):
        slider_a.reset()
        slider_b.reset()
        slider_c.reset()

    reset_btn.on_clicked(reset)

    # Initial draw
    update()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)
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
