"""
EEG Seizure Visualization with Quantum A-Gate Activation.

Integrates with QDNU seizure prediction pipeline to visualize:
- Real EEG data from Kaggle dataset
- Quantum neuron activation through A-Gate circuits
- Ictal vs Interictal state comparison
- Fidelity-based classification with Bloch sphere trajectories
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_loader import load_for_qdnu
from pn_dynamics import PNDynamics
from quantum_agate import create_single_channel_agate as create_single_channel_circuit
from visualization.agate_visualization import extract_visualization_data, draw_bloch_sphere
from visualization.fractal_generator import generate_fractal_from_state


class EEGQuantumVisualizer:
    """
    Visualizes EEG signals through quantum A-Gate neuron activation.

    Combines PN dynamics evolution with Bloch sphere visualization
    to show how seizure (ictal) and normal (interictal) EEG patterns
    produce distinct quantum states.
    """

    def __init__(self, num_channels: int = 4, lambda_a: float = 0.1,
                 lambda_c: float = 0.05, saturation_mode: str = 'symmetric'):
        self.num_channels = num_channels
        self.lambda_a = lambda_a
        self.lambda_c = lambda_c
        self.saturation_mode = saturation_mode

        # Initialize single PN dynamics processor
        self.pn = PNDynamics(
            lambda_a=lambda_a,
            lambda_c=lambda_c,
            saturation_mode=saturation_mode
        )

    def process_window(self, eeg_window: np.ndarray) -> dict:
        """
        Process an EEG window and extract quantum visualization data.

        Args:
            eeg_window: Shape (num_channels, window_size)

        Returns:
            dict with final params, circuits, and visualization data per channel
        """
        # Use the multichannel evolution method from QDNU's PNDynamics
        channel_params = self.pn.evolve_multichannel(eeg_window)

        # Extract quantum data for each channel
        results = {'channels': []}
        for ch, (a, b, c) in enumerate(channel_params):
            circuit = create_single_channel_circuit(a, b, c)
            viz = extract_visualization_data(circuit)

            results['channels'].append({
                'a': a, 'b': b, 'c': c,
                'bloch_E': viz['bloch_E'],
                'bloch_I': viz['bloch_I'],
                'concurrence': viz['concurrence'],
                'purity_E': viz['purity_E'],
                'purity_I': viz['purity_I'],
                'statevector': viz['statevector'],
                'probabilities': viz['probabilities'],
            })

        return results

    def compare_ictal_interictal(self, subject: str = 'Dog_1',
                                  n_ictal: int = 5, n_interictal: int = 5,
                                  window_size: int = 500):
        """
        Load and compare ictal vs interictal EEG windows.

        Creates visualization showing quantum state differences.
        """
        print(f"Loading {subject} EEG data...")
        ictal_windows, interictal_windows = load_for_qdnu(
            subject, num_channels=self.num_channels,
            window_size=window_size, n_ictal=n_ictal, n_interictal=n_interictal
        )

        # Process windows
        print("Processing ictal windows...")
        ictal_results = [self.process_window(w) for w in ictal_windows]
        print("Processing interictal windows...")
        interictal_results = [self.process_window(w) for w in interictal_windows]

        # Extract metrics for comparison
        ictal_entanglement = [np.mean([ch['concurrence'] for ch in r['channels']])
                              for r in ictal_results]
        interictal_entanglement = [np.mean([ch['concurrence'] for ch in r['channels']])
                                   for r in interictal_results]

        ictal_a = [np.mean([ch['a'] for ch in r['channels']]) for r in ictal_results]
        interictal_a = [np.mean([ch['a'] for ch in r['channels']]) for r in interictal_results]

        ictal_c = [np.mean([ch['c'] for ch in r['channels']]) for r in ictal_results]
        interictal_c = [np.mean([ch['c'] for ch in r['channels']]) for r in interictal_results]

        print(f"Ictal: a={np.mean(ictal_a):.4f}, c={np.mean(ictal_c):.4f}, ent={np.mean(ictal_entanglement):.4f}")
        print(f"Interictal: a={np.mean(interictal_a):.4f}, c={np.mean(interictal_c):.4f}, ent={np.mean(interictal_entanglement):.4f}")

        # Create comparison figure
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0A0A1A')
        fig.suptitle(f'Quantum A-Gate Activation: Ictal vs Interictal ({subject})',
                    fontsize=14, fontweight='bold')

        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Bloch spheres for first ictal and interictal
        ax_ictal_E = fig.add_subplot(gs[0, 0], projection='3d')
        ax_ictal_I = fig.add_subplot(gs[0, 1], projection='3d')
        ax_inter_E = fig.add_subplot(gs[0, 2], projection='3d')
        ax_inter_I = fig.add_subplot(gs[0, 3], projection='3d')

        # Draw Bloch spheres for channel 0
        ictal_ch0 = ictal_results[0]['channels'][0]
        inter_ch0 = interictal_results[0]['channels'][0]

        draw_bloch_sphere(ax_ictal_E, ictal_ch0['bloch_E'], color='#E74C3C',
                          label=f'Ictal E (ent={ictal_ch0["concurrence"]:.3f})')
        draw_bloch_sphere(ax_ictal_I, ictal_ch0['bloch_I'], color='#C0392B',
                          label=f'Ictal I')
        draw_bloch_sphere(ax_inter_E, inter_ch0['bloch_E'], color='#3498DB',
                          label=f'Interictal E (ent={inter_ch0["concurrence"]:.3f})')
        draw_bloch_sphere(ax_inter_I, inter_ch0['bloch_I'], color='#2980B9',
                          label=f'Interictal I')

        # Row 2: Fractals and parameter distributions
        ax_frac_ictal = fig.add_subplot(gs[1, 0])
        ax_frac_inter = fig.add_subplot(gs[1, 1])
        ax_params = fig.add_subplot(gs[1, 2])
        ax_ent = fig.add_subplot(gs[1, 3])

        # Fractals
        frac_ictal = generate_fractal_from_state(ictal_ch0['statevector'], width=200, height=200)
        frac_inter = generate_fractal_from_state(inter_ch0['statevector'], width=200, height=200)

        ax_frac_ictal.imshow(frac_ictal, cmap='hot', origin='lower')
        ax_frac_ictal.set_title('Ictal Fractal', fontsize=11)
        ax_frac_ictal.axis('off')

        ax_frac_inter.imshow(frac_inter, cmap='cool', origin='lower')
        ax_frac_inter.set_title('Interictal Fractal', fontsize=11)
        ax_frac_inter.axis('off')

        # Parameter comparison
        x = np.arange(2)
        width = 0.35
        ax_params.bar(x - width/2, [np.mean(ictal_a), np.mean(ictal_c)],
                     width, label='Ictal', color='#E74C3C', alpha=0.8)
        ax_params.bar(x + width/2, [np.mean(interictal_a), np.mean(interictal_c)],
                     width, label='Interictal', color='#3498DB', alpha=0.8)
        ax_params.set_xticks(x)
        ax_params.set_xticklabels(['a (E)', 'c (I)'])
        ax_params.set_ylabel('Mean Value')
        ax_params.set_title('PN Parameters', fontsize=11)
        ax_params.legend(fontsize=9)
        ax_params.set_facecolor('#1a1a2e')

        # Entanglement distribution
        ax_ent.boxplot([ictal_entanglement, interictal_entanglement],
                      labels=['Ictal', 'Interictal'],
                      patch_artist=True,
                      boxprops=dict(facecolor='#2a2a4a'))
        ax_ent.set_ylabel('Mean Concurrence')
        ax_ent.set_title('Entanglement Distribution', fontsize=11)
        ax_ent.set_facecolor('#1a1a2e')

        # Row 3: Multi-window trajectories
        ax_traj = fig.add_subplot(gs[2, :2])
        ax_probs = fig.add_subplot(gs[2, 2:])

        # a-c trajectory
        ax_traj.scatter(ictal_a, ictal_c, c='#E74C3C', s=100, alpha=0.7, label='Ictal', edgecolors='white')
        ax_traj.scatter(interictal_a, interictal_c, c='#3498DB', s=100, alpha=0.7, label='Interictal', edgecolors='white')
        ax_traj.set_xlabel('a (Excitatory)')
        ax_traj.set_ylabel('c (Inhibitory)')
        ax_traj.set_title('Parameter Space Trajectory', fontsize=11)
        ax_traj.legend(fontsize=10)
        ax_traj.set_facecolor('#1a1a2e')

        # Probability comparison
        states = ['|00>', '|01>', '|10>', '|11>']
        ictal_probs = np.mean([r['channels'][0]['probabilities'] for r in ictal_results], axis=0)
        inter_probs = np.mean([r['channels'][0]['probabilities'] for r in interictal_results], axis=0)

        x = np.arange(4)
        ax_probs.bar(x - width/2, ictal_probs, width, label='Ictal', color='#E74C3C', alpha=0.8)
        ax_probs.bar(x + width/2, inter_probs, width, label='Interictal', color='#3498DB', alpha=0.8)
        ax_probs.set_xticks(x)
        ax_probs.set_xticklabels(states)
        ax_probs.set_ylabel('Mean Probability')
        ax_probs.set_title('Measurement Probabilities', fontsize=11)
        ax_probs.legend(fontsize=9)
        ax_probs.set_facecolor('#1a1a2e')

        plt.tight_layout()
        output_path = Path(__file__).parent.parent / 'figures' / 'eeg_quantum_comparison.png'
        plt.savefig(output_path, dpi=200, facecolor='#0A0A1A', edgecolor='none')
        print(f"Saved: {output_path}")
        plt.show()

        return {
            'ictal': ictal_results,
            'interictal': interictal_results,
            'ictal_entanglement': ictal_entanglement,
            'interictal_entanglement': interictal_entanglement,
        }


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Compare ictal vs interictal
        viz = EEGQuantumVisualizer(num_channels=4)
        viz.compare_ictal_interictal(subject='Dog_1', n_ictal=10, n_interictal=10)

    else:
        print("Usage:")
        print("  python eeg_visualizer.py compare  - Compare ictal vs interictal quantum states")
