"""
QDNU Performance Dashboard

Visualizes quantum seizure prediction results with real EEG data.
Generates publication-quality figures with mathematical annotations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path

from eeg_loader import load_for_qdnu, list_available_subjects
from template_trainer import TemplateTrainer
from seizure_predictor import SeizurePredictor
from pn_dynamics import PNDynamics


def run_validation(subject='Dog_1', num_channels=4, n_samples=20):
    """
    Run QDNU validation on real EEG data.

    Returns dict with all results for plotting.
    """
    np.random.seed(42)  # Ensure reproducible results
    print(f"Loading data for {subject}...")
    ictal_windows, interictal_windows = load_for_qdnu(
        subject,
        num_channels=num_channels,
        window_size=500,
        n_ictal=n_samples,
        n_interictal=n_samples
    )

    # Train on first ictal sample (seizure template)
    print("Training template on ictal data...")
    trainer = TemplateTrainer(
        num_channels=num_channels,
        lambda_a=0.1,
        lambda_c=0.05,
        saturation_mode='symmetric'  # Paper: symmetric PN dynamics
        # Note: Uses default dt=0.01 for proper parameter evolution
    )
    trainer.train(ictal_windows[0])

    # Collect PN parameters for all samples
    pn = PNDynamics(lambda_a=0.1, lambda_c=0.05, saturation_mode='symmetric')  # Uses default dt=0.01

    ictal_params = []
    interictal_params = []

    for w in ictal_windows:
        normalized = trainer.normalize_eeg(w)
        params = pn.evolve_multichannel(normalized)
        ictal_params.append(params)

    for w in interictal_windows:
        normalized = trainer.normalize_eeg(w)
        params = pn.evolve_multichannel(normalized)
        interictal_params.append(params)

    # Run predictions
    print("Running predictions...")
    predictor = SeizurePredictor(trainer, threshold=0.82)  # Paper: optimized threshold

    ictal_fidelities = []
    interictal_fidelities = []

    for w in ictal_windows[1:]:  # Skip training sample
        _, fid, _ = predictor.predict(w)
        ictal_fidelities.append(fid)

    for w in interictal_windows:
        _, fid, _ = predictor.predict(w)
        interictal_fidelities.append(fid)

    # Evaluation metrics
    test_data = ictal_windows[1:] + interictal_windows
    labels = [1] * (len(ictal_windows) - 1) + [0] * len(interictal_windows)
    eval_metrics = predictor.evaluate(test_data, labels)

    return {
        'subject': subject,
        'num_channels': num_channels,
        'ictal_params': ictal_params,
        'interictal_params': interictal_params,
        'ictal_fidelities': ictal_fidelities,
        'interictal_fidelities': interictal_fidelities,
        'template_params': trainer.template_params,
        'circuit_qubits': trainer.template_circuit.num_qubits,
        'circuit_depth': trainer.template_circuit.depth(),
        'circuit_gates': trainer.template_circuit.size(),
        'eval_metrics': eval_metrics
    }


def create_dashboard(results, output_path='qdnu_dashboard.png'):
    """
    Create 5-panel dashboard figure.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"QDNU Quantum Seizure Prediction Dashboard - {results['subject']}",
                 fontsize=14, fontweight='bold')

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # === Panel 1: Fidelity Distribution ===
    ax1 = fig.add_subplot(gs[0, 0])

    ictal_fid = results['ictal_fidelities']
    inter_fid = results['interictal_fidelities']

    bins = np.linspace(
        min(min(ictal_fid), min(inter_fid)) - 0.01,
        max(max(ictal_fid), max(inter_fid)) + 0.01,
        20
    )

    ax1.hist(ictal_fid, bins=bins, alpha=0.7, label='Ictal', color='red', edgecolor='darkred')
    ax1.hist(inter_fid, bins=bins, alpha=0.7, label='Interictal', color='blue', edgecolor='darkblue')
    ax1.axvline(0.82, color='black', linestyle='--', label='Threshold (0.82)')
    ax1.set_xlabel('Fidelity F')
    ax1.set_ylabel('Count')
    ax1.set_title('Quantum Fidelity Distribution')
    ax1.legend(fontsize=8)

    # Math annotation
    ax1.text(0.02, 0.98, r'$F = |\langle\psi_{template}|\psi_{test}\rangle|^2$',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === Panel 2: PN Parameter Space ===
    ax2 = fig.add_subplot(gs[0, 1])

    # Extract mean (a, c) for each sample
    ictal_a = [np.mean([p[0] for p in params]) for params in results['ictal_params']]
    ictal_c = [np.mean([p[2] for p in params]) for params in results['ictal_params']]
    inter_a = [np.mean([p[0] for p in params]) for params in results['interictal_params']]
    inter_c = [np.mean([p[2] for p in params]) for params in results['interictal_params']]

    ax2.scatter(ictal_a, ictal_c, c='red', alpha=0.7, label='Ictal', s=50, edgecolors='darkred')
    ax2.scatter(inter_a, inter_c, c='blue', alpha=0.7, label='Interictal', s=50, edgecolors='darkblue')

    # Mark template
    template_a = np.mean([p[0] for p in results['template_params']])
    template_c = np.mean([p[2] for p in results['template_params']])
    ax2.scatter([template_a], [template_c], c='gold', s=200, marker='*',
                label='Template', edgecolors='black', zorder=5)

    ax2.set_xlabel('Excitatory (a)')
    ax2.set_ylabel('Inhibitory (c)')
    ax2.set_title('PN Parameter Space')
    ax2.legend(fontsize=8)

    # Math annotation
    ax2.text(0.02, 0.98, r'$\frac{da}{dt} = -\lambda_a a + f(t)(1-a)$' + '\n' +
             r'$\frac{dc}{dt} = +\lambda_c c + f(t)(1-c)$',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === Panel 3: Phase Coherence ===
    ax3 = fig.add_subplot(gs[0, 2])

    # Phase std (lower = more synchronized)
    ictal_phase_std = [np.std([p[1] for p in params]) for params in results['ictal_params']]
    inter_phase_std = [np.std([p[1] for p in params]) for params in results['interictal_params']]

    positions = [1, 2]
    bp = ax3.boxplot([ictal_phase_std, inter_phase_std], positions=positions, widths=0.6,
                     patch_artist=True)

    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('blue')
    bp['boxes'][1].set_alpha(0.7)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Ictal', 'Interictal'])
    ax3.set_ylabel('Phase Std (b)')
    ax3.set_title('Phase Coherence Across Channels')

    # Math annotation
    ax3.text(0.02, 0.98, r'Coherence $= \sigma_b = \sqrt{\frac{1}{M}\sum_i(b_i - \bar{b})^2}$',
             transform=ax3.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === Panel 4: Circuit Complexity ===
    ax4 = fig.add_subplot(gs[1, 0])

    M_values = [2, 4, 6, 8]
    qubits = [2*M + 1 for M in M_values]
    gates = [17*M - 2 for M in M_values]  # Formula from paper
    depth = [7 + 2*(M-1) + 2 for M in M_values]  # Approximate

    ax4.plot(M_values, qubits, 'o-', label=f'Qubits (2M+1)', color='green', linewidth=2)
    ax4.plot(M_values, gates, 's-', label=f'Gates (17M-2)', color='purple', linewidth=2)

    # Mark current config
    curr_M = results['num_channels']
    ax4.axvline(curr_M, color='gray', linestyle='--', alpha=0.5)
    ax4.scatter([curr_M], [results['circuit_qubits']], c='green', s=100, zorder=5, edgecolors='black')
    ax4.scatter([curr_M], [results['circuit_gates']], c='purple', s=100, zorder=5, edgecolors='black')

    ax4.set_xlabel('Channels (M)')
    ax4.set_ylabel('Count')
    ax4.set_title('Quantum Circuit Scaling')
    ax4.legend(fontsize=8)
    ax4.set_xticks(M_values)

    # Math annotation
    ax4.text(0.02, 0.98, r'$O(M)$ scaling vs $O(M^2)$ classical',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === Panel 5: Performance Metrics ===
    ax5 = fig.add_subplot(gs[1, 1])

    metrics = results['eval_metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['specificity']
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    bars = ax5.bar(metric_names, metric_values, color=colors, edgecolor='black', alpha=0.8)

    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel('Score')
    ax5.set_title('Classification Performance')
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # === Panel 6: Confusion Matrix Heatmap ===
    ax6 = fig.add_subplot(gs[1, 2])

    # Build confusion matrix
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])

    # Plot heatmap
    im = ax6.imshow(cm, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            # Choose text color based on background
            text_color = 'white' if val > cm.max() / 2 else 'black'
            ax6.text(j, i, f'{val}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=text_color)

    # Labels
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Pred: Interictal', 'Pred: Ictal'])
    ax6.set_yticklabels(['True: Interictal', 'True: Ictal'])
    ax6.set_xlabel('Predicted', fontsize=10)
    ax6.set_ylabel('Actual', fontsize=10)

    # Title with key stats
    sep = np.mean(ictal_fid) - np.mean(inter_fid)
    ax6.set_title(f'Confusion Matrix\nFidelity Sep: {sep:.3f} | {results["circuit_qubits"]}q, {results["circuit_gates"]}g',
                 fontsize=10)

    # Add colorbar
    plt.colorbar(im, ax=ax6, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nDashboard saved to {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("QDNU QUANTUM SEIZURE PREDICTION - DASHBOARD GENERATOR")
    print("=" * 60)

    # Check available subjects
    subjects = list_available_subjects()
    if not subjects:
        print("No EEG data found. Exiting.")
        exit(1)

    print(f"\nAvailable subjects: {subjects}")

    # Run on first available subject
    subject = subjects[0]

    # Run validation
    results = run_validation(subject=subject, num_channels=4, n_samples=15)

    # Create dashboard
    output_path = create_dashboard(results, 'figures/qdnu_dashboard.png')

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject: {results['subject']}")
    print(f"Channels: {results['num_channels']}")
    print(f"Circuit: {results['circuit_qubits']} qubits, {results['circuit_gates']} gates")
    print(f"Ictal fidelity: {np.mean(results['ictal_fidelities']):.4f}")
    print(f"Interictal fidelity: {np.mean(results['interictal_fidelities']):.4f}")
    print(f"Accuracy: {results['eval_metrics']['accuracy']:.2%}")
    print("=" * 60)
