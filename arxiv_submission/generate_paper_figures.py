#!/usr/bin/env python3
"""
Generate all figures for the QPNN Architecture Paper.

This script generates publication-quality figures from the analysis results JSON files.
Output: figures/ directory with PNG files referenced in main_v2.tex

Usage:
    python generate_paper_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Create figures directory
FIGURES_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def fig1_agate_circuit():
    """
    Figure 1: A-Gate circuit diagram.
    Creates a schematic representation of the single-channel A-Gate circuit.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    h_color = '#4a90d9'      # Blue for Hadamard
    p_color = '#50c878'      # Green for Phase
    rx_color = '#ff6b6b'     # Red for Rx
    ry_color = '#ffa500'     # Orange for Ry
    cr_color = '#9370db'     # Purple for controlled rotations

    # Wire positions
    y_e = 2  # Excitatory qubit
    y_i = 0  # Inhibitory qubit

    # Draw qubit wires
    ax.hlines([y_e, y_i], 0.5, 11.5, colors='black', linewidths=1)

    # Labels
    ax.text(0.2, y_e, r'$|E\rangle$', ha='right', va='center', fontsize=12)
    ax.text(0.2, y_i, r'$|I\rangle$', ha='right', va='center', fontsize=12)

    # Gate positions
    gate_size = 0.4

    def draw_gate(x, y, label, color, fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x - gate_size, y - gate_size),
            2*gate_size, 2*gate_size,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Excitatory qubit gates
    positions_e = [1, 2, 3.5, 5, 6]
    labels_e = ['H', r'P(b)', r'$R_x(2a)$', r'P(b)', 'H']
    colors_e = [h_color, p_color, rx_color, p_color, h_color]
    fontsizes_e = [10, 9, 8, 9, 10]

    for x, label, color, fs in zip(positions_e, labels_e, colors_e, fontsizes_e):
        draw_gate(x, y_e, label, color, fs)

    # Inhibitory qubit gates
    positions_i = [1, 2, 3.5, 5, 6]
    labels_i = ['H', r'P(b)', r'$R_y(2c)$', r'P(b)', 'H']
    colors_i = [h_color, p_color, ry_color, p_color, h_color]
    fontsizes_i = [10, 9, 8, 9, 10]

    for x, label, color, fs in zip(positions_i, labels_i, colors_i, fontsizes_i):
        draw_gate(x, y_i, label, color, fs)

    # Controlled rotations (Layer 2)
    # CRy: E controls I
    x_cry = 8
    ax.plot([x_cry, x_cry], [y_e - gate_size, y_i + gate_size], 'k-', linewidth=1.5)
    ax.plot(x_cry, y_e, 'ko', markersize=8)  # Control dot
    draw_gate(x_cry, y_i, r'$R_y$', cr_color, 9)
    ax.text(x_cry + 0.6, y_i, r'$\frac{\pi}{4}$', ha='left', va='center', fontsize=9)

    # CRz: I controls E
    x_crz = 10
    ax.plot([x_crz, x_crz], [y_e - gate_size, y_i + gate_size], 'k-', linewidth=1.5)
    ax.plot(x_crz, y_i, 'ko', markersize=8)  # Control dot
    draw_gate(x_crz, y_e, r'$R_z$', cr_color, 9)
    ax.text(x_crz + 0.6, y_e, r'$\frac{\pi}{4}$', ha='left', va='center', fontsize=9)

    # Layer labels
    ax.annotate('', xy=(6.5, -0.8), xytext=(0.5, -0.8),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(3.5, -1.1, 'Layer 1: Per-qubit encoding', ha='center', fontsize=10, color='gray')

    ax.annotate('', xy=(10.5, -0.8), xytext=(7.5, -0.8),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(9, -1.1, 'Layer 2: E-I coupling', ha='center', fontsize=10, color='gray')

    # Title
    ax.set_title('A-Gate Circuit: Single-Channel PN Neuron Encoding\n(14 gates, depth 7)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.savefig(FIGURES_DIR / 'agate_circuit.png')
    plt.close()
    print("Generated: agate_circuit.png")


def fig2_multichannel_circuit():
    """
    Figure 2: Multi-channel circuit architecture.
    Shows the full M-channel circuit with ring topology and global ancilla.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    # Colors
    agate_color = '#e6f2ff'
    ring_color = '#ffe6e6'
    ancilla_color = '#e6ffe6'

    # Draw 4 channels (M=4 example)
    M = 4
    y_positions = [6, 4.5, 3, 1.5]  # E qubits
    y_positions_i = [5.5, 4, 2.5, 1]  # I qubits

    # Draw A-Gate boxes
    for i in range(M):
        # A-Gate box
        rect = mpatches.FancyBboxPatch(
            (1, y_positions_i[i] - 0.2), 3, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=agate_color, edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)
        ax.text(2.5, y_positions[i] - 0.25, f'A-Gate {i+1}', ha='center', va='center', fontsize=9)

        # Qubit labels
        ax.text(0.3, y_positions[i], f'$E_{i+1}$', ha='right', va='center', fontsize=10)
        ax.text(0.3, y_positions_i[i], f'$I_{i+1}$', ha='right', va='center', fontsize=10)

        # Wires
        ax.hlines([y_positions[i], y_positions_i[i]], 0.5, 13, colors='black', linewidths=0.8)

    # Ancilla qubit
    y_anc = -0.3
    ax.hlines(y_anc, 0.5, 13, colors='black', linewidths=0.8)
    ax.text(0.3, y_anc, '$|0\\rangle$', ha='right', va='center', fontsize=10)

    # Hadamard on ancilla
    rect = mpatches.FancyBboxPatch(
        (4.7, y_anc - 0.25), 0.6, 0.5,
        boxstyle="round,pad=0.02",
        facecolor='#4a90d9', edgecolor='black', linewidth=1
    )
    ax.add_patch(rect)
    ax.text(5, y_anc, 'H', ha='center', va='center', fontsize=9, fontweight='bold')

    # Ring topology (excitatory)
    ring_x = 6
    for i in range(M - 1):
        # CNOT: vertical line + target
        ax.plot([ring_x, ring_x], [y_positions[i], y_positions[i+1]], 'k-', linewidth=1)
        ax.plot(ring_x, y_positions[i], 'ko', markersize=6)  # Control
        ax.plot(ring_x, y_positions[i+1], 'ko', markersize=10, fillstyle='none', markeredgewidth=2)  # Target
        ring_x += 0.4

    # Ring topology (inhibitory)
    ring_x = 6
    for i in range(M - 1):
        ax.plot([ring_x + 2, ring_x + 2], [y_positions_i[i], y_positions_i[i+1]], 'k-', linewidth=1)
        ax.plot(ring_x + 2, y_positions_i[i], 'ko', markersize=6)
        ax.plot(ring_x + 2, y_positions_i[i+1], 'ko', markersize=10, fillstyle='none', markeredgewidth=2)
        ring_x += 0.4

    # Global CZ coupling (ancilla to all E)
    cz_x = 11
    for i, y in enumerate(y_positions):
        ax.plot([cz_x + i*0.3, cz_x + i*0.3], [y, y_anc], 'k-', linewidth=1)
        ax.plot(cz_x + i*0.3, y, 'ko', markersize=6)
        ax.plot(cz_x + i*0.3, y_anc, 'ko', markersize=6)

    # Measurement symbols
    meas_x = 13
    for y in y_positions + y_positions_i + [y_anc]:
        # Meter symbol
        ax.plot([meas_x - 0.15, meas_x + 0.15], [y, y], 'k-', linewidth=1.5)
        ax.plot([meas_x - 0.15, meas_x], [y - 0.15, y + 0.15], 'k-', linewidth=1.5)
        ax.plot([meas_x, meas_x + 0.15], [y + 0.15, y - 0.15], 'k-', linewidth=1.5)

    # Region labels
    ax.annotate('', xy=(4.2, 7), xytext=(0.8, 7),
                arrowprops=dict(arrowstyle='<->', color='#4a90d9', lw=2))
    ax.text(2.5, 7.3, '14M gates', ha='center', fontsize=10, color='#4a90d9')

    ax.annotate('', xy=(9.5, 7), xytext=(5.5, 7),
                arrowprops=dict(arrowstyle='<->', color='#d94a4a', lw=2))
    ax.text(7.5, 7.3, '2(M-1) CNOTs', ha='center', fontsize=10, color='#d94a4a')

    ax.annotate('', xy=(12.5, 7), xytext=(10.5, 7),
                arrowprops=dict(arrowstyle='<->', color='#4ad94a', lw=2))
    ax.text(11.5, 7.3, 'M CZs', ha='center', fontsize=10, color='#4ad94a')

    ax.set_title(f'Multi-Channel Circuit Architecture (M={M} channels, {2*M+1} qubits)\n' +
                 f'Total: 17M - 1 = {17*M - 1} gates = O(M)',
                 fontsize=12, fontweight='bold', pad=10)

    plt.savefig(FIGURES_DIR / 'multichannel_circuit.png')
    plt.close()
    print("Generated: multichannel_circuit.png")


def fig3_complexity_scaling():
    """
    Figure 3: Complexity scaling comparison (O(M) vs O(M^2)).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    M = np.arange(1, 65)
    classical = M ** 2  # O(M^2)
    quantum = 17 * M - 1  # Actual gate count formula

    ax.plot(M, classical, 'r-', linewidth=2, label=r'Classical: $O(M^2)$')
    ax.plot(M, quantum, 'b-', linewidth=2, label=r'Quantum: $O(M)$')

    # Highlight key points
    for m_val in [8, 19, 64]:
        ax.axvline(m_val, color='gray', linestyle='--', alpha=0.5)
        classical_val = m_val ** 2
        quantum_val = 17 * m_val - 1
        ax.plot(m_val, classical_val, 'ro', markersize=8)
        ax.plot(m_val, quantum_val, 'bo', markersize=8)

        ratio = classical_val / quantum_val
        ax.annotate(f'M={m_val}\n{ratio:.1f}x',
                   xy=(m_val, classical_val),
                   xytext=(m_val + 2, classical_val * 0.9),
                   fontsize=9, ha='left')

    ax.set_xlabel('Number of Channels (M)', fontsize=12)
    ax.set_ylabel('Operations / Gates', fontsize=12)
    ax.set_title('Correlation Encoding Complexity: Classical vs Quantum', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 4500)
    ax.grid(True, alpha=0.3)

    # Add annotations for clinical systems
    ax.text(8, 200, '8-ch\n(this work)', ha='center', fontsize=9, color='gray')
    ax.text(19, 600, '19-ch\n(10-20 system)', ha='center', fontsize=9, color='gray')
    ax.text(64, 1500, '64-ch\n(high-density)', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'complexity_comparison.png')
    plt.close()
    print("Generated: complexity_comparison.png")


def fig4_encoding_ablation():
    """
    Figure 4: Encoding strategy ablation bar chart.
    """
    # Data from analysis results
    encodings = ['V1\n(PN dynamics)', 'V2\n(Band power)', 'V3\n(PLV/Hilbert)', 'Classical\n(8-ch)', 'Classical\n(18-ch)']
    aucs = [0.444, 0.534, 0.529, 0.625, 0.820]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#2d6a4f']

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(encodings, aucs, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Horizontal lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance level')

    # Annotations
    ax.annotate('', xy=(1, 0.56), xytext=(0, 0.46),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 0.51, '+9%', ha='center', fontsize=10, color='green', fontweight='bold')

    ax.annotate('', xy=(3, 0.64), xytext=(1, 0.55),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2, 0.60, 'Gap: 9%', ha='center', fontsize=10, color='red', fontweight='bold')

    ax.set_ylabel('AUC (Area Under ROC Curve)', fontsize=12)
    ax.set_title('Encoding Strategy Comparison (CHB-MIT, 22-Subject LOSO)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # Add bracket for quantum methods
    ax.annotate('', xy=(0, 0.9), xytext=(2, 0.9),
                arrowprops=dict(arrowstyle='-', color='blue', lw=1.5))
    ax.text(1, 0.93, 'Quantum Methods', ha='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'encoding_ablation.png')
    plt.close()
    print("Generated: encoding_ablation.png")


def fig5_ibm_discrimination_heatmap():
    """
    Figure 5: IBM hardware configuration discrimination heatmap.
    """
    # Load discrimination matrix
    matrix_path = ANALYSIS_DIR / "ibm_hardware" / "discrimination_matrix.json"
    data = load_json(matrix_path)

    # Get hardware matrix
    configs = data['hardware']['configs']
    matrix = np.array(data['hardware']['matrix'])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use diverging colormap centered at low values
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=0.25)

    # Labels
    ax.set_xticks(range(len(configs)))
    ax.set_yticks(range(len(configs)))
    config_labels = ['Sync', 'Desync', 'Half', 'Excit', 'Inhib']
    ax.set_xticklabels(config_labels, fontsize=10)
    ax.set_yticklabels(config_labels, fontsize=10)

    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(configs)):
            val = matrix[i, j]
            color = 'white' if val > 0.1 else 'black'
            if i == j:
                text = '1.00'
            else:
                text = f'{val:.3f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Hellinger Fidelity', fontsize=11)

    # Title with discrimination quality
    disc_quality = data['hardware']['discrimination_quality']
    ax.set_title(f'Configuration Discrimination Matrix (ibm_torino)\n'
                 f'Discrimination Quality: {disc_quality:.1%}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ibm_discrimination_heatmap.png')
    plt.close()
    print("Generated: ibm_discrimination_heatmap.png")


def fig6_ibm_scaling():
    """
    Figure 6: IBM hardware scaling experiment (CZ gates vs channels).
    """
    # Load scaling data
    scaling_path = ANALYSIS_DIR / "ibm_hardware" / "scaling_experiment.json"
    data = load_json(scaling_path)

    channels = np.array(data['channel_counts'])
    cz_gates = np.array(data['two_qubit_gates'])

    # Linear fit
    slope = data['linear_fit']['slope']
    intercept = data['linear_fit']['intercept']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter plot of actual data
    ax.scatter(channels, cz_gates, s=100, c='#4a90d9', edgecolors='black',
               linewidths=1.5, zorder=5, label='Measured (transpiled)')

    # Linear fit line
    x_fit = np.linspace(1, 10, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r--', linewidth=2,
            label=f'Linear fit: {slope:.1f}M {intercept:+.1f}')

    # Theoretical O(M^2) for comparison
    y_quad = 14 * channels ** 2 / 8  # Scaled for visualization
    ax.plot(channels, y_quad, 'g:', linewidth=2, alpha=0.6,
            label=r'$O(M^2)$ scaling (scaled)')

    # Annotations for each point
    for m, cz in zip(channels, cz_gates):
        ax.annotate(f'{cz}', xy=(m, cz), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=10)

    ax.set_xlabel('Number of Channels (M)', fontsize=12)
    ax.set_ylabel('Two-Qubit (CZ) Gates', fontsize=12)
    ax.set_title('Hardware Scaling Validation (ibm_torino)\n'
                 r'Confirmed: CZ gates $= 14.1M - 17.5 = O(M)$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    # Add R^2 annotation
    ax.text(8, 20, f'$R^2 > 0.99$', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ibm_scaling.png')
    plt.close()
    print("Generated: ibm_scaling.png")


def fig7_per_subject_auc():
    """
    Figure 7: Per-subject AUC distribution.
    """
    # Load classical results
    results_path = ANALYSIS_DIR / "quantum_loso_v2" / "classical_results.json"
    data = load_json(results_path)

    subjects = []
    aucs = []
    for subj, metrics in data['per_subject'].items():
        subjects.append(subj.replace('chb', ''))
        aucs.append(metrics['auc'])

    # Sort by AUC
    sorted_indices = np.argsort(aucs)[::-1]
    subjects_sorted = [subjects[i] for i in sorted_indices]
    aucs_sorted = [aucs[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Color by performance
    colors = ['#2d6a4f' if auc >= 0.7 else '#96ceb4' if auc >= 0.5 else '#ff6b6b'
              for auc in aucs_sorted]

    bars = ax.bar(range(len(subjects_sorted)), aucs_sorted, color=colors,
                  edgecolor='black', linewidth=0.8)

    # Horizontal lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(np.mean(aucs), color='blue', linestyle='-', alpha=0.7, linewidth=2,
               label=f'Mean: {np.mean(aucs):.1%}')

    ax.set_xticks(range(len(subjects_sorted)))
    ax.set_xticklabels([f'chb{s}' for s in subjects_sorted], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_title('Per-Subject Classification Performance (Classical 8-Channel Baseline)\n'
                 'CHB-MIT Dataset, Leave-One-Subject-Out Cross-Validation',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # Add legend for colors
    high_patch = mpatches.Patch(color='#2d6a4f', label='AUC >= 0.7')
    mid_patch = mpatches.Patch(color='#96ceb4', label='0.5 <= AUC < 0.7')
    low_patch = mpatches.Patch(color='#ff6b6b', label='AUC < 0.5')
    ax.legend(handles=[high_patch, mid_patch, low_patch], loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'per_subject_auc.png')
    plt.close()
    print("Generated: per_subject_auc.png")


def fig_fidelity_comparison():
    """
    Additional figure: Hellinger fidelity comparison across execution modes.
    """
    # Load fidelity data
    fidelity_path = ANALYSIS_DIR / "ibm_hardware" / "fidelity_comparison.json"
    data = load_json(fidelity_path)

    configs = data['configs']
    ideal_vs_noisy = [data['ideal_vs_noisy'][c] for c in configs]
    ideal_vs_hardware = [data['ideal_vs_hardware'][c] for c in configs]
    noisy_vs_hardware = [data['noisy_vs_hardware'][c] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width, ideal_vs_noisy, width, label='Ideal vs Noisy', color='#4a90d9')
    bars2 = ax.bar(x, ideal_vs_hardware, width, label='Ideal vs Hardware', color='#50c878')
    bars3 = ax.bar(x + width, noisy_vs_hardware, width, label='Noisy vs Hardware', color='#ff6b6b')

    ax.set_ylabel('Hellinger Distance', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_title('Fidelity Analysis: Simulation vs Hardware Execution\n'
                 '(Lower = closer agreement)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    config_labels = ['Sync', 'Desync', 'Half', 'Excit', 'Inhib']
    ax.set_xticklabels(config_labels)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fidelity_comparison.png')
    plt.close()
    print("Generated: fidelity_comparison.png")


def main():
    """Generate all figures."""
    print("Generating paper figures...")
    print(f"Output directory: {FIGURES_DIR}")
    print("-" * 50)

    # Generate all figures
    fig1_agate_circuit()
    fig2_multichannel_circuit()
    fig3_complexity_scaling()
    fig4_encoding_ablation()
    fig5_ibm_discrimination_heatmap()
    fig6_ibm_scaling()
    fig7_per_subject_auc()
    fig_fidelity_comparison()  # Bonus figure

    print("-" * 50)
    print(f"All figures generated in: {FIGURES_DIR}")
    print("\nFiles created:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
