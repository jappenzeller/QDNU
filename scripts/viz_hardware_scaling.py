#!/usr/bin/env python3
"""
================================================================================
VIZ-PREP — Hardware Scaling Visualization
================================================================================

Generates publication-quality figures for:
1. Simulation vs Hardware scaling curve
2. Gap trajectory analysis
3. Polarity calibration impact
4. Classical vs Quantum comparison

Outputs to: figures/hardware_scaling/
================================================================================
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "hardware_scaling"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'simulation': '#2196F3',      # Blue
    'hardware_raw': '#F44336',    # Red
    'hardware_cal': '#4CAF50',    # Green
    'classical': '#9C27B0',       # Purple
    'threshold': '#FF9800',       # Orange
}


def load_json(path: Path):
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
        return None


def plot_scaling_curve(sim_data, hw_data, calibrated_data, classical_data):
    """
    Figure 1: AUC vs Channel Count scaling curve.

    Shows simulation ceiling, hardware performance, and calibrated performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data points
    channels = []
    sim_aucs = []
    hw_raw_aucs = []
    hw_cal_aucs = []
    classical_aucs = []

    for ch_key in ['ch4', 'ch8', 'ch12', 'ch16']:
        n_ch = int(ch_key[2:])

        # Simulation
        if ch_key in sim_data and sim_data[ch_key].get('overall_auc'):
            channels.append(n_ch)
            sim_aucs.append(sim_data[ch_key]['overall_auc'])
        else:
            continue

        # Hardware raw
        if ch_key in hw_data and hw_data[ch_key].get('overall_auc'):
            hw_raw_aucs.append(hw_data[ch_key]['overall_auc'])
        else:
            hw_raw_aucs.append(None)

        # Classical
        if classical_data and ch_key in classical_data and classical_data[ch_key].get('overall_auc'):
            classical_aucs.append(classical_data[ch_key]['overall_auc'])
        else:
            classical_aucs.append(None)

    # Plot simulation line
    ax.plot(channels, sim_aucs, 'o-', color=COLORS['simulation'],
            linewidth=2, markersize=10, label='Simulation (ideal)')

    # Plot hardware raw points
    hw_channels = [c for c, h in zip(channels, hw_raw_aucs) if h is not None]
    hw_values = [h for h in hw_raw_aucs if h is not None]
    if hw_channels:
        ax.plot(hw_channels, hw_values, 's--', color=COLORS['hardware_raw'],
                linewidth=2, markersize=10, label='Hardware (raw)')

    # Plot classical baseline
    cl_channels = [c for c, cl in zip(channels, classical_aucs) if cl is not None]
    cl_values = [cl for cl in classical_aucs if cl is not None]
    if cl_channels:
        ax.plot(cl_channels, cl_values, '^-', color=COLORS['classical'],
                linewidth=2, markersize=10, label='Classical (Tier-2 MAX)')

    # Add calibrated point for CH8 if available
    if calibrated_data and 'ch8' in calibrated_data:
        cal_auc = calibrated_data['ch8'].get('calibrated_overall_auc')
        if cal_auc:
            ax.plot(8, cal_auc, 'D', color=COLORS['hardware_cal'],
                    markersize=12, label='Hardware (calibrated)', zorder=5)
            # Draw arrow from raw to calibrated
            raw_ch8 = hw_data.get('ch8', {}).get('overall_auc')
            if raw_ch8:
                ax.annotate('', xy=(8, cal_auc), xytext=(8, raw_ch8),
                           arrowprops=dict(arrowstyle='->', color=COLORS['hardware_cal'],
                                          lw=2, ls='--'))

    # Clinical threshold line
    ax.axhline(y=0.70, color=COLORS['threshold'], linestyle=':', linewidth=2,
               label='Clinical threshold (0.70)')

    # Random chance line
    ax.axhline(y=0.50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(16.2, 0.51, 'Random', color='gray', fontsize=9)

    # Labels and styling
    ax.set_xlabel('Number of EEG Channels', fontsize=12)
    ax.set_ylabel('AUC (LOSO CV)', fontsize=12)
    ax.set_title('Quantum Seizure Detection: Scaling Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks([4, 8, 12, 16])
    ax.set_xlim(2, 18)
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add qubit count as secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([4, 8, 12, 16])
    ax2.set_xticklabels(['9', '17', '25', '33'])
    ax2.set_xlabel('Number of Qubits', fontsize=10)

    plt.tight_layout()
    return fig


def plot_gap_analysis(sim_data, hw_data):
    """
    Figure 2: Hardware gap (Sim - HW) vs qubit count.

    Shows how hardware degradation scales with circuit size.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    qubits = []
    gaps = []
    gap_pcts = []

    for ch_key in ['ch4', 'ch8']:
        if ch_key in sim_data and ch_key in hw_data:
            sim_auc = sim_data[ch_key].get('overall_auc')
            hw_auc = hw_data[ch_key].get('overall_auc')
            n_qubits = sim_data[ch_key].get('n_qubits')

            if sim_auc and hw_auc and n_qubits:
                gap = sim_auc - hw_auc
                gap_pct = (gap / sim_auc) * 100

                qubits.append(n_qubits)
                gaps.append(gap)
                gap_pcts.append(gap_pct)

    if len(qubits) >= 2:
        # Plot gap values
        ax.bar(qubits, gaps, width=3, color=COLORS['hardware_raw'], alpha=0.7,
               edgecolor='black', linewidth=1.5)

        # Add percentage labels
        for q, g, p in zip(qubits, gaps, gap_pcts):
            ax.text(q, g + 0.002, f'{p:.1f}%', ha='center', fontsize=11, fontweight='bold')

        # Trend line
        coeffs = np.polyfit(qubits, gaps, 1)
        trend_x = np.array([5, 35])
        trend_y = coeffs[0] * trend_x + coeffs[1]
        ax.plot(trend_x, trend_y, '--', color='gray', linewidth=2, alpha=0.5,
                label=f'Trend: slope={coeffs[0]:.5f}')

        # Project to CH12/CH16
        for n_q, ch in [(25, 'CH12'), (33, 'CH16')]:
            pred_gap = coeffs[0] * n_q + coeffs[1]
            ax.scatter(n_q, pred_gap, marker='x', s=100, color='gray', zorder=5)
            ax.text(n_q, pred_gap + 0.003, f'{ch}\n(projected)', ha='center',
                   fontsize=9, color='gray')

    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Gap (Simulation - Hardware AUC)', fontsize=12)
    ax.set_title('Hardware Degradation vs Circuit Size', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.06)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_polarity_calibration(calibrated_data):
    """
    Figure 3: Per-subject polarity calibration impact.

    Shows raw vs calibrated AUC for each subject.
    """
    if not calibrated_data or 'ch8' not in calibrated_data:
        logger.warning("No calibration data available for plotting")
        return None

    ch8 = calibrated_data['ch8']
    per_subject = ch8.get('per_subject', {})

    if not per_subject:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    subjects = sorted(per_subject.keys())
    x = np.arange(len(subjects))
    width = 0.35

    raw_aucs = [per_subject[s]['raw'] for s in subjects]
    cal_aucs = [per_subject[s]['calibrated'] for s in subjects]
    polarities = [per_subject[s]['polarity'] for s in subjects]

    # Plot bars
    bars1 = ax.bar(x - width/2, raw_aucs, width, label='Raw AUC',
                   color=COLORS['hardware_raw'], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, cal_aucs, width, label='Calibrated AUC',
                   color=COLORS['hardware_cal'], alpha=0.7, edgecolor='black')

    # Highlight inverted subjects
    for i, (s, pol) in enumerate(zip(subjects, polarities)):
        if pol == 'inverted':
            ax.annotate('INV', (x[i] + width/2, cal_aucs[i] + 0.02),
                       ha='center', fontsize=8, fontweight='bold', color='red')

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.7, color=COLORS['threshold'], linestyle=':', linewidth=2, alpha=0.7)

    # Aggregate values
    raw_overall = ch8.get('raw_overall_auc', 0)
    cal_overall = ch8.get('calibrated_overall_auc', 0)
    lift = ch8.get('calibration_lift', 0)

    ax.text(0.98, 0.98, f'Raw Overall: {raw_overall:.3f}\nCalibrated: {cal_overall:.3f}\nLift: +{lift:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Polarity Calibration: Per-Subject Impact (CH8)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_classical_quantum_comparison(sim_data, hw_data, calibrated_data, classical_data):
    """
    Figure 4: Classical vs Quantum comparison at each channel count.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for grouped bar chart
    ch_keys = ['ch4', 'ch8', 'ch12']
    x = np.arange(len(ch_keys))
    width = 0.2

    classical_vals = []
    sim_vals = []
    hw_raw_vals = []
    hw_cal_vals = []

    for ch_key in ch_keys:
        # Classical
        if classical_data and ch_key in classical_data:
            classical_vals.append(classical_data[ch_key].get('overall_auc', 0))
        else:
            classical_vals.append(0)

        # Simulation
        if ch_key in sim_data:
            sim_vals.append(sim_data[ch_key].get('overall_auc', 0))
        else:
            sim_vals.append(0)

        # Hardware raw
        if ch_key in hw_data and hw_data[ch_key].get('overall_auc'):
            hw_raw_vals.append(hw_data[ch_key]['overall_auc'])
        else:
            hw_raw_vals.append(0)

        # Hardware calibrated (only for CH8)
        if ch_key == 'ch8' and calibrated_data and 'ch8' in calibrated_data:
            hw_cal_vals.append(calibrated_data['ch8'].get('calibrated_overall_auc', 0))
        else:
            hw_cal_vals.append(0)

    # Plot bars
    ax.bar(x - 1.5*width, classical_vals, width, label='Classical', color=COLORS['classical'], alpha=0.8)
    ax.bar(x - 0.5*width, sim_vals, width, label='Quantum (sim)', color=COLORS['simulation'], alpha=0.8)

    # Only plot hardware where we have data
    hw_raw_nonzero = [(i, v) for i, v in enumerate(hw_raw_vals) if v > 0]
    if hw_raw_nonzero:
        ax.bar([x[i] + 0.5*width for i, v in hw_raw_nonzero],
               [v for i, v in hw_raw_nonzero], width,
               label='Quantum (HW raw)', color=COLORS['hardware_raw'], alpha=0.8)

    hw_cal_nonzero = [(i, v) for i, v in enumerate(hw_cal_vals) if v > 0]
    if hw_cal_nonzero:
        ax.bar([x[i] + 1.5*width for i, v in hw_cal_nonzero],
               [v for i, v in hw_cal_nonzero], width,
               label='Quantum (HW cal)', color=COLORS['hardware_cal'], alpha=0.8)

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.7, color=COLORS['threshold'], linestyle=':', linewidth=2)

    ax.set_xlabel('Channel Configuration', fontsize=12)
    ax.set_ylabel('AUC (LOSO CV)', fontsize=12)
    ax.set_title('Classical vs Quantum Seizure Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['4 channels\n(9 qubits)', '8 channels\n(17 qubits)', '12 channels\n(25 qubits)'])
    ax.set_ylim(0.4, 0.85)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def main():
    logger.info("=" * 70)
    logger.info("VIZ-PREP: Hardware Scaling Visualization")
    logger.info("=" * 70)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    sim_data = load_json(RESULTS_DIR / "projection" / "simulation_scaling.json")
    hw_data = load_json(RESULTS_DIR / "projection" / "hardware_scaling.json")
    calibrated_data = load_json(RESULTS_DIR / "calibration" / "calibrated_gap_inputs.json")
    classical_data = load_json(RESULTS_DIR / "scaling" / "classical_channel_scaling.json")

    if not sim_data:
        logger.error("Missing simulation data")
        return
    if not hw_data:
        logger.error("Missing hardware data")
        return

    # Generate figures
    figures = []

    # Figure 1: Scaling curve
    logger.info("Generating scaling curve...")
    fig1 = plot_scaling_curve(sim_data, hw_data, calibrated_data, classical_data)
    fig1.savefig(FIGURES_DIR / "fig1_scaling_curve.png", dpi=300, bbox_inches='tight')
    fig1.savefig(FIGURES_DIR / "fig1_scaling_curve.pdf", bbox_inches='tight')
    plt.close(fig1)
    figures.append("fig1_scaling_curve")
    logger.info(f"  Saved: {FIGURES_DIR / 'fig1_scaling_curve.png'}")

    # Figure 2: Gap analysis
    logger.info("Generating gap analysis...")
    fig2 = plot_gap_analysis(sim_data, hw_data)
    fig2.savefig(FIGURES_DIR / "fig2_gap_analysis.png", dpi=300, bbox_inches='tight')
    fig2.savefig(FIGURES_DIR / "fig2_gap_analysis.pdf", bbox_inches='tight')
    plt.close(fig2)
    figures.append("fig2_gap_analysis")
    logger.info(f"  Saved: {FIGURES_DIR / 'fig2_gap_analysis.png'}")

    # Figure 3: Polarity calibration
    if calibrated_data:
        logger.info("Generating polarity calibration plot...")
        fig3 = plot_polarity_calibration(calibrated_data)
        if fig3:
            fig3.savefig(FIGURES_DIR / "fig3_polarity_calibration.png", dpi=300, bbox_inches='tight')
            fig3.savefig(FIGURES_DIR / "fig3_polarity_calibration.pdf", bbox_inches='tight')
            plt.close(fig3)
            figures.append("fig3_polarity_calibration")
            logger.info(f"  Saved: {FIGURES_DIR / 'fig3_polarity_calibration.png'}")

    # Figure 4: Classical vs Quantum
    if classical_data:
        logger.info("Generating classical vs quantum comparison...")
        fig4 = plot_classical_quantum_comparison(sim_data, hw_data, calibrated_data, classical_data)
        fig4.savefig(FIGURES_DIR / "fig4_classical_quantum.png", dpi=300, bbox_inches='tight')
        fig4.savefig(FIGURES_DIR / "fig4_classical_quantum.pdf", bbox_inches='tight')
        plt.close(fig4)
        figures.append("fig4_classical_quantum")
        logger.info(f"  Saved: {FIGURES_DIR / 'fig4_classical_quantum.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("VIZ-PREP COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(figures)} figures in {FIGURES_DIR}:")
    for fig_name in figures:
        print(f"  - {fig_name}.png / .pdf")
    print()


if __name__ == "__main__":
    main()
