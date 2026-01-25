"""
Diagnostic script to identify why fidelity shows no separation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from eeg_loader import load_for_qdnu, load_subject_data
from template_trainer import TemplateTrainer
from seizure_predictor import SeizurePredictor
from pn_dynamics import PNDynamics


def diagnose_failure(ictal_data, interictal_data, subject='Dog_1'):
    """Comprehensive diagnostic check."""

    print("=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1. Data sanity
    print("\n1. DATA SANITY")
    print(f"   Ictal shape: {ictal_data.shape}")
    print(f"   Interictal shape: {interictal_data.shape}")
    print(f"   Ictal range: [{ictal_data.min():.2f}, {ictal_data.max():.2f}]")
    print(f"   Interictal range: [{interictal_data.min():.2f}, {interictal_data.max():.2f}]")
    print(f"   Ictal mean/std: {ictal_data.mean():.2f} / {ictal_data.std():.2f}")
    print(f"   Interictal mean/std: {interictal_data.mean():.2f} / {interictal_data.std():.2f}")
    print(f"   Ictal NaN count: {np.isnan(ictal_data).sum()}")
    print(f"   Interictal NaN count: {np.isnan(interictal_data).sum()}")

    # 2. Spectral difference
    print("\n2. SPECTRAL CONTENT (Channel 0)")
    try:
        f, psd_ictal = welch(ictal_data[0], fs=400, nperseg=min(256, len(ictal_data[0])))
        f, psd_inter = welch(interictal_data[0], fs=400, nperseg=min(256, len(interictal_data[0])))

        # Energy in different bands
        delta_ictal = np.trapz(psd_ictal[(f >= 0.5) & (f <= 4)])
        delta_inter = np.trapz(psd_inter[(f >= 0.5) & (f <= 4)])
        theta_ictal = np.trapz(psd_ictal[(f >= 4) & (f <= 8)])
        theta_inter = np.trapz(psd_inter[(f >= 4) & (f <= 8)])
        alpha_ictal = np.trapz(psd_ictal[(f >= 8) & (f <= 13)])
        alpha_inter = np.trapz(psd_inter[(f >= 8) & (f <= 13)])
        beta_ictal = np.trapz(psd_ictal[(f >= 13) & (f <= 30)])
        beta_inter = np.trapz(psd_inter[(f >= 13) & (f <= 30)])

        print(f"   Delta (0.5-4 Hz):  ictal={delta_ictal:.2e}, inter={delta_inter:.2e}, diff={abs(delta_ictal-delta_inter):.2e}")
        print(f"   Theta (4-8 Hz):    ictal={theta_ictal:.2e}, inter={theta_inter:.2e}, diff={abs(theta_ictal-theta_inter):.2e}")
        print(f"   Alpha (8-13 Hz):   ictal={alpha_ictal:.2e}, inter={alpha_inter:.2e}, diff={abs(alpha_ictal-alpha_inter):.2e}")
        print(f"   Beta (13-30 Hz):   ictal={beta_ictal:.2e}, inter={beta_inter:.2e}, diff={abs(beta_ictal-beta_inter):.2e}")
        print(f"   Total power ratio: {psd_ictal.sum() / psd_inter.sum():.2f}x")
    except Exception as e:
        print(f"   Error computing spectra: {e}")

    # 3. PN parameters
    print("\n3. PN PARAMETERS")
    trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05, dt=0.001)
    ictal_norm = trainer.normalize_eeg(ictal_data)
    inter_norm = trainer.normalize_eeg(interictal_data)

    print(f"   Normalized ictal range: [{ictal_norm.min():.4f}, {ictal_norm.max():.4f}]")
    print(f"   Normalized inter range: [{inter_norm.min():.4f}, {inter_norm.max():.4f}]")

    ictal_params = trainer.pn_dynamics.evolve_multichannel(ictal_norm)
    inter_params = trainer.pn_dynamics.evolve_multichannel(inter_norm)

    print("\n   Ictal parameters:")
    for i, (a, b, c) in enumerate(ictal_params):
        print(f"     Ch{i}: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    print("\n   Interictal parameters:")
    for i, (a, b, c) in enumerate(inter_params):
        print(f"     Ch{i}: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    ictal_b_std = np.std([p[1] for p in ictal_params])
    inter_b_std = np.std([p[1] for p in inter_params])

    print(f"\n   Ictal phase coherence (b std): {ictal_b_std:.4f}")
    print(f"   Interictal phase coherence: {inter_b_std:.4f}")
    print(f"   Difference: {abs(ictal_b_std - inter_b_std):.4f}")

    # 4. Saturation check
    all_a = [p[0] for p in ictal_params] + [p[0] for p in inter_params]
    all_c = [p[2] for p in ictal_params] + [p[2] for p in inter_params]
    saturated_a = sum(1 for a in all_a if a > 0.99 or a < 0.01)
    saturated_c = sum(1 for c in all_c if c > 0.99 or c < 0.01)

    print(f"\n4. SATURATION CHECK")
    print(f"   Saturated a parameters: {saturated_a}/{len(all_a)}")
    print(f"   Saturated c parameters: {saturated_c}/{len(all_c)}")
    print(f"   a range: [{min(all_a):.4f}, {max(all_a):.4f}]")
    print(f"   c range: [{min(all_c):.4f}, {max(all_c):.4f}]")

    # 5. Fidelity
    print("\n5. QUANTUM FIDELITY")
    trainer.train(ictal_data)
    predictor = SeizurePredictor(trainer, threshold=0.5)

    _, fid_self, _ = predictor.predict(ictal_data)
    _, fid_inter, _ = predictor.predict(interictal_data)

    print(f"   Template vs itself: {fid_self:.6f} (should be ~1.0)")
    print(f"   Template vs interictal: {fid_inter:.6f}")
    print(f"   Separation: {fid_self - fid_inter:.6f}")

    # 6. Lambda sensitivity
    print("\n6. LAMBDA SENSITIVITY ANALYSIS")
    lambda_configs = [
        (0.01, 0.005, "very slow"),
        (0.05, 0.025, "slow"),
        (0.1, 0.05, "medium"),
        (0.2, 0.1, "fast"),
        (0.5, 0.25, "very fast"),
    ]

    for lambda_a, lambda_c, label in lambda_configs:
        trainer_test = TemplateTrainer(4, lambda_a, lambda_c, dt=0.001)

        ictal_norm = trainer_test.normalize_eeg(ictal_data)
        inter_norm = trainer_test.normalize_eeg(interictal_data)

        ictal_p = trainer_test.pn_dynamics.evolve_multichannel(ictal_norm)
        inter_p = trainer_test.pn_dynamics.evolve_multichannel(inter_norm)

        # Parameter differences
        a_diff = abs(np.mean([p[0] for p in ictal_p]) - np.mean([p[0] for p in inter_p]))
        b_diff = abs(np.mean([p[1] for p in ictal_p]) - np.mean([p[1] for p in inter_p]))
        c_diff = abs(np.mean([p[2] for p in ictal_p]) - np.mean([p[2] for p in inter_p]))

        print(f"   {label:12s} (λa={lambda_a}, λc={lambda_c}): a_diff={a_diff:.4f}, b_diff={b_diff:.4f}, c_diff={c_diff:.4f}")

    # 7. Recommendation
    print("\n7. DIAGNOSIS")
    issues = []

    if ictal_norm.max() - ictal_norm.min() < 0.2:
        issues.append("Normalization too aggressive - range < 0.2")

    if saturated_a + saturated_c > len(all_a) // 2:
        issues.append("Parameters saturating at bounds")

    if abs(ictal_b_std - inter_b_std) < 0.001:
        issues.append("No phase coherence difference detected")

    if fid_self < 0.99:
        issues.append(f"Self-fidelity too low ({fid_self:.4f}) - numerical issues")

    if abs(fid_self - fid_inter) < 0.001:
        issues.append("Fidelity shows no separation")

    # Check if parameters are nearly identical
    param_diff = sum([
        abs(ictal_params[i][j] - inter_params[i][j])
        for i in range(len(ictal_params))
        for j in range(3)
    ])
    if param_diff < 0.01:
        issues.append(f"PN parameters nearly identical (total diff={param_diff:.4f})")

    if issues:
        print("   ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   No obvious issues detected")

    print("=" * 60)

    return {
        'ictal_params': ictal_params,
        'inter_params': inter_params,
        'fid_self': fid_self,
        'fid_inter': fid_inter,
        'issues': issues
    }


def visualize_raw_difference(ictal_data, interictal_data, output_path='eeg_comparison.png'):
    """Create visual comparison of ictal vs interictal."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot first channel
    axes[0, 0].plot(ictal_data[0, :500], 'r-', alpha=0.7, label='Ictal', linewidth=0.8)
    axes[0, 0].plot(interictal_data[0, :500], 'b-', alpha=0.7, label='Interictal', linewidth=0.8)
    axes[0, 0].set_title('Channel 0 - Raw Signal (first 500 samples)')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')

    # Power spectral density
    try:
        f_ictal, psd_ictal = welch(ictal_data[0], fs=400, nperseg=256)
        f_inter, psd_inter = welch(interictal_data[0], fs=400, nperseg=256)
        axes[0, 1].semilogy(f_ictal, psd_ictal, 'r-', label='Ictal', linewidth=1.5)
        axes[0, 1].semilogy(f_inter, psd_inter, 'b-', label='Interictal', linewidth=1.5)
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('PSD')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 50)
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')

    # Cross-correlation between channels
    try:
        ictal_corr = np.corrcoef(ictal_data)
        inter_corr = np.corrcoef(interictal_data)

        x = np.arange(ictal_corr.shape[0])
        axes[1, 0].bar(x - 0.2, ictal_corr[0], 0.4, color='red', alpha=0.7, label='Ictal')
        axes[1, 0].bar(x + 0.2, inter_corr[0], 0.4, color='blue', alpha=0.7, label='Interictal')
        axes[1, 0].set_title('Cross-channel Correlation (Ch0 vs all)')
        axes[1, 0].set_xlabel('Channel')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].legend()
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')

    # Amplitude histogram
    axes[1, 1].hist(ictal_data[0].flatten(), bins=50, alpha=0.5, color='r', label='Ictal', density=True)
    axes[1, 1].hist(interictal_data[0].flatten(), bins=50, alpha=0.5, color='b', label='Interictal', density=True)
    axes[1, 1].set_title('Amplitude Distribution (Ch0)')
    axes[1, 1].set_xlabel('Amplitude')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    np.random.seed(42)

    print("Loading EEG data...")
    ictal_windows, interictal_windows = load_for_qdnu(
        'Dog_1', num_channels=4, window_size=500, n_ictal=15, n_interictal=15
    )

    # Use first windows for diagnosis
    ictal_data = ictal_windows[0]
    interictal_data = interictal_windows[0]

    # Run diagnostics
    results = diagnose_failure(ictal_data, interictal_data)

    # Create visualization
    visualize_raw_difference(ictal_data, interictal_data, 'eeg_comparison.png')

    print("\nDiagnostic complete!")
