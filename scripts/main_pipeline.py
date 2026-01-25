"""
Main Pipeline for Quantum Seizure Prediction

End-to-end integration of all components:
1. EEG data loading (synthetic or real)
2. PN dynamics evolution
3. Quantum circuit creation
4. Template training
5. Fidelity-based prediction
6. Performance evaluation

This script validates the complete quantum advantage pipeline.

Reference: Gupta, A., et al. (2024). Positive-negative neuron model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import os

from qdnu import (
    PNDynamics,
    create_single_channel_agate,
    MultichannelCircuit,
    TemplateTrainer,
    SeizurePredictor
)


def generate_synthetic_eeg(num_channels, time_steps, signal_type='normal', fs=256):
    """
    Generate synthetic EEG data with known characteristics.

    Signal types:
    - 'normal': Random noise, unsynchronized across channels
    - 'preictal': Synchronized oscillations (15 Hz) + noise
    - 'ictal': High-amplitude synchronized bursts

    Args:
        num_channels: Number of EEG channels
        time_steps: Number of time samples
        signal_type: 'normal', 'preictal', or 'ictal'
        fs: Sampling frequency (default: 256 Hz)

    Returns:
        numpy array: EEG data shape (num_channels, time_steps)
    """
    t = np.linspace(0, time_steps / fs, time_steps)
    eeg = np.zeros((num_channels, time_steps))

    if signal_type == 'normal':
        # Random unsynchronized activity
        for i in range(num_channels):
            # Mix of random frequencies
            freq = np.random.uniform(8, 12)  # Alpha band
            eeg[i] = (0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
                     + 0.5 * np.random.randn(time_steps))

    elif signal_type == 'preictal':
        # Synchronized 15 Hz oscillation (characteristic of pre-ictal)
        sync_signal = 2.0 * np.sin(2 * np.pi * 15 * t)
        for i in range(num_channels):
            # High synchronization + small noise
            phase_shift = np.random.uniform(-0.1, 0.1)  # Small phase variation
            amplitude_var = 1 + 0.1 * i  # Slight amplitude gradient
            eeg[i] = (sync_signal * amplitude_var * np.cos(phase_shift)
                     + 0.3 * np.random.randn(time_steps))

    elif signal_type == 'ictal':
        # High-amplitude synchronized bursts
        sync_signal = 4.0 * np.sin(2 * np.pi * 10 * t)  # Strong 10 Hz
        for i in range(num_channels):
            # Very high synchronization
            eeg[i] = sync_signal + 0.2 * np.random.randn(time_steps)

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    return eeg


def load_eeg_data(filepath=None, num_channels=4, duration_sec=10, fs=256):
    """
    Load EEG data from file or generate synthetic.

    Args:
        filepath: Path to EEG file (None for synthetic)
        num_channels: Number of channels (for synthetic)
        duration_sec: Duration in seconds (for synthetic)
        fs: Sampling frequency

    Returns:
        numpy array: EEG data shape (num_channels, time_steps)
    """
    if filepath is None:
        # Generate synthetic for testing
        time_steps = duration_sec * fs
        return generate_synthetic_eeg(num_channels, time_steps, 'preictal', fs)
    else:
        # TODO: Implement real EEG loading (.edf, .csv)
        raise NotImplementedError(
            f"Real EEG loading not yet implemented. "
            f"Requested file: {filepath}"
        )


def run_training_pipeline(preictal_data_list, config):
    """
    Run the training pipeline on pre-ictal data.

    Args:
        preictal_data_list: List of pre-ictal EEG arrays
        config: Configuration dictionary with:
            - num_channels: int
            - lambda_a: float
            - lambda_c: float
            - dt: float
            - template_path: str (optional)

    Returns:
        TemplateTrainer: Trained template trainer
    """
    # Initialize trainer
    trainer = TemplateTrainer(
        num_channels=config['num_channels'],
        lambda_a=config['lambda_a'],
        lambda_c=config['lambda_c'],
        dt=config['dt']
    )

    # Average pre-ictal data if multiple samples
    if len(preictal_data_list) > 1:
        # Ensure same shape
        min_time = min(d.shape[1] for d in preictal_data_list)
        truncated = [d[:, :min_time] for d in preictal_data_list]
        avg_preictal = np.mean(truncated, axis=0)
    else:
        avg_preictal = preictal_data_list[0]

    # Train
    trainer.train(avg_preictal)

    # Save template if path provided
    if 'template_path' in config and config['template_path']:
        trainer.save_template(config['template_path'])

    return trainer


def run_prediction_pipeline(test_data_list, trainer, config):
    """
    Run prediction pipeline on test data.

    Args:
        test_data_list: List of test EEG arrays
        trainer: Trained TemplateTrainer
        config: Configuration with 'threshold'

    Returns:
        list: List of result dictionaries
    """
    predictor = SeizurePredictor(trainer, threshold=config['threshold'])

    results = []
    for i, eeg in enumerate(test_data_list):
        pred, fid, metrics = predictor.predict(eeg)
        results.append({
            'index': i,
            'prediction': pred,
            'fidelity': fid,
            'metrics': metrics
        })

    return results


def visualize_results(results, labels=None, output_path=None):
    """
    Visualize prediction results.

    Args:
        results: List of result dictionaries from prediction
        labels: Optional ground truth labels
        output_path: Path to save figures (None = display only)
    """
    try:
        import matplotlib.pyplot as plt

        fidelities = [r['fidelity'] for r in results]
        predictions = [r['prediction'] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Fidelity scores
        ax1 = axes[0]
        colors = ['green' if p else 'red' for p in predictions]
        ax1.bar(range(len(fidelities)), fidelities, color=colors, alpha=0.7)
        ax1.axhline(y=0.7, color='black', linestyle='--', label='Threshold')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Fidelity')
        ax1.set_title('Fidelity Scores')
        ax1.legend()

        # Plot 2: Confusion matrix (if labels provided)
        if labels is not None:
            ax2 = axes[1]
            tp = sum(1 for p, l in zip(predictions, labels) if p and l)
            tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
            fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
            fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

            cm = np.array([[tn, fp], [fn, tp]])
            im = ax2.imshow(cm, cmap='Blues')
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(['Pred Neg', 'Pred Pos'])
            ax2.set_yticklabels(['True Neg', 'True Pos'])
            ax2.set_title('Confusion Matrix')

            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, cm[i, j], ha='center', va='center', fontsize=16)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Results saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("matplotlib not available - skipping visualization")


# === Main execution ===

if __name__ == "__main__":
    np.random.seed(42)

    # Configuration
    config = {
        'num_channels': 4,
        'lambda_a': 0.1,
        'lambda_c': 0.05,
        'dt': 0.001,
        'threshold': 0.5,  # Lower threshold for synthetic data
        'template_path': 'trained_template.npy'
    }

    print("=" * 50)
    print("QUANTUM SEIZURE PREDICTION PIPELINE")
    print("=" * 50)

    # === Test 1: Training Phase ===
    print("\n=== Test 1: Training Phase ===")

    # Generate 3 pre-ictal training samples
    preictal_samples = [
        generate_synthetic_eeg(4, 2560, 'preictal')
        for _ in range(3)
    ]

    trainer = run_training_pipeline(preictal_samples, config)
    print("[OK] Training complete")

    # === Test 2: Prediction Phase ===
    print("\n=== Test 2: Prediction Phase ===")

    test_cases = [
        ('preictal', True),
        ('normal', False),
        ('preictal', True),
        ('normal', False)
    ]

    predictor = SeizurePredictor(trainer, threshold=config['threshold'])
    correct = 0

    for signal_type, expected in test_cases:
        eeg = generate_synthetic_eeg(4, 2560, signal_type)
        pred, fid, _ = predictor.predict(eeg)

        if pred == expected:
            correct += 1
            status = "[OK]"
        else:
            status = "[X]"

        print(f"{status} {signal_type}: pred={pred}, fid={fid:.4f} (expected {expected})")

    accuracy = correct / len(test_cases)
    print(f"\nPrediction accuracy: {accuracy:.1%}")

    # === Test 3: Batch Evaluation ===
    print("\n=== Test 3: Batch Evaluation ===")

    test_segments = []
    labels = []

    # Generate test data
    for _ in range(20):
        test_segments.append(generate_synthetic_eeg(4, 2560, 'preictal'))
        labels.append(1)
    for _ in range(20):
        test_segments.append(generate_synthetic_eeg(4, 2560, 'normal'))
        labels.append(0)

    eval_results = predictor.evaluate(test_segments, labels)

    print(f"Accuracy:    {eval_results['accuracy']:.3f}")
    print(f"Precision:   {eval_results['precision']:.3f}")
    print(f"Recall:      {eval_results['recall']:.3f}")
    print(f"F1 Score:    {eval_results['f1']:.3f}")
    print(f"Specificity: {eval_results['specificity']:.3f}")
    print(f"Pre-ictal mean fidelity: {eval_results['mean_fidelity_positive']:.4f}")
    print(f"Normal mean fidelity:    {eval_results['mean_fidelity_negative']:.4f}")

    # === Test 4: Template Persistence ===
    print("\n=== Test 4: Template Persistence ===")

    # Save template
    trainer.save_template('test_template.npy')

    # Load into new trainer
    new_trainer = TemplateTrainer(num_channels=4)
    new_trainer.load_template('test_template.npy')

    # Predict with loaded template
    new_predictor = SeizurePredictor(new_trainer, threshold=0.5)
    test_eeg = generate_synthetic_eeg(4, 2560, 'preictal')
    pred, fid, _ = new_predictor.predict(test_eeg)

    print(f"[OK] Loaded template prediction: {pred}, fidelity: {fid:.4f}")

    # Clean up test file
    os.remove('test_template.npy')

    # === Test 5: Parameter Sensitivity ===
    print("\n=== Test 5: Parameter Sensitivity ===")
    print("Testing different lambda values:")

    for lambda_a in [0.05, 0.1, 0.15]:
        for lambda_c in [0.025, 0.05, 0.075]:
            trainer_test = TemplateTrainer(4, lambda_a, lambda_c, 0.001)
            trainer_test.train(generate_synthetic_eeg(4, 2560, 'preictal'))

            pred_test = SeizurePredictor(trainer_test, threshold=0.5)
            _, fid_preictal, _ = pred_test.predict(generate_synthetic_eeg(4, 2560, 'preictal'))
            _, fid_normal, _ = pred_test.predict(generate_synthetic_eeg(4, 2560, 'normal'))

            diff = fid_preictal - fid_normal
            print(f"lambda_a={lambda_a:.2f}, lambda_c={lambda_c:.3f}: "
                  f"preictal={fid_preictal:.4f}, normal={fid_normal:.4f}, diff={diff:.4f}")

    # === Test 6: Circuit Statistics ===
    print("\n=== Test 6: Circuit Statistics ===")

    print(f"Template circuit qubits: {trainer.template_circuit.num_qubits}")
    print(f"Template circuit depth:  {trainer.template_circuit.depth()}")
    print(f"Template circuit gates:  {trainer.template_circuit.size()}")

    # Gate count breakdown
    gate_counts = {}
    for instruction in trainer.template_circuit.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    print("Gate breakdown:")
    for gate, count in sorted(gate_counts.items()):
        print(f"  {gate}: {count}")

    # === Summary ===
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Channels:     {config['num_channels']}")
    print(f"Qubits:       {trainer.template_circuit.num_qubits}")
    print(f"Circuit depth: {trainer.template_circuit.depth()}")
    print(f"Threshold:    {config['threshold']}")
    print(f"Accuracy:     {eval_results['accuracy']:.1%}")
    print("=" * 50)
    print("ALL TESTS COMPLETE")
    print("=" * 50)
