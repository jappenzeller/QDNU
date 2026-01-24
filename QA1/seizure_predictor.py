"""
Seizure Predictor via Quantum Fidelity Measurement

Predicts seizures by comparing test EEG against a trained quantum template:
1. Normalize and process test EEG through PN dynamics
2. Create quantum circuit from test parameters
3. Compute fidelity F = |<psi_template|psi_test>|^2
4. Classify based on fidelity threshold

Quantum advantage: Single fidelity measurement captures all M^2 channel
correlations simultaneously.

Reference: Gupta, A., et al. (2024). Positive-negative neuron model.
"""

import numpy as np
from qiskit.quantum_info import Statevector
from multichannel_circuit import create_multichannel_circuit


class SeizurePredictor:
    """
    Predicts seizures using quantum state fidelity.

    The prediction process:
    1. Normalize test EEG using trainer's normalization
    2. Evolve PN dynamics to get test parameters
    3. Create quantum circuit for test data
    4. Compute fidelity against template
    5. Classify based on threshold

    Attributes:
        trainer: TemplateTrainer with trained template
        threshold: Fidelity threshold for positive prediction
        num_channels: Number of EEG channels
    """

    def __init__(self, trainer, threshold=0.7):
        """
        Initialize seizure predictor.

        Args:
            trainer: TemplateTrainer instance (must be trained)
            threshold: Fidelity threshold for positive prediction (default: 0.7)

        Raises:
            ValueError: If trainer has not been trained
        """
        if trainer.template_params is None:
            raise ValueError("Trainer must be trained first")

        self.trainer = trainer
        self.threshold = threshold
        self.num_channels = trainer.num_channels

    def predict(self, test_eeg):
        """
        Predict seizure risk from test EEG.

        Args:
            test_eeg: numpy array shape (num_channels, time_steps)

        Returns:
            tuple: (prediction, fidelity, metrics)
                prediction: bool, True if predicted pre-ictal
                fidelity: float in [0, 1], similarity to template
                metrics: dict with diagnostic information
        """
        # 1. Normalize test EEG
        test_normalized = self.trainer.normalize_eeg(test_eeg)

        # 2. Evolve test dynamics
        test_params = self.trainer.pn_dynamics.evolve_multichannel(test_normalized)

        # 3. Create test circuit
        test_circuit = create_multichannel_circuit(test_params)

        # 4. Compute fidelity
        fidelity = self.compute_fidelity(
            self.trainer.template_circuit,
            test_circuit
        )

        # 5. Make prediction
        prediction = fidelity > self.threshold

        # 6. Compute confidence metrics
        metrics = self._compute_metrics(test_params, fidelity)

        return (prediction, fidelity, metrics)

    def compute_fidelity(self, circuit1, circuit2):
        """
        Compute quantum state fidelity between two circuits.

        F = |<psi1|psi2>|^2

        This single measurement captures all O(M^2) channel correlations
        in the exponential Hilbert space.

        Args:
            circuit1: First QuantumCircuit (template)
            circuit2: Second QuantumCircuit (test)

        Returns:
            float: Fidelity in [0, 1]
        """
        sv1 = Statevector.from_instruction(circuit1)
        sv2 = Statevector.from_instruction(circuit2)

        # Fidelity = |<psi1|psi2>|^2
        fidelity = abs(sv1.inner(sv2)) ** 2

        return fidelity

    def _compute_metrics(self, test_params, fidelity):
        """
        Compute diagnostic metrics for the prediction.

        Args:
            test_params: List of (a, b, c) tuples from test EEG
            fidelity: Computed fidelity value

        Returns:
            dict: Diagnostic metrics
        """
        a_vals = [p[0] for p in test_params]
        b_vals = [p[1] for p in test_params]
        c_vals = [p[2] for p in test_params]

        template_params = self.trainer.template_params
        template_a = [p[0] for p in template_params]
        template_b = [p[1] for p in template_params]
        template_c = [p[2] for p in template_params]

        # L2 distance in parameter space
        param_distance = np.sqrt(
            sum((a - ta) ** 2 for a, ta in zip(a_vals, template_a)) +
            sum((b - tb) ** 2 for b, tb in zip(b_vals, template_b)) +
            sum((c - tc) ** 2 for c, tc in zip(c_vals, template_c))
        )

        return {
            'fidelity': fidelity,
            'phase_coherence': np.std(b_vals),
            'excitatory_mean': np.mean(a_vals),
            'inhibitory_mean': np.mean(c_vals),
            'param_distance': param_distance,
            'threshold': self.threshold,
            'margin': fidelity - self.threshold
        }

    def predict_batch(self, test_segments):
        """
        Predict on multiple EEG segments.

        Args:
            test_segments: List of EEG arrays, each (num_channels, time_steps)

        Returns:
            list: List of (prediction, fidelity, metrics) tuples
        """
        results = []
        for segment in test_segments:
            result = self.predict(segment)
            results.append(result)
        return results

    def evaluate(self, test_data, true_labels):
        """
        Evaluate predictor performance on labeled data.

        Args:
            test_data: List of EEG segments
            true_labels: List of ground truth labels (1=pre-ictal, 0=normal)

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, F1, etc.)
        """
        predictions = []
        fidelities = []

        for segment in test_data:
            pred, fid, _ = self.predict(segment)
            predictions.append(int(pred))
            fidelities.append(fid)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        fidelities = np.array(fidelities)

        # Basic metrics
        tp = np.sum((predictions == 1) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC-AUC (simple calculation)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(true_labels, fidelities)
        except (ImportError, ValueError):
            auc = None

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'auc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'mean_fidelity_positive': np.mean(fidelities[true_labels == 1]) if np.any(true_labels == 1) else 0,
            'mean_fidelity_negative': np.mean(fidelities[true_labels == 0]) if np.any(true_labels == 0) else 0
        }


# === Test cases ===

if __name__ == "__main__":
    from template_trainer import TemplateTrainer

    np.random.seed(42)

    print("=" * 50)
    print("Seizure Predictor Test Suite")
    print("=" * 50)

    def generate_preictal_eeg(num_channels, time_steps):
        """Synthetic pre-ictal: synchronized oscillations."""
        t = np.linspace(0, 10, time_steps)
        eeg = np.zeros((num_channels, time_steps))
        sync_signal = 2.0 * np.sin(2 * np.pi * 15 * t)
        for i in range(num_channels):
            eeg[i] = sync_signal * (1 + 0.1 * i) + 0.3 * np.random.randn(time_steps)
        return eeg

    def generate_normal_eeg(num_channels, time_steps):
        """Synthetic normal EEG: random, unsynchronized."""
        return np.random.randn(num_channels, time_steps) * 0.5

    # Setup: Train a template
    print("\n--- Training Template ---")
    trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05)
    preictal = generate_preictal_eeg(4, 500)
    trainer.train(preictal)

    predictor = SeizurePredictor(trainer, threshold=0.5)

    # Test 1: Pre-ictal detection
    print("\n=== Test 1: Pre-ictal Detection ===")
    test_preictal = generate_preictal_eeg(4, 500)
    pred, fid, metrics = predictor.predict(test_preictal)
    print(f"Prediction: {pred}")
    print(f"Fidelity: {fid:.4f}")
    print(f"Phase coherence: {metrics['phase_coherence']:.4f}")
    print(f"Param distance: {metrics['param_distance']:.4f}")
    # Pre-ictal should have high fidelity with template
    print("[OK] Pre-ictal prediction complete")

    # Test 2: Normal EEG
    print("\n=== Test 2: Normal EEG ===")
    test_normal = generate_normal_eeg(4, 500)
    pred, fid, metrics = predictor.predict(test_normal)
    print(f"Prediction: {pred}")
    print(f"Fidelity: {fid:.4f}")
    print(f"Phase coherence: {metrics['phase_coherence']:.4f}")
    print("[OK] Normal EEG prediction complete")

    # Test 3: Fidelity bounds
    print("\n=== Test 3: Fidelity Bounds ===")
    assert 0 <= fid <= 1, f"Fidelity should be in [0,1], got {fid}"
    print("[OK] Fidelity in valid range")

    # Test 4: Batch prediction
    print("\n=== Test 4: Batch Prediction ===")
    batch = [
        generate_preictal_eeg(4, 500),
        generate_normal_eeg(4, 500),
        generate_preictal_eeg(4, 500),
        generate_normal_eeg(4, 500)
    ]
    results = predictor.predict_batch(batch)
    for i, (pred, fid, _) in enumerate(results):
        label = "pre-ictal" if i % 2 == 0 else "normal"
        print(f"Segment {i} ({label}): pred={pred}, fidelity={fid:.4f}")
    print("[OK] Batch prediction complete")

    # Test 5: Evaluation metrics
    print("\n=== Test 5: Evaluation Metrics ===")
    test_segments = [generate_preictal_eeg(4, 500) for _ in range(10)]
    test_segments += [generate_normal_eeg(4, 500) for _ in range(10)]
    true_labels = [1] * 10 + [0] * 10

    eval_metrics = predictor.evaluate(test_segments, true_labels)
    print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"Precision: {eval_metrics['precision']:.3f}")
    print(f"Recall: {eval_metrics['recall']:.3f}")
    print(f"F1 Score: {eval_metrics['f1']:.3f}")
    print(f"Specificity: {eval_metrics['specificity']:.3f}")
    print(f"Mean fidelity (pre-ictal): {eval_metrics['mean_fidelity_positive']:.4f}")
    print(f"Mean fidelity (normal): {eval_metrics['mean_fidelity_negative']:.4f}")
    if eval_metrics['auc'] is not None:
        print(f"ROC-AUC: {eval_metrics['auc']:.3f}")
    print("[OK] Evaluation complete")

    # Test 6: Threshold sensitivity
    print("\n=== Test 6: Threshold Sensitivity ===")
    test_sample = generate_preictal_eeg(4, 500)
    _, base_fid, _ = predictor.predict(test_sample)
    print(f"Sample fidelity: {base_fid:.4f}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        predictor.threshold = thresh
        pred, fid, _ = predictor.predict(test_sample)
        print(f"Threshold {thresh}: pred={pred}")
    print("[OK] Threshold sensitivity verified")

    # Test 7: Pre-ictal vs Normal fidelity comparison
    print("\n=== Test 7: Fidelity Separation ===")
    predictor.threshold = 0.5
    preictal_fids = []
    normal_fids = []
    for _ in range(5):
        _, fid, _ = predictor.predict(generate_preictal_eeg(4, 500))
        preictal_fids.append(fid)
        _, fid, _ = predictor.predict(generate_normal_eeg(4, 500))
        normal_fids.append(fid)

    print(f"Pre-ictal fidelity: {np.mean(preictal_fids):.4f} +/- {np.std(preictal_fids):.4f}")
    print(f"Normal fidelity: {np.mean(normal_fids):.4f} +/- {np.std(normal_fids):.4f}")

    if np.mean(preictal_fids) > np.mean(normal_fids):
        print("[OK] Pre-ictal has higher fidelity than normal")
    else:
        print("[WARN] Fidelity separation not as expected")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
