"""
Performance Benchmark: Quantum vs Classical PN Neuron

Comprehensive comparison measuring:
- Classification accuracy (precision, recall, F1, AUC)
- Inference speed
- Scalability with channels O(M) vs O(M²)
- Memory usage

Validates quantum advantage for seizure prediction.
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from pn_dynamics import PNDynamics
from template_trainer import TemplateTrainer
from seizure_predictor import SeizurePredictor


class ClassicalPNBaseline:
    """
    Classical PN neuron baseline for comparison.

    Uses same PN dynamics but with explicit O(M²) feature extraction:
    - PN parameters (a, b, c) per channel
    - Phase Locking Value (PLV) - pairwise, O(M²)
    - Correlation matrix - pairwise, O(M²)
    - Statistical features per channel
    """

    def __init__(self, lambda_a=0.1, lambda_c=0.1, dt=0.001):
        self.lambda_a = lambda_a
        self.lambda_c = lambda_c
        self.dt = dt
        self.pn = PNDynamics(lambda_a, lambda_c, dt, saturation_mode='clamp')
        self.template_features = None
        self.classifier = None

    def extract_features(self, eeg_data):
        """
        Extract classical features from EEG.

        Feature count: 3M + M(M-1)/2 + M(M-1)/2 + 3M = 6M + M²-M = O(M²)
        """
        # Normalize
        eeg_norm = self._normalize(eeg_data)
        num_channels = eeg_data.shape[0]
        features = []

        # 1. PN parameters (3M features)
        params = self.pn.evolve_multichannel(eeg_norm)
        for a, b, c in params:
            features.extend([a, b, c])

        # 2. Phase Locking Value - O(M²)
        try:
            from scipy.signal import hilbert
            analytic = [hilbert(eeg_norm[i]) for i in range(num_channels)]
            phases = [np.angle(sig) for sig in analytic]

            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    phase_diff = phases[i] - phases[j]
                    plv = abs(np.mean(np.exp(1j * phase_diff)))
                    features.append(plv)
        except:
            # Fallback if hilbert fails
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    features.append(0.5)

        # 3. Correlation matrix - O(M²)
        corr = np.corrcoef(eeg_norm)
        corr = np.nan_to_num(corr, nan=0.0)
        triu_idx = np.triu_indices(num_channels, k=1)
        features.extend(corr[triu_idx])

        # 4. Statistical features (3M)
        for ch in range(num_channels):
            features.append(np.mean(eeg_norm[ch]))
            features.append(np.std(eeg_norm[ch]))
            features.append(np.max(np.abs(eeg_norm[ch])))

        return np.array(features, dtype=np.float32)

    def _normalize(self, eeg_data):
        """Z-score + tanh normalization (same as quantum)."""
        normalized = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            signal = eeg_data[ch]
            mean, std = np.mean(signal), np.std(signal)
            if std > 0:
                z = (signal - mean) / std
            else:
                z = signal - mean
            normalized[ch] = (np.tanh(z / 2) + 1) / 2
        return normalized

    def train(self, preictal_samples, normal_samples):
        """Train classifier on labeled samples."""
        print("Training classical baseline...")

        X, y = [], []

        for eeg in preictal_samples:
            X.append(self.extract_features(eeg))
            y.append(1)

        for eeg in normal_samples:
            X.append(self.extract_features(eeg))
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        print(f"  Feature dim: {X.shape[1]} (O(M²) where M={preictal_samples[0].shape[0]})")

        # Template = mean of preictal features
        self.template_features = np.mean(X[y == 1], axis=0)

        # Train SVM
        try:
            from sklearn.svm import SVC
            self.classifier = SVC(kernel='rbf', probability=True, C=1.0)
            self.classifier.fit(X, y)
            print("  [OK] SVM classifier trained")
        except ImportError:
            print("  [WARN] sklearn not available, using template matching")
            self.classifier = None

    def predict(self, test_eeg):
        """Predict seizure risk."""
        features = self.extract_features(test_eeg)

        if self.classifier is not None:
            prob = self.classifier.predict_proba([features])[0]
            prediction = prob[1] > 0.5
            confidence = prob[1]
        else:
            # Cosine similarity template matching
            sim = np.dot(features, self.template_features)
            sim /= (np.linalg.norm(features) * np.linalg.norm(self.template_features) + 1e-10)
            prediction = sim > 0.7
            confidence = (sim + 1) / 2  # Map [-1,1] to [0,1]

        return prediction, confidence


class PerformanceBenchmark:
    """Compare quantum vs classical PN implementations."""

    def __init__(self):
        self.results = {'quantum': {}, 'classical': {}}

    def benchmark_accuracy(self, quantum_predictor, classical_baseline,
                          test_samples, test_labels):
        """Measure classification accuracy."""
        print("\n" + "=" * 60)
        print("ACCURACY BENCHMARK")
        print("=" * 60)

        for name, predictor in [('quantum', quantum_predictor),
                                ('classical', classical_baseline)]:
            print(f"\nTesting {name}...")

            predictions, confidences = [], []

            for eeg in test_samples:
                if name == 'quantum':
                    pred, conf, _ = predictor.predict(eeg)
                else:
                    pred, conf = predictor.predict(eeg)
                predictions.append(int(pred))
                confidences.append(float(conf))

            predictions = np.array(predictions)
            confidences = np.array(confidences)
            test_labels_arr = np.array(test_labels)

            # Metrics
            tp = np.sum((predictions == 1) & (test_labels_arr == 1))
            tn = np.sum((predictions == 0) & (test_labels_arr == 0))
            fp = np.sum((predictions == 1) & (test_labels_arr == 0))
            fn = np.sum((predictions == 0) & (test_labels_arr == 1))

            acc = (tp + tn) / len(test_labels) if len(test_labels) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(test_labels, confidences)
            except:
                auc = 0.5

            self.results[name].update({
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'auc': auc,
                'predictions': predictions, 'confidences': confidences,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            })

            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall:    {rec:.3f}")
            print(f"  F1:        {f1:.3f}")
            print(f"  AUC:       {auc:.3f}")

    def benchmark_speed(self, quantum_predictor, classical_baseline,
                       test_sample, n_trials=10):
        """Measure inference speed."""
        print("\n" + "=" * 60)
        print("SPEED BENCHMARK")
        print("=" * 60)

        for name, predictor in [('quantum', quantum_predictor),
                                ('classical', classical_baseline)]:
            times = []
            for _ in range(n_trials):
                start = time.time()
                if name == 'quantum':
                    predictor.predict(test_sample)
                else:
                    predictor.predict(test_sample)
                times.append(time.time() - start)

            mean_t, std_t = np.mean(times), np.std(times)
            self.results[name]['time_mean'] = mean_t
            self.results[name]['time_std'] = std_t

            print(f"\n{name.capitalize()}:")
            print(f"  Mean: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
            print(f"  Throughput: {1/mean_t:.1f} samples/sec")

    def benchmark_scalability(self, channel_counts=[4, 6, 8]):
        """Test scaling with channels."""
        print("\n" + "=" * 60)
        print("SCALABILITY BENCHMARK")
        print("=" * 60)

        q_times, c_times = [], []

        for M in channel_counts:
            print(f"\nM = {M} channels:")

            # Generate data
            preictal = np.random.randn(M, 1000) * 0.5
            normal = np.random.randn(M, 1000) * 0.3

            # Quantum
            start = time.time()
            trainer = TemplateTrainer(num_channels=M, lambda_a=0.1, lambda_c=0.05)
            trainer.train(preictal)
            q_time = time.time() - start
            q_times.append(q_time)
            print(f"  Quantum: {q_time:.3f}s (gates: {trainer.template_circuit.size()})")

            # Classical
            start = time.time()
            classical = ClassicalPNBaseline()
            classical.train([preictal], [normal])
            c_time = time.time() - start
            c_times.append(c_time)
            n_features = 3*M + M*(M-1)//2 + M*(M-1)//2 + 3*M
            print(f"  Classical: {c_time:.3f}s (features: {n_features})")

        self.results['scalability'] = {
            'channels': channel_counts,
            'quantum_times': q_times,
            'classical_times': c_times
        }

    def visualize_results(self, test_labels, save_dir='benchmark_results'):
        """Generate visualization plots."""
        Path(save_dir).mkdir(exist_ok=True)

        # 1. ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, color in [('quantum', 'blue'), ('classical', 'red')]:
            if 'confidences' in self.results[name]:
                try:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(test_labels, self.results[name]['confidences'])
                    auc = self.results[name]['auc']
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'{name.capitalize()} (AUC={auc:.3f})')
                except:
                    pass

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve: Quantum vs Classical')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(f'{save_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/roc_curves.png")

        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, name in enumerate(['quantum', 'classical']):
            r = self.results[name]
            cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
            im = axes[idx].imshow(cm, cmap='Blues')
            axes[idx].set_title(f'{name.capitalize()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Normal', 'Ictal'])
            axes[idx].set_yticklabels(['Normal', 'Ictal'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/confusion_matrices.png")

        # 3. Scalability
        if 'scalability' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = self.results['scalability']
            M = np.array(sc['channels'])

            ax.plot(M, sc['quantum_times'], 'o-', lw=2, label='Quantum', color='blue')
            ax.plot(M, sc['classical_times'], 's-', lw=2, label='Classical', color='red')

            # Reference lines
            q0, c0 = sc['quantum_times'][0], sc['classical_times'][0]
            ax.plot(M, q0 * M / M[0], '--', alpha=0.5, color='blue', label='O(M)')
            ax.plot(M, c0 * (M / M[0])**2, '--', alpha=0.5, color='red', label='O(M²)')

            ax.set_xlabel('Channels (M)')
            ax.set_ylabel('Training Time (s)')
            ax.set_title('Scalability: Quantum O(M) vs Classical O(M²)')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.savefig(f'{save_dir}/scalability.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_dir}/scalability.png")

        # 4. Summary comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(len(metrics))
        width = 0.35

        q_vals = [self.results['quantum'].get(m, 0) for m in metrics]
        c_vals = [self.results['classical'].get(m, 0) for m in metrics]

        ax.bar(x - width/2, q_vals, width, label='Quantum', color='blue', alpha=0.8)
        ax.bar(x + width/2, c_vals, width, label='Classical', color='red', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        for i, (q, c) in enumerate(zip(q_vals, c_vals)):
            ax.text(i - width/2, q + 0.02, f'{q:.2f}', ha='center', fontsize=9)
            ax.text(i + width/2, c + 0.02, f'{c:.2f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/comparison.png")

        # 5. Generate report
        self._generate_report(save_dir)

    def _generate_report(self, save_dir):
        """Generate text report."""
        with open(f'{save_dir}/benchmark_report.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("QUANTUM vs CLASSICAL BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("ACCURACY METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Metric':<12} {'Quantum':>10} {'Classical':>10} {'Winner':>10}\n")
            f.write("-" * 60 + "\n")

            for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                q = self.results['quantum'].get(m, 0)
                c = self.results['classical'].get(m, 0)
                winner = 'Quantum' if q >= c else 'Classical'
                f.write(f"{m.capitalize():<12} {q:>10.3f} {c:>10.3f} {winner:>10}\n")

            f.write("\n\nSPEED (ms/sample)\n")
            f.write("-" * 60 + "\n")
            if 'time_mean' in self.results['quantum']:
                q_t = self.results['quantum']['time_mean'] * 1000
                c_t = self.results['classical']['time_mean'] * 1000
                f.write(f"Quantum:   {q_t:.2f} ms\n")
                f.write(f"Classical: {c_t:.2f} ms\n")

            f.write("\n\nCOMPLEXITY\n")
            f.write("-" * 60 + "\n")
            f.write("Quantum:   O(M) gates, O(2^M) Hilbert space\n")
            f.write("Classical: O(M²) features, O(M²) memory\n")

        print(f"Saved: {save_dir}/benchmark_report.txt")


# === Main execution ===

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("QUANTUM vs CLASSICAL BENCHMARK")
    print("=" * 60)

    # Generate synthetic data
    def gen_data(n, M=4, sync=False):
        samples = []
        for _ in range(n):
            if sync:
                t = np.linspace(0, 10, 1000)
                sig = 2 * np.sin(2 * np.pi * 15 * t)
                eeg = np.array([sig * (1 + 0.1*i) + 0.3*np.random.randn(1000) for i in range(M)])
            else:
                eeg = np.random.randn(M, 1000) * 0.5
            samples.append(eeg)
        return samples

    print("\n1. Generating dataset...")
    train_ictal = gen_data(5, sync=True)
    train_normal = gen_data(5, sync=False)
    test_ictal = gen_data(15, sync=True)
    test_normal = gen_data(15, sync=False)

    test_samples = test_ictal + test_normal
    test_labels = [1] * 15 + [0] * 15

    print("\n2. Training quantum model...")
    q_trainer = TemplateTrainer(num_channels=4)
    q_trainer.train(train_ictal[0])
    q_predictor = SeizurePredictor(q_trainer, threshold=0.5)

    print("\n3. Training classical baseline...")
    c_baseline = ClassicalPNBaseline()
    c_baseline.train(train_ictal, train_normal)

    print("\n4. Running benchmarks...")
    bench = PerformanceBenchmark()

    bench.benchmark_accuracy(q_predictor, c_baseline, test_samples, test_labels)
    bench.benchmark_speed(q_predictor, c_baseline, test_samples[0])
    bench.benchmark_scalability(channel_counts=[4, 6, 8])

    print("\n5. Generating visualizations...")
    bench.visualize_results(test_labels)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    print("\nResults in: benchmark_results/")
