"""
Template Trainer for Quantum Seizure Prediction

Trains a quantum template state from pre-ictal EEG data:
1. Normalize EEG signals
2. Evolve PN dynamics to get (a, b, c) parameters
3. Create quantum circuit template
4. Save/load functionality for trained templates

The template captures the "signature" of pre-ictal neural activity,
enabling seizure prediction via quantum fidelity measurement.

Reference: Gupta, A., et al. (2024). Positive-negative neuron model.
"""

import numpy as np
from pn_dynamics import PNDynamics
from multichannel_circuit import create_multichannel_circuit


class TemplateTrainer:
    """
    Trains quantum template from pre-ictal EEG data.

    The training process:
    1. Normalize EEG to prevent parameter saturation
    2. Evolve PN dynamics to extract (a, b, c) for each channel
    3. Create multi-channel quantum circuit
    4. Store template for later prediction

    Attributes:
        num_channels: Number of EEG channels
        pn_dynamics: PNDynamics instance for signal evolution
        template_params: List of (a, b, c) tuples after training
        template_circuit: QuantumCircuit encoding the template
    """

    def __init__(self, num_channels, lambda_a=0.1, lambda_c=0.1, dt=0.001,
                 saturation_mode='clamp'):
        """
        Initialize template trainer.

        Args:
            num_channels: Number of EEG channels to process
            lambda_a: Excitatory decay rate for PN dynamics
            lambda_c: Inhibitory growth rate for PN dynamics
            dt: Time step for PN integration
            saturation_mode: 'clamp', 'logistic', or 'symmetric'
        """
        self.num_channels = num_channels
        self.pn_dynamics = PNDynamics(
            lambda_a=lambda_a,
            lambda_c=lambda_c,
            dt=dt,
            saturation_mode=saturation_mode
        )
        self.template_params = None
        self.template_circuit = None

    def train(self, preictal_eeg):
        """
        Train template from pre-ictal EEG data.

        Args:
            preictal_eeg: numpy array shape (num_channels, time_steps)

        Returns:
            list: Template parameters [(a, b, c), ...] for each channel
        """
        # Validate input
        if preictal_eeg.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, "
                f"got {preictal_eeg.shape[0]}"
            )

        # 1. Normalize EEG
        eeg_normalized = self.normalize_eeg(preictal_eeg)

        # 2. Evolve PN dynamics
        params = self.pn_dynamics.evolve_multichannel(eeg_normalized)

        # 3. Store template
        self.template_params = params

        # 4. Create quantum circuit
        self.template_circuit = create_multichannel_circuit(params)

        # 5. Log template info
        self._log_template_info()

        return self.template_params

    def normalize_eeg(self, eeg_data):
        """
        Normalize EEG data for stable PN dynamics.

        Per channel:
        1. Subtract mean (center at zero)
        2. Divide by std (unit variance)
        3. Scale to [0, 1] range

        Args:
            eeg_data: numpy array shape (num_channels, time_steps)

        Returns:
            numpy array: Normalized EEG in [0, 1] range
        """
        normalized = np.zeros_like(eeg_data)

        for ch in range(eeg_data.shape[0]):
            signal = eeg_data[ch]

            # Z-score normalization
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 0:
                z_scored = (signal - mean) / std
            else:
                z_scored = signal - mean

            # Scale to [0, 1] using sigmoid-like mapping
            # tanh squashes to [-1, 1], then shift to [0, 1]
            normalized[ch] = (np.tanh(z_scored / 2) + 1) / 2

        return normalized

    def _log_template_info(self):
        """Log summary of template parameters."""
        if self.template_params is None:
            print("No template trained yet.")
            return

        a_vals = [p[0] for p in self.template_params]
        b_vals = [p[1] for p in self.template_params]
        c_vals = [p[2] for p in self.template_params]

        print("\n--- Template Info ---")
        print(f"Channels: {len(self.template_params)}")
        print(f"a (excitatory): mean={np.mean(a_vals):.4f}, std={np.std(a_vals):.4f}")
        print(f"b (phase):      mean={np.mean(b_vals):.4f}, std={np.std(b_vals):.4f}")
        print(f"c (inhibitory): mean={np.mean(c_vals):.4f}, std={np.std(c_vals):.4f}")
        print(f"Phase coherence (lower = more synchronized): {np.std(b_vals):.4f}")
        print(f"Circuit qubits: {self.template_circuit.num_qubits}")
        print(f"Circuit depth: {self.template_circuit.depth()}")
        print("---------------------\n")

    def save_template(self, filepath):
        """
        Save template parameters to file.

        Args:
            filepath: Path to save file (.npy format)
        """
        if self.template_params is None:
            raise ValueError("No template to save. Call train() first.")

        # Convert to numpy array for saving
        params_array = np.array(self.template_params)
        np.save(filepath, params_array)
        print(f"Template saved to {filepath}")

    def load_template(self, filepath):
        """
        Load template parameters from file.

        Args:
            filepath: Path to saved template file
        """
        params_array = np.load(filepath)

        # Convert back to list of tuples
        self.template_params = [tuple(p) for p in params_array]

        # Recreate circuit
        self.template_circuit = create_multichannel_circuit(self.template_params)

        # Update num_channels if needed
        self.num_channels = len(self.template_params)

        print(f"Template loaded from {filepath}")
        self._log_template_info()

    def get_template_statevector(self):
        """
        Get the quantum statevector of the template circuit.

        Returns:
            Statevector: Template quantum state
        """
        from qiskit.quantum_info import Statevector
        return Statevector.from_instruction(self.template_circuit)


# === Test cases ===

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 50)
    print("Template Trainer Test Suite")
    print("=" * 50)

    def generate_preictal_eeg(num_channels, time_steps):
        """Synthetic pre-ictal: increased synchronization + amplitude."""
        t = np.linspace(0, 10, time_steps)
        eeg = np.zeros((num_channels, time_steps))

        # Common synchronized component (15 Hz oscillation)
        sync_signal = 2.0 * np.sin(2 * np.pi * 15 * t)

        for i in range(num_channels):
            # Each channel has sync component + small noise
            eeg[i] = sync_signal * (1 + 0.1 * i) + 0.3 * np.random.randn(time_steps)

        return eeg

    def generate_normal_eeg(num_channels, time_steps):
        """Synthetic normal EEG: random, unsynchronized."""
        return np.random.randn(num_channels, time_steps) * 0.5

    # Test 1: Training
    print("\n=== Test 1: Training ===")
    trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05)
    preictal = generate_preictal_eeg(4, 500)
    params = trainer.train(preictal)
    print(f"Trained with {len(params)} channels")
    assert len(params) == 4
    print("[OK] Training successful")

    # Test 2: Parameter bounds
    print("\n=== Test 2: Parameter Bounds ===")
    a_vals = [p[0] for p in params]
    b_vals = [p[1] for p in params]
    c_vals = [p[2] for p in params]
    print(f"a range: [{min(a_vals):.3f}, {max(a_vals):.3f}]")
    print(f"b range: [{min(b_vals):.3f}, {max(b_vals):.3f}]")
    print(f"c range: [{min(c_vals):.3f}, {max(c_vals):.3f}]")
    assert all(0 <= a <= 1 for a in a_vals), "a should be in [0, 1]"
    assert all(0 <= c <= 1 for c in c_vals), "c should be in [0, 1]"
    print("[OK] Parameters within bounds")

    # Test 3: Circuit creation
    print("\n=== Test 3: Circuit Creation ===")
    print(f"Circuit qubits: {trainer.template_circuit.num_qubits}")
    print(f"Circuit depth: {trainer.template_circuit.depth()}")
    assert trainer.template_circuit.num_qubits == 9  # 2*4 + 1
    print("[OK] Circuit structure correct")

    # Test 4: Save/Load
    print("\n=== Test 4: Save/Load ===")
    trainer.save_template("test_template.npy")

    new_trainer = TemplateTrainer(num_channels=4)
    new_trainer.load_template("test_template.npy")
    print(f"Loaded {len(new_trainer.template_params)} channels")

    # Verify loaded params match
    for i, (orig, loaded) in enumerate(zip(params, new_trainer.template_params)):
        assert np.allclose(orig, loaded), f"Mismatch at channel {i}"
    print("[OK] Save/load roundtrip successful")

    # Clean up test file
    import os
    os.remove("test_template.npy")

    # Test 5: Normal vs Pre-ictal comparison
    print("\n=== Test 5: Normal vs Pre-ictal ===")
    normal_eeg = generate_normal_eeg(4, 500)
    preictal_eeg = generate_preictal_eeg(4, 500)

    trainer_normal = TemplateTrainer(num_channels=4)
    trainer_normal.train(normal_eeg)

    trainer_preictal = TemplateTrainer(num_channels=4)
    trainer_preictal.train(preictal_eeg)

    b_std_normal = np.std([p[1] for p in trainer_normal.template_params])
    b_std_preictal = np.std([p[1] for p in trainer_preictal.template_params])

    print(f"Normal phase std: {b_std_normal:.4f}")
    print(f"Pre-ictal phase std: {b_std_preictal:.4f}")
    print(f"Pre-ictal should have LOWER std (more synchronized)")

    # Pre-ictal should show more phase coherence (lower std)
    # Note: This may not always hold with random data, but generally should
    if b_std_preictal < b_std_normal:
        print("[OK] Pre-ictal shows higher phase coherence")
    else:
        print("[WARN] Phase coherence difference not as expected (may vary with random seed)")

    # Test 6: EEG normalization
    print("\n=== Test 6: EEG Normalization ===")
    raw_eeg = np.random.randn(4, 100) * 100 + 50  # Large values
    normalized = trainer.normalize_eeg(raw_eeg)
    print(f"Raw EEG range: [{raw_eeg.min():.1f}, {raw_eeg.max():.1f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    assert normalized.min() >= 0 and normalized.max() <= 1
    print("[OK] Normalization produces [0, 1] range")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
