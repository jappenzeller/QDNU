"""
Positive-Negative Dynamic Neural Unit (PNDNU) Evolution

Implements differential equations that map EEG signals to quantum gate
parameters (a, b, c) with configurable saturation modes.

Reference: Gupta, A., et al. (2024). Positive-negative neuron model.
"""

import numpy as np


class PNDynamics:
    """
    Positive-Negative Dynamic Neural Unit evolution.

    Implements differential equations that map EEG signals to
    quantum gate parameters (a, b, c) with proper saturation.

    Parameters:
        a (excitatory): Decays naturally with lambda_a, driven by signal
        b (phase): Pure integration, creates E-I coupling
        c (inhibitory): Grows naturally with lambda_c, driven by signal

    Saturation modes:
        'clamp': Hard clipping after integration (recommended)
        'logistic': Self-saturating logistic growth terms
        'symmetric': Classical PN with symmetric decay
    """

    def __init__(self, lambda_a=0.1, lambda_c=0.1, dt=0.001,
                 saturation_mode='clamp'):
        """
        Initialize PN dynamics parameters.

        Args:
            lambda_a: Excitatory decay rate (default: 0.1)
            lambda_c: Inhibitory growth rate (default: 0.1)
            dt: Time step for integration (default: 0.001)
            saturation_mode: 'clamp', 'logistic', or 'symmetric'
        """
        self.lambda_a = lambda_a
        self.lambda_c = lambda_c
        self.dt = dt
        self.saturation_mode = saturation_mode

        # Validation
        valid_modes = ['clamp', 'logistic', 'symmetric']
        if saturation_mode not in valid_modes:
            raise ValueError(
                f"Invalid saturation_mode: {saturation_mode}. "
                f"Must be one of {valid_modes}"
            )

    def evolve_single_channel(self, eeg_signal):
        """
        Evolve PN dynamics for a single EEG channel.

        Args:
            eeg_signal: 1D numpy array of EEG samples

        Returns:
            tuple: (a, b, c) final parameter values
                a: Excitatory state [0, 1]
                b: Phase parameter [0, inf)
                c: Inhibitory state [0, 1]
        """
        a, b, c = 0.0, 0.0, 0.0

        for t in range(len(eeg_signal)):
            # Rectify signal (EEG can be negative)
            f_t = abs(eeg_signal[t])

            # Compute derivatives based on saturation mode
            if self.saturation_mode == 'clamp':
                da, db, dc = self._derivatives_clamp(a, b, c, f_t)
            elif self.saturation_mode == 'logistic':
                da, db, dc = self._derivatives_logistic(a, b, c, f_t)
            else:  # symmetric
                da, db, dc = self._derivatives_symmetric(a, b, c, f_t)

            # Update parameters
            a += da
            b += db
            c += dc

            # Apply saturation
            a, b, c = self._apply_saturation(a, b, c)

        return (a, b, c)

    def evolve_multichannel(self, eeg_signals):
        """
        Evolve PN dynamics for multiple EEG channels.

        Args:
            eeg_signals: 2D numpy array (num_channels, time_steps)

        Returns:
            list: List of (a, b, c) tuples, one per channel
        """
        num_channels = eeg_signals.shape[0]
        params = []

        for ch in range(num_channels):
            params_ch = self.evolve_single_channel(eeg_signals[ch])
            params.append(params_ch)

        return params

    # === Derivative computation methods ===

    def _derivatives_clamp(self, a, b, c, f_t):
        """
        Standard PN dynamics with post-integration clamping.

        da/dt = -lambda_a * a + f(t) * (1 - a)
        db/dt = f(t) * (1 - b)
        dc/dt = +lambda_c * c + f(t) * (1 - c)
        """
        da = self.dt * (-self.lambda_a * a + f_t * (1 - a))
        db = self.dt * (f_t * (1 - b))
        dc = self.dt * (self.lambda_c * c + f_t * (1 - c))
        return da, db, dc

    def _derivatives_logistic(self, a, b, c, f_t):
        """
        Logistic saturation - growth terms self-saturate.

        da/dt = -lambda_a * a * (1-a) + f(t) * (1 - a)
        db/dt = f(t) * (1 - b)
        dc/dt = +lambda_c * c * (1-c) + f(t) * (1 - c)
        """
        da = self.dt * (-self.lambda_a * a * (1 - a) + f_t * (1 - a))
        db = self.dt * (f_t * (1 - b))
        dc = self.dt * (self.lambda_c * c * (1 - c) + f_t * (1 - c))
        return da, db, dc

    def _derivatives_symmetric(self, a, b, c, f_t):
        """
        Classical PN with symmetric decay (both a and c decay).

        da/dt = -lambda_a * a + f(t) * (1 - a)
        db/dt = f(t) * (1 - b)
        dc/dt = -lambda_c * c + f(t) * (1 - c)
        """
        da = self.dt * (-self.lambda_a * a + f_t * (1 - a))
        db = self.dt * (f_t * (1 - b))
        dc = self.dt * (-self.lambda_c * c + f_t * (1 - c))
        return da, db, dc

    def _apply_saturation(self, a, b, c):
        """
        Apply bounds to parameters.

        a, c: Clipped to [0, 1]
        b: Clipped to [0, 2*pi] (phase wraps at 2*pi for quantum circuits)
        """
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 2 * np.pi)  # Phase parameter wraps at 2*pi
        c = np.clip(c, 0.0, 1.0)
        return a, b, c


# === Test cases ===

if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility

    print("=" * 50)
    print("PN Dynamics Test Suite")
    print("=" * 50)

    # Test 1: Overflow prevention
    print("\n=== Test 1: Overflow Prevention ===")
    signal = np.ones(1000) * 0.5

    for mode in ['clamp', 'logistic', 'symmetric']:
        pn = PNDynamics(lambda_a=0.1, lambda_c=0.1, dt=0.01,
                        saturation_mode=mode)
        a, b, c = pn.evolve_single_channel(signal)
        print(f"{mode:12s}: a={a:.4f}, b={b:.4f}, c={c:.4f}")

        # Validate bounds
        assert 0 <= a <= 1, f"a overflow in {mode}: {a}"
        assert 0 <= c <= 1, f"c overflow in {mode}: {c}"
        assert b >= 0, f"b negative in {mode}: {b}"
        print(f"             [OK] No overflow")

    # Test 2: Multi-channel
    print("\n=== Test 2: Multi-Channel ===")
    signals = np.random.randn(4, 200) * 0.3 + 0.5
    pn = PNDynamics(saturation_mode='clamp')
    params = pn.evolve_multichannel(signals)

    print(f"Processed {len(params)} channels")
    for i, (a, b, c) in enumerate(params):
        print(f"  Ch {i}: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        assert 0 <= a <= 1 and 0 <= c <= 1, f"Overflow in channel {i}"
    print("[OK] All channels within bounds")

    # Test 3: Strong sustained input (stress test)
    print("\n=== Test 3: Strong Sustained Input (Stress Test) ===")
    signal_strong = np.ones(2000) * 0.9

    for mode in ['clamp', 'logistic', 'symmetric']:
        pn = PNDynamics(lambda_a=0.1, lambda_c=0.1, dt=0.01,
                        saturation_mode=mode)
        a, b, c = pn.evolve_single_channel(signal_strong)
        print(f"{mode:12s}: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        assert 0 <= a <= 1, f"a overflow in {mode}"
        assert 0 <= c <= 1, f"c overflow in {mode}"
    print("[OK] Stress test passed")

    # Test 4: Zero input
    print("\n=== Test 4: Zero Input ===")
    signal_zero = np.zeros(500)
    pn = PNDynamics(saturation_mode='clamp')
    a, b, c = pn.evolve_single_channel(signal_zero)
    print(f"Zero input: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    assert a == 0 and b == 0 and c == 0, "Parameters should stay at 0"
    print("[OK] Zero input handled correctly")

    # Test 5: Sinusoidal input
    print("\n=== Test 5: Sinusoidal Input ===")
    t = np.linspace(0, 10, 1000)
    signal_sin = np.sin(2 * np.pi * 0.5 * t) * 0.4 + 0.5
    pn = PNDynamics(lambda_a=0.1, lambda_c=0.05, dt=0.01)
    a, b, c = pn.evolve_single_channel(signal_sin)
    print(f"Sinusoidal: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    assert 0 <= a <= 1 and 0 <= c <= 1
    print("[OK] Sinusoidal input handled correctly")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
