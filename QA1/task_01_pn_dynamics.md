# Task 1: Implement PN Dynamics Evolution

## Objective
Implement the Positive-Negative (PN) neural dynamics differential equations that evolve EEG signals into quantum gate parameters (a, b, c).

## Background
From Gupta, Jin, Homma textbook - PN neuron has coupled excitatory and inhibitory dynamics:
- `a`: excitatory state (natural decay with λ_a)
- `b`: shared phase parameter (no intrinsic dynamics)
- `c`: inhibitory state (natural growth with λ_c)

## Requirements

### Create file: `pn_dynamics.py`

1. **Class: `PNDynamics`**
   - Constructor parameters: `lambda_a=0.1, lambda_c=0.1, dt=0.001`
   - Store as instance variables

2. **Method: `evolve_single_channel(eeg_signal)`**
   - Input: 1D numpy array (single EEG channel timeseries)
   - Initialize: `a=0.0, b=0.0, c=0.0`
   - Loop through each timestep:
     - Rectify signal: `f_t = abs(eeg_signal[t])`
     - Compute derivatives:
       ```python
       da = dt * (-lambda_a * a + f_t * (1 - a))
       db = dt * (f_t * (1 - b))
       dc = dt * (lambda_c * c + f_t * (1 - c))
       ```
     - Update: `a += da`, `b += db`, `c += dc`
   - Return: tuple `(a, b, c)`

3. **Method: `evolve_multichannel(eeg_signals)`**
   - Input: 2D numpy array shape `(num_channels, time_steps)`
   - Call `evolve_single_channel()` for each channel
   - Return: list of `(a, b, c)` tuples, one per channel

4. **Add docstrings** to all methods explaining inputs/outputs

## Test Cases

Create test in same file:
```python
if __name__ == "__main__":
    import numpy as np
    
    # Test 1: Single channel with constant signal
    signal = np.ones(100) * 0.5
    pn = PNDynamics(lambda_a=0.1, lambda_c=0.1, dt=0.01)
    a, b, c = pn.evolve_single_channel(signal)
    print(f"Test 1: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    # Expected: a should approach saturation, c should grow
    
    # Test 2: Multi-channel
    signals = np.random.randn(3, 200) * 0.5
    params = pn.evolve_multichannel(signals)
    print(f"Test 2: {len(params)} channels")
    for i, (a, b, c) in enumerate(params):
        print(f"  Ch {i}: a={a:.3f}, b={b:.3f}, c={c:.3f}")
```

## Acceptance Criteria
- [ ] File `pn_dynamics.py` created
- [ ] Class `PNDynamics` with correct constructor
- [ ] `evolve_single_channel()` returns valid (a,b,c) tuple
- [ ] `evolve_multichannel()` returns list of tuples
- [ ] Test cases run without errors
- [ ] Parameters stay bounded (should not exceed ~1.0 for unit input)

## Time Estimate
10 minutes

## Dependencies
- numpy

## Next Task
Task 2: Single-channel quantum A-gate implementation
