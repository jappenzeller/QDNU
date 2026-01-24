# Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Parameter Overflow (a, b, c > 1.0)

**Symptoms:**
```python
RuntimeError: Parameter values exceed bounds
a=1.523, b=2.341, c=0.876
```

**Root Cause:**
- EEG signal not normalized before dynamics integration
- Input amplitudes too large for PN dynamics saturation

**Solutions:**

1. **Increase normalization** in `TemplateTrainer.normalize_eeg()`:
```python
def normalize_eeg(self, eeg_data):
    normalized = np.zeros_like(eeg_data)
    for ch in range(self.num_channels):
        # Robust normalization
        ch_data = eeg_data[ch]
        ch_data = ch_data - np.mean(ch_data)
        ch_data = ch_data / (np.std(ch_data) + 1e-8)
        
        # Scale to [0, 1] with clipping
        ch_data = (ch_data - np.min(ch_data)) / (np.max(ch_data) - np.min(ch_data) + 1e-8)
        normalized[ch] = ch_data * 0.5  # Scale to [0, 0.5] for safety
    
    return normalized
```

2. **Adjust lambda parameters**:
- Decrease λ_a (slower excitatory decay)
- Decrease λ_c (slower inhibitory growth)
```python
trainer = TemplateTrainer(num_channels=4, lambda_a=0.05, lambda_c=0.025)
```

3. **Increase dt** (coarser time steps):
```python
trainer = TemplateTrainer(num_channels=4, dt=0.01)  # Instead of 0.001
```

---

### Issue 2: Fidelity Always Near 0.5

**Symptoms:**
```python
Pre-ictal fidelity: 0.523
Normal fidelity: 0.517
Cannot discriminate!
```

**Root Cause:**
- Template and test states essentially random
- Phase parameter b not capturing synchronization
- Entanglement not functioning correctly

**Diagnosis:**
```python
# Check phase coherence
b_values = [params[1] for params in template_params]
print(f"Phase std: {np.std(b_values)}")
# Should be < 0.1 for synchronized pre-ictal
```

**Solutions:**

1. **Verify EEG has actual synchronization**:
```python
from scipy.signal import hilbert

def check_synchronization(eeg_data):
    phases = []
    for ch in range(len(eeg_data)):
        analytic = hilbert(eeg_data[ch])
        phases.append(np.angle(analytic))
    
    # Compute mean phase coherence
    phase_diffs = []
    for i in range(len(phases)):
        for j in range(i+1, len(phases)):
            phase_diffs.append(phases[i] - phases[j])
    
    coherence = np.abs(np.mean(np.exp(1j * np.array(phase_diffs))))
    print(f"Input synchronization: {coherence:.3f}")
    # Should be > 0.7 for pre-ictal
    
    return coherence
```

2. **Increase integration window**:
```python
# Use longer EEG segments for training
# More time → better integration of phase
preictal_long = preictal_eeg[:, :5000]  # Instead of 500 samples
```

3. **Check circuit construction**:
```python
# Verify shared phase b is used correctly
def verify_agate(params):
    a, b, c = params[0]  # First channel
    qc = create_single_channel_agate(a, b, c)
    
    # Count P gates (should be 4: 2 for E, 2 for I)
    p_gates = sum(1 for gate in qc.data if gate.operation.name == 'p')
    print(f"Phase gates: {p_gates} (expected 4)")
```

---

### Issue 3: Low Prediction Accuracy

**Symptoms:**
```python
Evaluation results:
Accuracy:  0.532
Precision: 0.501
Recall:    0.498
```

**Root Causes & Solutions:**

**A. Threshold too high/low:**
```python
# Try different thresholds
for thresh in [0.5, 0.6, 0.7, 0.8]:
    predictor.threshold = thresh
    eval_results = predictor.evaluate(test_data, labels)
    print(f"Threshold {thresh}: Acc={eval_results['accuracy']:.3f}")
```

**B. Template not representative:**
```python
# Train on multiple pre-ictal examples
preictal_samples = [load_eeg(f) for f in preictal_files]
avg_preictal = np.mean(preictal_samples, axis=0)
trainer.train(avg_preictal)
```

**C. Test data different distribution:**
```python
# Check if test parameters in same range as template
print("Template ranges:")
print(f"  a: [{min(p[0] for p in template_params):.3f}, {max(p[0] for p in template_params):.3f}]")
print("Test ranges:")
test_params = trainer.pn_dynamics.evolve_multichannel(test_eeg)
print(f"  a: [{min(p[0] for p in test_params):.3f}, {max(p[0] for p in test_params):.3f}]")
```

---

### Issue 4: Circuit Too Deep / Slow Simulation

**Symptoms:**
```python
Circuit depth: 2847
Simulation taking > 5 minutes for 19 channels
```

**Solutions:**

1. **Reduce entanglement complexity**:
```python
# Simplify from all-to-all to ring only
def create_multichannel_circuit_shallow(params_list):
    # ... per-channel encoding ...
    
    # Only nearest-neighbor, skip global ancilla
    for i in range(num_channels - 1):
        qc.cnot(1 + 2*i, 1 + 2*(i+1))
```

2. **Use statevector simulator** (fastest):
```python
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')
# Faster than qasm_simulator for statevector operations
```

3. **Reduce number of channels** for testing:
```python
# Downsample to 4-8 channels initially
eeg_subset = eeg_data[:4, :]  # First 4 channels only
```

---

### Issue 5: Qiskit Import Errors

**Symptoms:**
```python
ImportError: cannot import name 'Statevector' from 'qiskit'
ModuleNotFoundError: No module named 'qiskit_algorithms'
```

**Solutions:**

1. **Check Qiskit version**:
```bash
pip show qiskit
# Should be >= 1.0.0
```

2. **Install required packages**:
```bash
pip install qiskit[all] qiskit-aer
```

3. **Update imports for Qiskit 1.0+**:
```python
# Correct imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter, ParameterVector
```

---

### Issue 6: NaN or Inf in Dynamics

**Symptoms:**
```python
Warning: NaN detected in parameter evolution
a=nan, b=inf, c=0.234
```

**Root Cause:**
- Division by zero in normalization
- Numerical instability in integration

**Solutions:**

1. **Add epsilon to prevent division by zero**:
```python
# In dynamics evolution
f_t = abs(eeg_signal[t]) + 1e-10
da = dt * (-lambda_a * a + f_t * (1 - a))
```

2. **Clip parameters**:
```python
# After each update
a = np.clip(a, 0.0, 1.0)
b = np.clip(b, 0.0, 1.0)
c = np.clip(c, 0.0, 1.0)
```

3. **Use smaller dt**:
```python
trainer = TemplateTrainer(dt=0.0001)  # Finer time steps
```

---

### Issue 7: Memory Error with Large Channel Count

**Symptoms:**
```python
MemoryError: Unable to allocate array for 2^40 statevector
```

**Root Cause:**
- 2M qubits = 2^(2M) complex numbers in memory
- 20 channels = 2^40 = 1TB of memory!

**Solutions:**

1. **Use simulator backend** (doesn't store full statevector):
```python
from qiskit import Aer
backend = Aer.get_backend('qasm_simulator')
# Or use IBM Quantum hardware
```

2. **Reduce channels**:
```python
# Clinical standard: 19 channels
# For simulation: 8-12 channels max
```

3. **Use approximate fidelity**:
```python
# Instead of full statevector overlap
def approximate_fidelity(params1, params2):
    # Compute parameter-space distance
    dist = 0
    for p1, p2 in zip(params1, params2):
        a1, b1, c1 = p1
        a2, b2, c2 = p2
        dist += (a1-a2)**2 + (b1-b2)**2 + (c1-c2)**2
    
    # Convert to fidelity-like score
    return np.exp(-dist)
```

---

## Debugging Checklist

When implementing each task, verify:

- [ ] All imports resolve correctly
- [ ] No NaN or Inf in intermediate values
- [ ] Parameters stay in valid ranges [0, 1]
- [ ] Circuit depth is reasonable (< 500 for simulation)
- [ ] Statevector norm = 1.0 (unitarity)
- [ ] Template and test use same normalization
- [ ] Phase coherence differs between pre-ictal and normal
- [ ] Fidelity scores are in valid range [0, 1]

## Performance Monitoring

Add logging to track:

```python
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# In critical sections
logger.info(f"Parameter ranges: a=[{a_min:.3f}, {a_max:.3f}]")
logger.info(f"Circuit depth: {qc.depth()}")
logger.info(f"Fidelity: {fidelity:.4f}")
```

## Getting Help

If issues persist:

1. Check test outputs match expected ranges
2. Visualize intermediate states:
   ```python
   qc.draw('mpl')
   plt.savefig('debug_circuit.png')
   ```
3. Compare against working examples in task test cases
4. Verify EEG data quality (not corrupted, reasonable amplitude)

## Advanced Debugging: Parameter Visualization

```python
import matplotlib.pyplot as plt

def visualize_parameter_evolution(eeg_signal, dt=0.001):
    """Track how a, b, c evolve over time"""
    pn = PNDynamics(lambda_a=0.1, lambda_c=0.05, dt=dt)
    
    a_history, b_history, c_history = [], [], []
    a, b, c = 0.0, 0.0, 0.0
    
    for t in range(len(eeg_signal)):
        f_t = abs(eeg_signal[t])
        a += dt * (-pn.lambda_a * a + f_t * (1 - a))
        b += dt * (f_t * (1 - b))
        c += dt * (pn.lambda_c * c + f_t * (1 - c))
        
        a_history.append(a)
        b_history.append(b)
        c_history.append(c)
    
    plt.figure(figsize=(12, 4))
    plt.plot(a_history, label='a (excitatory)')
    plt.plot(b_history, label='b (phase)')
    plt.plot(c_history, label='c (inhibitory)')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Parameter value')
    plt.title('PN Parameter Evolution')
    plt.savefig('parameter_evolution.png')
    plt.close()
```

---

**Remember**: Most issues stem from:
1. Improper normalization
2. Wrong parameter ranges
3. Memory limitations
4. Insufficient EEG synchronization

Start with small, synthetic data to validate each component before scaling up.
