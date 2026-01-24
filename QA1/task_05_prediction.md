# Task 5: Prediction via Quantum Fidelity Measurement

## Objective
Implement prediction pipeline that compares test EEG against trained template using quantum state fidelity.

## Background
Quantum advantage: Single fidelity measurement F = |⟨ψ_template|ψ_test⟩|² captures all channel correlations simultaneously (vs O(M²) classical comparisons).

## Requirements

### Create file: `seizure_predictor.py`

1. **Class: `SeizurePredictor`**
   
   **Constructor:**
   ```python
   def __init__(self, trainer, threshold=0.7):
       """
       Args:
           trainer: TemplateTrainer instance (already trained)
           threshold: fidelity threshold for positive prediction
       """
       if trainer.template_params is None:
           raise ValueError("Trainer must be trained first")
       
       self.trainer = trainer
       self.threshold = threshold
       self.num_channels = trainer.num_channels
   ```

2. **Method: `predict(test_eeg)`**
   - Input: `test_eeg` array shape `(num_channels, time_steps)`
   - Steps:
     ```python
     # 1. Normalize test EEG
     test_normalized = self.trainer.normalize_eeg(test_eeg)
     
     # 2. Evolve test dynamics
     test_params = self.trainer.pn_dynamics.evolve_multichannel(test_normalized)
     
     # 3. Create test circuit
     test_circuit = create_multichannel_circuit(test_params)
     
     # 4. Compute fidelity
     fidelity = self.compute_fidelity(self.trainer.template_circuit, test_circuit)
     
     # 5. Make prediction
     prediction = fidelity > self.threshold
     
     # 6. Compute confidence metrics
     metrics = self._compute_metrics(test_params, fidelity)
     ```
   - Return: `(prediction, fidelity, metrics)`

3. **Method: `compute_fidelity(circuit1, circuit2)`**
   - Input: two QuantumCircuits
   - Use Qiskit Statevector:
     ```python
     from qiskit.quantum_info import Statevector
     
     sv1 = Statevector(circuit1)
     sv2 = Statevector(circuit2)
     
     # Fidelity = |⟨ψ₁|ψ₂⟩|²
     fidelity = abs(sv1.inner(sv2))**2
     ```
   - Return: float in [0, 1]

4. **Method: `_compute_metrics(test_params, fidelity)`**
   - Compute additional diagnostic metrics:
     ```python
     {
         'fidelity': fidelity,
         'phase_coherence': std of b values,
         'excitatory_mean': mean of a values,
         'inhibitory_mean': mean of c values,
         'param_distance': L2 distance to template params
     }
     ```
   - Return: dictionary

5. **Method: `predict_batch(test_segments)`**
   - Input: list of EEG segments
   - Call `predict()` on each segment
   - Return: list of (prediction, fidelity, metrics) tuples

6. **Method: `evaluate(test_data, true_labels)`**
   - Input: test EEG segments and ground truth labels
   - Compute:
     - Accuracy
     - Precision
     - Recall
     - F1 score
     - ROC-AUC
   - Return: evaluation dictionary

## Test Cases

```python
if __name__ == "__main__":
    import numpy as np
    from template_trainer import TemplateTrainer
    from multichannel_circuit import create_multichannel_circuit
    
    # Setup: Train a template
    def generate_preictal_eeg(num_channels, time_steps):
        t = np.linspace(0, 10, time_steps)
        eeg = np.zeros((num_channels, time_steps))
        sync_signal = 2.0 * np.sin(2*np.pi*15*t)
        for i in range(num_channels):
            eeg[i] = sync_signal * (1 + 0.1*i) + 0.3*np.random.randn(time_steps)
        return eeg
    
    def generate_normal_eeg(num_channels, time_steps):
        return np.random.randn(num_channels, time_steps) * 0.5
    
    trainer = TemplateTrainer(num_channels=4)
    preictal = generate_preictal_eeg(4, 500)
    trainer.train(preictal)
    
    predictor = SeizurePredictor(trainer, threshold=0.7)
    
    # Test 1: Predict on pre-ictal (should be positive)
    print("=== Test 1: Pre-ictal Detection ===")
    test_preictal = generate_preictal_eeg(4, 500)
    pred, fid, metrics = predictor.predict(test_preictal)
    print(f"Prediction: {pred} (expected True)")
    print(f"Fidelity: {fid:.3f}")
    print(f"Metrics: {metrics}")
    
    # Test 2: Predict on normal (should be negative)
    print("\n=== Test 2: Normal EEG ===")
    test_normal = generate_normal_eeg(4, 500)
    pred, fid, metrics = predictor.predict(test_normal)
    print(f"Prediction: {pred} (expected False)")
    print(f"Fidelity: {fid:.3f}")
    
    # Test 3: Batch prediction
    print("\n=== Test 3: Batch Prediction ===")
    batch = [
        generate_preictal_eeg(4, 500),
        generate_normal_eeg(4, 500),
        generate_preictal_eeg(4, 500),
        generate_normal_eeg(4, 500)
    ]
    results = predictor.predict_batch(batch)
    for i, (pred, fid, _) in enumerate(results):
        print(f"Segment {i}: pred={pred}, fidelity={fid:.3f}")
    
    # Test 4: Evaluation metrics
    print("\n=== Test 4: Evaluation ===")
    test_segments = [generate_preictal_eeg(4, 500) for _ in range(10)]
    test_segments += [generate_normal_eeg(4, 500) for _ in range(10)]
    true_labels = [1]*10 + [0]*10  # 1=preictal, 0=normal
    
    eval_metrics = predictor.evaluate(test_segments, true_labels)
    print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"Precision: {eval_metrics['precision']:.3f}")
    print(f"Recall: {eval_metrics['recall']:.3f}")
    
    # Test 5: Threshold sensitivity
    print("\n=== Test 5: Threshold Analysis ===")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        predictor.threshold = thresh
        pred, fid, _ = predictor.predict(test_preictal)
        print(f"Threshold {thresh}: pred={pred}, fid={fid:.3f}")
```

## Acceptance Criteria
- [ ] File `seizure_predictor.py` created
- [ ] `SeizurePredictor` class with all methods
- [ ] `predict()` returns (bool, float, dict) tuple
- [ ] `compute_fidelity()` returns value in [0, 1]
- [ ] Pre-ictal test data yields higher fidelity than normal
- [ ] Batch prediction handles multiple segments
- [ ] `evaluate()` computes standard ML metrics
- [ ] Different thresholds affect prediction outcomes

## Time Estimate
12 minutes

## Dependencies
- numpy
- qiskit
- sklearn (for evaluation metrics)
- template_trainer.py (Task 4)
- multichannel_circuit.py (Task 3)

## Notes
- Fidelity measurement is the quantum "kernel" - computes similarity in exponential Hilbert space
- Threshold tuning is crucial for sensitivity/specificity tradeoff
- Metrics help diagnose why predictions succeed/fail

## Next Task
Task 6: Integration testing and validation
