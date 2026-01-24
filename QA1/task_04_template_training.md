# Task 4: Template Training from Pre-Ictal EEG

## Objective
Create training pipeline that learns quantum template state from pre-ictal EEG data.

## Background
Template matching approach:
1. Train: Integrate PN dynamics over pre-ictal window → quantum state
2. Test: Integrate new EEG → quantum state  
3. Measure: Quantum fidelity indicates match

## Requirements

### Create file: `template_trainer.py`

1. **Class: `TemplateTrainer`**
   
   **Constructor:**
   ```python
   def __init__(self, num_channels, lambda_a=0.1, lambda_c=0.1, dt=0.001):
       self.num_channels = num_channels
       self.pn_dynamics = PNDynamics(lambda_a, lambda_c, dt)
       self.template_params = None
       self.template_circuit = None
   ```

2. **Method: `train(preictal_eeg)`**
   - Input: `preictal_eeg` numpy array shape `(num_channels, time_steps)`
   - Steps:
     ```python
     # 1. Normalize EEG (important for stability)
     eeg_normalized = self.normalize_eeg(preictal_eeg)
     
     # 2. Evolve PN dynamics
     params = self.pn_dynamics.evolve_multichannel(eeg_normalized)
     
     # 3. Store template
     self.template_params = params
     
     # 4. Create quantum circuit
     self.template_circuit = create_multichannel_circuit(params)
     
     # 5. Log template info
     self._log_template_info()
     ```
   - Return: `self.template_params`

3. **Method: `normalize_eeg(eeg_data)`**
   - Input: raw EEG array
   - Per channel:
     - Subtract mean
     - Divide by std
     - Scale to [0, 1] range
   - Return: normalized array

4. **Method: `_log_template_info()`**
   - Print summary of template parameters
   - Compute and display:
     - Mean/std of a, b, c across channels
     - Phase coherence (std of b values - lower = more synchronized)

5. **Method: `save_template(filepath)`**
   - Save template_params to disk (use numpy.save or pickle)
   
6. **Method: `load_template(filepath)`**
   - Load template_params from disk
   - Recreate template_circuit

## Test Cases

```python
if __name__ == "__main__":
    import numpy as np
    from pn_dynamics import PNDynamics
    from multichannel_circuit import create_multichannel_circuit
    
    # Generate synthetic pre-ictal EEG
    def generate_preictal_eeg(num_channels, time_steps):
        """Synthetic pre-ictal: increased sync + amplitude"""
        t = np.linspace(0, 10, time_steps)
        eeg = np.zeros((num_channels, time_steps))
        
        # Common synchronized component
        sync_signal = 2.0 * np.sin(2*np.pi*15*t)
        
        for i in range(num_channels):
            # Each channel has sync component + small noise
            eeg[i] = sync_signal * (1 + 0.1*i) + 0.3*np.random.randn(time_steps)
        
        return eeg
    
    # Test 1: Train on pre-ictal data
    print("=== Test 1: Training ===")
    trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05)
    preictal = generate_preictal_eeg(4, 500)
    params = trainer.train(preictal)
    print(f"Trained with {len(params)} channels")
    
    # Test 2: Check template parameters
    print("\n=== Test 2: Template Parameters ===")
    a_vals = [p[0] for p in params]
    b_vals = [p[1] for p in params]
    c_vals = [p[2] for p in params]
    print(f"a: mean={np.mean(a_vals):.3f}, std={np.std(a_vals):.3f}")
    print(f"b: mean={np.mean(b_vals):.3f}, std={np.std(b_vals):.3f}")
    print(f"c: mean={np.mean(c_vals):.3f}, std={np.std(c_vals):.3f}")
    print(f"Phase coherence (std of b): {np.std(b_vals):.3f}")
    # Expect low std in b for synchronized signal
    
    # Test 3: Circuit creation
    print("\n=== Test 3: Circuit ===")
    print(f"Circuit qubits: {trainer.template_circuit.num_qubits}")
    print(f"Circuit depth: {trainer.template_circuit.depth()}")
    
    # Test 4: Save/load
    print("\n=== Test 4: Save/Load ===")
    trainer.save_template("test_template.npy")
    
    new_trainer = TemplateTrainer(num_channels=4)
    new_trainer.load_template("test_template.npy")
    print(f"Loaded template with {len(new_trainer.template_params)} channels")
    
    # Test 5: Different EEG types
    print("\n=== Test 5: Normal vs Pre-ictal ===")
    normal_eeg = np.random.randn(4, 500) * 0.5  # Random, unsynchronized
    preictal_eeg = generate_preictal_eeg(4, 500)
    
    trainer_normal = TemplateTrainer(num_channels=4)
    trainer_normal.train(normal_eeg)
    
    trainer_preictal = TemplateTrainer(num_channels=4)
    trainer_preictal.train(preictal_eeg)
    
    b_std_normal = np.std([p[1] for p in trainer_normal.template_params])
    b_std_preictal = np.std([p[1] for p in trainer_preictal.template_params])
    
    print(f"Normal b coherence: {b_std_normal:.3f}")
    print(f"Pre-ictal b coherence: {b_std_preictal:.3f}")
    print(f"Pre-ictal should have LOWER std (more synchronized)")
```

## Acceptance Criteria
- [ ] File `template_trainer.py` created
- [ ] `TemplateTrainer` class with all methods
- [ ] `train()` successfully processes multi-channel EEG
- [ ] EEG normalization prevents parameter overflow
- [ ] Template info logging shows parameter statistics
- [ ] Save/load roundtrip preserves parameters
- [ ] Pre-ictal data shows lower phase coherence std than normal
- [ ] Template circuit is valid QuantumCircuit

## Time Estimate
10 minutes

## Dependencies
- numpy
- pn_dynamics.py (Task 1)
- multichannel_circuit.py (Task 3)

## Notes
- Normalization is critical - raw EEG can saturate parameters
- Phase coherence (std of b values) is key discriminator
- Template represents "signature" of pre-ictal state

## Next Task
Task 5: Prediction via quantum fidelity measurement
