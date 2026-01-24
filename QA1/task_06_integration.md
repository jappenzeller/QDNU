# Task 6: Integration Testing and End-to-End Pipeline

## Objective
Create main pipeline that integrates all components and validates the complete quantum seizure prediction system.

## Requirements

### Create file: `main_pipeline.py`

1. **Function: `load_eeg_data(filepath)`**
   - Placeholder for loading real EEG data
   - For now, return synthetic data
   - Should support common formats (.edf, .csv)
   ```python
   def load_eeg_data(filepath=None, num_channels=4, duration_sec=10, fs=256):
       """
       Args:
           filepath: path to EEG file (if None, generate synthetic)
           num_channels: number of EEG channels
           duration_sec: duration in seconds
           fs: sampling frequency
       Returns:
           eeg_data: array (num_channels, time_steps)
       """
       if filepath is None:
           # Generate synthetic for testing
           time_steps = duration_sec * fs
           return generate_synthetic_eeg(num_channels, time_steps)
       else:
           # TODO: Implement real EEG loading
           raise NotImplementedError("Real EEG loading not yet implemented")
   ```

2. **Function: `run_training_pipeline(preictal_files, config)`**
   - Input: list of pre-ictal EEG file paths, configuration dict
   - Steps:
     ```python
     # 1. Load and concatenate pre-ictal data
     preictal_data = []
     for f in preictal_files:
         eeg = load_eeg_data(f)
         preictal_data.append(eeg)
     
     # 2. Initialize trainer
     trainer = TemplateTrainer(
         num_channels=config['num_channels'],
         lambda_a=config['lambda_a'],
         lambda_c=config['lambda_c'],
         dt=config['dt']
     )
     
     # 3. Train on concatenated or averaged data
     avg_preictal = np.mean(preictal_data, axis=0)
     trainer.train(avg_preictal)
     
     # 4. Save template
     trainer.save_template(config['template_path'])
     
     # 5. Return trainer
     return trainer
     ```

3. **Function: `run_prediction_pipeline(test_files, trainer, config)`**
   - Input: test EEG files, trained TemplateTrainer, config
   - Steps:
     ```python
     predictor = SeizurePredictor(trainer, threshold=config['threshold'])
     
     results = []
     for f in test_files:
         eeg = load_eeg_data(f)
         pred, fid, metrics = predictor.predict(eeg)
         results.append({
             'file': f,
             'prediction': pred,
             'fidelity': fid,
             'metrics': metrics
         })
     
     return results
     ```

4. **Function: `visualize_results(results, output_path)`**
   - Create plots:
     - Fidelity scores over time/files
     - Confusion matrix (if labels available)
     - ROC curve
   - Save figures to output_path

5. **Function: `generate_synthetic_eeg(num_channels, time_steps, signal_type='normal')`**
   - Generate test data with known characteristics
   - `signal_type` in ['normal', 'preictal', 'ictal']
   - Return: array (num_channels, time_steps)

6. **Main execution block:**
   ```python
   if __name__ == "__main__":
       # Configuration
       config = {
           'num_channels': 4,
           'lambda_a': 0.1,
           'lambda_c': 0.05,
           'dt': 0.001,
           'threshold': 0.7,
           'template_path': 'trained_template.npy'
       }
       
       # Full pipeline test
       print("="*50)
       print("QUANTUM SEIZURE PREDICTION PIPELINE")
       print("="*50)
       
       # ... (see test cases below)
   ```

## Test Cases

```python
# In main execution block:

# Test 1: End-to-end with synthetic data
print("\n=== Test 1: Training Phase ===")
preictal_files = [None, None, None]  # 3 synthetic pre-ictal samples
trainer = run_training_pipeline(preictal_files, config)
print("✓ Training complete")

# Test 2: Prediction on various signal types
print("\n=== Test 2: Prediction Phase ===")
test_cases = [
    ('preictal', True),   # Expected positive
    ('normal', False),     # Expected negative
    ('preictal', True),
    ('normal', False)
]

correct = 0
for signal_type, expected in test_cases:
    # Generate test EEG
    eeg = generate_synthetic_eeg(4, 2560, signal_type=signal_type)
    
    # Predict
    predictor = SeizurePredictor(trainer, threshold=config['threshold'])
    pred, fid, _ = predictor.predict(eeg)
    
    # Check
    if pred == expected:
        correct += 1
        status = "✓"
    else:
        status = "✗"
    
    print(f"{status} {signal_type}: pred={pred}, fid={fid:.3f} (expected {expected})")

accuracy = correct / len(test_cases)
print(f"\nAccuracy: {accuracy:.1%}")

# Test 3: Batch evaluation
print("\n=== Test 3: Batch Evaluation ===")
test_segments = []
labels = []

for _ in range(20):
    test_segments.append(generate_synthetic_eeg(4, 2560, 'preictal'))
    labels.append(1)
for _ in range(20):
    test_segments.append(generate_synthetic_eeg(4, 2560, 'normal'))
    labels.append(0)

predictor = SeizurePredictor(trainer, threshold=config['threshold'])
eval_results = predictor.evaluate(test_segments, labels)

print(f"Accuracy:  {eval_results['accuracy']:.3f}")
print(f"Precision: {eval_results['precision']:.3f}")
print(f"Recall:    {eval_results['recall']:.3f}")
print(f"F1 Score:  {eval_results['f1']:.3f}")

# Test 4: Save/load template and predict
print("\n=== Test 4: Template Persistence ===")
trainer.save_template('test_template.npy')
new_trainer = TemplateTrainer(num_channels=4)
new_trainer.load_template('test_template.npy')
new_predictor = SeizurePredictor(new_trainer, threshold=0.7)

test_eeg = generate_synthetic_eeg(4, 2560, 'preictal')
pred, fid, _ = new_predictor.predict(test_eeg)
print(f"✓ Loaded template prediction: {pred}, fidelity: {fid:.3f}")

# Test 5: Parameter sensitivity analysis
print("\n=== Test 5: Parameter Sensitivity ===")
for lambda_a in [0.05, 0.1, 0.15]:
    for lambda_c in [0.025, 0.05, 0.075]:
        config_test = config.copy()
        config_test['lambda_a'] = lambda_a
        config_test['lambda_c'] = lambda_c
        
        trainer_test = TemplateTrainer(4, lambda_a, lambda_c, 0.001)
        trainer_test.train(generate_synthetic_eeg(4, 2560, 'preictal'))
        
        pred_test = SeizurePredictor(trainer_test, threshold=0.7)
        _, fid, _ = pred_test.predict(generate_synthetic_eeg(4, 2560, 'preictal'))
        
        print(f"λ_a={lambda_a:.2f}, λ_c={lambda_c:.3f}: fidelity={fid:.3f}")

print("\n" + "="*50)
print("ALL TESTS COMPLETE")
print("="*50)
```

## Acceptance Criteria
- [ ] File `main_pipeline.py` created
- [ ] All 5 test cases pass
- [ ] Training pipeline produces valid template
- [ ] Prediction pipeline classifies pre-ictal vs normal correctly
- [ ] Batch evaluation shows >70% accuracy on synthetic data
- [ ] Template save/load preserves prediction capability
- [ ] Parameter sensitivity analysis runs without errors
- [ ] Code is well-documented with docstrings

## Time Estimate
15 minutes

## Dependencies
- All previous task modules
- numpy
- matplotlib (for visualization)
- sklearn (for metrics)

## Expected Output

```
==================================================
QUANTUM SEIZURE PREDICTION PIPELINE
==================================================

=== Test 1: Training Phase ===
Trained with 4 channels
Template parameters:
  a: mean=X.XXX, std=X.XXX
  b: mean=X.XXX, std=X.XXX
  c: mean=X.XXX, std=X.XXX
✓ Training complete

=== Test 2: Prediction Phase ===
✓ preictal: pred=True, fid=0.XXX (expected True)
✓ normal: pred=False, fid=0.XXX (expected False)
✓ preictal: pred=True, fid=0.XXX (expected True)
✓ normal: pred=False, fid=0.XXX (expected False)

Accuracy: 100.0%

=== Test 3: Batch Evaluation ===
Accuracy:  0.XXX
Precision: 0.XXX
Recall:    0.XXX
F1 Score:  0.XXX

=== Test 4: Template Persistence ===
✓ Loaded template prediction: True, fidelity: 0.XXX

=== Test 5: Parameter Sensitivity ===
...

==================================================
ALL TESTS COMPLETE
==================================================
```

## Notes
- This validates the entire quantum advantage pipeline
- Synthetic data allows controlled testing
- Real EEG integration is next step (separate task)
- Success criteria: Pre-ictal detection with >70% accuracy

## Next Steps
- Task 7: Real EEG data integration (.edf format)
- Task 8: Quantum hardware deployment (real backend)
- Task 9: Performance benchmarking vs classical methods
