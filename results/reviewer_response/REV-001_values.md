# REV-001 reviewer-response values

Generated: 2026-05-28T18:16:26.861752

Paste each value into the response letter where the bracketed key appears.

```
R1.6_xgb_8ch:    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, objective=binary:logistic (XGBoost default for binary), eval_metric=logloss (n_estimators=200, max_depth=6, reg_alpha=0.1, reg_lambda=1.0, n_bags=5, early_stopping=none, random_state=42+bag)
R1.6_xgb_18ch:   learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, objective=binary:logistic (XGBoost default for binary), eval_metric=logloss (identical config to 8-ch baseline; 18 input channels, 2016 features)
R1.7_cz:         CZ(12)=162, CZ(16)=233, CZ(24)=404, CZ(32)=595 ; refit a=19.365, b=-50.990, R2=0.9910 (over M in {2,4,6,8,12,16,24,32}) ; original-range refit a=14.100, b=-17.500, R2=0.9906 -- matches published 14.1M - 17.5
R1.8_delong_p:   0.002676 (V2 vs classical-8ch, AUC_V2=0.5344, AUC_cls8=0.6253, deltaAUC=-0.0909, z=-3.0027, n=504)
R1.8_perm_p:     V1=0.0142 (AUC=0.4444), V2=0.09119 (AUC=0.5344), V3=0.1293 (AUC=0.5291) [K=10,000 stratified label permutations; one-sided in direction of observed AUC]
R1.8_ci:         V1=[0.3935,0.4955], V2=[0.4838,0.5847], V3=[0.4770,0.5792], cls8=[0.5770,0.6738], cls18=[0.7806,0.8574] [B=10,000 stratified percentile bootstrap]
R1.8_run_sd:     classical_8ch_runs=['0.5912', '0.6169', '0.6576', '0.6141', '0.6276'], SD=0.0241 ; quantum=0.0000 (method: statevector simulation; deterministic dual-template; shot noise N/A (no measurement sampling).)
R1.10_mode:      noiseless statevector simulation (no shot sampling, no noise model). The classification pipeline (scripts/run_quantum_loso*.py, V1/V2/V3) calls qiskit.quantum_info.Statevector via QA1.multichannel_circuit.get_statevector() to evaluate ictal/interictal fidelity per segment. Hardware execution in this paper covers ONLY the circuit-validation experiments (Hellinger fidelity, discrimination matrix); the AUC numbers in Table 9 are statevector results.
R1.10_noiseless_auc: already noiseless (V2 = 0.5344, matches published 0.534 within +-0.005). No additional run required: the encoding-bottleneck argument is unaffected by noise because device noise is not in this experiment.
R2.5_repro:      qiskit=2.3.0, seed_transpiler=42, rng_seed=42+bag (XGBoost bags 0..4); permutation/bootstrap rng_seed=1729, shots_classification=N/A (statevector), shots_config_disc=8192, torino_calibration_date=not recoverable from logs (hardware run completed 2026-02-15T14:36:22.375999; backend properties were not serialized at run time)
```