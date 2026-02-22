## Per-Patient Quantum Performance — Polarity-Aware Analysis

### Aggregate Impact of Polarity Calibration
| Metric                 | Value  |
|------------------------|--------|
| Raw overall HW AUC     | 0.5241 |
| Calibrated overall AUC | 0.6365 |
| N patients inverted    | 3 / 7 |
| Calibration lift       | +0.1124 |

### Mandelbrot Concordance
Patients in both analyses: 4
Polarity matches Mandelbrot direction: 2 / 4 (50%)
Interpretation: inconclusive

### Patient Classification Summary
| Category         | N | Avg Calibrated AUC | Notes |
|------------------|---|--------------------|-------|
| STRONG           | 2 | 0.676              |  |
| STRONG-INVERTED  | 1 | 0.717              | polarity flip needed |
| MODERATE         | 4 | 0.597              |  |
| UNTESTED-PROMISING | 11 | N/A                |  |
| UNTESTED         | 4 | N/A                |  |

### Per-Patient Table (sorted by calibrated HW AUC descending)
| Subject | Classical | Raw HW | Cal HW | Polarity | Mandelbrot | Category |
|---------|-----------|--------|--------|----------|------------|----------|
| chb11   | 0.783     | 0.283  | 0.717  | inv      | N/A        | STRONG-I |
| chb01   | 1.000     | 0.686  | 0.686  | std      | ictal_insi | STRONG   |
| chb07   | 0.933     | 0.667  | 0.667  | std      | ictal_insi | STRONG   |
| chb21   | 0.743     | 0.387  | 0.613  | inv      | N/A        | MODERATE |
| chb05   | 1.000     | 0.610  | 0.610  | std      | ictal_outs | MODERATE |
| chb14   | 0.636     | 0.600  | 0.600  | std      | N/A        | MODERATE |
| chb03   | 0.686     | 0.436  | 0.564  | inv      | ictal_insi | MODERATE |
| chb02   | 0.820     | N/A    | N/A    | N/A      | ictal_outs | UNTESTED |
| chb04   | 0.787     | N/A    | N/A    | N/A      | ictal_outs | UNTESTED |
| chb06   | 0.463     | N/A    | N/A    | N/A      | ictal_outs | UNTESTED |
| chb08   | 0.760     | N/A    | N/A    | N/A      | ictal_outs | UNTESTED |
| chb09   | 0.938     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb10   | 0.914     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb13   | 0.679     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb15   | 0.758     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb16   | 0.840     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb17   | 0.783     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb18   | 0.792     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb19   | 0.800     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb20   | 0.969     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb22   | 1.000     | N/A    | N/A    | N/A      | N/A        | UNTESTED |
| chb23   | 0.957     | N/A    | N/A    | N/A      | N/A        | UNTESTED |

### Data Sparsity Notes
- Hardware runs per patient: min=0, max=1
- Subjects with hardware data: 7 / 22
- Single-run hardware patients: 7