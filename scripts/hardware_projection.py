#!/usr/bin/env python3
"""
================================================================================
PROMPT 010-FIX — Hardware Readiness Projection (Corrected)
================================================================================

Projects when IBM Quantum hardware will achieve clinically meaningful seizure
detection performance by analyzing:

1. Regime-specific comparators (not a single threshold)
2. Hardware performance with polarity calibration
3. Gap trajectory excluding degenerate CH4
4. Two-dimensional scaling: channels AND window size
5. Simulation-hardware inversion handling

Key corrections from PROMPT 018/017-SIM discoveries:
- Simulation can UNDERPERFORM hardware (signed gaps)
- Feature dimensionality determines fair comparisons
- Window size has larger impact than channel count for classical
- Polarity statistics vary with encoding parameters

Inputs:
  - results/projection/simulation_scaling.json (PROMPT 008)
  - results/projection/hardware_scaling.json (PROMPT 009)
  - results/calibration/calibrated_gap_inputs.json (PROMPT 012)
  - results/full_cohort/full_cohort_polarity.json (PROMPT 018)
  - results/window_analysis/quantum_20s_simulation.json (PROMPT 017-SIM)
  - results/window_analysis/window_size_impact.json (PROMPT 015)
  - results/hardware_validation/ch12_plv_hardware.json (PROMPT 013, pending)
  - results/hardware_validation/ch16_plv_hardware.json (PROMPT 014, pending)
  - results/hardware_validation/ch8_20s_hardware.json (PROMPT 017, pending)

Outputs:
  - results/projection/hardware_projection.json
  - results/projection/hardware_projection.md

================================================================================
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "projection"

# Input files - existing
SIM_SCALING = RESULTS_DIR / "projection" / "simulation_scaling.json"
HW_SCALING = RESULTS_DIR / "projection" / "hardware_scaling.json"
CALIBRATED_GAP = RESULTS_DIR / "calibration" / "calibrated_gap_inputs.json"
CLASSICAL_SCALING = RESULTS_DIR / "scaling" / "classical_channel_scaling.json"

# Input files - new from PROMPT 015-018
FULL_COHORT = RESULTS_DIR / "full_cohort" / "full_cohort_polarity.json"
QUANTUM_20S_SIM = RESULTS_DIR / "window_analysis" / "quantum_20s_simulation.json"
WINDOW_IMPACT = RESULTS_DIR / "window_analysis" / "window_size_impact.json"

# Input files - pending (March 22+)
CH12_HARDWARE = RESULTS_DIR / "hardware_validation" / "ch12_plv_hardware.json"
CH16_HARDWARE = RESULTS_DIR / "hardware_validation" / "ch16_plv_hardware.json"
CH8_20S_HARDWARE = RESULTS_DIR / "hardware_validation" / "ch8_20s_hardware.json"

# =============================================================================
# FIX 1: REGIME-SPECIFIC COMPARATORS (replaces CLINICAL_THRESHOLD = 0.70)
# =============================================================================

COMPARATORS = {
    'tier2b_full': {
        'label': 'Classical Ceiling (336 features)',
        'auc': 0.952,
        'n_features': 336,
        'n_patients': 22,
        'source': 'PROMPT 018',
        'note': 'Not a fair quantum comparator — feature count exceeds encoding capacity'
    },
    'tier3_calibrated': {
        'label': 'Riemannian Calibrated (36 features)',
        'auc': 0.659,
        'n_features': 36,
        'n_patients': 22,
        'source': 'PROMPT 018',
        'note': 'Geometry-preserving classical method with polarity calibration'
    },
    'tier2a_eigenvalues': {
        'label': 'Eigenvalues Only (8 features)',
        'auc': 0.628,
        'n_features': 8,
        'n_patients': 22,
        'source': 'PROMPT 018',
        'note': 'Fair quantum comparator — matched feature dimensionality'
    }
}

# Minimum AUC to be considered non-degenerate (FIX 2)
NON_DEGENERATE_THRESHOLD = 0.52


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return None


def is_non_degenerate(sim_auc: Optional[float], hw_auc: Optional[float]) -> bool:
    """
    FIX 2: Check if a data point is non-degenerate.

    Returns True if EITHER simulation or hardware AUC > 0.52.
    CH4 fails this filter (both at 0.50).
    """
    if sim_auc is not None and sim_auc > NON_DEGENERATE_THRESHOLD:
        return True
    if hw_auc is not None and hw_auc > NON_DEGENERATE_THRESHOLD:
        return True
    return False


def compute_signed_gap(sim_auc: Optional[float], hw_auc: Optional[float]) -> Optional[float]:
    """
    FIX 3: Compute signed gap between simulation and hardware.

    Positive gap = simulation beats hardware (expected)
    Negative gap = hardware beats simulation (inversion)
    """
    if sim_auc is None or hw_auc is None:
        return None
    return sim_auc - hw_auc


# =============================================================================
# DATA POINT COLLECTION (FIX 4: 2D tracking)
# =============================================================================

def collect_data_points(
    sim_data: Optional[Dict],
    hw_data: Optional[Dict],
    calibrated_data: Optional[Dict],
    quantum_20s_sim: Optional[Dict],
    ch12_hw: Optional[Dict],
    ch16_hw: Optional[Dict],
    ch8_20s_hw: Optional[Dict]
) -> List[Dict]:
    """
    Collect all data points in a flat list with 2D coordinates (channels, window_size).

    FIX 4: Track both channel count and window size as dimensions.
    FIX 5: Include polarity statistics per point.
    """
    points = []

    # CH4 @ 1.95s - V2 encoding (degenerate)
    if hw_data and 'ch4' in hw_data:
        ch4_hw = hw_data['ch4']
        ch4_sim = sim_data.get('ch4', {}) if sim_data else {}
        points.append({
            'id': 'ch4_1.95s_hw',
            'channels': 4,
            'qubits': ch4_hw.get('n_qubits', 9),
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': ch4_hw.get('encoding', 'V2_band_power'),
            'raw_auc': ch4_hw.get('overall_auc'),
            'calibrated_auc': ch4_hw.get('overall_auc'),  # No calibration data for CH4
            'n_inverted': 0,
            'calibration_lift': 0,
            'status': 'degenerate',
            'note': 'All subjects at 0.500 AUC — no signal'
        })
        if ch4_sim.get('overall_auc'):
            points.append({
                'id': 'ch4_1.95s_sim',
                'channels': 4,
                'qubits': ch4_sim.get('n_qubits', 9),
                'window_s': 1.95,
                'backend': 'simulation',
                'encoding': 'V2_band_power',
                'raw_auc': ch4_sim.get('overall_auc'),
                'calibrated_auc': ch4_sim.get('overall_auc'),
                'n_inverted': None,
                'calibration_lift': None,
                'status': 'degenerate',
                'note': 'Excluded from trend fitting'
            })

    # CH8 @ 1.95s - PLV encoding (primary hardware result)
    if hw_data and 'ch8' in hw_data:
        ch8_hw = hw_data['ch8']
        ch8_sim = sim_data.get('ch8', {}) if sim_data else {}

        # Get calibrated data
        cal_data = calibrated_data.get('ch8', {}) if calibrated_data else {}
        cal_auc = cal_data.get('calibrated_overall_auc', ch8_hw.get('overall_auc'))
        n_inv = cal_data.get('n_inverted', 0)
        lift = cal_data.get('calibration_lift', 0)

        points.append({
            'id': 'ch8_1.95s_hw',
            'channels': 8,
            'qubits': 17,
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': ch8_hw.get('encoding', 'PLV_theta_alpha'),
            'raw_auc': ch8_hw.get('overall_auc'),
            'calibrated_auc': cal_auc,
            'n_inverted': n_inv,
            'n_subjects': cal_data.get('n_subjects', 7),
            'calibration_lift': lift,
            'status': 'available',
            'note': 'Primary hardware result with polarity calibration'
        })

        # Simulation for CH8 @ 1.95s (V2 encoding, not PLV!)
        if ch8_sim.get('overall_auc'):
            points.append({
                'id': 'ch8_1.95s_sim',
                'channels': 8,
                'qubits': 17,
                'window_s': 1.95,
                'backend': 'simulation',
                'encoding': 'V2_band_power',  # Note: different encoding!
                'raw_auc': ch8_sim.get('overall_auc'),
                'calibrated_auc': ch8_sim.get('overall_auc'),
                'n_inverted': None,
                'calibration_lift': None,
                'status': 'available',
                'note': 'Encoding mismatch with hardware (V2 vs PLV)'
            })

    # CH8 @ 20s - PLV simulation (PROMPT 017-SIM)
    if quantum_20s_sim:
        points.append({
            'id': 'ch8_20s_sim',
            'channels': 8,
            'qubits': 17,
            'window_s': 20.0,
            'backend': 'simulation',
            'encoding': quantum_20s_sim.get('config', {}).get('encoding', 'PLV_theta_alpha'),
            'raw_auc': quantum_20s_sim.get('overall_raw_auc'),
            'calibrated_auc': quantum_20s_sim.get('overall_calibrated_auc'),
            'n_inverted': quantum_20s_sim.get('n_inverted'),
            'n_subjects': 7,
            'calibration_lift': quantum_20s_sim.get('calibration_lift'),
            'status': 'available',
            'note': 'Simulation inverts at 20s — worse raw AUC than 1.95s'
        })

    # CH8 @ 20s - Hardware (PROMPT 017, pending)
    if ch8_20s_hw:
        points.append({
            'id': 'ch8_20s_hw',
            'channels': 8,
            'qubits': 17,
            'window_s': 20.0,
            'backend': 'hardware',
            'encoding': ch8_20s_hw.get('encoding', 'PLV_theta_alpha'),
            'raw_auc': ch8_20s_hw.get('overall_raw_auc'),
            'calibrated_auc': ch8_20s_hw.get('overall_calibrated_auc'),
            'n_inverted': ch8_20s_hw.get('n_inverted'),
            'calibration_lift': ch8_20s_hw.get('calibration_lift'),
            'status': 'available',
            'note': 'March 22+ data'
        })
    else:
        points.append({
            'id': 'ch8_20s_hw',
            'channels': 8,
            'qubits': 17,
            'window_s': 20.0,
            'backend': 'hardware',
            'encoding': 'PLV_theta_alpha',
            'raw_auc': None,
            'calibrated_auc': None,
            'n_inverted': None,
            'calibration_lift': None,
            'status': 'pending',
            'note': 'Scheduled for March 22+'
        })

    # CH12 @ 1.95s - Simulation
    if sim_data and 'ch12' in sim_data:
        ch12_sim = sim_data['ch12']
        if ch12_sim.get('feasible') and ch12_sim.get('overall_auc'):
            points.append({
                'id': 'ch12_1.95s_sim',
                'channels': 12,
                'qubits': 25,
                'window_s': 1.95,
                'backend': 'simulation',
                'encoding': 'V2_band_power',
                'raw_auc': ch12_sim.get('overall_auc'),
                'calibrated_auc': ch12_sim.get('overall_auc'),  # No calibration for sim
                'n_inverted': None,
                'calibration_lift': None,
                'status': 'available',
                'note': '7-subject subset (limited by memory)'
            })

    # CH12 @ 1.95s - Hardware (PROMPT 013, pending)
    if ch12_hw:
        points.append({
            'id': 'ch12_1.95s_hw',
            'channels': 12,
            'qubits': 25,
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': ch12_hw.get('encoding', 'PLV_theta_alpha'),
            'raw_auc': ch12_hw.get('overall_raw_auc'),
            'calibrated_auc': ch12_hw.get('overall_calibrated_auc'),
            'n_inverted': ch12_hw.get('n_inverted'),
            'calibration_lift': ch12_hw.get('calibration_lift'),
            'status': 'available',
            'note': 'March 22+ data'
        })
    else:
        points.append({
            'id': 'ch12_1.95s_hw',
            'channels': 12,
            'qubits': 25,
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': 'PLV_theta_alpha',
            'raw_auc': None,
            'calibrated_auc': None,
            'n_inverted': None,
            'calibration_lift': None,
            'status': 'pending',
            'note': 'Scheduled for March 22+'
        })

    # CH16 @ 1.95s - Simulation (not feasible)
    if sim_data and 'ch16' in sim_data:
        ch16_sim = sim_data['ch16']
        points.append({
            'id': 'ch16_1.95s_sim',
            'channels': 16,
            'qubits': 33,
            'window_s': 1.95,
            'backend': 'simulation',
            'encoding': 'V2_band_power',
            'raw_auc': None,
            'calibrated_auc': None,
            'n_inverted': None,
            'calibration_lift': None,
            'status': 'infeasible',
            'note': ch16_sim.get('failure_reason', 'Requires 128GB RAM')
        })

    # CH16 @ 1.95s - Hardware (PROMPT 014, pending)
    if ch16_hw:
        points.append({
            'id': 'ch16_1.95s_hw',
            'channels': 16,
            'qubits': 33,
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': ch16_hw.get('encoding', 'PLV_theta_alpha'),
            'raw_auc': ch16_hw.get('overall_raw_auc'),
            'calibrated_auc': ch16_hw.get('overall_calibrated_auc'),
            'n_inverted': ch16_hw.get('n_inverted'),
            'calibration_lift': ch16_hw.get('calibration_lift'),
            'status': 'available',
            'note': 'March 22+ data'
        })
    else:
        points.append({
            'id': 'ch16_1.95s_hw',
            'channels': 16,
            'qubits': 33,
            'window_s': 1.95,
            'backend': 'hardware',
            'encoding': 'PLV_theta_alpha',
            'raw_auc': None,
            'calibrated_auc': None,
            'n_inverted': None,
            'calibration_lift': None,
            'status': 'pending',
            'note': 'Scheduled for March 22+'
        })

    return points


# =============================================================================
# GAP ANALYSIS (FIX 2 + FIX 3)
# =============================================================================

def compute_gap_analysis(data_points: List[Dict]) -> Dict:
    """
    Compute gaps against all three comparators.

    FIX 1: Three comparators instead of single threshold
    FIX 2: Exclude degenerate points from trend fitting
    FIX 3: Signed gaps (hardware can beat simulation)
    """
    # Find best hardware result (calibrated)
    hw_points = [p for p in data_points
                 if p['backend'] == 'hardware'
                 and p['status'] == 'available'
                 and p.get('calibrated_auc') is not None]

    if not hw_points:
        return {'error': 'No available hardware data points'}

    best_hw = max(hw_points, key=lambda p: p['calibrated_auc'])

    # Gap analysis against each comparator
    gap_analysis = {
        'quantum_best': {
            'id': best_hw['id'],
            'calibrated_auc': best_hw['calibrated_auc'],
            'raw_auc': best_hw['raw_auc'],
            'channels': best_hw['channels'],
            'window_s': best_hw['window_s'],
            'encoding': best_hw['encoding']
        },
        'gaps': {}
    }

    for comp_key, comp in COMPARATORS.items():
        gap = best_hw['calibrated_auc'] - comp['auc']
        status = 'EXCEEDED' if gap > 0 else ('nearly_there' if gap > -0.05 else 'below')

        gap_analysis['gaps'][comp_key] = {
            'comparator_label': comp['label'],
            'comparator_auc': comp['auc'],
            'gap': round(gap, 4),
            'status': status,
            'note': comp['note']
        }

    return gap_analysis


def fit_gap_model(data_points: List[Dict]) -> Dict:
    """
    Fit a model to predict gap vs qubit count.

    FIX 2: Exclude degenerate CH4 from fitting.
    Only fit on non-degenerate points.
    """
    # Get paired (sim, hw) points at same (channels, window)
    pairs = []

    hw_by_key = {(p['channels'], p['window_s']): p
                 for p in data_points
                 if p['backend'] == 'hardware' and p.get('raw_auc') is not None}

    sim_by_key = {(p['channels'], p['window_s']): p
                  for p in data_points
                  if p['backend'] == 'simulation' and p.get('raw_auc') is not None}

    for key in hw_by_key:
        if key in sim_by_key:
            hw = hw_by_key[key]
            sim = sim_by_key[key]

            # FIX 2: Check non-degenerate
            if not is_non_degenerate(sim['raw_auc'], hw['raw_auc']):
                continue

            gap = compute_signed_gap(sim['raw_auc'], hw['raw_auc'])
            if gap is not None:
                pairs.append({
                    'channels': hw['channels'],
                    'qubits': hw['qubits'],
                    'window_s': hw['window_s'],
                    'sim_auc': sim['raw_auc'],
                    'hw_auc': hw['raw_auc'],
                    'gap': gap,
                    'inverted': gap < 0  # FIX 3: Track inversion
                })

    if len(pairs) < 1:
        return {
            'model': 'no_data',
            'n_points': 0,
            'note': 'No non-degenerate sim-hw pairs available'
        }

    if len(pairs) < 2:
        # Single point — report but don't fit
        return {
            'model': 'single_point',
            'n_points': 1,
            'data_points': pairs,
            'note': 'Need CH12/CH16 hardware data (March 22+) for trend analysis'
        }

    # Linear fit if we have enough points
    qubits = np.array([p['qubits'] for p in pairs])
    gaps = np.array([p['gap'] for p in pairs])

    coeffs = np.polyfit(qubits, gaps, 1)
    slope, intercept = coeffs

    # Predictions for future qubit counts
    predictions = {}
    for n_qubits in [25, 33, 41, 49]:
        pred_gap = slope * n_qubits + intercept
        predictions[f'q{n_qubits}'] = {
            'qubits': n_qubits,
            'predicted_gap': float(pred_gap),
            'channels': (n_qubits - 1) // 2
        }

    return {
        'model': 'linear',
        'n_points': len(pairs),
        'slope': float(slope),
        'intercept': float(intercept),
        'data_points': pairs,
        'predictions': predictions,
        'interpretation': 'positive slope = gap worsens with qubits; negative = gap improves',
        'note': 'Based on non-degenerate points only (CH4 excluded)'
    }


# =============================================================================
# CHANNEL SCALING CURVE
# =============================================================================

def compute_channel_scaling(data_points: List[Dict]) -> Dict:
    """Compute channel scaling curve at fixed 1.95s window."""
    hw_1_95 = [p for p in data_points
               if p['backend'] == 'hardware'
               and p['window_s'] == 1.95]

    hw_1_95.sort(key=lambda p: p['channels'])

    return {
        'window_s': 1.95,
        'points': [{
            'channels': p['channels'],
            'qubits': p['qubits'],
            'raw_auc': p.get('raw_auc'),
            'calibrated_auc': p.get('calibrated_auc'),
            'status': p['status'],
            'encoding': p.get('encoding')
        } for p in hw_1_95]
    }


def compute_window_scaling(data_points: List[Dict]) -> Dict:
    """Compute window size impact at fixed CH8."""
    ch8_points = [p for p in data_points if p['channels'] == 8]

    by_window = {}
    for p in ch8_points:
        key = p['window_s']
        if key not in by_window:
            by_window[key] = {'window_s': key, 'simulation': None, 'hardware': None}

        if p['backend'] == 'simulation':
            by_window[key]['simulation'] = {
                'raw_auc': p.get('raw_auc'),
                'calibrated_auc': p.get('calibrated_auc'),
                'status': p['status']
            }
        else:
            by_window[key]['hardware'] = {
                'raw_auc': p.get('raw_auc'),
                'calibrated_auc': p.get('calibrated_auc'),
                'n_inverted': p.get('n_inverted'),
                'calibration_lift': p.get('calibration_lift'),
                'status': p['status']
            }

    return {
        'channels': 8,
        'points': [by_window[k] for k in sorted(by_window.keys())]
    }


# =============================================================================
# POLARITY TRACKING (FIX 5)
# =============================================================================

def compute_polarity_tracking(data_points: List[Dict], full_cohort_data: Optional[Dict]) -> Dict:
    """
    Track polarity statistics across data points.

    FIX 5: Add polarity stats per data point.
    """
    tracking = {
        'quantum_hw_points': [],
        'classical_reference': {}
    }

    # Quantum hardware polarity stats
    for p in data_points:
        if p['backend'] == 'hardware' and p.get('n_inverted') is not None:
            tracking['quantum_hw_points'].append({
                'id': p['id'],
                'channels': p['channels'],
                'window_s': p['window_s'],
                'n_inverted': p['n_inverted'],
                'n_subjects': p.get('n_subjects'),
                'calibration_lift': p.get('calibration_lift')
            })

    # Classical reference from PROMPT 018
    if full_cohort_data:
        t2a = full_cohort_data.get('tier2a_eigenvalues', {})
        t3 = full_cohort_data.get('tier3', {})

        tracking['classical_reference'] = {
            'tier2a': {
                'n_inverted': t2a.get('n_inverted', 0),
                'n_subjects': len(t2a.get('per_subject', {})),
                'calibration_lift': t2a.get('calibration_lift', 0)
            },
            'tier3': {
                'n_inverted': t3.get('n_inverted', 0),
                'n_subjects': len(t3.get('per_subject', {})),
                'calibration_lift': t3.get('calibration_lift', 0)
            }
        }

    return tracking


# =============================================================================
# SIMULATION RELIABILITY ASSESSMENT (FIX 3)
# =============================================================================

def assess_simulation_reliability(data_points: List[Dict], quantum_20s_sim: Optional[Dict]) -> Dict:
    """
    Assess reliability of simulation as a proxy for hardware.

    FIX 3: Simulation can underperform hardware — track when this happens.
    """
    findings = []

    # Check CH8 1.95s: sim vs hw
    ch8_hw = next((p for p in data_points if p['id'] == 'ch8_1.95s_hw'), None)
    ch8_sim = next((p for p in data_points if p['id'] == 'ch8_1.95s_sim'), None)

    if ch8_hw and ch8_sim and ch8_hw.get('calibrated_auc') and ch8_sim.get('raw_auc'):
        gap = ch8_sim['raw_auc'] - ch8_hw['calibrated_auc']
        if gap < 0:
            findings.append({
                'observation': 'Simulation underestimates hardware (PLV 1.95s)',
                'sim_auc': ch8_sim['raw_auc'],
                'hw_calibrated_auc': ch8_hw['calibrated_auc'],
                'gap': round(gap, 4),
                'note': 'Hardware with calibration beats simulation'
            })

    # Check 20s simulation behavior
    if quantum_20s_sim:
        sim_1_95 = next((p for p in data_points if p['id'] == 'ch8_1.95s_sim'), None)
        if sim_1_95:
            sim_change = quantum_20s_sim.get('overall_raw_auc', 0) - sim_1_95.get('raw_auc', 0)
            if sim_change < 0:
                findings.append({
                    'observation': 'Simulation worsens with longer windows',
                    'sim_1.95s_auc': sim_1_95.get('raw_auc'),
                    'sim_20s_auc': quantum_20s_sim.get('overall_raw_auc'),
                    'change': round(sim_change, 4),
                    'note': 'Raw AUC drops, but calibration recovers'
                })

    # Encoding mismatch warning
    findings.append({
        'observation': 'Encoding mismatch between sim and hardware',
        'sim_encoding': 'V2_band_power',
        'hw_encoding': 'PLV_theta_alpha',
        'note': 'Direct gap comparison is not reliable'
    })

    return {
        'findings': findings,
        'recommendation': 'Do not use simulation for encoding selection or performance projection. '
                          'Hardware with calibration is the ground truth.',
        'inversion_observed': any(f.get('gap', 0) < 0 for f in findings if 'gap' in f)
    }


# =============================================================================
# MAIN PROJECTION
# =============================================================================

def project_hardware_readiness(
    sim_data: Optional[Dict],
    hw_data: Optional[Dict],
    calibrated_data: Optional[Dict],
    full_cohort_data: Optional[Dict],
    quantum_20s_sim: Optional[Dict],
    window_impact: Optional[Dict],
    ch12_hw: Optional[Dict],
    ch16_hw: Optional[Dict],
    ch8_20s_hw: Optional[Dict]
) -> Dict:
    """
    Project when hardware will achieve various performance targets.

    Implements all FIX 1-6 corrections.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'prompt': 'PROMPT 010-FIX',
        'comparators': COMPARATORS
    }

    # Collect all data points (FIX 4: 2D tracking)
    data_points = collect_data_points(
        sim_data, hw_data, calibrated_data, quantum_20s_sim,
        ch12_hw, ch16_hw, ch8_20s_hw
    )
    results['data_points'] = data_points

    # Gap analysis (FIX 1: three comparators)
    results['gap_analysis'] = compute_gap_analysis(data_points)

    # Gap model (FIX 2: exclude CH4, FIX 3: signed gaps)
    results['gap_model'] = fit_gap_model(data_points)

    # Channel scaling curve
    results['channel_scaling'] = compute_channel_scaling(data_points)

    # Window size impact
    results['window_scaling'] = compute_window_scaling(data_points)

    # Polarity tracking (FIX 5)
    results['polarity_tracking'] = compute_polarity_tracking(data_points, full_cohort_data)

    # Simulation reliability (FIX 3)
    results['simulation_reliability'] = assess_simulation_reliability(data_points, quantum_20s_sim)

    # Key conclusions
    gap_analysis = results['gap_analysis']
    if 'gaps' in gap_analysis:
        conclusions = []

        t2a_gap = gap_analysis['gaps'].get('tier2a_eigenvalues', {}).get('gap', -999)
        t3_gap = gap_analysis['gaps'].get('tier3_calibrated', {}).get('gap', -999)
        t2b_gap = gap_analysis['gaps'].get('tier2b_full', {}).get('gap', -999)

        if t2a_gap >= 0:
            conclusions.append(f"1. Quantum EXCEEDS matched-dimensionality classical (Tier 2A) by {t2a_gap:+.3f}")
        else:
            conclusions.append(f"1. Quantum is {abs(t2a_gap):.3f} below matched-dimensionality classical (Tier 2A)")

        if abs(t3_gap) < 0.05:
            conclusions.append(f"2. Gap to Tier 3 calibrated is small ({t3_gap:+.3f}) — may close with 20s windows")
        else:
            conclusions.append(f"2. Gap to Tier 3 calibrated: {t3_gap:+.3f}")

        conclusions.append(f"3. Gap to Tier 2B (classical ceiling) is large ({t2b_gap:+.3f}) — not closeable on near-term hardware")
        conclusions.append("4. Simulation is unreliable proxy — hardware with calibration is ground truth")

        results['conclusions'] = conclusions

    # Pending data summary
    pending = [p for p in data_points if p['status'] == 'pending']
    results['pending_data'] = {
        'count': len(pending),
        'ids': [p['id'] for p in pending],
        'expected_date': 'March 22, 2026+',
        'note': 'Run PROMPT 010-R after March 22 data arrives'
    }

    return results


# =============================================================================
# REPORT GENERATION (FIX 6: Updated structure)
# =============================================================================

def generate_markdown_report(results: Dict) -> str:
    """Generate markdown report with FIX 6 structure."""
    lines = [
        "# Hardware Readiness Projection — Updated",
        "",
        f"Generated: {results['timestamp']}",
        "",
        "## Current State (pre-March 22)",
        "",
        "| Data Point | Channels | Window | Backend | Calibrated AUC | Status |",
        "|------------|----------|--------|---------|----------------|--------|"
    ]

    for p in results.get('data_points', []):
        cal = f"{p['calibrated_auc']:.4f}" if p.get('calibrated_auc') else "—"
        lines.append(f"| {p['id']} | {p['channels']} | {p['window_s']}s | {p['backend']} | {cal} | {p['status']} |")

    lines.extend(["", "## Gap Analysis (against three comparators)", ""])

    gap = results.get('gap_analysis', {})
    if 'quantum_best' in gap:
        qb = gap['quantum_best']
        lines.extend([
            f"**Quantum Best**: {qb['id']} — {qb['calibrated_auc']:.4f} calibrated AUC",
            "",
            "| Comparator | AUC | Quantum Best | Gap | Status |",
            "|------------|-----|--------------|-----|--------|"
        ])

        for comp_key, g in gap.get('gaps', {}).items():
            status_emoji = "+" if g['status'] == 'EXCEEDED' else ("~" if g['status'] == 'nearly_there' else "-")
            lines.append(f"| {g['comparator_label']} | {g['comparator_auc']:.3f} | {qb['calibrated_auc']:.3f} | {g['gap']:+.3f} | {status_emoji} {g['status']} |")

    lines.extend(["", "## Channel Scaling Curve", ""])

    ch_scale = results.get('channel_scaling', {})
    lines.extend([
        f"Fixed window: {ch_scale.get('window_s', 1.95)}s",
        "",
        "| Channels | Qubits | Raw AUC | Calibrated AUC | Status |",
        "|----------|--------|---------|----------------|--------|"
    ])

    for p in ch_scale.get('points', []):
        raw = f"{p['raw_auc']:.4f}" if p.get('raw_auc') else "—"
        cal = f"{p['calibrated_auc']:.4f}" if p.get('calibrated_auc') else "—"
        lines.append(f"| {p['channels']} | {p['qubits']} | {raw} | {cal} | {p['status']} |")

    lines.extend(["", "## Window Size Impact", ""])

    win_scale = results.get('window_scaling', {})
    lines.extend([
        f"Fixed channels: {win_scale.get('channels', 8)}",
        "",
        "| Window | Sim Raw | Sim Cal | HW Raw | HW Cal | HW Status |",
        "|--------|---------|---------|--------|--------|-----------|"
    ])

    for p in win_scale.get('points', []):
        sim = p.get('simulation', {})
        hw = p.get('hardware', {})
        sim_raw = f"{sim.get('raw_auc', 0):.3f}" if sim.get('raw_auc') else "—"
        sim_cal = f"{sim.get('calibrated_auc', 0):.3f}" if sim.get('calibrated_auc') else "—"
        hw_raw = f"{hw.get('raw_auc', 0):.3f}" if hw.get('raw_auc') else "—"
        hw_cal = f"{hw.get('calibrated_auc', 0):.3f}" if hw.get('calibrated_auc') else "—"
        hw_status = hw.get('status', 'unknown')
        lines.append(f"| {p['window_s']}s | {sim_raw} | {sim_cal} | {hw_raw} | {hw_cal} | {hw_status} |")

    lines.extend(["", "## Simulation Reliability Assessment", ""])

    sim_rel = results.get('simulation_reliability', {})
    for f in sim_rel.get('findings', []):
        lines.append(f"- **{f['observation']}**")
        if 'gap' in f:
            lines.append(f"  - Gap: {f['gap']:+.4f}")
        if 'note' in f:
            lines.append(f"  - {f['note']}")
        lines.append("")

    lines.append(f"**Recommendation**: {sim_rel.get('recommendation', 'N/A')}")

    lines.extend(["", "## Polarity Tracking", ""])

    pol = results.get('polarity_tracking', {})
    lines.append("**Quantum Hardware Points:**")
    lines.append("")
    lines.append("| Point | Channels | Window | N Inverted | Calibration Lift |")
    lines.append("|-------|----------|--------|------------|------------------|")

    for p in pol.get('quantum_hw_points', []):
        n_inv = str(p.get('n_inverted', '—'))
        n_subj = p.get('n_subjects')
        if n_subj:
            n_inv = f"{p.get('n_inverted', '—')}/{n_subj}"
        lift = f"{p['calibration_lift']:.3f}" if p.get('calibration_lift') else "—"
        lines.append(f"| {p['id']} | {p['channels']} | {p['window_s']}s | {n_inv} | {lift} |")

    cl_ref = pol.get('classical_reference', {})
    if cl_ref:
        lines.extend([
            "",
            "**Classical Reference (PROMPT 018, 22 patients):**",
            f"- Tier 2A: {cl_ref.get('tier2a', {}).get('n_inverted', 0)}/22 inverted, lift {cl_ref.get('tier2a', {}).get('calibration_lift', 0):.3f}",
            f"- Tier 3: {cl_ref.get('tier3', {}).get('n_inverted', 0)}/22 inverted, lift {cl_ref.get('tier3', {}).get('calibration_lift', 0):.3f}"
        ])

    lines.extend(["", "## Projections (after March 22 data)", ""])

    pending = results.get('pending_data', {})
    if pending.get('count', 0) > 0:
        lines.extend([
            f"Pending data points: {pending['count']}",
            f"- {', '.join(pending.get('ids', []))}",
            f"- Expected: {pending.get('expected_date', 'TBD')}",
            "",
            "After March 22 hardware data arrives:",
            "1. Run `python scripts/hardware_projection.py` (PROMPT 010-R)",
            "2. 4-point channel scaling curve will be available",
            "3. 2-point window scaling (1.95s vs 20s) will be available",
            "4. Combined CH16 @ 20s projection can be computed"
        ])
    else:
        lines.append("All data points available. Projections computed above.")

    lines.extend(["", "## Key Conclusions", ""])

    for c in results.get('conclusions', []):
        lines.append(c)

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PROMPT 010-FIX: Hardware Readiness Projection (Corrected)")
    logger.info("=" * 70)

    # Load input data
    logger.info("Loading input data...")

    sim_data = load_json(SIM_SCALING)
    hw_data = load_json(HW_SCALING)
    calibrated_data = load_json(CALIBRATED_GAP)
    full_cohort_data = load_json(FULL_COHORT)
    quantum_20s_sim = load_json(QUANTUM_20S_SIM)
    window_impact = load_json(WINDOW_IMPACT)

    # Pending March 22+ data
    ch12_hw = load_json(CH12_HARDWARE)
    ch16_hw = load_json(CH16_HARDWARE)
    ch8_20s_hw = load_json(CH8_20S_HARDWARE)

    logger.info(f"Simulation data: {SIM_SCALING.exists()}")
    logger.info(f"Hardware data: {HW_SCALING.exists()}")
    logger.info(f"Full cohort data: {FULL_COHORT.exists()}")
    logger.info(f"Quantum 20s sim: {QUANTUM_20S_SIM.exists()}")
    logger.info(f"CH12 hardware: {CH12_HARDWARE.exists()}")
    logger.info(f"CH16 hardware: {CH16_HARDWARE.exists()}")
    logger.info(f"CH8 20s hardware: {CH8_20S_HARDWARE.exists()}")

    # Run projection
    logger.info("\nRunning projection analysis...")
    results = project_hardware_readiness(
        sim_data, hw_data, calibrated_data, full_cohort_data,
        quantum_20s_sim, window_impact, ch12_hw, ch16_hw, ch8_20s_hw
    )

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "hardware_projection.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Generate and save markdown report
    md_report = generate_markdown_report(results)
    md_path = OUTPUT_DIR / "hardware_projection.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    logger.info(f"Saved: {md_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("HARDWARE READINESS PROJECTION SUMMARY")
    print("=" * 70)

    print("\nComparators (from PROMPT 018):")
    for key, comp in COMPARATORS.items():
        print(f"  {comp['label']}: {comp['auc']:.3f}")

    gap = results.get('gap_analysis', {})
    if 'quantum_best' in gap:
        qb = gap['quantum_best']
        print(f"\nQuantum Best: {qb['calibrated_auc']:.4f} (calibrated, {qb['id']})")
        print("\nGap Analysis:")
        for comp_key, g in gap.get('gaps', {}).items():
            print(f"  vs {g['comparator_label']}: {g['gap']:+.4f} ({g['status']})")

    pending = results.get('pending_data', {})
    print(f"\nPending: {pending.get('count', 0)} data points (March 22+)")

    print("\n" + "=" * 70)
    print("Conclusions:")
    for c in results.get('conclusions', []):
        print(f"  {c}")
    print("=" * 70)

    logger.info("Projection complete.")


if __name__ == "__main__":
    main()
