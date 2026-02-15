#!/usr/bin/env python3
"""
IBM Quantum Hardware Validation of the A-Gate Multichannel Circuit

Validates that the QPNN A-Gate architecture produces physically meaningful
quantum states on real IBM Quantum hardware. Measures fidelity degradation
compared to ideal simulation.

This is NOT a classification experiment - it validates circuit behavior.

Output:
    analysis_results/ibm_hardware/
        transpilation_report.json
        ideal_results.json
        noisy_results.json
        hardware_results.json
        fidelity_comparison.json
        discrimination_matrix.json
        scaling_experiment.json (if run)

Usage:
    python scripts/ibm_hardware_validation.py [--skip-hardware] [--scaling]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from QA1.multichannel_circuit import create_multichannel_circuit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

SHOTS = 8192  # Standard for IBM Quantum
OUTPUT_DIR = Path("analysis_results/ibm_hardware")
SEED = 42

# Test configurations that exercise different circuit behaviors
TEST_CONFIGS = {
    'synchronized': {
        'description': 'All channels synchronized (identical phases)',
        'params': [(0.5, 1.0, 0.5)] * 8
    },
    'desynchronized': {
        'description': 'All channels desynchronized (random phases)',
        'params': [(0.5, b, 0.5) for b in [0.3, 1.7, 4.2, 2.8, 5.1, 0.9, 3.5, 2.1]]
    },
    'half_sync': {
        'description': 'Half synchronized, half random',
        'params': [(0.5, 1.0, 0.5)] * 4 + [(0.5, b, 0.5) for b in [3.1, 0.7, 4.8, 2.3]]
    },
    'excitatory': {
        'description': 'Strong excitatory (high a, low c)',
        'params': [(0.8, 1.0, 0.2)] * 8
    },
    'inhibitory': {
        'description': 'Strong inhibitory (low a, high c)',
        'params': [(0.2, 1.0, 0.8)] * 8
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_json(data: Dict, filename: str):
    """Save data to JSON file."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def counts_to_dict(counts) -> Dict[str, int]:
    """Convert Qiskit counts object to serializable dict."""
    if hasattr(counts, 'items'):
        return dict(counts)
    return counts


def hellinger_fidelity(p: Dict[str, int], q: Dict[str, int]) -> float:
    """
    Compute Hellinger fidelity between two count distributions.

    F_H = (sum_x sqrt(p(x) * q(x)))^2

    This is a classical measure of distribution similarity, not quantum
    state fidelity, but it's what we can measure on hardware.
    """
    # Normalize to probabilities
    total_p = sum(p.values())
    total_q = sum(q.values())

    if total_p == 0 or total_q == 0:
        return 0.0

    # Get all keys
    all_keys = set(p.keys()) | set(q.keys())

    # Compute Hellinger fidelity
    sqrt_sum = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0) / total_p
        q_prob = q.get(key, 0) / total_q
        sqrt_sum += np.sqrt(p_prob * q_prob)

    return sqrt_sum ** 2


def count_two_qubit_gates(circuit, gate_names: List[str] = None) -> int:
    """Count two-qubit gates in a circuit."""
    if gate_names is None:
        gate_names = ['cz', 'ecr', 'cx', 'cnot', 'swap', 'iswap', 'rzz', 'rxx', 'ryy']

    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in gate_names)


# =============================================================================
# CIRCUIT BUILDING
# =============================================================================

def build_test_circuits() -> Dict[str, Any]:
    """
    Build test circuits for all configurations.

    Returns:
        Dict with circuit info for each configuration
    """
    from qiskit import QuantumCircuit

    circuits = {}

    for name, config in TEST_CONFIGS.items():
        params = config['params']

        # Create circuit without measurements
        qc = create_multichannel_circuit(params)

        # Create version with measurements
        qc_meas = qc.copy()
        qc_meas.measure_all()

        circuits[name] = {
            'circuit': qc,
            'circuit_with_meas': qc_meas,
            'params': params,
            'description': config['description'],
            'n_qubits': qc.num_qubits,
            'original_depth': qc.depth(),
            'original_gates': dict(qc.count_ops())
        }

        logger.info(f"Built circuit '{name}': {qc.num_qubits} qubits, depth {qc.depth()}")

    return circuits


# =============================================================================
# TRANSPILATION
# =============================================================================

def transpile_circuits(circuits: Dict, backend) -> Tuple[Dict, Dict]:
    """
    Transpile circuits for the target backend.

    Returns:
        transpiled_circuits: Dict of transpiled circuits
        transpilation_report: Dict with gate counts and depths
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info(f"Transpiling for backend: {backend.name}")
    logger.info(f"Basis gates: {backend.basis_gates}")

    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=SEED
    )

    transpiled = {}
    report = {
        'backend': backend.name,
        'basis_gates': list(backend.basis_gates),
        'backend_qubits': backend.num_qubits,
        'circuits': {}
    }

    for name, info in circuits.items():
        qc = info['circuit_with_meas']
        t_qc = pm.run(qc)
        transpiled[name] = t_qc

        two_q = count_two_qubit_gates(t_qc)

        # Get layout info
        layout = None
        if t_qc.layout is not None:
            try:
                layout = list(t_qc.layout.final_index_layout())
            except:
                layout = "unavailable"

        report['circuits'][name] = {
            'original_depth': info['original_depth'],
            'transpiled_depth': t_qc.depth(),
            'original_gates': info['original_gates'],
            'transpiled_gates': dict(t_qc.count_ops()),
            'two_qubit_gates': two_q,
            'physical_qubits': t_qc.num_qubits,
            'layout': layout
        }

        logger.info(f"  {name}: depth {info['original_depth']} -> {t_qc.depth()}, "
                   f"2Q gates: {two_q}")

    return transpiled, report


# =============================================================================
# IDEAL SIMULATOR
# =============================================================================

def run_ideal_simulator(circuits: Dict) -> Dict[str, Dict[str, int]]:
    """
    Run circuits on ideal statevector simulator and sample.

    Returns:
        Dict of counts for each configuration
    """
    from qiskit.quantum_info import Statevector

    logger.info("Running ideal statevector simulation...")
    results = {}

    for name, info in circuits.items():
        qc = info['circuit']  # Without measurements
        sv = Statevector.from_instruction(qc)
        counts = sv.sample_counts(SHOTS)
        results[name] = counts_to_dict(counts)

        # Log top outcomes
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        logger.info(f"  {name}: top outcomes = {top}")

    return results


# =============================================================================
# NOISY SIMULATOR
# =============================================================================

def run_noisy_simulator(transpiled: Dict, backend) -> Dict[str, Dict[str, int]]:
    """
    Run transpiled circuits on noisy simulator using backend noise model.

    Returns:
        Dict of counts for each configuration
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    logger.info("Running noisy simulation with backend noise model...")

    noise_model = NoiseModel.from_backend(backend)
    noisy_sim = AerSimulator(noise_model=noise_model)

    results = {}

    for name, t_qc in transpiled.items():
        job = noisy_sim.run(t_qc, shots=SHOTS)
        counts = job.result().get_counts()
        results[name] = counts_to_dict(counts)

        # Log entropy as measure of noise spread
        probs = np.array(list(counts.values())) / sum(counts.values())
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        logger.info(f"  {name}: {len(counts)} unique outcomes, entropy={entropy:.2f} bits")

    return results


# =============================================================================
# REAL HARDWARE
# =============================================================================

def run_hardware(transpiled: Dict, backend) -> Dict[str, Dict[str, int]]:
    """
    Run transpiled circuits on real IBM Quantum hardware.

    Returns:
        Dict of counts for each configuration
    """
    from qiskit_ibm_runtime import SamplerV2, Batch

    logger.info(f"Submitting jobs to hardware: {backend.name}")

    results = {}

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        jobs = {}

        for name, t_qc in transpiled.items():
            logger.info(f"  Submitting {name}...")
            job = sampler.run([t_qc], shots=SHOTS)
            jobs[name] = job
            logger.info(f"    Job ID: {job.job_id()}")

        logger.info("Waiting for hardware results...")

        for name, job in jobs.items():
            logger.info(f"  Retrieving {name}...")
            result = job.result()

            # SamplerV2 returns PrimitiveResult
            # Access counts from the first (and only) pub result
            pub_result = result[0]

            # The data is in pub_result.data.<register_name>
            # For measure_all(), the register is typically 'meas'
            try:
                counts = pub_result.data.meas.get_counts()
            except AttributeError:
                # Try alternative access patterns
                try:
                    counts = pub_result.data.c.get_counts()
                except:
                    # Fallback: get all available data
                    logger.warning(f"  Non-standard data format for {name}")
                    counts = {}
                    for attr in dir(pub_result.data):
                        if not attr.startswith('_'):
                            try:
                                data = getattr(pub_result.data, attr)
                                if hasattr(data, 'get_counts'):
                                    counts = data.get_counts()
                                    break
                            except:
                                pass

            results[name] = counts_to_dict(counts)
            logger.info(f"    Got {len(counts)} unique outcomes")

    return results


# =============================================================================
# FIDELITY ANALYSIS
# =============================================================================

def compute_fidelity_comparison(
    ideal: Dict[str, Dict],
    noisy: Dict[str, Dict],
    hardware: Dict[str, Dict] = None
) -> Dict:
    """
    Compute fidelity comparisons between all result sets.

    Returns:
        Dict with fidelity matrices
    """
    configs = list(ideal.keys())

    comparison = {
        'configs': configs,
        'ideal_vs_noisy': {},
        'ideal_vs_hardware': {},
        'noisy_vs_hardware': {},
        'summary': {}
    }

    for name in configs:
        # Ideal vs Noisy
        f_noisy = hellinger_fidelity(ideal[name], noisy[name])
        comparison['ideal_vs_noisy'][name] = f_noisy

        if hardware:
            # Ideal vs Hardware
            f_hw = hellinger_fidelity(ideal[name], hardware[name])
            comparison['ideal_vs_hardware'][name] = f_hw

            # Noisy vs Hardware
            f_n_hw = hellinger_fidelity(noisy[name], hardware[name])
            comparison['noisy_vs_hardware'][name] = f_n_hw

    # Compute summary statistics
    comparison['summary']['ideal_vs_noisy'] = {
        'mean': np.mean(list(comparison['ideal_vs_noisy'].values())),
        'std': np.std(list(comparison['ideal_vs_noisy'].values())),
        'min': min(comparison['ideal_vs_noisy'].values()),
        'max': max(comparison['ideal_vs_noisy'].values())
    }

    if hardware:
        comparison['summary']['ideal_vs_hardware'] = {
            'mean': np.mean(list(comparison['ideal_vs_hardware'].values())),
            'std': np.std(list(comparison['ideal_vs_hardware'].values())),
            'min': min(comparison['ideal_vs_hardware'].values()),
            'max': max(comparison['ideal_vs_hardware'].values())
        }
        comparison['summary']['noisy_vs_hardware'] = {
            'mean': np.mean(list(comparison['noisy_vs_hardware'].values())),
            'std': np.std(list(comparison['noisy_vs_hardware'].values())),
            'min': min(comparison['noisy_vs_hardware'].values()),
            'max': max(comparison['noisy_vs_hardware'].values())
        }

    return comparison


def compute_discrimination_matrix(results: Dict[str, Dict]) -> Dict:
    """
    Compute cross-configuration fidelity matrix.

    This measures how well the circuit discriminates between different
    input configurations. Low cross-fidelity = good discrimination.

    Returns:
        Dict with discrimination matrix
    """
    configs = list(results.keys())
    n = len(configs)

    matrix = np.zeros((n, n))

    for i, cfg_i in enumerate(configs):
        for j, cfg_j in enumerate(configs):
            matrix[i, j] = hellinger_fidelity(results[cfg_i], results[cfg_j])

    # Convert to serializable format
    return {
        'configs': configs,
        'matrix': matrix.tolist(),
        'diagonal': np.diag(matrix).tolist(),
        'off_diagonal_mean': float(np.mean(matrix[~np.eye(n, dtype=bool)])),
        'discrimination_quality': float(np.mean(np.diag(matrix)) -
                                        np.mean(matrix[~np.eye(n, dtype=bool)]))
    }


# =============================================================================
# SCALING EXPERIMENT
# =============================================================================

def run_scaling_experiment(backend) -> Dict:
    """
    Run scaling experiment with different channel counts.

    Empirically validates the O(M) scaling claim by measuring:
    - Transpiled 2Q gate count vs M
    - Circuit depth vs M

    Returns:
        Dict with scaling data
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info("Running scaling experiment...")

    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=SEED
    )

    channel_counts = [2, 4, 6, 8]
    results = {
        'channel_counts': channel_counts,
        'qubits': [],
        'original_depth': [],
        'transpiled_depth': [],
        'two_qubit_gates': [],
        'total_gates': []
    }

    for M in channel_counts:
        params = [(0.5, 1.0, 0.5)] * M  # Synchronized
        qc = create_multichannel_circuit(params)
        qc.measure_all()

        t_qc = pm.run(qc)
        two_q = count_two_qubit_gates(t_qc)

        results['qubits'].append(2 * M + 1)
        results['original_depth'].append(qc.depth())
        results['transpiled_depth'].append(t_qc.depth())
        results['two_qubit_gates'].append(two_q)
        results['total_gates'].append(t_qc.size())

        logger.info(f"  M={M}: qubits={2*M+1}, 2Q_gates={two_q}, "
                   f"depth={t_qc.depth()}")

    # Fit linear model: 2Q_gates = a * M + b
    M_arr = np.array(channel_counts)
    gates_arr = np.array(results['two_qubit_gates'])

    if len(M_arr) > 1:
        coeffs = np.polyfit(M_arr, gates_arr, 1)
        results['linear_fit'] = {
            'slope': float(coeffs[0]),
            'intercept': float(coeffs[1]),
            'formula': f"2Q_gates ~ {coeffs[0]:.1f}*M + {coeffs[1]:.1f}"
        }
        logger.info(f"  Linear fit: {results['linear_fit']['formula']}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def connect_to_ibm() -> Any:
    """Connect to IBM Quantum and select backend."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    logger.info("Connecting to IBM Quantum...")

    try:
        service = QiskitRuntimeService()
    except Exception as e:
        logger.error(f"Failed to connect to IBM Quantum: {e}")
        logger.error("Make sure you have saved your IBM Quantum credentials:")
        logger.error("  from qiskit_ibm_runtime import QiskitRuntimeService")
        logger.error("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        raise

    # Get least busy backend with sufficient qubits
    logger.info("Finding least busy backend with >= 17 qubits...")

    try:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=17
        )
        logger.info(f"Selected backend: {backend.name}")
        logger.info(f"  Qubits: {backend.num_qubits}")
        logger.info(f"  Basis gates: {backend.basis_gates}")
        return backend
    except Exception as e:
        logger.warning(f"Could not find hardware backend: {e}")
        logger.info("Falling back to simulator backend for testing...")

        # Use a fake backend for testing
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
        logger.info(f"Using fake backend: {backend.name}")
        return backend


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='IBM Quantum Hardware Validation of A-Gate Circuit'
    )
    parser.add_argument(
        '--skip-hardware',
        action='store_true',
        help='Skip real hardware execution (use noisy simulator only)'
    )
    parser.add_argument(
        '--scaling',
        action='store_true',
        help='Run scaling experiment'
    )
    parser.add_argument(
        '--fake-backend',
        action='store_true',
        help='Use fake backend instead of real hardware (for testing)'
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("IBM Quantum Hardware Validation of A-Gate Multichannel Circuit")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")

    ensure_output_dir()

    # Step 1: Build test circuits
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Building test circuits")
    logger.info("=" * 70)
    circuits = build_test_circuits()

    # Step 2: Connect to IBM Quantum
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Connecting to IBM Quantum")
    logger.info("=" * 70)

    if args.fake_backend:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
        logger.info(f"Using fake backend: {backend.name}")
    else:
        backend = connect_to_ibm()

    # Step 3: Transpile circuits
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Transpiling circuits for backend")
    logger.info("=" * 70)
    transpiled, transpilation_report = transpile_circuits(circuits, backend)
    save_json(transpilation_report, 'transpilation_report.json')

    # Step 4: Run ideal simulator
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Running ideal statevector simulation")
    logger.info("=" * 70)
    ideal_results = run_ideal_simulator(circuits)
    save_json(ideal_results, 'ideal_results.json')

    # Step 5: Run noisy simulator
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Running noisy simulation")
    logger.info("=" * 70)
    noisy_results = run_noisy_simulator(transpiled, backend)
    save_json(noisy_results, 'noisy_results.json')

    # Step 6: Run on real hardware (optional)
    hardware_results = None
    if not args.skip_hardware and not args.fake_backend:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Running on real IBM Quantum hardware")
        logger.info("=" * 70)
        try:
            hardware_results = run_hardware(transpiled, backend)
            save_json(hardware_results, 'hardware_results.json')
        except Exception as e:
            logger.error(f"Hardware execution failed: {e}")
            logger.info("Continuing with simulator results only...")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Skipping hardware execution")
        logger.info("=" * 70)

    # Step 7: Compute fidelity comparisons
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Computing fidelity comparisons")
    logger.info("=" * 70)

    fidelity_comparison = compute_fidelity_comparison(
        ideal_results, noisy_results, hardware_results
    )
    save_json(fidelity_comparison, 'fidelity_comparison.json')

    # Print fidelity table
    print("\n" + "=" * 80)
    print("FIDELITY COMPARISON")
    print("=" * 80)
    print(f"{'Config':<20} {'Ideal vs Noisy':<18} ", end="")
    if hardware_results:
        print(f"{'Ideal vs Hardware':<18} {'Noisy vs Hardware':<18}")
    else:
        print()
    print("-" * 80)

    for name in TEST_CONFIGS.keys():
        f_noisy = fidelity_comparison['ideal_vs_noisy'][name]
        print(f"{name:<20} {f_noisy:<18.4f} ", end="")
        if hardware_results:
            f_hw = fidelity_comparison['ideal_vs_hardware'][name]
            f_n_hw = fidelity_comparison['noisy_vs_hardware'][name]
            print(f"{f_hw:<18.4f} {f_n_hw:<18.4f}")
        else:
            print()

    # Step 8: Compute discrimination matrices
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Computing discrimination matrices")
    logger.info("=" * 70)

    discrimination = {
        'ideal': compute_discrimination_matrix(ideal_results),
        'noisy': compute_discrimination_matrix(noisy_results)
    }
    if hardware_results:
        discrimination['hardware'] = compute_discrimination_matrix(hardware_results)

    save_json(discrimination, 'discrimination_matrix.json')

    # Print discrimination summary
    print("\n" + "=" * 80)
    print("DISCRIMINATION QUALITY (higher = better)")
    print("=" * 80)
    print(f"{'Source':<20} {'Diagonal Mean':<18} {'Off-Diag Mean':<18} {'Quality':<18}")
    print("-" * 80)

    for source, data in discrimination.items():
        diag_mean = np.mean(data['diagonal'])
        off_diag = data['off_diagonal_mean']
        quality = data['discrimination_quality']
        print(f"{source:<20} {diag_mean:<18.4f} {off_diag:<18.4f} {quality:<18.4f}")

    # Step 9: Scaling experiment (optional)
    if args.scaling:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 9: Running scaling experiment")
        logger.info("=" * 70)
        scaling_results = run_scaling_experiment(backend)
        save_json(scaling_results, 'scaling_experiment.json')

        print("\n" + "=" * 80)
        print("SCALING EXPERIMENT")
        print("=" * 80)
        print(f"{'Channels':<12} {'Qubits':<10} {'2Q Gates':<12} {'Depth':<10}")
        print("-" * 50)
        for i, M in enumerate(scaling_results['channel_counts']):
            print(f"{M:<12} {scaling_results['qubits'][i]:<10} "
                  f"{scaling_results['two_qubit_gates'][i]:<12} "
                  f"{scaling_results['transpiled_depth'][i]:<10}")

        if 'linear_fit' in scaling_results:
            print(f"\nLinear fit: {scaling_results['linear_fit']['formula']}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    summary = {
        'completed': datetime.now().isoformat(),
        'backend': backend.name,
        'configs_tested': list(TEST_CONFIGS.keys()),
        'shots': SHOTS,
        'fidelity_summary': fidelity_comparison['summary'],
        'discrimination_quality': {
            source: data['discrimination_quality']
            for source, data in discrimination.items()
        },
        'hardware_executed': hardware_results is not None
    }
    save_json(summary, 'summary.json')

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Backend: {backend.name}")
    print(f"Configurations tested: {len(TEST_CONFIGS)}")
    print(f"Shots per circuit: {SHOTS}")

    print(f"\nIdeal vs Noisy fidelity: "
          f"{fidelity_comparison['summary']['ideal_vs_noisy']['mean']:.4f} "
          f"± {fidelity_comparison['summary']['ideal_vs_noisy']['std']:.4f}")

    if hardware_results:
        print(f"Ideal vs Hardware fidelity: "
              f"{fidelity_comparison['summary']['ideal_vs_hardware']['mean']:.4f} "
              f"± {fidelity_comparison['summary']['ideal_vs_hardware']['std']:.4f}")

    print(f"\nDiscrimination quality:")
    for source, quality in summary['discrimination_quality'].items():
        print(f"  {source}: {quality:.4f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
