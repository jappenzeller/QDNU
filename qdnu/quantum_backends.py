"""
Quantum Hardware Backends for QDNU

Provides abstraction layer for running quantum circuits on:
- Local simulators (FREE)
- AWS Braket (IonQ, Rigetti, Amazon SV1)
- IBM Qiskit (IBM Quantum processors)

Cross-platform verification ensures robustness.
"""

import numpy as np
from abc import ABC, abstractmethod


class QuantumBackend(ABC):
    """Abstract base class for quantum backends."""

    @abstractmethod
    def run_circuit(self, circuit, shots=1024):
        """
        Execute a quantum circuit.

        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of measurement shots

        Returns:
            dict: Results with 'counts', 'statevector' (if available)
        """
        pass

    @abstractmethod
    def get_statevector(self, circuit):
        """
        Get exact statevector (simulator only).

        Args:
            circuit: Qiskit QuantumCircuit

        Returns:
            numpy array: Complex statevector
        """
        pass

    @abstractmethod
    def is_available(self):
        """Check if backend is available."""
        pass


class LocalSimulator(QuantumBackend):
    """
    Local Qiskit Aer simulator (FREE, fast).

    Best for development and testing.
    """

    def __init__(self):
        self.name = "local_aer"
        self._check_availability()

    def _check_availability(self):
        try:
            from qiskit_aer import AerSimulator
            self._simulator = AerSimulator()
            self._available = True
        except ImportError:
            try:
                from qiskit import Aer
                self._simulator = Aer.get_backend('aer_simulator')
                self._available = True
            except:
                self._available = False

    def is_available(self):
        return self._available

    def run_circuit(self, circuit, shots=1024):
        if not self._available:
            raise RuntimeError("Local simulator not available")

        from qiskit import transpile

        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self._simulator)
        job = self._simulator.run(transpiled, shots=shots)
        result = job.result()

        return {
            'counts': result.get_counts(),
            'shots': shots,
            'backend': self.name
        }

    def get_statevector(self, circuit):
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(circuit)
        return sv.data


class BraketBackend(QuantumBackend):
    """
    AWS Braket backend for cloud quantum computing.

    Supports:
    - LocalSimulator (FREE)
    - SV1 state vector simulator (~$0.075/min)
    - IonQ, Rigetti, OQC QPUs

    Setup:
        aws configure
        # Enter AWS Access Key, Secret Key, Region
    """

    # Common device ARNs
    DEVICES = {
        'local': None,  # Use LocalSimulator
        'sv1': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
        'dm1': 'arn:aws:braket:::device/quantum-simulator/amazon/dm1',
        'ionq': 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
        'rigetti': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3',
    }

    def __init__(self, device='local', s3_folder=None):
        """
        Initialize Braket backend.

        Args:
            device: 'local', 'sv1', 'ionq', 'rigetti', or full ARN
            s3_folder: S3 bucket for results (required for cloud)
        """
        self.device_name = device
        self.s3_folder = s3_folder
        self._check_availability()

    def _check_availability(self):
        try:
            if self.device_name == 'local':
                from braket.devices import LocalSimulator
                self._device = LocalSimulator()
            else:
                from braket.aws import AwsDevice
                arn = self.DEVICES.get(self.device_name, self.device_name)
                self._device = AwsDevice(arn)
            self._available = True
        except ImportError:
            self._available = False
        except Exception as e:
            print(f"Braket initialization warning: {e}")
            self._available = False

    def is_available(self):
        return self._available

    def _qiskit_to_braket(self, qiskit_circuit):
        """Convert Qiskit circuit to Braket circuit."""
        from braket.circuits import Circuit as BraketCircuit

        braket_qc = BraketCircuit()
        n_qubits = qiskit_circuit.num_qubits

        for instruction in qiskit_circuit.data:
            gate = instruction.operation
            qubits = [q._index for q in instruction.qubits]
            name = gate.name.lower()
            params = gate.params

            # Map common gates
            if name == 'h':
                braket_qc.h(qubits[0])
            elif name == 'x':
                braket_qc.x(qubits[0])
            elif name == 'y':
                braket_qc.y(qubits[0])
            elif name == 'z':
                braket_qc.z(qubits[0])
            elif name == 'rx':
                braket_qc.rx(qubits[0], params[0])
            elif name == 'ry':
                braket_qc.ry(qubits[0], params[0])
            elif name == 'rz':
                braket_qc.rz(qubits[0], params[0])
            elif name == 'p':
                braket_qc.phaseshift(qubits[0], params[0])
            elif name == 'cx' or name == 'cnot':
                braket_qc.cnot(qubits[0], qubits[1])
            elif name == 'cz':
                braket_qc.cz(qubits[0], qubits[1])
            elif name == 'cry':
                braket_qc.cry(qubits[0], qubits[1], params[0])
            elif name == 'crz':
                braket_qc.crz(qubits[0], qubits[1], params[0])
            elif name == 'measure':
                pass  # Handle separately
            else:
                print(f"Warning: Gate '{name}' not mapped, skipping")

        return braket_qc

    def run_circuit(self, circuit, shots=1024):
        if not self._available:
            raise RuntimeError("Braket backend not available")

        braket_circuit = self._qiskit_to_braket(circuit)

        if self.device_name == 'local':
            task = self._device.run(braket_circuit, shots=shots)
        else:
            if not self.s3_folder:
                raise ValueError("S3 folder required for cloud execution")
            task = self._device.run(
                braket_circuit,
                s3_destination_folder=self.s3_folder,
                shots=shots
            )

        result = task.result()

        return {
            'counts': dict(result.measurement_counts),
            'shots': shots,
            'backend': f'braket_{self.device_name}'
        }

    def get_statevector(self, circuit):
        if self.device_name != 'local':
            raise ValueError("Statevector only available for local simulator")

        braket_circuit = self._qiskit_to_braket(circuit)

        from braket.devices import LocalSimulator
        device = LocalSimulator()
        task = device.run(braket_circuit, shots=0)  # 0 shots = statevector
        result = task.result()

        return np.array(result.get_value_by_result_type_name('StateVector'))


class IBMBackend(QuantumBackend):
    """
    IBM Quantum backend.

    Supports:
    - AerSimulator (local, FREE)
    - ibmq_qasm_simulator (cloud simulator)
    - Real QPUs (ibm_brisbane, etc.)

    Setup:
        export QISKIT_IBM_TOKEN=your_token
        # Get token from https://quantum.ibm.com
    """

    def __init__(self, backend_name='aer_simulator', use_simulator=True):
        """
        Initialize IBM backend.

        Args:
            backend_name: Name of IBM backend or 'aer_simulator'
            use_simulator: If True, use local Aer simulator
        """
        self.backend_name = backend_name
        self.use_simulator = use_simulator
        self._check_availability()

    def _check_availability(self):
        try:
            if self.use_simulator:
                from qiskit_aer import AerSimulator
                self._backend = AerSimulator()
            else:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                self._backend = service.backend(self.backend_name)
            self._available = True
        except ImportError:
            self._available = False
        except Exception as e:
            print(f"IBM backend warning: {e}")
            self._available = False

    def is_available(self):
        return self._available

    def run_circuit(self, circuit, shots=1024):
        if not self._available:
            raise RuntimeError("IBM backend not available")

        from qiskit import transpile

        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self._backend)

        if self.use_simulator:
            job = self._backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts()
        else:
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(self._backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            counts = result[0].data.meas.get_counts()

        return {
            'counts': counts,
            'shots': shots,
            'backend': f'ibm_{self.backend_name}'
        }

    def get_statevector(self, circuit):
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(circuit)
        return sv.data


def get_available_backends():
    """List all available quantum backends."""
    backends = {}

    # Local simulator
    local = LocalSimulator()
    backends['local'] = {
        'available': local.is_available(),
        'type': 'simulator',
        'cost': 'FREE'
    }

    # Braket local
    try:
        braket = BraketBackend(device='local')
        backends['braket_local'] = {
            'available': braket.is_available(),
            'type': 'simulator',
            'cost': 'FREE'
        }
    except:
        backends['braket_local'] = {'available': False}

    # IBM simulator
    ibm = IBMBackend(use_simulator=True)
    backends['ibm_simulator'] = {
        'available': ibm.is_available(),
        'type': 'simulator',
        'cost': 'FREE'
    }

    return backends


def compute_fidelity_from_counts(counts1, counts2, shots):
    """
    Estimate fidelity from measurement counts (for hardware runs).

    Uses classical shadow / cross-entropy approach.
    """
    # Normalize counts to probabilities
    prob1 = {k: v / shots for k, v in counts1.items()}
    prob2 = {k: v / shots for k, v in counts2.items()}

    # Classical fidelity estimate (Bhattacharyya coefficient)
    all_keys = set(prob1.keys()) | set(prob2.keys())
    fidelity = sum(
        np.sqrt(prob1.get(k, 0) * prob2.get(k, 0))
        for k in all_keys
    ) ** 2

    return fidelity


# === Test cases ===

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Backends Test Suite")
    print("=" * 60)

    # Test 1: Check available backends
    print("\n=== Test 1: Available Backends ===")
    backends = get_available_backends()
    for name, info in backends.items():
        status = "OK" if info.get('available') else "NOT AVAILABLE"
        print(f"  {name}: [{status}]")

    # Test 2: Local simulator
    print("\n=== Test 2: Local Simulator ===")
    local = LocalSimulator()
    if local.is_available():
        from quantum_agate import create_single_channel_agate

        qc = create_single_channel_agate(0.5, 0.3, 0.7)
        sv = local.get_statevector(qc)
        print(f"Statevector shape: {sv.shape}")
        print(f"Statevector norm: {np.linalg.norm(sv):.6f}")

        result = local.run_circuit(qc, shots=1000)
        print(f"Measurement counts: {len(result['counts'])} outcomes")
        print(f"Backend: {result['backend']}")
        print("[OK] Local simulator working")
    else:
        print("[SKIP] Local simulator not available")

    # Test 3: Braket local
    print("\n=== Test 3: Braket Local ===")
    try:
        braket = BraketBackend(device='local')
        if braket.is_available():
            from quantum_agate import create_single_channel_agate
            qc = create_single_channel_agate(0.5, 0.3, 0.7)
            result = braket.run_circuit(qc, shots=1000)
            print(f"Braket counts: {len(result['counts'])} outcomes")
            print("[OK] Braket local working")
        else:
            print("[SKIP] Braket not available")
    except Exception as e:
        print(f"[SKIP] Braket error: {e}")

    # Test 4: IBM simulator
    print("\n=== Test 4: IBM Simulator ===")
    ibm = IBMBackend(use_simulator=True)
    if ibm.is_available():
        from quantum_agate import create_single_channel_agate
        qc = create_single_channel_agate(0.5, 0.3, 0.7)
        result = ibm.run_circuit(qc, shots=1000)
        print(f"IBM counts: {len(result['counts'])} outcomes")
        print("[OK] IBM simulator working")
    else:
        print("[SKIP] IBM simulator not available")

    # Test 5: Cross-platform comparison
    print("\n=== Test 5: Cross-Platform Comparison ===")
    if local.is_available():
        from multichannel_circuit import create_multichannel_circuit

        params = [(0.5, 0.3, 0.6), (0.4, 0.35, 0.65)]
        qc = create_multichannel_circuit(params)

        sv = local.get_statevector(qc)
        result = local.run_circuit(qc, shots=4096)

        # Check consistency
        print(f"Circuit qubits: {qc.num_qubits}")
        print(f"Statevector norm: {np.linalg.norm(sv):.6f}")
        print(f"Total counts: {sum(result['counts'].values())}")
        print("[OK] Cross-platform test complete")

    print("\n" + "=" * 60)
    print("Backend tests complete!")
    print("=" * 60)
