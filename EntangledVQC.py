from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Define parameters for the variational circuit
theta = Parameter('θ')
phi = Parameter('ϕ')

# Create a quantum circuit with 2 qubits
qc = QuantumCircuit(2)

# Step 1: Entangle the two qubits (create a Bell state)
qc.h(0)          # Hadamard on qubit 0
qc.cx(0, 1)      # CNOT with qubit 0 as control, qubit 1 as target

# Step 2: Add parameterized variational gates
qc.rx(theta, 0)  # Apply RX rotation on qubit 0
qc.ry(phi, 1)    # Apply RY rotation on qubit 1

# Optional: Add another CNOT for more entanglement
qc.cx(0, 1)

# Draw the circuit
qc.draw('mpl')
