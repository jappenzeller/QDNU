from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_bloch_multivector

# Create a quantum circuit with 3 qubits (2 main qubits + 1 ancillary)
qc = QuantumCircuit(3)

# Step 1: Create an entangled state (e.g., Bell state)
qc.h(0)
qc.cx(0, 1)

# Step 2: Use an ancillary qubit to control a rotation on the entangled qubits
qc.cx(0, 2)  # Entangle control qubit with ancilla
qc.ry(0.5, 1).c_if(qc.clbits, 0)  # Conditional rotation on qubit 1 (example)
qc.cx(0, 2)  # Uncompute ancilla

# Optional: Measurement (to observe the effect)
qc.measure_all()

# Visualize the circuit
print(qc.draw())

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
result = execute(qc, simulator).result()
statevector = result.get_statevector()

# Plot the final state on the Bloch sphere
plot_bloch_multivector(statevector)
