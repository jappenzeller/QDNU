from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import plot_circuit_layout

# Define parameters for the QDNU
input_theta1 = Parameter('θ1')  # Input parameter for qubit 1
input_theta2 = Parameter('θ2')  # Input parameter for qubit 2
weight1 = Parameter('w1')       # Weight parameter for qubit 1
weight2 = Parameter('w2')       # Weight parameter for qubit 2
bias1 = Parameter('b1')         # Bias parameter for qubit 1
bias2 = Parameter('b2')         # Bias parameter for qubit 2

# Create a quantum circuit
qc = QuantumCircuit(2)  # Two-qubit circuit
qc.ry(input_theta1, 0)   # Encode input data into qubit 1
qc.ry(input_theta2, 1)   # Encode input data into qubit 2
qc.rz(weight1, 0)        # Apply weight as a rotation to qubit 1
qc.rz(weight2, 1)        # Apply weight as a rotation to qubit 2
qc.rx(bias1, 0)          # Apply bias as a rotation to qubit 1
qc.rx(bias2, 1)          # Apply bias as a rotation to qubit 2
qc.cx(0, 1)              # Add a controlled-X gate for entanglement
qc.measure_all()         # Add measurements

# Visualize the circuit
qc.draw('mpl')
