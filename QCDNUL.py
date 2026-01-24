from qiskit import  QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.visualization import plot_circuit_layout, circuit_drawer
import numpy as np

# Define a parameterized quantum circuit (QDNU)
theta1 = Parameter('θ1')  # Parameter for the first qubit
theta2 = Parameter('θ2')  # Parameter for the second qubit

# Create the quantum circuit
qc = QuantumCircuit(2)
qc.ry(theta1, 0)  # Apply a Ry rotation with θ1 on qubit 0
qc.ry(theta2, 1)  # Apply a Ry rotation with θ2 on qubit 1
qc.cx(0, 1)       # Add a controlled-X gate


circuit_drawer(qc, output="mpl", filename="quantum_circuit.png")

# Define an observable for two qubits (e.g., Z tensor Z)
observable = SparsePauliOp.from_list([("ZZ", 1.0)])  # Measure Z ⊗ Z on both qubits

estimator = StatevectorEstimator()

# Define an objective function
def objective_function(params):
    # Bind parameters to the circuit
    bound_circuit = qc.assign_parameters({theta1: params[0], theta2: params[1]})
    
    # Use the Qiskit Estimator primitive to evaluate the circuit
    
    result = estimator.run(bound_circuit, observable)
    expectation_value = result.result().values[0]
    
    # Define the target and calculate loss
    target = 1.0  # Example target value
    loss = (expectation_value - target) ** 2
    return loss

# Initialize the optimizer
optimizer = COBYLA(maxiter=100)

# Initial guess for parameters
initial_params = [0.5, 0.5]

# Optimize the circuit parameters
result = optimizer.minimize(fun=objective_function, x0=initial_params)

# Print results
print("Optimized Parameters:", result.x)
print("Minimum Loss:", result.fun)