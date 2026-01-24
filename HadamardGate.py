# Import necessary modules from Qiskit
from qiskit import QuantumCircuit#, Aer, execute
from qiskit.visualization import plot_circuit_layout, circuit_drawer
#from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
# Create a quantum circuit with one qubit
qc = QuantumCircuit(2)

# Apply the Hadamard gate (H gate) to the qubit
qc.h(0)
qc.cx(0,1)
#qc.measure(0,0)
# Draw the quantum circuit
#print(qc.draw())

circuit_drawer(qc, output="mpl", filename="quantum_circuit_h.png")

#service = QiskitRuntimeService(channel="ibm_quantum", token="2a0d4a18284dc693b0c5f15cad5a137409156972657574ea16b3e3fd93aaed0d74a1b947b1c614403373c223db196e74ef40dc3a6f9628e3ca192369713df370")
#backend = service.least_busy(operational=True, simulator=False)

#sampler = Sampler(backend)

#job = sampler.run([qc])
#result = job.result()

# Simulate the circuit to observe the output state
#backend = Aer.get_backend('statevector_simulator')
#result = execute(qc, backend).result()
#statevector = result.get_statevector()

#print("\nStatevector after applying Hadamard gate:")
#print(statevector)
