import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import QuantumInstance

# Define the quantum circuit for the QNN
def create_quantum_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits)

    # Define parameter vectors for inputs and weights
    input_params = ParameterVector('x', num_qubits)
    weight_params = ParameterVector('w', num_qubits)

    # Encode input data into the quantum circuit
    for i in range(num_qubits):
        circuit.ry(input_params[i], i)

    # Apply parameterized rotations as trainable weights
    for i in range(num_qubits):
        circuit.rz(weight_params[i], i)

    # Add entangling gates
    for i in range(num_qubits - 1):
        circuit.cz(i, i + 1)

    return circuit, input_params, weight_params

# Build the QRNN model
class QuantumRNN(nn.Module):
    def __init__(self, seq_length, num_qubits):
        super(QuantumRNN, self).__init__()
        self.seq_length = seq_length
        self.num_qubits = num_qubits

        # Create the quantum circuit
        self.qc, self.input_params, self.weight_params = create_quantum_circuit(num_qubits)

        # Define the QNN
        self.quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))
        self.qnn = CircuitQNN(self.qc,
                              input_params=self.input_params,
                              weight_params=self.weight_params,
                              quantum_instance=self.quantum_instance,
                              interpret=lambda x: np.array([np.mean(x)]),
                              output_shape=(1,))

        # Connector to PyTorch
        self.qnn_torch = TorchConnector(self.qnn)

        # Linear layer for final output
        self.fc = nn.Linear(seq_length, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []

        for t in range(self.seq_length):
            # Get input at time step t
            xi = x[:, t, :]
            # Pass through the QNN
            qnn_output = self.qnn_torch(xi)
            outputs.append(qnn_output)

        # Concatenate outputs from all time steps
        outputs = torch.cat(outputs, dim=1)
        # Pass through the linear layer
        out = self.fc(outputs)
        return out

# Generate synthetic sequential data
def generate_data(num_samples, seq_length, num_features):
    X = np.random.rand(num_samples, seq_length, num_features)
    y = np.sum(X, axis=(1, 2)) > (seq_length * num_features * 0.5)
    y = y.astype(np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define parameters
seq_length = 5
num_qubits = 2  # Number of qubits in the quantum circuit
num_features = num_qubits  # For simplicity, match features to qubits

# Initialize the model, loss function, and optimizer
model = QuantumRNN(seq_length=seq_length, num_qubits=num_qubits)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Prepare data
X_train, y_train = generate_data(50, seq_length, num_features)
X_test, y_test = generate_data(10, seq_length, num_features)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
    accuracy = (predictions == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")