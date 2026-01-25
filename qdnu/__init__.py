"""
QDNU - Quantum Positive-Negative Neuron Library

Core quantum computing architecture for multi-channel EEG analysis.
"""

from .quantum_agate import (
    create_single_channel_agate,
    visualize_agate,
    get_statevector,
    compute_fidelity,
)
from .pn_dynamics import PNDynamics
from .multichannel_circuit import (
    create_multichannel_circuit,
    get_qubit_indices,
    add_measurements,
)
from .template_trainer import TemplateTrainer
from .seizure_predictor import SeizurePredictor
from .quantum_backends import (
    QuantumBackend,
    LocalSimulator,
    get_available_backends,
)

__all__ = [
    # A-Gate circuit
    'create_single_channel_agate',
    'visualize_agate',
    'get_statevector',
    'compute_fidelity',
    # PN dynamics
    'PNDynamics',
    # Multi-channel circuit
    'create_multichannel_circuit',
    'get_qubit_indices',
    'add_measurements',
    # Template and prediction
    'TemplateTrainer',
    'SeizurePredictor',
    # Backends
    'QuantumBackend',
    'LocalSimulator',
    'get_available_backends',
]

__version__ = '0.1.0'
