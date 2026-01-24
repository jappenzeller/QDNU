from numpy import array
from numpy import matmul
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector
from numpy import sqrt

import matplotlib.pyplot as plt

ket0 = array([1, 0])
ket1 = array([0, 1])

M1 = array([[1, 1], [0, 0]])
M2 = array([[1, 1], [1, 0]])

u = Statevector([1 / sqrt(2), 1 / sqrt(2)])
v = Statevector([(1 + 2.0j) / 3, -2 / 3])
w = Statevector([1 / 3, 2 / 3])

v = Statevector([(1 + 2.0j) / 3, -2 / 3])


v.draw("latex")

