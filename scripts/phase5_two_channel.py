import numpy as np, mne
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # circuit drawings are Unicode; Windows consoles default to cp1252
from scipy.signal import butter, filtfilt, iirnotch
from pyriemann.estimation import Covariances
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, Operator
from qiskit_aer import AerSimulator
mne.set_log_level("ERROR"); np.set_printoptions(precision=4, suppress=True)

f = "data/eegmmidb/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R02.edf"
raw = mne.io.read_raw_edf(f, preload=True, verbose=False); mne.datasets.eegbci.standardize(raw); raw.pick(["O1","O2"])
x = raw.get_data()*1e6; fs = raw.info["sfreq"]; x = x - x.mean(1, keepdims=True)
b,a = butter(4,[0.5/(fs/2),40/(fs/2)],btype="band"); x = filtfilt(b,a,x,axis=1)
b,a = iirnotch(60,Q=30,fs=fs); x = filtfilt(b,a,x,axis=1)
w = x[:, int(10*fs):int(14*fs)]
Sigma = Covariances(estimator="lwf").transform(w[None])[0]
print("S001, eyes closed, O1/O2, seconds 10-14, 4 s at 160 Hz = 640 samples per channel\n")
print("covariance Sigma (uV^2), rows/cols = O1, O2:\n", Sigma)
rho = Sigma/np.trace(Sigma)
print("\nrho = Sigma / tr(Sigma):\n", rho, "\n  trace =", round(float(np.trace(rho)),6))
lam, V = np.linalg.eigh(rho); lam, V = lam[::-1], V[:, ::-1]
print("\nweights lambda:", lam, "  sum =", round(float(lam.sum()),6))
print("patterns V (columns = eigenvectors, rows = O1, O2):\n", V)
print("purity tr(rho^2) =", round(float(np.trace(rho@rho)),4), "  entropy =", round(float(-(lam*np.log(lam)).sum()),4), "nats  (max ln 2 =", round(np.log(2),4), ")")

qc = QuantumCircuit(2)                                    # q0 = system, q1 = ancilla
theta = 2*np.arctan2(np.sqrt(lam[1]), np.sqrt(lam[0]))    # Ry(theta)|0> = sqrt(lam0)|0> + sqrt(lam1)|1>
qc.ry(theta, 1); qc.cx(1, 0); qc.unitary(Operator(V), [0], label="V")
psi = Statevector(qc)
print("\ncircuit (q0 system, q1 ancilla):"); print(qc.draw(output="text"))
print("|Psi> amplitudes, basis order |q1 q0> = |anc sys>:", psi.data.real.round(4))
red = partial_trace(psi, [1]).data.real
print("\nreduced state on the system (ancilla traced out):\n", red)
print("max |reduced - rho| =", np.abs(red - rho).max())

Z = np.diag([1.,-1.]); classical = float(np.trace(rho@Z).real)
print("\ntr(rho Z) = rho11 - rho22 =", round(classical,4))
qc_m = qc.copy(); qc_m.measure_all()
counts = AerSimulator().run(qc_m, shots=1000, seed_simulator=7).result().get_counts()
p0 = sum(v for k,v in counts.items() if k[-1]=="0")/1000
print("counts over 1000 shots (bitstring = anc sys):", dict(sorted(counts.items())))
print("shot estimate <Z> = P(sys=0) - P(sys=1) =", round(2*p0-1,4), "   expected shot noise ~", round(float(np.sqrt((1-classical**2)/1000)),4))
