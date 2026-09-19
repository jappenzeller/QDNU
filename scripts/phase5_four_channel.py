"""Phase 5, worked example 2: four channels -> 4x4 density matrix -> 2 system qubits + 2 ancilla.
Same window as the two-channel example (S001 R02, seconds 10-14). Channels Pz Oz O1 O2.
Checks: reduced state == rho; shot estimates of Z1, Z2, Z1Z2 vs tr(rho O); ancilla marginal vs lambda;
transpiled gate count for the whole preparation."""
import numpy as np, mne
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # circuit drawings are Unicode; Windows consoles default to cp1252
from scipy.signal import butter, filtfilt, iirnotch
from pyriemann.estimation import Covariances
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, partial_trace, Operator, DensityMatrix
from qiskit_aer import AerSimulator
mne.set_log_level("ERROR"); np.set_printoptions(precision=4, suppress=True, linewidth=120)

CH = ["Pz", "Oz", "O1", "O2"]
f = "data/eegmmidb/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R02.edf"
raw = mne.io.read_raw_edf(f, preload=True, verbose=False); mne.datasets.eegbci.standardize(raw); raw.pick(CH)
x = raw.get_data()*1e6; fs = raw.info["sfreq"]; x = x - x.mean(1, keepdims=True)
b,a = butter(4,[0.5/(fs/2),40/(fs/2)],btype="band"); x = filtfilt(b,a,x,axis=1)
b,a = iirnotch(60,Q=30,fs=fs); x = filtfilt(b,a,x,axis=1)
w = x[:, int(10*fs):int(14*fs)]
Sigma = Covariances(estimator="lwf").transform(w[None])[0]
print("S001 eyes closed, channels", CH, "seconds 10-14\n")
print("Sigma (uV^2):\n", Sigma, "\n  tr =", round(float(np.trace(Sigma)),2))
rho = Sigma/np.trace(Sigma)
print("\nrho:\n", rho)
lam, V = np.linalg.eigh(rho); lam, V = lam[::-1], V[:, ::-1]
print("\nlambda:", lam, " sum =", round(float(lam.sum()),6))
print("V (columns = patterns, rows =", CH, "):\n", V)
S = float(-(lam*np.log(lam)).sum())
print("purity =", round(float(np.trace(rho@rho)),4), " entropy =", round(S,4), "nats (max ln 4 =", round(np.log(4),4), ")  e^S =", round(np.exp(S),3))

# ---- circuit: q0,q1 = system ; q2,q3 = ancilla
qc = QuantumCircuit(4)
qc.append(StatePreparation(np.sqrt(lam)), [2, 3])     # ancilla amplitudes sqrt(lambda_k), k in basis |q3 q2>
qc.cx(2, 0); qc.cx(3, 1)                              # copy ancilla index onto system
qc.unitary(Operator(V), [0, 1], label="V")            # rotate computational basis into patterns
psi = Statevector(qc)
red = partial_trace(psi, [2, 3]).data.real
print("\nreduced system state:\n", red)
print("max |reduced - rho| =", np.abs(red - rho).max())
anc = partial_trace(psi, [0, 1]).data.real
print("ancilla marginal diagonal:", np.diag(anc), " (should equal lambda)")

# ---- observables. Qubit q0 is the least-significant bit of the channel index: index = 2*q1 + q0
# channel order [Pz,Oz,O1,O2] -> index 0..3 -> |q1 q0> = 00,01,10,11
Z = np.diag([1.,-1.]); I2 = np.eye(2)
obs = {"Z0 (Pz,O1 vs Oz,O2)": np.kron(I2, Z),     # acts on q0 -> index bit 0
       "Z1 (Pz,Oz vs O1,O2)": np.kron(Z, I2),     # acts on q1 -> index bit 1
       "Z0Z1 (Pz,O2 vs Oz,O1)": np.kron(Z, Z)}
exact = {k: float(np.trace(rho@O).real) for k,O in obs.items()}

sim = AerSimulator()
qc_m = qc.copy(); qc_m.measure_all(); qc_m = transpile(qc_m, sim, optimization_level=0)
res = sim.run(qc_m, shots=1000, seed_simulator=7).result().get_counts()
# bitstring order is q3 q2 q1 q0
def est(fn): return sum(v*fn(k) for k,v in res.items())/1000
shot = {"Z0 (Pz,O1 vs Oz,O2)": est(lambda k: 1-2*int(k[-1])),
        "Z1 (Pz,Oz vs O1,O2)": est(lambda k: 1-2*int(k[-2])),
        "Z0Z1 (Pz,O2 vs Oz,O1)": est(lambda k: (1-2*int(k[-1]))*(1-2*int(k[-2])))}
print("\n%-24s %9s %9s %9s" % ("observable","exact","shots","noise~"))
for k in obs: print("%-24s %9.4f %9.4f %9.4f" % (k, exact[k], shot[k], np.sqrt((1-exact[k]**2)/1000)))

anc_counts = {}
for k,v in res.items(): anc_counts[k[:2]] = anc_counts.get(k[:2],0)+v
print("\nancilla counts |q3 q2| ->", dict(sorted(anc_counts.items())))
print("ancilla fractions vs lambda:")
for i,l in enumerate(lam):
    key = format(i, "02b")            # k in |q3 q2> with q2 = LSB -> string is q3 q2
    print("  k=%d  %s  shots %.3f  lambda %.4f" % (i, key, anc_counts.get(key,0)/1000, l))
sys_counts = {}
for k,v in res.items(): sys_counts[k[2:]] = sys_counts.get(k[2:],0)+v
print("system marginal |q1 q0| ->", dict(sorted(sys_counts.items())), " diag(rho) =", np.diag(rho))

# ---- what hardware would see
for basis in (["rz","sx","x","cx"],):
    t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=1)
    ops = t.count_ops()
    print("\ntranspiled to", basis, "->", dict(ops), " depth", t.depth(), " CX count", ops.get("cx",0))
    tV = transpile(QuantumCircuit(2).compose(QuantumCircuit(2)), basis_gates=basis)
qV = QuantumCircuit(2); qV.unitary(Operator(V),[0,1]); tV = transpile(qV, basis_gates=["rz","sx","x","cx"], optimization_level=3)
qA = QuantumCircuit(2); qA.append(StatePreparation(np.sqrt(lam)),[0,1]); tA = transpile(qA, basis_gates=["rz","sx","x","cx"], optimization_level=3)
print("  of which: V alone CX =", tV.count_ops().get("cx",0), "; ancilla prep alone CX =", tA.count_ops().get("cx",0), "; copy CX = 2")
print("\ncircuit:"); print(qc.draw(output="text"))
