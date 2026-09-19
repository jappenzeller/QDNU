"""Phase 5, worked example 2: eight channels -> 8x8 density matrix -> 3 system qubits + 3 ancilla.
Same window as the two-channel example (S001 R02, seconds 10-14). DSP-000 "P" subset.
Checks: reduced state == rho; shot estimates of Z1, Z2, Z1Z2 vs tr(rho O); ancilla marginal vs lambda;
transpiled gate count for the whole preparation."""
import numpy as np, mne
from scipy.signal import butter, filtfilt, iirnotch
from pyriemann.estimation import Covariances
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, partial_trace, Operator, DensityMatrix
from qiskit_aer import AerSimulator
mne.set_log_level("ERROR"); np.set_printoptions(precision=4, suppress=True, linewidth=120)

CH = ["Fz", "Cz", "Pz", "Oz", "O1", "O2", "P3", "P4"]
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
print("purity =", round(float(np.trace(rho@rho)),4), " entropy =", round(S,4), "nats (max ln 8 =", round(np.log(8),4), ")  e^S =", round(np.exp(S),3))

# ---- circuit: q0,q1 = system ; q2,q3 = ancilla
qc = QuantumCircuit(6)
qc.append(StatePreparation(np.sqrt(lam)), [3, 4, 5])  # ancilla amplitudes sqrt(lambda_k), k in basis |q5 q4 q3>
qc.cx(3, 0); qc.cx(4, 1); qc.cx(5, 2)                 # copy ancilla index onto system
qc.unitary(Operator(V), [0, 1, 2], label="V")         # rotate computational basis into patterns
psi = Statevector(qc)
red = partial_trace(psi, [3, 4, 5]).data.real
print("\nreduced system state:\n", red)
print("max |reduced - rho| =", np.abs(red - rho).max())
print("purity of system from statevector:", round(float(np.trace(red@red)),4))
anc = partial_trace(psi, [0, 1, 2]).data.real
print("ancilla marginal diagonal:", np.diag(anc), " (should equal lambda)")

# ---- observables. Qubit q0 is the least-significant bit of the channel index: index = 2*q1 + q0
# channel order [Pz,Oz,O1,O2] -> index 0..3 -> |q1 q0> = 00,01,10,11
Z = np.diag([1.,-1.]); I2 = np.eye(2)
def zs(bits):  # bits: which system qubits carry Z, e.g. (0,), (1,2)
    M = np.array([[1.]]); 
    for q in (2,1,0): M = np.kron(M, Z if q in bits else I2)
    return M
def label(bits):
    plus = [CH[i] for i in range(8) if sum((i>>q)&1 for q in bits)%2==0]
    return "Z"+"".join(str(b) for b in bits)+" (+: "+",".join(plus)+")"
obs = {label(b): zs(b) for b in [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]}
exact = {k: float(np.trace(rho@O).real) for k,O in obs.items()}

sim = AerSimulator()
qc_m = qc.copy(); qc_m.measure_all(); qc_m = transpile(qc_m, sim, optimization_level=0)
res = sim.run(qc_m, shots=1000, seed_simulator=7).result().get_counts()
# bitstring order is q3 q2 q1 q0
def est(fn): return sum(v*fn(k) for k,v in res.items())/1000
def parity(k, bits): return (-1)**sum(int(k[-1-b]) for b in bits)
shot = {label(b): est(lambda k, b=b: parity(k,b)) for b in [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]}
print("\n%-30s %9s %9s %9s" % ("observable","exact","shots","noise~"))
for k in obs: print("%-30s %9.4f %9.4f %9.4f" % (k, exact[k], shot[k], np.sqrt((1-exact[k]**2)/1000)))

anc_counts = {}
for k,v in res.items(): anc_counts[k[:3]] = anc_counts.get(k[:3],0)+v
print("\nancilla counts |q5 q4 q3| ->", dict(sorted(anc_counts.items())))
print("ancilla fractions vs lambda:")
for i,l in enumerate(lam):
    key = format(i, "03b")
    print("  k=%d  %s  shots %.3f  lambda %.4f" % (i, key, anc_counts.get(key,0)/1000, l))
sys_counts = {}
for k,v in res.items(): sys_counts[k[3:]] = sys_counts.get(k[3:],0)+v
print("system marginal |q2 q1 q0| ->", dict(sorted(sys_counts.items())), " diag(rho) =", np.diag(rho))

# ---- what hardware would see
for basis in (["rz","sx","x","cx"],):
    t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=1)
    ops = t.count_ops()
    print("\ntranspiled to", basis, "->", dict(ops), " depth", t.depth(), " CX count", ops.get("cx",0))
    tV = transpile(QuantumCircuit(2).compose(QuantumCircuit(2)), basis_gates=basis)
qV = QuantumCircuit(3); qV.unitary(Operator(V),[0,1,2]); tV = transpile(qV, basis_gates=["rz","sx","x","cx"], optimization_level=3)
qA = QuantumCircuit(3); qA.append(StatePreparation(np.sqrt(lam)),[0,1,2]); tA = transpile(qA, basis_gates=["rz","sx","x","cx"], optimization_level=3)
print("  of which: V alone CX =", tV.count_ops().get("cx",0), "; ancilla prep alone CX =", tA.count_ops().get("cx",0), "; copy CX = 3")
print("\ndet V =", round(float(np.linalg.det(V)),4))
