"""
Symmetry-preserving ansatz for Track A (Limitations: 'nine of ten tracks use the same hardware-efficient
ansatz, not the symmetry-preserving circuits available for several of them (A/B/D/G)').

The staggered-fermion Schwinger Hamiltonian conserves total charge; its exact ground state lies in the
half-filling (total-Z = 0) sector for every task tested (variance ~1e-31, checked at 4 and 6 qubits). So
PennyLane's particle-conserving ParticleConservingU2 template, started from a half-filled reference state, can
represent the ground state and never leaves the right sector. Same Q-MAML protocol as the notebook (Learner ->
weights, Algorithm 1 pretraining, Algorithm 2 adaptation), with all five init schemes applied to the U2
weights, so the comparison is between init schemes WITHIN this ansatz.

Configs (size, layers, n_test, adapt_iters, pretrain epochs, lr): L=2 and L=3 use the notebook protocol; L=4
uses the reduced protocol of run_schwinger_l4_scan.py (3 tasks, 150 iters, 100 epochs, lr 0.002), since that is
the size where the hardware-efficient ansatz failed. Seeds 0,1,2. Resumable.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn, pennylane as qml
from hamiltonians_schwinger import schwinger_hamiltonian, exact_ground_energy, sample_task_space
from schwinger_qmaml import classical_init
from models.learner import Learner

OUT = os.path.join(os.path.dirname(__file__), "results", "schwinger", "symmetric_ansatz_U2.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
S = ["qmaml", "zero", "pi", "uniform", "gaussian"]

class U2PQC(nn.Module):
    def __init__(self, n, layers):
        super().__init__()
        self.n, self.layers = n, layers
        self.shape = tuple(qml.ParticleConservingU2.shape(layers, n))
        self.init = np.array([1, 0] * (n // 2))
        dev = qml.device("default.qubit", wires=n)
        self.dev = dev
    def cost(self, w, H):
        n, init = self.n, self.init
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def c(w):
            qml.ParticleConservingU2(w, wires=range(n), init_state=init)
            return qml.expval(H)
        return c(w.to(torch.float64))

def run(seed, n, layers, n_test, iters, ep, plr, alr=0.02):
    torch.manual_seed(seed); np.random.seed(seed)
    train = sample_task_space(16, seed=seed)
    test = sample_task_space(n_test, seed=seed + 1000)
    E0 = [exact_ground_energy(schwinger_hamiltonian(n, m0, g), n) for m0, g in test]
    mabs = float(np.mean(np.abs(E0)))
    pqc = U2PQC(n, layers)
    learner = Learner(2, pqc.shape, hidden=256).double()
    opt = torch.optim.Adam(learner.parameters(), lr=plr)
    gn_last = None
    for e in range(ep):
        gns = []
        for m0, g in train:
            H = schwinger_hamiltonian(n, m0, g)
            opt.zero_grad()
            c = pqc.cost(0.3 * learner(torch.tensor([m0, g], dtype=torch.float64)), H)
            c.backward()
            gns.append(sum(float(p.grad.pow(2).sum()) for p in learner.parameters() if p.grad is not None) ** 0.5)
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0); opt.step()
        gn_last = float(np.mean(gns))
    gaps = {k: [] for k in S}
    for (m0, g), e0 in zip(test, E0):
        H = schwinger_hamiltonian(n, m0, g)
        for s in S:
            th0 = (0.3 * learner(torch.tensor([m0, g], dtype=torch.float64))).detach() if s == "qmaml" \
                else classical_init(pqc.shape, s, layers)
            th = nn.Parameter(th0.clone().double()); o = torch.optim.Adam([th], lr=alr); costs = []
            for _ in range(iters):
                o.zero_grad(); cc = pqc.cost(th, H); cc.backward(); o.step(); costs.append(cc.item())
            gaps[s].append(np.log(np.abs(np.array(costs) - e0) + 1e-8))
    final = {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mabs) for k, v in gaps.items()}
    final["_pretrain_gradnorm_last"] = gn_last
    return final

configs = [("L2", 4, 3, 6, 300, 40, 0.005), ("L3_l3", 6, 3, 6, 300, 40, 0.005), ("L3_l6", 6, 6, 6, 300, 40, 0.005),
           ("L4_l6", 8, 6, 3, 150, 100, 0.002)]
for name, n, layers, nt, iters, ep, plr in configs:
    for seed in [0, 1, 2]:
        key = f"{name}/seed{seed}"
        if key in out: continue
        t0 = time.time()
        out[key] = run(seed, n, layers, nt, iters, ep, plr)
        json.dump(out, open(OUT, "w"))
        print(key, f"{time.time()-t0:.0f}s", {k: round(v, 5) for k, v in out[key].items()}, flush=True)
print("U2 DONE", flush=True)
