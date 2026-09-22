"""
Track B follow-ups (resumable, saves after every finished piece):
  full   : gauge-invariant HVA and non-gauge-invariant control ansatz at Table-5-style FULL budget
           (6 held-out tasks, 300 adaptation iterations; the earlier replication used 4 tasks / 200 iters), seeds 0,1,2.
  anneal : the untried 'anneal V during pretraining' idea on the headline hardware-efficient ansatz (Z2LGTPQC),
           evaluated at V=10 with the headline protocol (6 tasks, 300 iters). Two schedules, geometric in V over
           the pretraining epochs: 'up' 1->10 and 'down' 30->10. Seeds 0,1,2.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from hamiltonians_z2lgt import z2_lgt_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
from z2lgt_qmaml import (classical_init, adapt_task_z2lgt, Z2LGTPQC, Z2LGTHVAPQC, Z2LGTHWEffZZPQC,
                          pretrain_learner_z2lgt_ansatz)
from models.learner import Learner

mode = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "results", f"z2_{mode}.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(out, open(OUT, "w"))
S = ["qmaml", "zero", "pi", "uniform", "gaussian"]
V, DEPTH, ELR = 10.0, 3, 0.02

def evaluate(pqc, learner, test, E0s, iters):
    mabs = float(np.mean(np.abs(E0s)))
    gaps = {k: [] for k in S}
    shape = pqc.shape if hasattr(pqc, "shape") else (DEPTH, NUM_QUBITS, 3)
    for (J, m, mu), E0 in zip(test, E0s):
        H = z2_lgt_hamiltonian(J, m, mu, V=V)
        for s in S:
            th0 = (0.3 * learner(torch.tensor([J, m, mu], dtype=torch.float64))).detach() if s == "qmaml" \
                else classical_init(shape, s, DEPTH)
            _, costs = adapt_task_z2lgt(pqc, th0, H, iters=iters, lr=ELR)
            gaps[s].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    return {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mabs) for k, v in gaps.items()}

def pretrain_annealed(pqc, train, epochs, lr, v_start, v_end):
    learner = Learner(3, (DEPTH, NUM_QUBITS, 3), hidden=256).double()
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    hist = []
    for e in range(epochs):
        v = v_start * (v_end / v_start) ** (e / max(epochs - 1, 1))
        tot = 0.0
        for J, m, mu in train:
            H = z2_lgt_hamiltonian(J, m, mu, V=v)
            opt.zero_grad()
            cost = pqc.cost(0.3 * learner(torch.tensor([J, m, mu], dtype=torch.float64)), H)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
            opt.step(); tot += cost.item()
        hist.append(tot / len(train))
    return learner, hist

for seed in [0, 1, 2]:
    torch.manual_seed(seed); np.random.seed(seed)
    train = sample_task_space(16, seed=seed)
    test = sample_task_space(6, seed=seed + 1000)
    E0s = [exact_ground_energy(z2_lgt_hamiltonian(J, m, mu, V=V)) for J, m, mu in test]
    if mode == "full":
        for tag, Cls in [("gauge_invariant_hva", Z2LGTHVAPQC), ("hardware_efficient_zz", Z2LGTHWEffZZPQC)]:
            key = f"{tag}/seed{seed}"
            if key in out: continue
            torch.manual_seed(seed); np.random.seed(seed)
            pqc = Cls(NUM_QUBITS, DEPTH)
            learner, _ = pretrain_learner_z2lgt_ansatz(pqc, task_dim=3, train_tasks=train, epochs=40, lr=0.005,
                                                        w0_scale=0.3, log_every=999, gauss_V=V, tag=tag)
            out[key] = evaluate(pqc, learner, test, E0s, 300); save()
            print(key, out[key], flush=True)
    else:
        for tag, (va, vb) in {"up_1to10": (1.0, 10.0), "down_30to10": (30.0, 10.0)}.items():
            key = f"{tag}/seed{seed}"
            if key in out: continue
            torch.manual_seed(seed); np.random.seed(seed)
            pqc = Z2LGTPQC(NUM_QUBITS, DEPTH)
            learner, hist = pretrain_annealed(pqc, train, 40, 0.005, va, vb)
            r = evaluate(pqc, learner, test, E0s, 300); r["_pretrain_loss_last"] = hist[-1]
            out[key] = r; save()
            print(key, r, flush=True)
print("Z2 DONE", mode, flush=True)
