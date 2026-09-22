"""
Test whether Track D's N=4 Q-MAML failures are bad local minima of the META-objective that pretraining
restarts can avoid. Diagnostics (results/su2_n4_diagnostics.json) showed seed 2 converges, at both pretraining
settings, to a plateau with training meta-loss ~0.95 (ground energies ~ -0.3) whose predicted inits are
stationary points (adapted energy == init energy). Here: for the two bad data seeds (2, then 0) run 6 pretraining
restarts that differ ONLY in the Learner's random initialization (torch seed), same tasks, original setting
(40 epochs, lr 0.01, depth 6, 300 adaptation iters). Record final TRAINING meta-loss (available without test
data) and test relative error for each restart, so selection-by-training-loss can be evaluated honestly.
Resumable.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from hamiltonians_su2lgt import su2_lgt_hamiltonian, exact_ground_energy, sample_task_space
from su2lgt_qmaml import classical_init, pretrain_learner_su2lgt, adapt_task_su2lgt

NQ, N, DEPTH, ITERS, LR = 8, 4, 6, 300, 0.05
OUT = os.path.join(os.path.dirname(__file__), "results", "su2_n4_restarts.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(out, open(OUT, "w"))

def rel_err(costs_final, E0, mean_abs):
    return float(np.exp(np.mean(np.log(np.abs(np.array(costs_final) - np.array(E0)) + 1e-8))) / mean_abs)

for dseed in [2, 0]:
    train = sample_task_space(16, seed=dseed)
    test = sample_task_space(6, seed=dseed + 1000)
    E0 = [exact_ground_energy(su2_lgt_hamiltonian(N, m, x), NQ) for m, x in test]
    mabs = float(np.mean(np.abs(E0)))
    for R in range(6):
        key = f"dseed{dseed}/restart{R}"
        if key in out: continue
        torch.manual_seed(1000 * (dseed + 1) + R); np.random.seed(1000 * (dseed + 1) + R)
        learner, pqc, hist = pretrain_learner_su2lgt(NQ, DEPTH, train, epochs=40, lr=0.01, w0_scale=0.3, log_every=999)
        finals, inits = [], []
        for (m, x) in test:
            H = su2_lgt_hamiltonian(N, m, x)
            th0 = (0.3 * learner(torch.tensor([m, x], dtype=torch.float64))).detach()
            inits.append(float(pqc.cost(th0, H).item()))
            _, costs = adapt_task_su2lgt(pqc, th0, H, iters=ITERS, lr=LR)
            finals.append(float(costs[-1]))
        out[key] = {"train_loss_last": hist["loss"][-1], "train_loss_min": min(hist["loss"]),
                    "gradnorm_last": hist["grad_norm"][-1], "mean_init_E": float(np.mean(inits)),
                    "test_rel_err": rel_err(finals, E0, mabs)}
        save()
        print(key, {k: round(v, 4) for k, v in out[key].items()}, flush=True)
    # pi reference for this data seed
    th = classical_init((DEPTH, NQ, 3), "pi", DEPTH); fin = []
    for (m, x) in test:
        _, c = adapt_task_su2lgt(pqc, th, su2_lgt_hamiltonian(N, m, x), iters=ITERS, lr=LR); fin.append(float(c[-1]))
    out[f"dseed{dseed}/pi"] = rel_err(fin, E0, mabs); save()
print("RESTARTS DONE", flush=True)
