"""
Diagnose Track D's seed-unstable Q-MAML at N=4 (8 qubits, depth 6). For seeds 0,1,2 and two pretraining
settings (A: 40 epochs lr 0.01 = original; B: 100 epochs lr 0.002), record: the full pretraining history
(mean energy, gradient norm per epoch), and per held-out test task the exact E0, the energy at Q-MAML's
predicted init BEFORE adaptation, the energy after adaptation, and pi's for reference. Also compares the
two settings' ADAPTED parameters and energies task-by-task (same-basin test: if adapted energies agree to
many digits across very different pretraining, adaptation is landing in one basin regardless of init).
Also reports each task's contribution to the relative-error figure, to show whether one task dominates.
Resumable: skips seeds already saved.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from hamiltonians_su2lgt import su2_lgt_hamiltonian, exact_ground_energy, sample_task_space
from su2lgt_qmaml import classical_init, pretrain_learner_su2lgt, adapt_task_su2lgt

NQ, N, DEPTH, ITERS, LR = 8, 4, 6, 300, 0.05
OUT = os.path.join(os.path.dirname(__file__), "results", "su2_n4_diagnostics.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}

for seed in [0, 1, 2]:
    if str(seed) in out: continue
    torch.manual_seed(seed); np.random.seed(seed)
    train = sample_task_space(16, seed=seed)
    test = sample_task_space(6, seed=seed + 1000)
    E0 = [exact_ground_energy(su2_lgt_hamiltonian(N, m, x), NQ) for m, x in test]
    mean_abs = float(np.mean(np.abs(E0)))
    rec = {"E0": E0, "mean_abs_E0": mean_abs, "settings": {}}
    thetas = {}
    for tag, (ep, plr) in {"A_40ep_lr0.01": (40, 0.01), "B_100ep_lr0.002": (100, 0.002)}.items():
        torch.manual_seed(seed); np.random.seed(seed)
        learner, pqc, hist = pretrain_learner_su2lgt(NQ, DEPTH, train, epochs=ep, lr=plr, w0_scale=0.3, log_every=999)
        per = []
        thetas[tag] = []
        for (m, x), e0 in zip(test, E0):
            H = su2_lgt_hamiltonian(N, m, x)
            th0 = (0.3 * learner(torch.tensor([m, x], dtype=torch.float64))).detach()
            init_E = float(pqc.cost(th0, H).item())
            th, costs = adapt_task_su2lgt(pqc, th0, H, iters=ITERS, lr=LR)
            thetas[tag].append(th.numpy())
            per.append({"init_E": init_E, "final_E": float(costs[-1]), "abs_err": abs(costs[-1] - e0),
                        "err_over_mean_absE0": abs(costs[-1] - e0) / mean_abs})
        rec["settings"][tag] = {"pretrain_loss": hist["loss"], "pretrain_gradnorm": hist["grad_norm"], "per_task": per}
    # same-basin test
    ta, tb = thetas["A_40ep_lr0.01"], thetas["B_100ep_lr0.002"]
    rec["same_basin"] = [{
        "final_E_A": rec["settings"]["A_40ep_lr0.01"]["per_task"][i]["final_E"],
        "final_E_B": rec["settings"]["B_100ep_lr0.002"]["per_task"][i]["final_E"],
        "theta_dist": float(np.linalg.norm(ta[i] - tb[i]))} for i in range(len(test))]
    # pi reference
    pi_final = []
    for (m, x), e0 in zip(test, E0):
        H = su2_lgt_hamiltonian(N, m, x)
        th0 = classical_init((DEPTH, NQ, 3), "pi", DEPTH)
        _, costs = adapt_task_su2lgt(pqc, th0, H, iters=ITERS, lr=LR)
        pi_final.append({"final_E": float(costs[-1]), "err_over_mean_absE0": abs(costs[-1] - e0) / mean_abs})
    rec["pi"] = pi_final
    out[str(seed)] = rec
    json.dump(out, open(OUT, "w"))
    print(f"seed {seed} saved", flush=True)
print("DIAG DONE", flush=True)
