"""
Track J future-work item: "the clear next step is a larger J (more qubits)." J=3.5 gives 2J+1=8 states
exactly (3 qubits, no Hilbert-space padding needed) -- the same reasoning the original J=1.5 choice used
(2J+1=4, "no padding needed"), so this is the next clean step up with zero new padding-correctness risk.
Verifies construction at J=3.5 first (Hermiticity, su(2) algebra, V=0 exact limit -- the same three checks
hamiltonians_lmg.verify_construction runs, parameterized here for the new J), then runs the standard
Q-MAML vs. classical-init protocol, single seed first.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import torch.nn as nn
from hamiltonians_lmg import lmg_hamiltonian, exact_ground_energy, sample_task_space, _collective_ops
from lmg_qmaml import classical_init, LMGPQC, adapt_task_lmg
from models.learner import Learner

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "lmg")
os.makedirs(RESULTS_DIR, exist_ok=True)

J_NEW = 3.5
NUM_QUBITS_NEW = 3  # 2*3.5+1 = 8 = 2^3, exact
DEPTH = 3
N_TRAIN_TASKS = 16
N_TEST_TASKS = 6
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.01
ADAPT_ITERS = 150
ADAPT_LR = 0.05
INIT_SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]


def verify_construction_at_j(J, atol=1e-8):
    ok = True
    H = lmg_hamiltonian(epsilon=1.0, V=0.6, J=J)
    assert H.shape == (2 ** NUM_QUBITS_NEW, 2 ** NUM_QUBITS_NEW), f"dim mismatch: {H.shape}"
    herm_err = np.max(np.abs(H - H.conj().T))
    print(f"  Hermiticity (J={J}): max|H-H^dag|={herm_err:.2e}  {'OK' if herm_err < atol else 'FAIL'}")
    ok = ok and herm_err < atol

    Jz, Jp, Jm = _collective_ops(J)
    c1_err = np.max(np.abs(Jz @ Jp - Jp @ Jz - Jp))
    c2_err = np.max(np.abs(Jp @ Jm - Jm @ Jp - 2 * Jz))
    algebra_ok = c1_err < atol and c2_err < atol
    print(f"  su(2) algebra: err1={c1_err:.2e} err2={c2_err:.2e}  {'OK' if algebra_ok else 'FAIL'}")
    ok = ok and algebra_ok

    H_v0 = lmg_hamiltonian(epsilon=1.3, V=0.0, J=J)
    E0_numeric = exact_ground_energy(H_v0)
    E0_analytic = -1.3 * J
    v0_ok = abs(E0_numeric - E0_analytic) < atol
    print(f"  V=0 limit: numeric={E0_numeric:.6f} analytic={E0_analytic:.6f}  {'OK' if v0_ok else 'FAIL'}")
    ok = ok and v0_ok
    return ok


def pretrain_learner_lmg_at_j(num_qubits, depth, train_tasks, epochs, lr, J, w0_scale=0.3):
    """pretrain_learner_lmg, but with J threaded through to lmg_hamiltonian (the shared function
    hardcodes the module's default J_SPIN, which silently mismatches any non-default qubit count)."""
    shape = (depth, num_qubits, 3)
    learner = Learner(2, shape, hidden=256).double()
    pqc = LMGPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    for epoch in range(epochs):
        for epsilon, V in train_tasks:
            H = lmg_hamiltonian(epsilon, V, J=J)
            opt.zero_grad()
            theta = w0_scale * learner(torch.tensor([epsilon, V], dtype=torch.float64))
            cost = pqc.cost(theta, H)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
            opt.step()
    return learner, pqc


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=seed)
    test_tasks = sample_task_space(N_TEST_TASKS, seed=seed + 1000)
    test_E0 = [exact_ground_energy(lmg_hamiltonian(eps, V, J=J_NEW)) for eps, V in test_tasks]
    mean_absE0 = np.mean(np.abs(test_E0))

    learner, pqc = pretrain_learner_lmg_at_j(NUM_QUBITS_NEW, DEPTH, train_tasks, epochs=PRETRAIN_EPOCHS,
                                              lr=PRETRAIN_LR, J=J_NEW, w0_scale=0.3)
    shape = (DEPTH, NUM_QUBITS_NEW, 3)
    gaps = {k: [] for k in INIT_SCHEMES}
    for (eps, V), E0 in zip(test_tasks, test_E0):
        H = lmg_hamiltonian(eps, V, J=J_NEW)
        for scheme in INIT_SCHEMES:
            theta0 = (0.3 * learner(torch.tensor([eps, V], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, DEPTH)
            _, costs = adapt_task_lmg(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    final = {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mean_absE0) for k, v in gaps.items()}
    print(f"[lmg J={J_NEW} seed={seed}] final rel err: {final}")
    return final


if __name__ == "__main__":
    ok = verify_construction_at_j(J_NEW)
    print("construction check:", "PASSED" if ok else "FAILED")
    assert ok, "Do not proceed -- construction failed verification at the new J"

    results = {}
    for seed in [0, 1, 2]:
        results[seed] = run_seed(seed)
    schemes = list(results[0].keys())
    summary = {k: [results[s][k] for s in [0, 1, 2]] for k in schemes}
    with open(os.path.join(RESULTS_DIR, f"larger_J_{J_NEW}_multiseed.json"), "w") as f:
        json.dump(summary, f)

    print(f"\n=== LMG J={J_NEW} multi-seed (n=3) summary ===")
    for k in schemes:
        vals = np.array(summary[k])
        print(f"  {k:10s} mean={vals.mean():.4f}  std={vals.std():.4f}  per-seed={np.round(vals,4).tolist()}")
