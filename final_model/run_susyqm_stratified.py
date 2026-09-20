"""
Track F future-work item: "9 held-out tasks split unevenly across three superpotentials by random draw
(2-4 each), not a balanced/stratified split." Reruns the identical single-seed protocol from
research_notebooks/11_SUSY_QM_QMAML.ipynb with a stratified (exactly 3 HO / 3 AHO / 3 DW) held-out test
set instead of the original random draw, everything else (train tasks, hyperparameters, seed) unchanged.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from hamiltonians_susyqm import susyqm_hamiltonian, exact_ground_energy, sample_task_space, SUPERPOTENTIALS
from susyqm_qmaml import classical_init, pretrain_learner_susyqm, adapt_task_susyqm, encode_task, NUM_QUBITS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "susyqm")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEPTH = 3
N_TRAIN_TASKS = 18
N_TEST_PER_SP = 3
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.01
ADAPT_ITERS = 200
ADAPT_LR = 0.05
SEED = 0
INIT_SCHEMES = ["qmaml", "basis_paper", "zero", "pi", "uniform", "gaussian"]


def stratified_test_tasks(n_per_sp, seed, m_range=(0.5, 1.5), g_range=(0.5, 1.5), mu_range=(0.5, 1.5)):
    rng = np.random.default_rng(seed)
    tasks = []
    for sp in SUPERPOTENTIALS:
        m = rng.uniform(*m_range, size=n_per_sp)
        g = rng.uniform(*g_range, size=n_per_sp)
        mu = rng.uniform(*mu_range, size=n_per_sp)
        tasks += [(sp, float(m[k]), float(g[k]), float(mu[k])) for k in range(n_per_sp)]
    rng.shuffle(tasks)
    return tasks


def fermion_occupied_for(sp):
    return sp in ("HO", "AHO")


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=SEED)
    test_tasks = stratified_test_tasks(N_TEST_PER_SP, seed=SEED + 1000)
    test_E0 = [exact_ground_energy(susyqm_hamiltonian(sp, m=m, g=g, mu=mu)) for sp, m, g, mu in test_tasks]
    print("stratified test set:", [t[0] for t in test_tasks])

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_susyqm(NUM_QUBITS, DEPTH, train_tasks, epochs=PRETRAIN_EPOCHS,
                                               lr=PRETRAIN_LR, w0_scale=0.3, log_every=10)
    print(f"pretrain done in {time.time()-t0:.1f}s")

    shape = (DEPTH, NUM_QUBITS, 3)
    gaps = {k: [] for k in INIT_SCHEMES}
    for (sp, m, g, mu), E0 in zip(test_tasks, test_E0):
        H = susyqm_hamiltonian(sp, m=m, g=g, mu=mu)
        for scheme in INIT_SCHEMES:
            if scheme == "qmaml":
                theta0 = (0.3 * learner(encode_task(sp, m, g, mu))).detach()
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            elif scheme == "basis_paper":
                theta0 = classical_init(shape, "basis_paper", DEPTH)
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR,
                                              fermion_occupied=fermion_occupied_for(sp))
            else:
                theta0 = classical_init(shape, scheme, DEPTH)
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in INIT_SCHEMES:
        gaps[k] = np.stack(gaps[k])

    print("Final log-gap, stratified test set (3 HO / 3 AHO / 3 DW):")
    result = {}
    for k in INIT_SCHEMES:
        final_gap = float(gaps[k][:, -1].mean())
        result[k] = final_gap
        print(f"  {k:12s} log-gap={final_gap:.3f}")

    by_sp = {sp: {k: [] for k in INIT_SCHEMES} for sp in SUPERPOTENTIALS}
    for ti, (sp, m, g, mu) in enumerate(test_tasks):
        for k in INIT_SCHEMES:
            by_sp[sp][k].append(float(gaps[k][ti, -1]))
    print("Per-superpotential (n=3 each):")
    for sp in SUPERPOTENTIALS:
        row = "  ".join(f"{k}={np.mean(by_sp[sp][k]):.2f}" for k in INIT_SCHEMES)
        print(f"  {sp}: {row}")

    with open(os.path.join(RESULTS_DIR, "stratified_test_split.json"), "w") as f:
        json.dump({"final_log_gap": result, "by_superpotential": by_sp}, f)
    print("saved -> stratified_test_split.json")
