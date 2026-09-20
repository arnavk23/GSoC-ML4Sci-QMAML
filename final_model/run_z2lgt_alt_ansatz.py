"""
Executes research_notebooks/08_Z2_LGT_QMAML.ipynb Section 8 (cells 19-24), written but never run: the
gauge-invariant Hamiltonian-variational ansatz (Z2LGTHVAPQC) and hardware-efficient-ZZ ansatz
(Z2LGTHWEffZZPQC) comparison flagged as "the most promising untried follow-up" to Track B's loss. Mirrors
the notebook's own protocol exactly (same hyperparameters, same reduced N_TEST_TASKS_ALT=4/ADAPT_ITERS_ALT=
200 budget, documented there for tractability), reusing the already-pretrained Section 2-3 learner/pqc only
implicitly (via the same SEED/train_tasks/test_tasks) since this script pretrains its own HVA/HWEffZZ
Learners (a fixed-shape default ansatz's Learner cannot be reused for a different ansatz's parameter shape).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from hamiltonians_z2lgt import z2_lgt_hamiltonian, exact_ground_energy, sample_task_space, verify_gauge_invariance, NUM_QUBITS
from z2lgt_qmaml import (classical_init, pretrain_learner_z2lgt, adapt_task_z2lgt,
                          Z2LGTHVAPQC, Z2LGTHWEffZZPQC, pretrain_learner_z2lgt_ansatz)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "z2_lgt")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEPTH = 3
N_TRAIN_TASKS = 16
N_TEST_TASKS = 6
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.005
ADAPT_LR = 0.02
GAUSS_V = 10.0
SEED = 0

N_TEST_TASKS_ALT = 4
ADAPT_ITERS_ALT = 200
INIT_SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]

if __name__ == "__main__":
    ok = verify_gauge_invariance(J=0.8, m=1.2, mu=0.6, V=10.0)
    print("gauge invariance check:", "PASSED" if ok else "FAILED")
    assert ok, "Do not proceed -- Hamiltonian is not gauge invariant"

    torch.manual_seed(SEED); np.random.seed(SEED)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=SEED)
    test_tasks = sample_task_space(N_TEST_TASKS, seed=SEED + 1000)
    test_tasks_alt = test_tasks[:N_TEST_TASKS_ALT]
    test_E0_alt = [exact_ground_energy(z2_lgt_hamiltonian(J, m, mu, V=GAUSS_V)) for J, m, mu in test_tasks_alt]
    mean_absE0_alt = np.mean(np.abs(test_E0_alt))

    ansatz_results = {}
    for tag, PQCClass in [("gauge_invariant_hva", Z2LGTHVAPQC), ("hardware_efficient_zz", Z2LGTHWEffZZPQC)]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        pqc = PQCClass(NUM_QUBITS, DEPTH)
        t0 = time.time()
        learner, hist = pretrain_learner_z2lgt_ansatz(
            pqc, task_dim=3, train_tasks=train_tasks, epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
            w0_scale=0.3, log_every=10, gauss_V=GAUSS_V, tag=tag,
        )
        print(f"[{tag}] pretrain done in {time.time()-t0:.1f}s  final gradnorm={hist['grad_norm'][-1]:.3f}")

        gaps = {k: [] for k in INIT_SCHEMES}
        for (J, m, mu), E0 in zip(test_tasks_alt, test_E0_alt):
            H = z2_lgt_hamiltonian(J, m, mu, V=GAUSS_V)
            for scheme in INIT_SCHEMES:
                if scheme == "qmaml":
                    with torch.no_grad():
                        theta0 = 0.3 * learner(torch.tensor([J, m, mu], dtype=torch.float64))
                else:
                    theta0 = classical_init(pqc.shape, scheme, DEPTH)
                _, costs = adapt_task_z2lgt(pqc, theta0, H, iters=ADAPT_ITERS_ALT, lr=ADAPT_LR)
                gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
        for k in INIT_SCHEMES:
            gaps[k] = np.stack(gaps[k])
        ansatz_results[tag] = gaps

        print(f"  Final relative energy error, {tag}:")
        for k in INIT_SCHEMES:
            final_gap = gaps[k][:, -1].mean()
            rel_err = np.exp(final_gap) / mean_absE0_alt
            print(f"    {k:10s} log-gap={final_gap:.3f}   ~relative energy error={rel_err:.2%}")

    with open(os.path.join(RESULTS_DIR, "alt_ansatz_comparison.json"), "w") as f:
        json.dump({name: {k: g.tolist() for k, g in gaps.items()} for name, gaps in ansatz_results.items()}, f)
    print("saved -> alt_ansatz_comparison.json")
