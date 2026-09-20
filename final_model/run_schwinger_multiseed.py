import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

from hamiltonians_schwinger import schwinger_hamiltonian, exact_ground_energy, sample_task_space, param_shape
from schwinger_qmaml import classical_init, pretrain_learner_schwinger, adapt_task_schwinger

NUM_QUBITS = 4
DEPTH = 3
N_TRAIN_TASKS = 16
N_TEST_TASKS = 6
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.005
ADAPT_ITERS = 300
ADAPT_LR = 0.02
INIT_SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "schwinger")

def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=seed)
    test_tasks = sample_task_space(N_TEST_TASKS, seed=seed + 1000)
    test_E0 = [exact_ground_energy(schwinger_hamiltonian(NUM_QUBITS, m0, g), NUM_QUBITS) for m0, g in test_tasks]
    learner, pqc, _ = pretrain_learner_schwinger(NUM_QUBITS, DEPTH, train_tasks, epochs=PRETRAIN_EPOCHS,
                                                  lr=PRETRAIN_LR, w0_scale=0.3, log_every=999)
    shape = param_shape(NUM_QUBITS, DEPTH)
    gaps = {k: [] for k in INIT_SCHEMES}
    for (m0, g), E0 in zip(test_tasks, test_E0):
        H = schwinger_hamiltonian(NUM_QUBITS, m0, g)
        for scheme in INIT_SCHEMES:
            theta0 = (0.3 * learner(torch.tensor([m0, g], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, DEPTH)
            _, costs = adapt_task_schwinger(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in INIT_SCHEMES:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = {k: float(np.exp(gaps[k][:, -1].mean()) / mean_abs_E0) for k in INIT_SCHEMES}
    print(f"seed={seed} final rel err: {final}")
    return final, gaps

if __name__ == "__main__":
    multiseed_final_err = {k: [] for k in INIT_SCHEMES}
    # seed 0 reuses the existing single-seed adaptation_gaps.json (Section 3 run)
    with open(os.path.join(RESULTS_DIR, "adaptation_gaps.json")) as f:
        existing = json.load(f)
    test_tasks0 = sample_task_space(N_TEST_TASKS, seed=0 + 1000)
    test_E0_0 = [exact_ground_energy(schwinger_hamiltonian(NUM_QUBITS, m0, g), NUM_QUBITS) for m0, g in test_tasks0]
    mean_absE0_0 = np.mean(np.abs(test_E0_0)) + 1e-8
    for k in INIT_SCHEMES:
        arr = np.array(existing[k])
        multiseed_final_err[k].append(float(np.exp(arr[:, -1].mean()) / mean_absE0_0))

    for seed in [1, 2]:
        final, _ = run_seed(seed)
        for k in INIT_SCHEMES:
            multiseed_final_err[k].append(final[k])

    with open(os.path.join(RESULTS_DIR, "multiseed_L2.json"), "w") as f:
        json.dump(multiseed_final_err, f)

    print("\nMulti-seed (n=3) final relative energy error, L=2, depth=3:")
    for k in INIT_SCHEMES:
        vals = np.array(multiseed_final_err[k])
        print(f"  {k:10s} mean={vals.mean():.4f}  std={vals.std():.4f}  per-seed={np.round(vals,4).tolist()}")
