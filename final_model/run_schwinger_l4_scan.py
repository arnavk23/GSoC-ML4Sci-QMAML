"""
Track A future-work item: "A joint depth x pretrain-budget x qubit-count scan is needed" to resolve the
L=4 (8-qubit) failure, where the original run (PRETRAIN_EPOCHS=40, PRETRAIN_LR=0.005, depth=12) showed
qmaml losing to pi (31.81% vs 8.56%) with pretraining itself not converging (gradient norm flat, mean
energy stalled far from true ground energies). This is a pretrain_epochs x pretrain_lr grid at depth=12
(the L=3-derived heuristic depth), plus one larger-depth spot-check at the best (epochs, lr) found -- a
bounded diagnostic slice of the full 3D scan, not the complete thing, but direct evidence on whether
pretrain budget alone (holding depth fixed) can recover the L=3 pattern at L=4.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from hamiltonians_schwinger import schwinger_hamiltonian, exact_ground_energy, sample_task_space, param_shape
from schwinger_qmaml import classical_init, pretrain_learner_schwinger, adapt_task_schwinger

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "schwinger")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_QUBITS_L4 = 8
N_TRAIN_TASKS = 16
N_TEST_TASKS = 3
ADAPT_ITERS = 150
ADAPT_LR = 0.02
SEED = 0
INIT_SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]


def run_config(depth, pretrain_epochs, pretrain_lr):
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=SEED)
    test_tasks = sample_task_space(N_TEST_TASKS, seed=SEED + 1000)
    test_E0 = [exact_ground_energy(schwinger_hamiltonian(NUM_QUBITS_L4, m0, g), NUM_QUBITS_L4) for m0, g in test_tasks]
    mean_absE0 = np.mean(np.abs(test_E0))

    t0 = time.time()
    learner, pqc, hist = pretrain_learner_schwinger(NUM_QUBITS_L4, depth, train_tasks, epochs=pretrain_epochs,
                                                      lr=pretrain_lr, w0_scale=0.3, log_every=999)
    pretrain_time = time.time() - t0

    shape = param_shape(NUM_QUBITS_L4, depth)
    gaps = {k: [] for k in INIT_SCHEMES}
    for (m0, g), E0 in zip(test_tasks, test_E0):
        H = schwinger_hamiltonian(NUM_QUBITS_L4, m0, g)
        for scheme in INIT_SCHEMES:
            theta0 = (0.3 * learner(torch.tensor([m0, g], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_schwinger(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    final = {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mean_absE0) for k, v in gaps.items()}
    print(f"[depth={depth} epochs={pretrain_epochs} lr={pretrain_lr}] pretrain {pretrain_time:.1f}s "
          f"final_gradnorm={hist['grad_norm'][-1]:.4f} final_meanE={hist['loss'][-1]:.4f}  rel_err={final}")
    return final, hist


if __name__ == "__main__":
    grid_results = {}
    # Step 1: pretrain_epochs x pretrain_lr grid at depth=12 (the L=3-derived heuristic depth)
    for epochs, lr in [(40, 0.005), (100, 0.005), (100, 0.002)]:
        if True:
            key = f"depth12_epochs{epochs}_lr{lr}"
            final, hist = run_config(depth=12, pretrain_epochs=epochs, pretrain_lr=lr)
            grid_results[key] = {"final": final, "grad_norm_last": hist["grad_norm"][-1], "loss_last": hist["loss"][-1]}
            with open(os.path.join(RESULTS_DIR, "l4_depth_pretrain_scan_partial.json"), "w") as f:
                json.dump(grid_results, f)

    # Step 2: pick the config with the lowest final qmaml relative error, then also try depth=18 there
    best_key = min(grid_results, key=lambda k: grid_results[k]["final"]["qmaml"])
    best_epochs = int(best_key.split("epochs")[1].split("_lr")[0])
    best_lr = float(best_key.split("_lr")[1])
    print(f"\nBest depth=12 config: {best_key}")
    final_d18, hist_d18 = run_config(depth=18, pretrain_epochs=best_epochs, pretrain_lr=best_lr)
    grid_results[f"depth18_epochs{best_epochs}_lr{best_lr}"] = {
        "final": final_d18, "grad_norm_last": hist_d18["grad_norm"][-1], "loss_last": hist_d18["loss"][-1]}

    with open(os.path.join(RESULTS_DIR, "l4_depth_pretrain_scan.json"), "w") as f:
        json.dump(grid_results, f)

    print("\n=== L=4 depth x pretrain-budget scan summary ===")
    for key, r in grid_results.items():
        print(f"  {key}: qmaml={r['final']['qmaml']:.2%} pi={r['final']['pi']:.2%} "
              f"gradnorm_final={r['grad_norm_last']:.4f}")
