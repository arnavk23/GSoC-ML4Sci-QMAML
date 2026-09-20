"""
Corrected re-run of research_notebooks/09_Scalar_Field_QMAML.ipynb cell 16 (phase-stratified multi-seed).
The executed cell has a real aggregation bug: seed 0's value is one gap averaged over all 6 held-out tasks
(matching Section 5's methodology), but seeds 1 and 2 each append 6 *individual per-task* relative errors
instead of one seed-level average -- so the printed "mean +/- std over 3 seeds" (n=13 samples: 1 + 6 + 6)
silently mixes one aggregate with twelve raw per-task values, understating how seed-to-seed variation
actually looks. This script reruns seeds 0/1/2 with per-seed aggregation applied uniformly (mean of the
per-task log-gap across all 6 held-out tasks, converted to relative error once, exactly like Section 5) for
both extrapolation directions.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from hamiltonians_scalar import scalar_field_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
from scalar_qmaml import classical_init, pretrain_learner_scalar, adapt_task_scalar

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "scalar_field")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEPTH = 3
N_TRAIN_TASKS = 16
N_TEST_TASKS = 6
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.005
ADAPT_ITERS = 300
ADAPT_LR = 0.02
INIT_SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]
SYMMETRIC_RANGE = (0.3, 1.0)
BROKEN_RANGE = (-2.5, -1.7)
LAMBDA_RANGE = (0.5, 1.5)


def run_direction(seed, train_range, test_range):
    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(N_TRAIN_TASKS, seed=seed, m0_sq_range=train_range, lambda0_range=LAMBDA_RANGE)
    test_tasks = sample_task_space(N_TEST_TASKS, seed=seed + 1000, m0_sq_range=test_range, lambda0_range=LAMBDA_RANGE)
    test_E0 = [exact_ground_energy(scalar_field_hamiltonian(m0_sq, lambda0)) for m0_sq, lambda0 in test_tasks]
    mean_absE0 = np.mean(np.abs(test_E0))

    learner, pqc, _ = pretrain_learner_scalar(NUM_QUBITS, DEPTH, train_tasks, epochs=PRETRAIN_EPOCHS,
                                               lr=PRETRAIN_LR, w0_scale=0.3, log_every=999)
    shape = (DEPTH, NUM_QUBITS, 3)
    gaps = {k: [] for k in INIT_SCHEMES}
    for (m0_sq, lambda0), E0 in zip(test_tasks, test_E0):
        H = scalar_field_hamiltonian(m0_sq, lambda0)
        for scheme in INIT_SCHEMES:
            theta0 = (0.3 * learner(torch.tensor([m0_sq, lambda0], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, DEPTH)
            _, costs = adapt_task_scalar(pqc, theta0, H, iters=ADAPT_ITERS, lr=ADAPT_LR)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    return {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mean_absE0) for k, v in gaps.items()}


if __name__ == "__main__":
    result = {"symmetric_to_broken": {k: [] for k in INIT_SCHEMES},
              "broken_to_symmetric": {k: [] for k in INIT_SCHEMES}}
    for seed in [0, 1, 2]:
        for direction, (train_range, test_range) in [
            ("symmetric_to_broken", (SYMMETRIC_RANGE, BROKEN_RANGE)),
            ("broken_to_symmetric", (BROKEN_RANGE, SYMMETRIC_RANGE)),
        ]:
            r = run_direction(seed, train_range, test_range)
            print(f"seed={seed} [{direction}]: {r}")
            for k in INIT_SCHEMES:
                result[direction][k].append(r[k])

    with open(os.path.join(RESULTS_DIR, "phase_stratified_multiseed_FIXED.json"), "w") as f:
        json.dump(result, f)

    print("\nCorrected phase-stratified final relative energy error, mean +/- std over 3 seeds:")
    for direction in result:
        print(f"  [{direction}]")
        for k in INIT_SCHEMES:
            vals = np.array(result[direction][k])
            print(f"    {k:10s} {vals.mean():.2%} +/- {vals.std():.2%}  per-seed={[f'{100*v:.2f}%' for v in vals]}")
