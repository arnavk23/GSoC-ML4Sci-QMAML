"""
Limitations future-work item: "Barren-plateau diagnostic scope: single reference-parameter gradient per
sample (standard McClean protocol), not a full Fisher-information/effective-dimension analysis."

This adds a second, independent diagnostic on top of research_notebooks/03_Barren_Plateau_Scaling.ipynb's
single-reference-gradient variance scan: the quantum Fisher information matrix (QFIM), via PennyLane's
`qml.metric_tensor` (the Fubini-Study metric tensor, proportional to the QFIM by a factor of 4), evaluated
at the full parameter vector rather than one reference coordinate. Effective dimension is summarized as the
participation ratio PR = (sum(lambda))^2 / sum(lambda^2) of the QFIM's eigenvalue spectrum, normalized by
the number of parameters -- a standard single-number summary of "how many directions in parameter space
the circuit can actually distinguish" (Abbas et al. 2021's effective-dimension framing), independent of and
complementary to a single-coordinate gradient-variance estimate: a circuit can have near-zero variance at
one reference parameter while still having a non-degenerate QFIM overall (or vice versa), which the
existing single-reference-gradient scan cannot distinguish.

Uses the same Heisenberg XYZ brick ansatz (IsingXX/YY/ZZ layers) as the existing VQE-domain barren-plateau
notebook, at a reduced qubit grid ([4, 6, 8]) for tractability (metric_tensor's cost scales with the number
of circuit evaluations needed per parameter, materially more expensive per sample than a single gradient
component).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp

from vqe_heisenberg import sample_task_space
from vqe_qmaml import classical_init, pretrain_learner_vqe

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "barren_plateau")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUBIT_GRID = [4, 6, 8]
DEPTH = 3
N_SAMPLES = 8       # QFIM samples per (qubit count, scheme) -- expensive, kept small
PRETRAIN_EPOCHS = 15
PRETRAIN_TASKS = 16
SCHEMES = ["qmaml", "uniform", "gaussian"]
SEED = 0


def make_ansatz_qnode(num_qubits, depth):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        for l in range(depth):
            for n in range(num_qubits - 1):
                qml.IsingXX(weights[l, n, 0], wires=[n, n + 1])
                qml.IsingYY(weights[l, n, 1], wires=[n, n + 1])
                qml.IsingZZ(weights[l, n, 2], wires=[n, n + 1])
        return qml.expval(qml.PauliZ(0))
    return circuit


def participation_ratio(eigvals, atol=1e-10):
    lam = np.clip(eigvals, 0, None)
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    if s2 < atol:
        return 0.0
    return float((s1 ** 2) / s2)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    results = {(nq, s): [] for nq in QUBIT_GRID for s in SCHEMES}

    for nq in QUBIT_GRID:
        print(f"\n=== qubits={nq} ===")
        circuit = make_ansatz_qnode(nq, DEPTH)
        shape = (DEPTH, nq - 1, 3)
        n_params = DEPTH * (nq - 1) * 3

        torch.manual_seed(SEED); np.random.seed(SEED)
        train_tasks = sample_task_space(PRETRAIN_TASKS, seed=SEED)
        t0 = time.time()
        learner, _, _ = pretrain_learner_vqe(nq, DEPTH, train_tasks, epochs=PRETRAIN_EPOCHS, lr=0.01)
        print(f"  pretrain {time.time()-t0:.1f}s")

        for scheme in SCHEMES:
            prs = []
            for s in range(N_SAMPLES):
                if scheme == "qmaml":
                    Jx, Jy, Jz = rng.uniform(0.2, 2.0, size=3)
                    with torch.no_grad():
                        theta = learner(torch.tensor([Jx, Jy, Jz], dtype=torch.float64)).numpy().reshape(shape)
                else:
                    theta = classical_init(shape, scheme, DEPTH).numpy()
                theta = pnp.array(theta, requires_grad=True)
                # block-diagonal approximation: avoids needing an ancilla wire for the Hadamard test,
                # the standard choice from Stokes et al. 2020's quantum natural gradient formulation.
                mt = qml.metric_tensor(circuit, approx="block-diag")(theta)
                mt_flat = np.array(mt).reshape(n_params, n_params)
                eigvals = np.linalg.eigvalsh(mt_flat)
                pr = participation_ratio(eigvals) / n_params  # normalize by max possible rank
                prs.append(pr)
            results[(nq, scheme)] = prs
            print(f"  {scheme:8s} effective-dim (participation ratio / n_params): "
                  f"mean={np.mean(prs):.4f} std={np.std(prs):.4f}  n_params={n_params}")

    out = {f"{nq}_{s}": v for (nq, s), v in results.items()}
    with open(os.path.join(RESULTS_DIR, "fisher_info_effective_dim.json"), "w") as f:
        json.dump(out, f)
    print("\nsaved -> fisher_info_effective_dim.json")
