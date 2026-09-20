"""
Multi-seed replication (and, for two tracks, a larger-qubit-count extension) for the seven Hamiltonian
tracks (D, E, F, G, H, I, J) that the paper's own Limitations section (Table 9's "single seed except where
noted" caveat) flags as not yet seed-replicated, unlike Tracks A/B/C. Mirrors the exact protocol already
used in research_notebooks/07_Schwinger_Model_QMAML.ipynb's Section 8 (same hyperparameters, same train/test
task-sampling convention, same classical_init/pretrain/adapt functions from each track's own <name>_qmaml.py
module) -- no new modeling choices, just more seeds and, for D/E, one additional verified qubit count.

Usage:
    python multiseed_validate.py --track su2lgt
    python multiseed_validate.py --track all
"""
import argparse
import json
import os
import time

import numpy as np
import torch

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")


def _save(track_dir, name, obj):
    d = os.path.join(RESULTS_ROOT, track_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"  saved -> {path}")


def _rel_err(final_gaps, mean_abs_E0):
    return {k: float(np.exp(np.mean(v)) / mean_abs_E0) for k, v in final_gaps.items()}


# ---------------------------------------------------------------------------
# Track D: SU(2) lattice gauge theory
# ---------------------------------------------------------------------------
def run_su2lgt(seed, num_qubits=4, depth=4, adapt_iters=250, adapt_lr=0.05,
               pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6, verify_first=True):
    from hamiltonians_su2lgt import su2_lgt_hamiltonian, exact_ground_energy, sample_task_space, verify_construction
    from su2lgt_qmaml import classical_init, pretrain_learner_su2lgt, adapt_task_su2lgt

    N = num_qubits // 2
    if verify_first:
        assert verify_construction(N=N), f"SU(2) LGT construction failed at N={N}"

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(su2_lgt_hamiltonian(N, m, x), num_qubits) for m, x in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_su2lgt(num_qubits, depth, train_tasks, epochs=pretrain_epochs,
                                               lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [su2lgt seed={seed} N={N}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, num_qubits, 3)
    gaps = {k: [] for k in schemes}
    for (m, x), E0 in zip(test_tasks, test_E0):
        H = su2_lgt_hamiltonian(N, m, x)
        for scheme in schemes:
            theta0 = (0.3 * learner(torch.tensor([m, x], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_su2lgt(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [su2lgt seed={seed} N={N}] final rel err: {final}")
    return final


# ---------------------------------------------------------------------------
# Track E: collective neutrino oscillations
# ---------------------------------------------------------------------------
def run_neutrino(seed, num_qubits=4, depth=3, adapt_iters=200, adapt_lr=0.05,
                  pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6, verify_first=True):
    from hamiltonians_neutrino import neutrino_hamiltonian, exact_ground_energy, sample_task_space, verify_construction
    from neutrino_qmaml import classical_init, pretrain_learner_neutrino, adapt_task_neutrino

    if verify_first:
        assert verify_construction(N=num_qubits), f"Neutrino construction failed at N={num_qubits}"

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(neutrino_hamiltonian(num_qubits, t, mu), num_qubits) for t, mu in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_neutrino(num_qubits, depth, train_tasks, epochs=pretrain_epochs,
                                                 lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [neutrino seed={seed} N={num_qubits}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, num_qubits, 3)
    gaps = {k: [] for k in schemes}
    for (t_mix, mu), E0 in zip(test_tasks, test_E0):
        H = neutrino_hamiltonian(num_qubits, t_mix, mu)
        for scheme in schemes:
            theta0 = (0.3 * learner(torch.tensor([t_mix, mu], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_neutrino(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [neutrino seed={seed} N={num_qubits}] final rel err: {final}")
    return final


# ---------------------------------------------------------------------------
# Track F: SUSY QM
# ---------------------------------------------------------------------------
def run_susyqm(seed, depth=3, adapt_iters=200, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01,
               n_train=18, n_test=9):
    from hamiltonians_susyqm import susyqm_hamiltonian, exact_ground_energy, sample_task_space
    from susyqm_qmaml import classical_init, pretrain_learner_susyqm, adapt_task_susyqm, encode_task, NUM_QUBITS

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(susyqm_hamiltonian(sp, m=m, g=g, mu=mu)) for sp, m, g, mu in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_susyqm(NUM_QUBITS, depth, train_tasks, epochs=pretrain_epochs,
                                               lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [susyqm seed={seed}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "basis_paper", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, NUM_QUBITS, 3)

    def fermion_occupied_for(sp):
        return sp in ("HO", "AHO")

    gaps = {k: [] for k in schemes}
    for (sp, m, g, mu), E0 in zip(test_tasks, test_E0):
        H = susyqm_hamiltonian(sp, m=m, g=g, mu=mu)
        for scheme in schemes:
            if scheme == "qmaml":
                theta0 = (0.3 * learner(encode_task(sp, m, g, mu))).detach()
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            elif scheme == "basis_paper":
                theta0 = classical_init(shape, "basis_paper", depth)
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr,
                                              fermion_occupied=fermion_occupied_for(sp))
            else:
                theta0 = classical_init(shape, scheme, depth)
                _, costs = adapt_task_susyqm(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    final_log_gap = {k: float(np.mean(gaps[k][:, -1])) for k in schemes}
    print(f"  [susyqm seed={seed}] final log-gap: {final_log_gap}")
    return final_log_gap  # log-gap is the metric that matters here (E0~0 for HO tasks)


# ---------------------------------------------------------------------------
# Track G: non-Abelian D8 lattice gauge theory
# ---------------------------------------------------------------------------
def run_d8lgt(seed, depth=3, adapt_iters=200, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01,
              n_train=16, n_test=6):
    from hamiltonians_d8lgt import d8_lgt_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
    from d8lgt_qmaml import classical_init, pretrain_learner_d8lgt, adapt_task_d8lgt

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(d8_lgt_hamiltonian(M, lam, eps)) for M, lam, eps in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_d8lgt(NUM_QUBITS, depth, train_tasks, epochs=pretrain_epochs,
                                              lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [d8lgt seed={seed}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, NUM_QUBITS, 3)
    gaps = {k: [] for k in schemes}
    for (M, lam, eps), E0 in zip(test_tasks, test_E0):
        H = d8_lgt_hamiltonian(M, lam, eps)
        for scheme in schemes:
            theta0 = (0.3 * learner(torch.tensor([M, lam, eps], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_d8lgt(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [d8lgt seed={seed}] final rel err: {final}")
    return final


# ---------------------------------------------------------------------------
# Track H: light nuclei (UCCSD)
# ---------------------------------------------------------------------------
def run_nuclear(seed, depth=2, adapt_iters=150, adapt_lr=0.1, pretrain_epochs=30, pretrain_lr=0.05,
                 n_train=18, n_test=9):
    from hamiltonians_nuclear import nuclear_hamiltonian, exact_ground_energy, sample_task_space
    from nuclear_qmaml import classical_init, pretrain_learner_nuclear, adapt_task_nuclear, encode_task

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(nuclear_hamiltonian(nuc, t, C, D), nuc) for nuc, t, C, D in test_tasks]

    t0 = time.time()
    learners, pqcs, _ = pretrain_learner_nuclear(depth, train_tasks, epochs=pretrain_epochs, lr=pretrain_lr,
                                                  w0_scale=0.3, log_every=999)
    print(f"  [nuclear seed={seed}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    gaps = {k: [] for k in schemes}
    for (nuc, t, C, D), E0 in zip(test_tasks, test_E0):
        H = nuclear_hamiltonian(nuc, t, C, D)
        pqc = pqcs[nuc]
        shape = (depth, pqc.n_excitations)
        for scheme in schemes:
            theta0 = (0.3 * learners[nuc](encode_task(nuc, t, C, D))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_nuclear(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [nuclear seed={seed}] final rel err: {final}")
    return final


# ---------------------------------------------------------------------------
# Track I: scalar Yukawa coupling
# ---------------------------------------------------------------------------
def run_yukawa(seed, depth=3, adapt_iters=150, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01,
               n_train=16, n_test=6):
    from hamiltonians_yukawa import yukawa_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
    from yukawa_qmaml import classical_init, pretrain_learner_yukawa, adapt_task_yukawa

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(yukawa_hamiltonian(eta, m)) for eta, m in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_yukawa(NUM_QUBITS, depth, train_tasks, epochs=pretrain_epochs,
                                               lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [yukawa seed={seed}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, NUM_QUBITS, 3)
    gaps = {k: [] for k in schemes}
    for (eta, m), E0 in zip(test_tasks, test_E0):
        H = yukawa_hamiltonian(eta, m)
        for scheme in schemes:
            theta0 = (0.3 * learner(torch.tensor([eta, m], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_yukawa(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [yukawa seed={seed}] final rel err: {final}")
    return final


# ---------------------------------------------------------------------------
# Track J: Lipkin-Meshkov-Glick model
# ---------------------------------------------------------------------------
def run_lmg(seed, depth=3, adapt_iters=150, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01,
            n_train=16, n_test=6):
    from hamiltonians_lmg import lmg_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
    from lmg_qmaml import classical_init, pretrain_learner_lmg, adapt_task_lmg

    torch.manual_seed(seed); np.random.seed(seed)
    train_tasks = sample_task_space(n_train, seed=seed)
    test_tasks = sample_task_space(n_test, seed=seed + 1000)
    test_E0 = [exact_ground_energy(lmg_hamiltonian(eps, V)) for eps, V in test_tasks]

    t0 = time.time()
    learner, pqc, _ = pretrain_learner_lmg(NUM_QUBITS, depth, train_tasks, epochs=pretrain_epochs,
                                            lr=pretrain_lr, w0_scale=0.3, log_every=999)
    print(f"  [lmg seed={seed}] pretrain {time.time()-t0:.1f}s")

    schemes = ["qmaml", "zero", "pi", "uniform", "gaussian"]
    shape = (depth, NUM_QUBITS, 3)
    gaps = {k: [] for k in schemes}
    for (eps, V), E0 in zip(test_tasks, test_E0):
        H = lmg_hamiltonian(eps, V)
        for scheme in schemes:
            theta0 = (0.3 * learner(torch.tensor([eps, V], dtype=torch.float64))).detach() if scheme == "qmaml" \
                else classical_init(shape, scheme, depth)
            _, costs = adapt_task_lmg(pqc, theta0, H, iters=adapt_iters, lr=adapt_lr)
            gaps[scheme].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
    for k in schemes:
        gaps[k] = np.stack(gaps[k])
    mean_abs_E0 = np.mean(np.abs(test_E0)) + 1e-8
    final = _rel_err({k: gaps[k][:, -1] for k in schemes}, mean_abs_E0)
    print(f"  [lmg seed={seed}] final rel err: {final}")
    return final


TRACKS = {
    "su2lgt": (run_su2lgt, "su2_lgt"),
    "neutrino": (run_neutrino, "neutrino"),
    "susyqm": (run_susyqm, "susyqm"),
    "d8lgt": (run_d8lgt, "d8_lgt"),
    "nuclear": (run_nuclear, "nuclear"),
    "yukawa": (run_yukawa, "yukawa"),
    "lmg": (run_lmg, "lmg"),
}


def multiseed(track_name, seeds=(0, 1, 2)):
    fn, dirname = TRACKS[track_name]
    results = {}
    for s in seeds:
        results[s] = fn(seed=s)
    schemes = list(next(iter(results.values())).keys())
    summary = {k: [results[s][k] for s in seeds] for k in schemes}
    _save(dirname, "multiseed_results.json", summary)
    print(f"\n=== {track_name} multi-seed (n={len(seeds)}) summary ===")
    for k in schemes:
        vals = np.array(summary[k])
        print(f"  {k:12s} mean={vals.mean():.4f}  std={vals.std():.4f}  per-seed={np.round(vals,4).tolist()}")
    return summary


def qubit_extension_su2lgt(seed=0):
    print("\n--- SU(2) LGT qubit-count extension: N=3 (6 qubits) ---")
    final = run_su2lgt(seed=seed, num_qubits=6, depth=4, adapt_iters=250, adapt_lr=0.05)
    _save("su2_lgt", "adaptation_final_N3.json", final)
    return final


def qubit_extension_neutrino(seed=0):
    print("\n--- Neutrino qubit-count extension: N=6 (6 qubits) ---")
    final = run_neutrino(seed=seed, num_qubits=6, depth=3, adapt_iters=200, adapt_lr=0.05)
    _save("neutrino", "adaptation_final_N6.json", final)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True,
                        choices=list(TRACKS.keys()) + ["all", "qubit_ext"])
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    if args.track == "all":
        for name in TRACKS:
            print(f"\n{'='*10} {name} {'='*10}")
            multiseed(name, seeds)
    elif args.track == "qubit_ext":
        qubit_extension_su2lgt()
        qubit_extension_neutrino()
    else:
        multiseed(args.track, seeds)
