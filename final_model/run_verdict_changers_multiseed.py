"""
Multi-seed (seeds 0,1,2) replication of the two single-seed findings that changed a track's verdict:
  (1) Track B: gauge-invariant HVA vs. non-gauge-invariant control ansatz (same reduced protocol as
      run_z2lgt_alt_ansatz.py: 4 held-out tasks, 200 adaptation iterations, V=10). Seed 0 is reused from
      results/z2_lgt/alt_ansatz_comparison.json; seeds 1,2 are new.
  (2) Track D: SU(2) LGT at N=3 and N=4 (seed 0 reused from adaptation_final_N{3,4}.json; seeds 1,2 new).
Saves after every finished piece so a restart cannot lose completed work.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, f"verdict_changers_multiseed_{sys.argv[1]}.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(out, open(OUT, "w"))

SCHEMES = ["qmaml", "zero", "pi", "uniform", "gaussian"]

def z2_seed(seed):
    from hamiltonians_z2lgt import z2_lgt_hamiltonian, exact_ground_energy, sample_task_space, NUM_QUBITS
    from z2lgt_qmaml import (classical_init, adapt_task_z2lgt, Z2LGTHVAPQC, Z2LGTHWEffZZPQC,
                              pretrain_learner_z2lgt_ansatz)
    V, DEPTH = 10.0, 3
    torch.manual_seed(seed); np.random.seed(seed)
    train = sample_task_space(16, seed=seed)
    test = sample_task_space(6, seed=seed + 1000)[:4]
    E0s = [exact_ground_energy(z2_lgt_hamiltonian(J, m, mu, V=V)) for J, m, mu in test]
    mabs = np.mean(np.abs(E0s))
    res = {}
    for tag, Cls in [("gauge_invariant_hva", Z2LGTHVAPQC), ("hardware_efficient_zz", Z2LGTHWEffZZPQC)]:
        torch.manual_seed(seed); np.random.seed(seed)
        pqc = Cls(NUM_QUBITS, DEPTH)
        learner, _ = pretrain_learner_z2lgt_ansatz(pqc, task_dim=3, train_tasks=train, epochs=40, lr=0.005,
                                                    w0_scale=0.3, log_every=999, gauss_V=V, tag=tag)
        gaps = {k: [] for k in SCHEMES}
        for (J, m, mu), E0 in zip(test, E0s):
            H = z2_lgt_hamiltonian(J, m, mu, V=V)
            for s in SCHEMES:
                th0 = (0.3 * learner(torch.tensor([J, m, mu], dtype=torch.float64))).detach() if s == "qmaml" \
                    else classical_init(pqc.shape, s, DEPTH)
                _, costs = adapt_task_z2lgt(pqc, th0, H, iters=200, lr=0.02)
                gaps[s].append(np.log(np.abs(np.array(costs) - E0) + 1e-8))
        res[tag] = {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mabs) for k, v in gaps.items()}
        print(f"[z2 seed={seed} {tag}] {res[tag]}", flush=True)
    return res

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "z2":
        # seed 0 from the existing run
        alt = json.load(open(os.path.join(RES, "z2_lgt", "alt_ansatz_comparison.json")))
        from hamiltonians_z2lgt import z2_lgt_hamiltonian, exact_ground_energy, sample_task_space
        test0 = sample_task_space(6, seed=1000)[:4]
        mabs0 = np.mean(np.abs([exact_ground_energy(z2_lgt_hamiltonian(J, m, mu, V=10.0)) for J, m, mu in test0]))
        s0 = {t: {k: float(np.exp(np.mean(np.array(v)[:, -1])) / mabs0) for k, v in d.items()} for t, d in alt.items()}
        out["z2"] = {"0": s0}; save()
        for seed in [1, 2]:
            out["z2"][str(seed)] = z2_seed(seed); save()
    else:
        from multiseed_validate import run_su2lgt
        for N, nq, depth in [(3, 6, 4), (4, 8, 6)]:
            key = f"su2_N{N}"
            out[key] = {"0": json.load(open(os.path.join(RES, "su2_lgt", f"adaptation_final_N{N}.json")))}; save()
            for seed in [1, 2]:
                out[key][str(seed)] = run_su2lgt(seed=seed, num_qubits=nq, depth=depth, adapt_iters=250 if N == 3 else 300,
                                                  adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, verify_first=False)
                save()
        print("SU2 DONE", flush=True)
