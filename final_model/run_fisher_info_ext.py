"""Extends run_fisher_info_probe.py: qubits {4,6,8,10,12} and 20 samples per point (was {4,6,8}, 8). Resumable per qubit count."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, pennylane as qml
from pennylane import numpy as pnp
import run_fisher_info_probe as F
from vqe_heisenberg import sample_task_space
from vqe_qmaml import classical_init, pretrain_learner_vqe

OUT = os.path.join(os.path.dirname(__file__), "results", "barren_plateau", "fisher_info_effective_dim_ext.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
NS, DEPTH, SEED = 20, 3, 0
rng = np.random.default_rng(SEED)
for nq in [4, 6, 8, 10, 12]:
    if str(nq) in out: continue
    circuit = F.make_ansatz_qnode(nq, DEPTH)
    shape = (DEPTH, nq - 1, 3); n_params = DEPTH * (nq - 1) * 3
    torch.manual_seed(SEED); np.random.seed(SEED)
    learner, _, _ = pretrain_learner_vqe(nq, DEPTH, sample_task_space(16, seed=SEED), epochs=15, lr=0.01)
    rec = {}
    for scheme in ["qmaml", "uniform", "gaussian"]:
        prs = []
        for _ in range(NS):
            if scheme == "qmaml":
                Jx, Jy, Jz = rng.uniform(0.2, 2.0, size=3)
                with torch.no_grad():
                    th = learner(torch.tensor([Jx, Jy, Jz], dtype=torch.float64)).numpy().reshape(shape)
            else:
                th = classical_init(shape, scheme, DEPTH).numpy()
            mt = qml.metric_tensor(circuit, approx="block-diag")(pnp.array(th, requires_grad=True))
            ev = np.linalg.eigvalsh(np.array(mt).reshape(n_params, n_params))
            prs.append(F.participation_ratio(ev) / n_params)
        rec[scheme] = prs
        print(f"qubits={nq} {scheme}: mean={np.mean(prs):.4f} std={np.std(prs):.4f}", flush=True)
    out[str(nq)] = rec; json.dump(out, open(OUT, "w"))
print("FISHER DONE", flush=True)
