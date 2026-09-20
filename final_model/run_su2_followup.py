"""
Follow-ups to the SU(2) N=3/N=4 findings (results/verdict_changers_multiseed_su2.json):
  n4lr : N=4 (8 qubits, depth 6, 300 iters) with pretrain lr 0.002 (vs 0.01) and 100 epochs (vs 40), seeds 0,1,2.
         Tests whether N=4's seed-unstable Q-MAML (23% / 2.9% / 450%) is a pretraining failure, as at Track A L=4.
  n3d6 : N=3 (6 qubits) with depth 6 and 300 iters (vs depth 4 / 250), seeds 0,1,2, pretrain lr 0.01 as before.
         Tests whether the N=3 'nothing converges' result was a budget artifact.
Saves after every seed.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from multiseed_validate import run_su2lgt

which = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "results", f"su2_followup_{which}.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for seed in [0, 1, 2]:
    if str(seed) in out: continue
    if which == "n4lr":
        r = run_su2lgt(seed=seed, num_qubits=8, depth=6, adapt_iters=300, adapt_lr=0.05,
                       pretrain_epochs=100, pretrain_lr=0.002, verify_first=False)
    else:
        r = run_su2lgt(seed=seed, num_qubits=6, depth=6, adapt_iters=300, adapt_lr=0.05,
                       pretrain_epochs=40, pretrain_lr=0.01, verify_first=False)
    out[str(seed)] = r
    json.dump(out, open(OUT, "w"))
print("FOLLOWUP DONE", which, flush=True)
