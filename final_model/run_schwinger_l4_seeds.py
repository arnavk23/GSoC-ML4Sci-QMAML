"""
Track A L=4 follow-up (same reduced protocol as run_schwinger_l4_scan.py: 3 held-out tasks, 150 iters).
Answers the two caveats left open there, resumable:
  (1) depth 18 at the ORIGINAL pretrain lr 0.005 (100 epochs), seed 0 -> is the lr fix needed at depth 18?
  (2) depth 12 and depth 18 at lr 0.002 / 100 epochs for seeds 1 and 2 -> does the depth-18 win replicate?
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import run_schwinger_l4_scan as L
OUT = os.path.join(os.path.dirname(__file__), "results", "schwinger", "l4_seeds_followup.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
jobs = [(0, 18, 100, 0.005), (1, 12, 100, 0.002), (1, 18, 100, 0.002), (2, 12, 100, 0.002), (2, 18, 100, 0.002)]
for seed, depth, ep, lr in jobs:
    key = f"seed{seed}_depth{depth}_ep{ep}_lr{lr}"
    if key in out: continue
    L.SEED = seed
    final, hist = L.run_config(depth, ep, lr)
    out[key] = {"final": final, "grad_norm_last": hist["grad_norm"][-1], "loss_last": hist["loss"][-1]}
    json.dump(out, open(OUT, "w"))
print("L4SEEDS DONE", flush=True)
