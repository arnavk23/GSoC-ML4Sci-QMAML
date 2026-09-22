"""Track E at N=6 (6 qubits): seeds 1,2 added to the existing seed-0 result (adaptation_final_N6.json). Resumable."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from multiseed_validate import run_neutrino
OUT = os.path.join(os.path.dirname(__file__), "results", "neutrino_N6_multiseed.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
if "0" not in out:
    out["0"] = json.load(open(os.path.join(os.path.dirname(__file__), "results", "neutrino", "adaptation_final_N6.json")))
    json.dump(out, open(OUT, "w"))
for seed in [1, 2]:
    if str(seed) in out: continue
    out[str(seed)] = run_neutrino(seed=seed, num_qubits=6, depth=3, adapt_iters=200, adapt_lr=0.05, verify_first=False)
    json.dump(out, open(OUT, "w"))
print("N6 DONE", flush=True)
