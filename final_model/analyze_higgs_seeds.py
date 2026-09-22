"""
Merge HIGGS results (seeds 0-3 from results/higgs_v3/raw_results.json, seeds 4-7 from results/higgs_v3_extra/) and
recompute the paper's Table 3/4 statistics: final validation accuracy mean +/- std over seeds, Welch t-test and
Cohen's d (pooled s_p = sqrt((s1^2+s2^2)/2)) of qmaml_paper vs gaussian, vs qmaml_existing (GSoC baseline), and vs the
post-hoc best classical scheme (best mean among zero/uniform/gaussian at that qubit count). Only seeds present for ALL
five variants at a qubit count are used, so comparisons are paired by seed set.
Usage: python analyze_higgs_seeds.py [max_seed]   (default: use every complete seed)
"""
import sys, os, json, ast, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = {}
for p in [os.path.join(HERE, "results", "higgs_v3", "raw_results.json")] + \
         [os.path.join(HERE, "results", "higgs_v3_extra", f"raw_results_q{q}.json") for q in (4, 6, 8)]:
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            RAW[ast.literal_eval(k)] = v
V = ["zero", "uniform", "gaussian", "qmaml_existing", "qmaml_paper"]
max_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 99
def welch(a, b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return p, (a.mean() - b.mean()) / sp
out = {}
for nq in (4, 6, 8):
    seeds = sorted({s for (q, s, v) in RAW if q == nq and s <= max_seed and all((nq, s, x) in RAW for x in V)})
    acc = {v: [RAW[(nq, s, v)]["val_accuracy"][-1] for s in seeds] for v in V}
    best = max(["zero", "uniform", "gaussian"], key=lambda v: np.mean(acc[v]))
    row = {"seeds": seeds, "mean": {v: float(np.mean(acc[v])) for v in V}, "std": {v: float(np.std(acc[v], ddof=1)) for v in V}}
    for name, other in [("gaussian", "gaussian"), ("gsoc_baseline", "qmaml_existing"), (f"best_classical({best})", best)]:
        p, d = welch(acc["qmaml_paper"], acc[other]); row[f"vs_{name}"] = {"p": float(p), "d": float(d)}
    out[nq] = row
    print(f"\n=== {nq} qubits, n={len(seeds)} seeds {seeds}")
    print("  " + "  ".join(f"{v}={row['mean'][v]:.3f}±{row['std'][v]:.3f}" for v in V))
    for k, x in row.items():
        if k.startswith("vs_"): print(f"  qmaml_paper {k}: p={x['p']:.3f}  d={x['d']:.2f}")
json.dump(out, open(os.path.join(HERE, "results", f"higgs_analysis_maxseed{max_seed}.json"), "w"), indent=1)
