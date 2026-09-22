"""Emit LaTeX tables for the Springer manuscript directly from saved results (no hand-transcribed numbers)."""
import os, json, ast
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "final_model", "results")
J = lambda *p: json.load(open(os.path.join(R, *p)))
CLASSICAL = ["zero", "pi", "uniform", "gaussian"]
DISP = {"zero": "Zero", "pi": r"$\pi$", "uniform": "Uniform", "gaussian": "Gaussian", "basis_paper": "basis", "qmaml": "Q-MAML"}


def fmt_pct(x):
    """x is a fraction; return a percentage string with sensible precision."""
    p = 100 * x
    if p == 0: return "0"
    if abs(p) >= 100: return f"{p:.0f}"
    if abs(p) >= 10: return f"{p:.1f}"
    if abs(p) >= 1: return f"{p:.2f}"
    if abs(p) >= 0.01: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\!\\times\\!10^{{{int(e)}}}$"


def track_rows():
    z2 = J("verdict_changers_multiseed_z2.json")["z2"]
    sets = [
        ("A", "Schwinger, $L{=}2$", J("schwinger", "multiseed_L2.json"), False),
        ("B", "$\\mathbb{Z}_2$ LGT, penalty ansatz", J("z2_lgt", "multiseed_V10.json"), False),
        ("B", "$\\mathbb{Z}_2$ LGT, gauge-inv.\\ ansatz",
         {k: [z2[str(i)]["gauge_invariant_hva"][k] for i in range(3)] for k in z2["0"]["gauge_invariant_hva"]}, False),
        ("C", "$\\phi^4$ scalar field", J("scalar_field", "multiseed_uniform.json"), False),
        ("D", "SU(2) LGT, $N{=}2$", J("su2_lgt", "multiseed_results.json"), False),
        ("E", "Neutrinos, $N{=}4$", J("neutrino", "multiseed_results.json"), False),
        ("F", "SUSY QM", J("susyqm", "multiseed_results.json"), True),
        ("G", "$\\mathbb{D}_8$ LGT", J("d8_lgt", "multiseed_results.json"), False),
        ("H", "Light nuclei (UCCSD)", J("nuclear", "multiseed_results.json"), False),
        ("I", "Scalar Yukawa", J("yukawa", "multiseed_results.json"), False),
        ("J", "LMG, $J{=}3/2$", J("lmg", "multiseed_results.json"), False),
    ]
    rows = []
    for tid, name, d, gap in sets:
        cl = [k for k in d if k != "qmaml"]
        best = min(cl, key=lambda k: np.mean(d[k]))
        # per-seed win / tie / loss against the best classical scheme at that seed, 10% tolerance on the error ratio
        wtl = [0, 0, 0]
        for i in range(3):
            b_i = min(d[k][i] for k in cl)
            r = np.exp(d["qmaml"][i] - b_i) if gap else d["qmaml"][i] / b_i
            at_floor = (not gap) and d["qmaml"][i] < 1e-6 and b_i < 1e-6   # both at the precision floor: a tie
            wtl[1 if at_floor else (0 if r < 0.9 else (2 if r > 1.1 else 1))] += 1
        wtl_s = f"{wtl[0]}/{wtl[1]}/{wtl[2]}"
        if gap:
            q, b = f"{np.mean(d['qmaml']):.2f}", f"{np.mean(d[best]):.2f}"
            rows.append((tid, name, q + " (log-gap)", f"{b} ({DISP[best]})", wtl_s))
        else:
            q = f"{fmt_pct(np.mean(d['qmaml']))}"
            b = f"{fmt_pct(np.mean(d[best]))} ({DISP[best]})"
            rows.append((tid, name, q, b, wtl_s))
    return rows


def write_tracks_table():
    rows = track_rows()
    L = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
         "Track & System & Q-MAML error (\\%) & Best classical (\\%) & W/T/L \\\\", "\\midrule"]
    for tid, name, q, b, w in rows:
        L.append(f"{tid} & {name} & {q} & {b} & {w} \\\\")
    L += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(HERE, "tracks_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return rows


def load_higgs():
    raw = {}
    paths = [os.path.join(R, "higgs_v3", "raw_results.json")] + \
            [os.path.join(R, "higgs_v3_extra", f"raw_results_q{q}.json") for q in (4, 6, 8)]
    for p in paths:
        if os.path.exists(p):
            for k, v in json.load(open(p)).items(): raw[ast.literal_eval(k)] = v
    return raw


def welch(a, b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return p, (a.mean() - b.mean()) / sp


def write_higgs_table():
    raw = load_higgs(); V = ["zero", "uniform", "gaussian", "qmaml_existing", "qmaml_paper"]
    VN = {"zero": "Zero", "uniform": "Uniform", "gaussian": "Gaussian", "qmaml_existing": "First-order variant",
          "qmaml_paper": "\\textbf{Q-MAML}"}
    info, cols = {}, {}
    for nq in (4, 6, 8):
        seeds = sorted({s for (q, s, v) in raw if q == nq and all((nq, s, x) in raw for x in V)})
        acc = {v: np.array([raw[(nq, s, v)]["val_accuracy"][-1] for s in seeds]) for v in V}
        best = max(["zero", "uniform", "gaussian"], key=lambda v: acc[v].mean())
        pg, dg = welch(acc["qmaml_paper"], acc["gaussian"]); pb, db = welch(acc["qmaml_paper"], acc[best])
        cols[nq] = (seeds, acc, best, pg, dg, pb, db)
        info[nq] = {"n": len(seeds), "p_gauss": pg, "d_gauss": dg, "p_best": pb, "d_best": db, "best": best,
                    "means": {v: float(acc[v].mean()) for v in V}}
    fp = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}"
    L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "& 4 qubits & 6 qubits & 8 qubits \\\\", "\\midrule",
         "Seeds ($n$) & " + " & ".join(str(len(cols[q][0])) for q in (4, 6, 8)) + " \\\\", "\\midrule"]
    for v in V:
        cells = []
        for q in (4, 6, 8):
            a = cols[q][1][v]; c = f"{a.mean():.3f}\\,$\\pm$\\,{a.std(ddof=1):.3f}"
            cells.append("\\textbf{" + c + "}" if v == "qmaml_paper" else c)
        L.append(VN[v] + " & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    L.append("Welch $p$ ($d$), Q-MAML vs Gaussian & " + " & ".join(f"{fp(cols[q][3])} ({cols[q][4]:.1f})" for q in (4, 6, 8)) + " \\\\")
    L.append("Welch $p$ ($d$), vs best classical & " + " & ".join(
        f"{fp(cols[q][5])} ({cols[q][6]:.1f}; {DISP[cols[q][2]]})" for q in (4, 6, 8)) + " \\\\")
    L += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(HERE, "higgs_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return info


if __name__ == "__main__":
    for r in write_tracks_table(): print(r)
    print(json.dumps(write_higgs_table(), indent=1))
