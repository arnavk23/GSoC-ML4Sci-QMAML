"""Emit LaTeX tables for the supplementary material (per-seed raw numbers behind the main text's means)."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "final_model", "results")
J = lambda *p: json.load(open(os.path.join(R, *p)))
S = ["qmaml", "zero", "pi", "uniform", "gaussian"]
DISP = {"qmaml": "Q-MAML", "zero": "Zero", "pi": "$\\pi$", "uniform": "Uniform", "gaussian": "Gaussian"}


def fmt(x, gap=False):
    if gap: return f"{x:.2f}"
    p = 100 * x
    if p == 0: return "0"
    if abs(p) >= 100: return f"{p:.0f}"
    if abs(p) >= 1: return f"{p:.2f}"
    if abs(p) >= 0.01: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\!\\times\\!10^{{{int(e)}}}$"


def write_full_track_table():
    z2 = J("verdict_changers_multiseed_z2.json")["z2"]
    sets = [
        ("A", "Schwinger, $L{=}2$", J("schwinger", "multiseed_L2.json"), False),
        ("B", "$\\mathbb{Z}_2$ LGT, penalty ansatz", J("z2_lgt", "multiseed_V10.json"), False),
        ("B", "$\\mathbb{Z}_2$ LGT, gauge-inv.\\ ansatz",
         {k: [z2[str(i)]["gauge_invariant_hva"][k] for i in range(3)] for k in z2["0"]["gauge_invariant_hva"]}, False),
        ("C", "$\\phi^4$ scalar field", J("scalar_field", "multiseed_uniform.json"), False),
        ("D", "SU(2) LGT, $N{=}2$", J("su2_lgt", "multiseed_results.json"), False),
        ("E", "Neutrinos, $N{=}4$", J("neutrino", "multiseed_results.json"), False),
        ("F", "SUSY QM (log-gap)", J("susyqm", "multiseed_results.json"), True),
        ("G", "$\\mathbb{D}_8$ LGT", J("d8_lgt", "multiseed_results.json"), False),
        ("H", "Light nuclei (UCCSD)", J("nuclear", "multiseed_results.json"), False),
        ("I", "Scalar Yukawa", J("yukawa", "multiseed_results.json"), False),
        ("J", "LMG, $J{=}3/2$", J("lmg", "multiseed_results.json"), False),
    ]
    L = ["\\begin{longtable}{@{}llcccccc@{}}", "\\caption{Per-seed relative error (\\%; log-gap for Track F) for every scheme and system at base size.}\\label{tab:supp_tracks}\\\\",
         "\\toprule", "Track & System & Seed & Q-MAML & Zero & $\\pi$ & Uniform & Gaussian \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Track & System & Seed & Q-MAML & Zero & $\\pi$ & Uniform & Gaussian \\\\", "\\midrule", "\\endhead"]
    for tid, name, d, gap in sets:
        n = len(d["qmaml"])
        for i in range(n):
            row = [tid if i == 0 else "", name if i == 0 else "", str(i)] + [fmt(d[k][i], gap) for k in S]
            L.append(" & ".join(row) + " \\\\")
    L += ["\\botrule", "\\end{longtable}"]
    open(os.path.join(HERE, "supp_tracks_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_vsweep_table():
    v = J("z2_lgt", "v_sweep.json")
    L = ["\\begin{table}[!htb]", "\\centering",
         "\\caption{$\\mathbb{Z}_2$ gauge theory, penalty-enforced ansatz: final energy by penalty strength $V$ (single seed, reduced budget).}\\label{tab:supp_vsweep}",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "$V$ & Q-MAML & Zero & $\\pi$ & Uniform & Gaussian & Q-MAML / best classical \\\\", "\\midrule"]
    for Vs in ["1.0", "3.0", "10.0", "30.0"]:
        d = v[Vs]
        fin = {k: float(np.mean([row[-1] for row in arr])) for k, arr in d.items()}
        cl = {k: fin[k] for k in fin if k != "qmaml"}
        best = min(cl, key=cl.get)
        ratio = fin["qmaml"] / cl[best]
        cells = [f"{fin[k]:.3f}" for k in S]
        L.append(f"{Vs.rstrip('0').rstrip('.')} & " + " & ".join(cells) + f" & {ratio:.2f} ({DISP[best]}) \\\\")
    L += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(HERE, "supp_vsweep_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_l4scan_table():
    d = J("schwinger", "l4_depth_pretrain_scan.json")
    L = ["\\begin{table}[!htb]", "\\centering",
         "\\caption{Lattice Schwinger model, $L{=}4$: final relative error (\\%) by depth and pretraining setting (single seed, reduced budget).}\\label{tab:supp_l4}",
         "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         "Pretraining setting & Q-MAML & Zero & $\\pi$ & Uniform & Gaussian \\\\", "\\midrule"]
    labels = {"depth12_epochs40_lr0.005": "Depth 12, 40 ep, lr 0.005",
              "depth12_epochs100_lr0.005": "Depth 12, 100 ep, lr 0.005",
              "depth12_epochs100_lr0.002": "Depth 12, 100 ep, lr 0.002",
              "depth18_epochs100_lr0.002": "Depth 18, 100 ep, lr 0.002"}
    for k, lab in labels.items():
        f = d[k]["final"]
        L.append(lab + " & " + " & ".join(fmt(f[s]) for s in S) + " \\\\")
    L += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(HERE, "supp_l4scan_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_quarkgluon_table():
    import csv
    rows = list(csv.DictReader(open(os.path.join(R, "quarkgluon", "summary.csv"))))
    lab = {"zero": "Zero", "uniform": "Uniform", "gaussian": "Gaussian", "qmaml_paper": "Q-MAML"}
    L = ["\\begin{table}[!htb]", "\\centering",
         "\\caption{Quark--gluon jet tagging, six qubits ($n=2$ seeds): final validation accuracy and meta-loss.}\\label{tab:supp_qg}",
         "\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Scheme & Validation accuracy & Meta-loss \\\\", "\\midrule"]
    for r in rows:
        v = r["variant"]
        if v not in lab: continue
        L.append(f"{lab[v]} & {float(r['val_accuracy_mean']):.3f}\\,$\\pm$\\,{float(r['val_accuracy_std']):.3f} & "
                  f"{float(r['meta_loss_mean']):.3f}\\,$\\pm$\\,{float(r['meta_loss_std']):.3f} \\\\")
    L += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(HERE, "supp_quarkgluon_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    write_full_track_table(); write_vsweep_table(); write_l4scan_table(); write_quarkgluon_table()
    print("supplementary tables written")
