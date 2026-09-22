"""Generate every figure for the Springer Nature manuscript from saved results (vector PDF)."""
import os, json, ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "final_model", "results")
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
J = lambda *p: json.load(open(os.path.join(R, *p)))

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = {"qmaml": "#D55E00", "pi": "#0072B2", "uniform": "#009E73", "gaussian": "#CC79A7", "zero": "#7F7F7F",
       "basis_paper": "#E69F00", "qmaml_paper": "#D55E00", "qmaml_existing": "#56B4E9"}
NAME = {"qmaml": "Q-MAML", "pi": r"$\pi$", "uniform": "Uniform", "gaussian": "Gaussian", "zero": "Zero",
        "qmaml_paper": "Q-MAML", "qmaml_existing": "First-order"}
W2, W1 = 5.4, 3.0   # inches: full text width (Springer single-column, ~12 cm) and narrow figures
CLASSICAL = ["zero", "pi", "uniform", "gaussian"]


def panel(ax, s, dx=-0.13, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- calibration
def fig_calibration():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    for ax, path, title in [(axs[0], ("vqe_heisenberg", "adaptation_gaps.json"), "Heisenberg XYZ, 6 qubits"),
                            (axs[1], ("vqe_molecule", "adaptation_gaps.json"), r"H$_2$ molecule, 4 qubits")]:
        d = J(*path)
        for s in ["zero", "pi", "uniform", "gaussian", "qmaml"]:
            a = np.array(d[s]); m, sd = a.mean(0), a.std(0); x = np.arange(a.shape[1])
            ax.plot(x, m, color=COL[s], label=NAME[s], lw=1.6 if s == "qmaml" else 1.0, zorder=3 if s == "qmaml" else 2)
            ax.fill_between(x, m - sd / 2, m + sd / 2, color=COL[s], alpha=0.12, lw=0)
        ax.set_xlabel("Adaptation iteration"); ax.set_title(title, loc="left")
    axs[0].set_ylabel(r"$\log|E_{\mathrm{PQC}}-E_0|$"); axs[0].legend(frameon=False, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.62))
    panel(axs[0], "a"); panel(axs[1], "b")
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf")); plt.close(fig)


# ---------------------------------------------------------------- HIGGS
def load_higgs():
    raw = {}
    paths = [os.path.join(R, "higgs_v3", "raw_results.json")] + \
            [os.path.join(R, "higgs_v3_extra", f"raw_results_q{q}.json") for q in (4, 6, 8)]
    for p in paths:
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                raw[ast.literal_eval(k)] = v
    return raw


def fig_higgs():
    raw = load_higgs(); V = ["zero", "uniform", "gaussian", "qmaml_existing", "qmaml_paper"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True); nmax = 0
    for ax, nq in zip(axs, (4, 6, 8)):
        seeds = sorted({s for (q, s, v) in raw if q == nq and all((nq, s, x) in raw for x in V)})
        nmax = max(nmax, len(seeds))
        for i, v in enumerate(V):
            acc = np.array([raw[(nq, s, v)]["val_accuracy"][-1] for s in seeds])
            ax.bar(i, acc.mean(), color=COL[v], alpha=0.85 if v == "qmaml_paper" else 0.55, width=0.7, lw=0)
            ax.errorbar(i, acc.mean(), acc.std(ddof=1), color="k", capsize=2, lw=0.8)
            ax.scatter(np.full(len(acc), i) + np.linspace(-0.18, 0.18, len(acc)), acc, s=6, color="k", zorder=4, lw=0)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_xticks(range(len(V))); ax.set_xticklabels([NAME[v] for v in V], rotation=35, ha="right")
        ax.set_title(f"{nq} qubits ($n={len(seeds)}$ seeds)", loc="left"); ax.set_ylim(0.42, 0.66)
    axs[0].set_ylabel("Validation accuracy")
    for a, s in zip(axs, "abc"):
        panel(a, s, dx=-0.2 if s == "a" else -0.12)
    fig.savefig(os.path.join(OUT, "fig_higgs.pdf")); plt.close(fig)
    return nmax


# ---------------------------------------------------------------- scoreboard / scaling (forest plots)
def ratios(d, per_seed=None, gap=False):
    """Q-MAML error / best classical error, per seed. gap=True: inputs are log-gaps."""
    if per_seed is None:
        n = len(d["qmaml"]); per_seed = [{k: d[k][i] for k in d} for i in range(n)]
    out = []
    for r in per_seed:
        cl = [r[k] for k in r if k != "qmaml" and not k.startswith("_")]
        out.append(np.exp(r["qmaml"] - min(cl)) if gap else r["qmaml"] / min(cl))
    return np.array(out)


def forest(rows, path, height, xlim, annotate=False):
    fig, ax = plt.subplots(figsize=(W1 + 0.7, height))
    ax.axvspan(-8, 0, color=COL["qmaml"], alpha=0.06, lw=0); ax.axvline(0, color="k", lw=0.7)
    for i, (lab, r) in enumerate(rows):
        y = len(rows) - 1 - i; lr = np.log10(r)
        ax.scatter(lr, np.full(len(lr), y) + np.linspace(-0.15, 0.15, len(lr)), s=13, color=COL["qmaml"], zorder=3, lw=0)
        ax.plot([lr.mean()] * 2, [y - 0.28, y + 0.28], color="k", lw=1.4, zorder=4)
        ax.axhline(y, color="0.9", lw=0.5, zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"$\log_{10}$ (Q-MAML error / best classical error)")
    if annotate:
        ax.text(xlim[0] + 0.1, len(rows) - 0.55, "Q-MAML better", fontsize=7, color=COL["qmaml"], va="top")
        ax.text(xlim[1] - 0.1, len(rows) - 0.55, "classical better", fontsize=7, color="0.3", va="top", ha="right")
    fig.savefig(path); plt.close(fig)


def fig_scoreboard():
    z2 = J("verdict_changers_multiseed_z2.json")["z2"]
    rows = [("A  Schwinger, $L{=}2$", ratios(J("schwinger", "multiseed_L2.json"))),
            ("B  $\\mathbb{Z}_2$ LGT, penalty ansatz", ratios(J("z2_lgt", "multiseed_V10.json"))),
            ("B  $\\mathbb{Z}_2$ LGT, gauge-inv. ansatz",
             ratios(None, [z2[str(i)]["gauge_invariant_hva"] for i in range(3)])),
            ("C  $\\phi^4$ scalar field", ratios(J("scalar_field", "multiseed_uniform.json"))),
            ("D  SU(2) LGT, $N{=}2$", ratios(J("su2_lgt", "multiseed_results.json"))),
            ("E  Neutrinos, $N{=}4$", ratios(J("neutrino", "multiseed_results.json"))),
            ("F  SUSY QM", ratios(J("susyqm", "multiseed_results.json"), gap=True)),
            ("G  $\\mathbb{D}_8$ LGT", ratios(J("d8_lgt", "multiseed_results.json"))),
            ("H  Light nuclei (UCCSD)", ratios(J("nuclear", "multiseed_results.json"))),
            ("I  Scalar Yukawa", ratios(J("yukawa", "multiseed_results.json"))),
            ("J  LMG, $J{=}3/2$", ratios(J("lmg", "multiseed_results.json")))]
    forest(rows, os.path.join(OUT, "fig_scoreboard.pdf"), 3.5, (-7.4, 3.3), annotate=True)


def fig_scaling():
    su = J("verdict_changers_multiseed_su2.json"); n3d6 = J("su2_followup_n3d6.json"); n6 = J("neutrino_N6_multiseed.json")
    ps = lambda dd: [dd[str(i)] for i in sorted(map(int, dd))]
    l4 = J("schwinger", "l4_depth_pretrain_scan.json"); l4b = J("schwinger", "l4_seeds_followup.json")
    a18 = [l4["depth18_epochs100_lr0.002"]["final"]] + \
          [v["final"] for k, v in sorted(l4b.items()) if "depth18_ep100_lr0.002" in k]
    rows = [("A  Schwinger $L{=}2$ (4 qubits)", ratios(J("schwinger", "multiseed_L2.json"))),
            ("A  Schwinger $L{=}4$, depth 18 (8)", ratios(None, a18)),
            ("D  SU(2) $N{=}2$ (4)", ratios(J("su2_lgt", "multiseed_results.json"))),
            ("D  SU(2) $N{=}3$, depth 6 (6)", ratios(None, ps(n3d6))),
            ("D  SU(2) $N{=}4$ (8)", ratios(None, ps(su["su2_N4"]))),
            ("E  Neutrinos $N{=}4$ (4)", ratios(J("neutrino", "multiseed_results.json"))),
            ("E  Neutrinos $N{=}6$ (6)", ratios(None, ps(n6)))]
    forest(rows, os.path.join(OUT, "fig_scaling.pdf"), 2.7, (-7.4, 2.9), annotate=True)


# ---------------------------------------------------------------- gauge invariance (Track B)
def fig_gauge():
    z2 = J("verdict_changers_multiseed_z2.json")["z2"]; head = J("z2_lgt", "multiseed_V10.json"); an = J("z2_anneal.json")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.4})
    ax = axs[0]; S = ["qmaml", "pi", "uniform", "gaussian", "zero"]
    groups = [("Penalty,\nhardware-eff.", [{k: head[k][i] for k in head} for i in range(3)]),
              ("Control\nansatz", [z2[str(i)]["hardware_efficient_zz"] for i in range(3)]),
              ("Gauge-inv.\nHVA ansatz", [z2[str(i)]["gauge_invariant_hva"] for i in range(3)])]
    xt, xl = [], []
    for g, (lab, ps_) in enumerate(groups):
        for j, s in enumerate(S):
            x = g * (len(S) + 1.2) + j; v = np.array([p[s] for p in ps_])
            ax.bar(x, v.mean(), width=0.75, color=COL[s], alpha=0.85 if s == "qmaml" else 0.5, lw=0)
            ax.scatter(np.full(3, x) + np.array([-0.18, 0, 0.18]), v, s=6, color="k", zorder=4, lw=0)
        xt.append(g * (len(S) + 1.2) + (len(S) - 1) / 2); xl.append(lab)
    ax.set_yscale("log"); ax.set_ylim(3e-6, 120); ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("Relative energy error"); panel(ax, "a", dx=-0.13)
    ax.legend(handles=[Patch(color=COL[s], alpha=0.8, label=NAME[s]) for s in S], frameon=False, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=0.9, handletextpad=0.4,
              fontsize=6.2)
    ax = axs[1]
    q = lambda k: np.array([an[f"{k}/seed{i}"]["qmaml"] for i in range(3)])
    xs = {"fixed\n$V{=}10$": np.array(head["qmaml"]), "anneal\n$1\\to10$": q("up_1to10"), "anneal\n$30\\to10$": q("down_30to10")}
    for i, (lab, v) in enumerate(xs.items()):
        ax.bar(i, v.mean() * 100, color=COL["qmaml"], alpha=0.55, width=0.65, lw=0)
        ax.scatter(np.full(3, i) + np.array([-0.15, 0, 0.15]), v * 100, s=8, color="k", zorder=4, lw=0)
    bestc = np.array([min(head[s][i] for s in CLASSICAL) for i in range(3)]) * 100
    ax.axhline(bestc.mean(), color=COL["uniform"], lw=1.2, ls="--")
    ax.text(-0.45, bestc.mean() - 1.5, "best classical", ha="left", va="top", fontsize=6.5, color=COL["uniform"])
    ax.set_xticks(range(3)); ax.set_xticklabels(list(xs)); ax.set_ylabel("Q-MAML error (%)"); panel(ax, "b", dx=-0.3)
    fig.savefig(os.path.join(OUT, "fig_gauge.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_calibration(); n = fig_higgs(); fig_scoreboard(); fig_gauge(); fig_scaling()
    print("figures written:", sorted(os.listdir(OUT)), " HIGGS seeds in figure:", n)
