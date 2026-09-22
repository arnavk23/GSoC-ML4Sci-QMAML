"""Additional figures: system pictograms, adaptation trajectories, pretraining diagnostics."""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import make_figures as mf
from make_figures import J, COL, NAME, OUT, W2, panel

INK, SOFT = "#2b2b2b", "#9aa5b1"
ACC = mf.COL["qmaml"]


# ---------------------------------------------------------------- pictograms of the ten systems
def _chain(ax):
    xs = np.linspace(-1.05, 1.05, 4)
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, 0), 0.16, fc=ACC if i % 2 == 0 else "white", ec=INK, lw=0.8))
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 0.2, 0), (xs[i + 1] - 0.2, 0), arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.8))


def _plaquette(ax):
    ax.add_patch(Rectangle((-0.6, -0.6), 1.2, 1.2, fc="#eef3f8", ec=INK, lw=0.9))
    for (x, y) in [(-0.6, -0.6), (0.6, -0.6), (0.6, 0.6), (-0.6, 0.6)]:
        ax.add_patch(Circle((x, y), 0.11, fc=ACC, ec=INK, lw=0.7))
    for (x, y, s) in [(0, -0.6, "$\\sigma^z$"), (0, 0.6, "$\\sigma^z$"), (-0.6, 0, "$\\sigma^z$"), (0.6, 0, "$\\sigma^z$")]:
        ax.text(x, y, s, ha="center", va="center", fontsize=5.5, bbox=dict(fc="white", ec="none", pad=0.6))


def _double_well(ax):
    x = np.linspace(-1.3, 1.3, 200); y = (x ** 2 - 0.85) ** 2 * 0.9 - 0.4
    ax.plot(x, y, color=ACC, lw=1.4); ax.axhline(-0.5, color=SOFT, lw=0.5)
    ax.plot([-0.92, 0.92], [-0.4, -0.4], "o", color=INK, ms=2.5)


def _su2(ax):
    r = 0.75
    pts = [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 4)[:-1] + np.pi / 2]
    ax.add_patch(Polygon(pts, closed=True, fc="#eef3f8", ec=INK, lw=0.9))
    for k, p in enumerate(pts):
        ax.add_patch(Circle(p, 0.15, fc=ACC if k else "white", ec=INK, lw=0.8))
    ax.text(0, 0, "SU(2)", ha="center", va="center", fontsize=5.5, color=INK)


def _neutrino(ax):
    rng = np.random.RandomState(3)
    for k in range(6):
        a = 2 * np.pi * k / 6; c = (0.62 * np.cos(a), 0.62 * np.sin(a)); th = a + np.pi / 2 + rng.uniform(-0.4, 0.4)
        ax.add_patch(Circle(c, 0.13, fc="white", ec=INK, lw=0.7))
        ax.add_patch(FancyArrowPatch((c[0] - 0.2 * np.cos(th), c[1] - 0.2 * np.sin(th)), (c[0] + 0.2 * np.cos(th), c[1] + 0.2 * np.sin(th)),
                                     arrowstyle="-|>", mutation_scale=5, color=ACC, lw=0.9))


def _susy(ax):
    x = np.linspace(-1.2, 1.2, 200)
    ax.plot(x, 0.55 * x ** 2 - 0.35, color=SOFT, lw=1.1)
    ax.plot(x, 0.32 * (x ** 2 - 0.5) ** 2 * 1.6 - 0.35, color=ACC, lw=1.3)
    ax.text(1.02, 0.42, "$W$", fontsize=6, color=SOFT); ax.text(-1.2, 0.5, "$V$", fontsize=6, color=ACC)


def _d8(ax):
    a = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8
    pts = [(0.72 * np.cos(t), 0.72 * np.sin(t)) for t in a]
    ax.add_patch(Polygon(pts, closed=True, fc="#eef3f8", ec=INK, lw=0.9))
    for k, p in enumerate(pts):
        ax.add_patch(Circle(p, 0.08, fc=ACC if k == 0 else INK, ec="none"))
    ax.add_patch(FancyArrowPatch((0.32, 0.05), (0.32, 0.4), connectionstyle="arc3,rad=-1.2", arrowstyle="-|>", mutation_scale=5, color=INK, lw=0.7))
    ax.text(0, -0.02, "$D_8$", ha="center", va="center", fontsize=6.5, color=INK)


def _nuclei(ax):
    rng = np.random.RandomState(1); pos = [(-0.2, 0.15), (0.25, 0.25), (0.05, -0.25), (0.42, -0.1), (-0.35, -0.2), (0.0, 0.5)]
    for k, p in enumerate(pos):
        ax.add_patch(Circle(p, 0.2, fc=ACC if k % 2 == 0 else "white", ec=INK, lw=0.8, alpha=0.9))


def _yukawa(ax):
    xs = np.linspace(0.05, 1.0, 60)
    ax.plot(xs, 0.11 * np.sin(xs * 38), color=ACC, lw=1.1)
    ax.plot([-1, 0.05], [0.55, 0], color=INK, lw=1.0); ax.plot([-1, 0.05], [-0.55, 0], color=INK, lw=1.0)
    ax.add_patch(Circle((0.05, 0), 0.07, fc=INK))
    ax.text(-0.9, 0.0, "$\\psi$", fontsize=6.5, ha="center"); ax.text(0.7, 0.3, "$\\varphi$", fontsize=6.5, color=ACC)


def _lmg(ax):
    a = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2; pts = np.array([[0.72 * np.cos(t), 0.72 * np.sin(t)] for t in a])
    for i in range(5):
        for j in range(i + 1, 5):
            ax.plot(*zip(pts[i], pts[j]), color=SOFT, lw=0.6)
    for p in pts:
        ax.add_patch(Circle(p, 0.12, fc=ACC, ec=INK, lw=0.7))


PICT = [("A", "Schwinger", "4 qubits", _chain), ("B", "$\\mathbb{Z}_2$ gauge", "8 qubits", _plaquette),
        ("C", "$\\phi^4$ scalar", "6 qubits", _double_well), ("D", "SU(2) gauge", "4 qubits", _su2),
        ("E", "Neutrinos", "4 qubits", _neutrino), ("F", "SUSY QM", "4 qubits", _susy),
        ("G", "$\\mathbb{D}_8$ gauge", "7 qubits", _d8), ("H", "Light nuclei", "8 qubits", _nuclei),
        ("I", "Yukawa", "4 qubits", _yukawa), ("J", "LMG", "2 qubits", _lmg)]


def fig_systems():
    fig, axs = plt.subplots(2, 5, figsize=(W2, 2.35), gridspec_kw={"wspace": 0.05, "hspace": 0.32})
    for ax, (t, name, nq, fn) in zip(axs.ravel(), PICT):
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.0, 1.0); ax.set_aspect("equal"); ax.axis("off")
        fn(ax)
        ax.text(-1.25, 0.98, t, fontsize=8.5, fontweight="bold", va="top", color=ACC)
        ax.text(0, -0.84, name, ha="center", va="center", fontsize=6.6, color=INK)
        ax.text(0, -1.16, nq, ha="center", va="center", fontsize=5.8, color="0.45")
    fig.savefig(os.path.join(OUT, "fig_systems.pdf")); plt.close(fig)


# ---------------------------------------------------------------- adaptation trajectories
def fig_traj():
    tracks = [("A", "Schwinger", "schwinger"), ("C", "$\\phi^4$ scalar", "scalar_field"), ("D", "SU(2), $N{=}2$", "su2_lgt"),
              ("E", "Neutrinos", "neutrino"), ("F", "SUSY QM", "susyqm"), ("G", "$\\mathbb{D}_8$", "d8_lgt"),
              ("H", "Nuclei", "nuclear"), ("I", "Yukawa", "yukawa"), ("J", "LMG", "lmg")]
    fig, axs = plt.subplots(3, 3, figsize=(W2, 4.5), gridspec_kw={"hspace": 0.55, "wspace": 0.32, "top": 0.9})
    order = ["zero", "pi", "uniform", "gaussian", "qmaml"]
    for ax, (t, name, path) in zip(axs.ravel(), tracks):
        d = J(path, "adaptation_gaps.json")
        for s in order:
            a = np.array(d[s]); m = a.mean(0); x = np.arange(len(m))
            ax.plot(x, m, color=COL[s], lw=1.5 if s == "qmaml" else 0.9, zorder=3 if s == "qmaml" else 2, label=NAME[s])
        ax.set_title(f"{t}  {name}", loc="left", fontsize=7.2)
        ax.tick_params(labelsize=6)
    for ax in axs[:, 0]: ax.set_ylabel(r"$\log|E-E_0|$", fontsize=7)
    for ax in axs[-1]: ax.set_xlabel("Iteration", fontsize=7)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handlelength=1.6)
    fig.savefig(os.path.join(OUT, "fig_traj.pdf")); plt.close(fig)


# ---------------------------------------------------------------- pretraining diagnostics (SU(2) N=4) and Fisher probe
def fig_diag():
    dg = J("su2_n4_diagnostics.json"); rs = J("su2_n4_restarts.json"); fi = J("barren_plateau", "fisher_info_effective_dim_ext.json")
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.5), gridspec_kw={"hspace": 0.5, "wspace": 0.38})
    SC = {"0": "#0072B2", "1": "#009E73", "2": "#D55E00"}

    ax = axs[0, 0]
    for sd in "012":
        for setting, ls in (("A_40ep_lr0.01", "-"), ("B_100ep_lr0.002", "--")):
            y = np.array(dg[sd]["settings"][setting]["pretrain_loss"])
            ax.plot(np.linspace(0, 1, len(y)), y, color=SC[sd], ls=ls, lw=1.1)
    ax.set_xlabel("Pretraining progress"); ax.set_ylabel("Training meta-loss"); ax.set_ylim(-0.9, 10.5)
    ax.legend(handles=[plt.Line2D([], [], color=SC[s], label=f"seed {s}") for s in "012"] +
              [plt.Line2D([], [], color="k", ls="-", label="40 ep, lr 0.01"), plt.Line2D([], [], color="k", ls="--", label="100 ep, lr 0.002")],
              frameon=False, fontsize=5.6, ncol=2, loc="upper right", handlelength=1.5, columnspacing=0.8, bbox_to_anchor=(1.03, 1.02))
    panel(ax, "a", dx=-0.2)

    ax = axs[0, 1]
    for sd in "012":
        pt = dg[sd]["settings"]["A_40ep_lr0.01"]["per_task"]
        E0 = np.array(dg[sd]["E0"])
        ax.scatter([p["init_E"] for p in pt], [p["final_E"] for p in pt], s=9, color=SC[sd], lw=0, label=f"seed {sd}", zorder=3)
        ax.scatter(E0, E0, s=9, marker="x", color=SC[sd], lw=0.6, zorder=2)
    lim = ax.get_xlim(); ax.plot(lim, lim, color="0.6", lw=0.6, zorder=1); ax.set_xlim(lim); ax.set_ylim(min(lim[0], -0.6), max(lim[1], 1.4))
    ax.set_xlabel("Energy at Q-MAML initialization"); ax.set_ylabel("Energy after adaptation")
    ax.text(0.04, 0.95, "$\\times$ exact ground energy", transform=ax.transAxes, fontsize=5.8, va="top", color="0.35")
    panel(ax, "b", dx=-0.22)

    ax = axs[1, 0]
    pi_e = {"0": 0.042, "2": 0.035}
    for sd_key, m in (("dseed0", "o"), ("dseed2", "s")):
        keys = [k for k in rs if k.startswith(sd_key + "/") and isinstance(rs[k], dict)]
        x = [rs[k]["train_loss_last"] for k in keys]; y = [100 * rs[k]["test_rel_err"] for k in keys]
        ax.scatter(x, y, s=16, marker=m, color=SC["0" if sd_key == "dseed0" else "2"], lw=0, label=f"data seed {sd_key[-1]}", zorder=3)
    ax.axhline(4.2, color=SC["0"], lw=0.7, ls="--"); ax.axhline(3.5, color=SC["2"], lw=0.7, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Final training meta-loss"); ax.set_ylabel("Held-out error (%)")
    ax.text(0.98, 0.06, "dashed: $\\pi$ baseline", transform=ax.transAxes, ha="right", fontsize=5.8, color="0.35")
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", handletextpad=0.2)
    panel(ax, "c", dx=-0.2)

    ax = axs[1, 1]
    nq = sorted(map(int, fi))
    for s, c, lab in (("qmaml", COL["qmaml"], "Q-MAML"), ("uniform", COL["uniform"], "Uniform"), ("gaussian", COL["gaussian"], "Gaussian")):
        m = np.array([np.mean(fi[str(q)][s]) for q in nq]); sd = np.array([np.std(fi[str(q)][s]) for q in nq])
        ax.plot(nq, m, "-o", color=c, ms=2.8, lw=1.2, label=lab); ax.fill_between(nq, m - sd, m + sd, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Qubits"); ax.set_ylabel("Effective dimension / #parameters"); ax.set_ylim(0.6, 1.0); ax.set_xticks(nq)
    ax.legend(frameon=False, fontsize=6, loc="center right")
    panel(ax, "d", dx=-0.3)
    fig.savefig(os.path.join(OUT, "fig_diag.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_systems(); fig_traj(); fig_diag()
    print(sorted(os.listdir(OUT)))
