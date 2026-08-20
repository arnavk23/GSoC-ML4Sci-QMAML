# Research notebooks

Read in this order:

1. **`01_Higgs_QMAML_Paper_Faithful.ipynb`** — quick-validation pass (5 epochs, 1 seed, small task pool) on
   real HIGGS data: classical PQC inits vs. the existing GSoC codebase's `outer_loop_qmaml` vs. our
   paper-faithful `pretrain_learner_paper`. Confirms the harness works end-to-end; validation-accuracy
   numbers here are too noisy to cite in the paper (see its own Findings cell) — use `01b` instead for
   actual results.
2. **`01b_Higgs_QMAML_Paper_Faithful_Scaled.ipynb`** — same comparison, paper-scale (more tasks, 10 epochs,
   4 seeds, qubits 4/6/8, mean ± std plots, Welch's t-test + Mann-Whitney + Cohen's d). **This is the primary
   classification-domain result to cite.** `qmaml_paper` has the highest mean accuracy at every qubit count
   tested, with large effect sizes throughout, and beats `gaussian` init and the prior codebase's own
   `qmaml_existing` with p<0.05 at all three qubit counts. Note: an intermediate 2-seed pass had suggested
   the advantage collapses to parity by 8 qubits (tracking notebook 04's barren-plateau collapse point almost
   too neatly) — running at 4 seeds showed that specific claim was noise, not signal. Kept as an explicit
   correction in the notebook's Findings cell and in `paper/main.tex` Section 6.1, rather than quietly fixed.
3. **`02_VQE_Heisenberg_QMAML_Study.ipynb`** — calibrates our system against the original paper's own
   setting (Heisenberg XYZ VQE, reconstructed ansatz) as ground truth before trusting the classification
   extension. Result: Q-MAML reaches near-converged solution quality in ~250-500 iterations vs. ~1500-2000
   for classical schemes (~4-6x fewer circuit evaluations) — a convergence-*speed* advantage; the final
   asymptotic gap largely closes given the full 2000-iteration budget.
4. **`05_VQE_Molecule_QMAML_Study.ipynb`** — second VQE calibration (H2 bond-length task space), this time
   using the paper's *exact* specified ansatz (no reconstruction caveat) and its full depth-7/2000-iteration
   budget. Strongest result of the whole suite: Q-MAML reaches machine-precision convergence by iteration
   ~100 vs. ~1500-2000 for classical schemes (~15-20x fewer circuit evaluations), with all non-degenerate
   schemes converging to nearly the same final answer.
5. **`03_Barren_Plateau_Scaling.ipynb`** — McClean et al.-style gradient-variance-vs-qubit-count diagnostic
   in the VQE domain. Result: no evidence Q-MAML changes the variance-scaling exponent; whatever advantage
   it has is about landscape placement, not resisting the plateau itself.
6. **`04_Barren_Plateau_Scaling_Higgs.ipynb`** — same diagnostic in the classification domain. Result: a much
   sharper, textbook barren plateau (~8 orders of magnitude, 4→10 qubits) than the VQE domain — classification
   cost functions plateau faster, a real challenge for the whole QMLHEP direction, not specific to Q-MAML.
7. **`06_Inner_Steps_Ablation.ipynb`** — sweeps `INNER_STEPS` from 0 to 20 in the prior codebase's
   `outer_loop_qmaml`, isolating why `qmaml_paper` (0 inner steps, Algorithm 1) beats `qmaml_existing`
   (`INNER_STEPS=5`) in notebook 01b — turns a speculative Discussion-section explanation into an actual
   controlled experiment. **Result: the Discussion's hypothesis is not supported.** Final meta-loss is flat
   across the sweep and final validation accuracy actually *increases* with `INNER_STEPS` (0.536 at 0 to
   0.583 at 20, 2 seeds), the opposite of the predicted "noisier signal degrades performance" story. Gradient
   norm scales up strongly with `INNER_STEPS` (0.13→2.49) and correlates with *better*, not worse, accuracy.
   Also surfaced an open discrepancy: `INNER_STEPS=0` here doesn't closely match notebook 01b's
   `qmaml_paper` numbers (0.68 vs. 0.364 meta-loss) even accounting for the known support/query evaluation
   split difference, meaning the real explanation for `qmaml_paper` beating `qmaml_existing` is still open —
   most likely the evaluation-split difference or another structural difference between the two code paths,
   not `INNER_STEPS` itself. The paper's Discussion section is updated to retract the original speculation
   accordingly.
8. **`07_Schwinger_Model_QMAML.ipynb`** — the first test of Q-MAML on a genuine HEP Hamiltonian (not a
   classification dataset with an HEP label, and not a spin chain/molecule): the lattice Schwinger model
   (1+1D QED), task space `(m0, g)`. At MVP scale (`NUM_QUBITS=4`, L=2) this is the strongest generalization
   result in the suite: the Learner's held-out-task init starts at log-gap -3.29 before any adaptation step
   (better than where every classical scheme *starts*), reaches <1% relative energy error by iteration 4 vs.
   192-266 for `pi`/`gaussian` and never for `uniform` within the 300-iteration budget, and — unlike
   notebooks 02/05 — the final-accuracy gap does not close within the shared budget either. `zero`-init
   reproduces the trivial-fixed-point degeneracy (exactly flat, zero gradient) seen elsewhere in the project.
   **Honest caveat (Section 5, L=3/6 qubits):** repeating the identical protocol at `NUM_QUBITS=6` with the
   same fixed depth=3/300-iteration budget, the advantage narrows and `pi` slightly overtakes `qmaml` by the
   final iteration (32.95% vs. 38.08% relative error) — likely fixed ansatz depth becoming insufficiently
   expressive at the larger Hilbert space (pretrain gradient norm stays healthy, so not a barren plateau)
   rather than the mechanism itself failing to scale. **Confirmed directly, not left as speculation (Section
   6):** re-running L=3 at depth 6 and depth 9 recovers `qmaml`'s lead (2.63% vs. `pi`'s 3.06% at depth 6)
   and then strengthens it sharply (0.05% vs. `pi`'s 0.57%, ~10x better, at depth 9) — the L=3 dip was a
   depth-insufficiency artifact of reusing the L=2 ansatz depth, not a breakdown of the mechanism at larger
   qubit counts. `zero`-init stays at the identical, exactly-flat error at every depth tested (697.64% at
   L=3, all depths), confirming its fixed point is algebraic. **But the depth story doesn't simply
   extrapolate (Section 7, L=4/8 qubits)**: extending the "depth is roughly 1.5x qubits" heuristic from L=3
   to depth 12 at L=4, `qmaml` (31.81% error) loses clearly to `pi` (8.56%) — and this time pretraining
   itself shows real strain (gradient norm does not decrease over 40 epochs, mean energy stalls far from the
   true ground energies), a materially different signature from L=3's clean depth-insufficiency case. The
   honest read: the pretraining budget/learning rate, not just ansatz depth, likely needs to scale with qubit
   count too, and a real depth-x-pretrain-budget scan (not single spot-checks) is needed before drawing any
   qubit-scaling conclusion beyond L=3. A systematic version of that scan, a symmetry-preserving ansatz, and
   cross-referencing notebook 08's Z2 LGT (a genuinely different, negative result — see below) are open
   follow-ups in its Findings cell.

9. **`08_Z2_LGT_QMAML.ipynb`** — Track B: a single-plaquette Z2 lattice gauge theory with dynamical matter
   (8 qubits, task space `(J, m, mu)`), the first genuinely different HEP Hamiltonian family in this project
   (real plaquette/magnetic term, local per-site Gauss's law enforced by an energy penalty rather than
   eliminated analytically). Gauge invariance verified exactly ([H,G_l]=0, G_l²=I to machine precision)
   before trusting the Hamiltonian. **Result is the opposite of notebook 07's**: `qmaml` is the *worst*
   scheme at final accuracy (59.16% relative error vs. 23-25% for `pi`/`uniform`/`gaussian`), nearly frozen
   for the full 300-iteration budget. Diagnosed with a follow-up script, not left as speculation: the
   Learner's init has a small-but-nonzero gradient (5.73) that no learning rate (0.02/0.1/0.3) escapes within
   budget — a genuine landscape plateau, not `zero`-init's exact fixed point. **Leading hypothesis (Gauss-law
   penalty V=10 dominates pretraining) tested directly with a V-sweep (V∈{1,3,10,30}) and largely refuted**:
   `qmaml` plateaus at *every* V tested, not just V=10 — the convergence plot shows it as a flat line in all
   four panels, with no monotonic recovery as V shrinks (V=3 is actually its worst relative showing). The
   mechanism is narrowed to "the Learner reliably lands in flat regions on this Hamiltonian family regardless
   of penalty strength" rather than confirmed as penalty-specific — an honest correction to the original
   diagnosis. The gauge-invariant Hamiltonian-variational ansatz comparison (structural rather than
   penalty-enforced gauge invariance) is now the most promising untried follow-up.
10. **`09_Scalar_Field_QMAML.ipynb`** — Track C: phi^4 lattice scalar field theory (6 qubits, `N_SITES=2`,
    task space `(m0_sq, lambda0)`), the first bosonic Hamiltonian in this project — no gauge field, no
    fermion encoding, and (unlike Tracks A/B) H is a dense field-amplitude-basis matrix measured via
    `qml.Hermitian` rather than a Pauli sum. Construction verified against the source paper's own analytic
    checks (Π eigenvalues match the Eq. 16-17 formula exactly; the reconstructed harmonic-oscillator spectrum
    matches n+1/2 to <2% on the low-lying levels). An initial `[Φ,Π]=i` "bulk" verification attempt gave a
    large, non-shrinking-with-truncation-size residual and looked like a bug — turned out to be the wrong
    diagnostic (targeting the wrong subspace), documented in the module rather than silently dropped.
    **Result: `qmaml` wins cleanly again** — 5.86% final relative energy error, ~3x faster than
    `zero`/`uniform`/`gaussian` (which mostly catch up by iteration 300) and ~6x better than `pi` (36.06%,
    the worst scheme here despite being competitive or best on Tracks A/B — no single classical scheme is
    reliably good across all three HEP families; `qmaml` is). **Also corrects a claim made elsewhere in this
    project**: `zero`-init is *not* stuck here (it descends from log-gap 1.37 to -1.89, tying `qmaml`) — the
    first Hamiltonian in this whole suite where the "exact fixed point at theta=0" pattern (notebooks 02-08)
    doesn't hold, so that pattern is Hamiltonian-structure-dependent, not universal to
    `StronglyEntanglingLayers`. **Phase-stratified generalization test (Section 5), the sharpest follow-up
    this Hamiltonian family enables**: train on one phase (symmetric or broken-symmetry-tending
    $m_0^2$ range), extrapolate to the other. The interpolation-regime advantage does *not* carry over:
    training on the symmetric phase and testing on the broken-symmetry-tending regime, `qmaml` (3.13% error)
    is beaten by `zero`/`uniform`/`gaussian` (1.81% each, tied) and only edges out `pi` (5.04%); in the
    reverse direction every scheme fails badly (85-97% error), `qmaml` included, nobody generalizing well.
    Mechanistically, `qmaml`'s prediction actually starts *behind* the task-independent classical schemes in
    the symmetric→broken direction — a Learner trained only on one phase is miscalibrated, not just
    unhelpful, for a qualitatively different regime it never saw. This qualifies the project's broader
    "qmaml is the only consistently competitive scheme" claim: true for interpolation, not established for
    extrapolation across a genuine phase transition.

**Also attempted and honestly reported as not working**: combining Q-MAML with identity-block
initialization (Grant et al. 2019) to directly counteract the barren-plateau collapse found in notebook 04.
The construction was verified correct (exact identity at initialization, confirmed via autodiff and finite
differences), but produced exactly-zero gradients everywhere rather than the theorem's predicted non-vanishing ones.

## Running

```bash
pip install -r ../requirements.txt
cd research_notebooks
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output <name>.ipynb <name>.ipynb
```

All notebooks cache the HIGGS subsample after first run (`final_model/data_higgs/`) and checkpoint results
to `final_model/results/<name>/raw_results.json` incrementally per cell — safe to interrupt and resume.
