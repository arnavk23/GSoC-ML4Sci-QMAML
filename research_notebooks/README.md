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
   it has is about landscape placement, not resisting the plateau itself. **A second, independent diagnostic
   gives "landscape placement" quantitative teeth** (`final_model/run_fisher_info_probe.py`): the single-
   reference-parameter gradient above can't distinguish "degenerate everywhere" from "degenerate along one
   direction, rich along others," so this computes the quantum Fisher information matrix (PennyLane
   `metric_tensor`, block-diagonal approximation) at sampled parameter points and summarizes it as an
   effective dimension (participation ratio / parameter count). Qubits {4,6,8}, depth 3, Heisenberg XYZ
   ansatz, 8 samples/point. **`qmaml` sits at consistently higher effective dimension (93-96% of max rank)
   than `uniform` (68-69%) or `gaussian` (72-73%) at every qubit count, and the gap doesn't shrink with
   qubit count** — if anything it widens slightly at 8 qubits. By this metric, `qmaml`'s init isn't just
   comparable in gradient magnitude to classical schemes, it sits in a genuinely less-degenerate region of
   parameter space — a distinction the single-coordinate gradient scan alone can't make. Reduced qubit range
   (4-8 vs. 4-14) and sample count (8 vs. 80/point) for tractability, since `metric_tensor` is markedly more
   expensive per sample than one gradient component.
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
   accordingly. **Section 8's multi-seed cells (written but not executed at the time of the initial writeup)
   have since been run**: `qmaml` L=2 final relative error 0.18%±0.13% (3 seeds), vs. `pi` 0.22%±0.11% (a
   close second, not a clean gap), `gaussian` 0.45%±0.19%, `uniform` 1.72%±0.31%, `zero` 311%±63% -- `qmaml`
   wins or ties `pi` at every seed, confirming the L=2 headline is not a single-seed fluke.
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
   follow-ups in its Findings cell. **The L=4 failure was then scanned and resolved**
   (`final_model/run_schwinger_l4_scan.py`; single seed, reduced protocol: 3 held-out tasks, 150 iterations,
   so absolute numbers aren't comparable to the 31.81%/8.56% above, but the 40-epoch row reproduces the
   original pretraining signature: mean energy 7.21, gradient norm 24.6). Two causes, not one: (1) more
   pretrain epochs alone do nothing (24.5% -> 23.9% at 100 epochs, pretraining still stalled); (2) lowering
   the pretrain learning rate 0.005 -> 0.002 fixes pretraining itself (mean energy -1.95, inside the true
   ground-energy range; gradient norm 24 -> 7.9) but only modestly helps the adapted result (21.2%, still
   behind `pi`'s 17.9%); (3) then raising depth 12 -> 18 at the same budget gives `qmaml` 2.43% vs. `pi`
   8.65%, `uniform` 11.0%, `gaussian` 11.8% — the lead is back, 3.6x over `pi`. Caveats: depth 18 was only run
   at lr 0.002 (untested whether lr 0.005 would still fail at depth 18), single seed, reduced protocol.

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
   penalty-enforced gauge invariance) is now the most promising untried follow-up. **3-seed replication
   (Section 6) confirms the negative result is robust, not a single-seed fluke**: `qmaml` mean 49.38%
   (std 13.70%, per-seed 59.16%/30.01%/58.96%) vs. `uniform` 20.34%/`gaussian` 19.43%/`pi` 27.35% (all with
   much tighter std, 3-5%) — `qmaml`'s *best* seed is still worse than every classical scheme's *worst* seed,
   so the ranking holds seed-by-seed, though the exact severity of the gap varies considerably with seed.
   **The gauge-invariant-ansatz follow-up was executed and it works** (`final_model/run_z2lgt_alt_ansatz.py`,
   executing notebook cells 19-24 which had been written but never run): built `Z2LGTHVAPQC` (gauge-invariant
   Hamiltonian-variational ansatz — each layer Trotterizes the *physical* Hamiltonian's own terms, so gauge
   invariance is structural, not penalty-enforced) and `Z2LGTHWEffZZPQC` (a same-complexity control ansatz
   that is *not* gauge-invariant), both starting from the Gauss-law vacuum, single seed, reduced budget (4
   held-out tasks, 200 iterations, documented for tractability), then **replicated at 3 seeds**
   (`final_model/run_verdict_changers_multiseed.py z2`). **On the gauge-invariant HVA, `qmaml` flips from
   worst scheme to best at every one of three seeds**: 0.0022%+/-0.0011% mean relative error vs. 0.011-0.014%
   for every classical scheme; per seed it is 3x-10x ahead of the best classical scheme (0.0031% vs. 0.0148%,
   0.0029% vs. 0.0088%, 0.0007% vs. 0.0073%). Same qualitative pattern as Tracks A and D at N=2, which share
   this track's penalty-free structural gauge invariance (Track G, also gauge-eliminated, replicated only to a
   tie, so the pattern is not universal). **On the control ansatz, `qmaml` is indistinguishable from
   `uniform`/`gaussian` at every seed** (2.58%+/-1.16% each) and `pi` is the control's most variable scheme
   (best at seeds 0 and 2, worst at seed 1) — ruling out "any different ansatz helps". This supports
   structural (not penalty-enforced) gauge invariance as the operative variable, but the two ansatze are
   different published circuits, not a minimal pair, so this is the best-supported reading, not a controlled
   ablation. Still at the reduced budget and one system size.
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
    extrapolation across a genuine phase transition. **3-seed replication of the interpolation-split result
    (Section 6) confirms both headline findings robustly**: `qmaml` 6.05%±1.60%, `zero` 6.12%±1.54% —
    essentially tied at every seed, confirming `zero` genuinely isn't stuck — while `uniform`/`gaussian` sit
    in the same tight ~6-7% cluster and `pi` is a robust, seed-independent outlier at 27.00%±9.09% (even its
    best seed, 14.58%, is ~2x worse than every other scheme's worst seed). **The phase-stratified
    (extrapolation) multiseed cell (Section 6's counterpart for the symmetric/broken-symmetry test, not the
    interpolation-split above) had a real aggregation bug**: seed 0's stored value was one gap correctly
    averaged over all 6 held-out tasks, but seeds 1 and 2 each appended 6 *individual per-task* values
    instead of one seed-level average — so the notebook's own printed "mean ± std over 3 seeds" (n=13:
    1 + 6 + 6) silently mixed one aggregate with twelve raw per-task numbers. Corrected and rerun
    (`final_model/run_scalar_phase_multiseed_fixed.py`, per-seed aggregation applied uniformly): **both
    directions replicate cleanly with proper 3-seed statistics.** Symmetric→broken: `qmaml` 2.23%±0.65%,
    `zero`/`uniform`/`gaussian` a tight 1.78%±0.09% cluster, `pi` worst at 5.65%±0.74% — `qmaml` is clearly,
    robustly *behind* the classical cluster here, confirming the single-seed finding with much tighter error
    bars than the buggy pass gave. Broken→symmetric: every scheme fails badly (75-94%), `uniform` relatively
    best (74.91%±9.44%), `qmaml` mid-pack (91.90%±1.19%, its own tightest spread) — "nobody generalizes well
    here" holds robustly.

11. **`10_QuarkGluon_QMAML.ipynb`** — generalization check for the classification-domain headline result
    (notebook 01b) on a second, physically distinct dataset: quark/gluon jet tagging (jet-substructure
    features, 6 qubits, 2 seeds, 4 variants). **`qmaml_paper` beats every classical scheme by a wide
    margin**: val accuracy 0.776±0.005 vs. 0.594-0.599 for zero/uniform/gaussian (all three clustered
    together, no distinctive `zero`-init behavior here — unlike the VQE-domain fixed-point pattern,
    consistent with the same absence on HIGGS). Meta-loss drops from 0.62 to 0.31 for `qmaml_paper` while
    every classical scheme stays pinned within 0.01 of ln 2 (chance-level) for the full 10 epochs. Gradient
    norm again tracks accuracy, not "moderate is best" (1.452 vs. 0.096-0.205, same wrinkle as HIGGS). Only
    2 seeds (effect size ~6.8 in raw terms, but not a powered significance test) — read as directional
    confirmation that the HIGGS result generalizes, not a sharper result than HIGGS's own 4-seed numbers.
    One reproducibility oddity, not a scientific finding: the `gaussian`/seed-1 run took ~11.8 hours in this
    session's execution vs. ~7 minutes for `qmaml_paper` at the same settings, a large seed-specific
    slowdown in the classical-baseline inner-loop path not seen at seed 0 — flagged for follow-up, not
    investigated further here.
12. **`11_SUSY_QM_QMAML.ipynb`** — Track F: supersymmetric quantum mechanics (4 qubits: 3 boson + 1
    fermion, cutoff Λ=8), task space (superpotential ∈ {HO, AHO, DW}, m, g, μ) — the first track with a
    *discrete* task label (which superpotential) alongside continuous coefficients, encoded as a one-hot +
    (m,g,μ) 6-vector. Construction verified: Hermiticity for all three superpotentials; the harmonic-
    oscillator boson block reproduces the analytic n+1/2 spectrum; the double-well superpotential's
    classical potential genuinely has two minima (the source paper's rendered equation lost a sign in
    extraction — verified directly with the sign that produces an actual double well, not assumed).
    **`qmaml` wins cleanly**: best starting point (log-gap -2.202 vs. 0.622-3.150 for classical schemes)
    and best final value (-4.867 vs. -4.663 to -4.752 for uniform/gaussian, its closest competitors). The
    source paper's own hand-picked basis-state initialization (`basis_paper`) is *not* competitive — worst
    starting point of all six schemes, only partial recovery over 200 iterations, a genuine negative result
    for "physics-derived point-estimate init beats generic random init." `zero`-init drifts slowly rather
    than staying exactly flat, unlike the *exact* fixed point on every Pauli-sum Hamiltonian tested
    (Heisenberg, molecule, Schwinger, Z2 LGT) — and unlike Track C's Hermitian-matrix scalar field, where
    `zero` wasn't stuck at all. Since SUSY QM and the scalar field are both Hermitian-matrix Hamiltonians
    with opposite `zero`-init behavior, whether this fixed point occurs is not simply a function of
    Pauli-sum vs. Hermitian-matrix Hamiltonians, narrowing that claim further. Per-superpotential
    breakdown: `qmaml`'s clearest margin is on the double well, the source paper's own hardest case
    (-5.09 vs. -1.06 to -4.58 for classical schemes); on the anharmonic oscillator, `qmaml`/`uniform`/
    `gaussian` converge to a near-identical final value, consistent with this project's repeated
    "different inits, same final answer, different speed" pattern rather than a permanent-optimum
    advantage there. **Metric caveat surfaced explicitly**: SUSY ground states have exact energy 0 when
    supersymmetry is unbroken (confirmed for the HO case, `E0=0.0000` at every HO test task), which makes
    this project's usual mean-normalized relative-error percentage explode into meaningless values here —
    log-gap is the metric to read for this track. **3-seed replication (`final_model/multiseed_validate.py
    --track susyqm`) softens the "wins cleanly" claim**: mean final log-gap over 3 seeds is `uniform`
    -4.75+/-0.11, `qmaml` -4.48+/-0.57, `gaussian` -4.55+/-0.14 -- uniform's tighter spread actually edges
    out qmaml's mean, driven by one weak qmaml seed (-3.67 vs. its other two seeds' -4.87/-4.90). `qmaml`
    still wins head-to-head in 2 of 3 seeds and stays clearly ahead of `basis_paper`/`pi`/`zero` at every
    seed -- the honest revision is "qmaml/uniform/gaussian form an indistinguishable top cluster," not
    "qmaml wins." **Balancing the held-out split confirms the same conclusion a different way**
    (`final_model/run_susyqm_stratified.py`, exactly 3 HO / 3 AHO / 3 DW instead of the original random 2-4
    each, single seed): `gaussian` -7.99, `qmaml` -7.67, `uniform` -7.52 -- gaussian now edges out qmaml
    slightly, reversing the original top two rather than just narrowing the gap. Per-superpotential, gaussian
    also wins double well under the balanced split (-5.40 vs. qmaml's -3.80), so the original "qmaml's
    clearest margin is on double well" claim doesn't survive a fairer split either. `basis_paper`/`pi` stay
    clearly behind regardless of split, so that part of the finding is robust.

13. **`12_SU2_LGT_QMAML.ipynb`** — Track D: SU(2) lattice gauge theory with dynamical matter (Atas, Zohar
    et al., Nature Communications 2021), N=2 (4 qubits, un-reduced Hamiltonian — does not reproduce the
    paper's own qubit-reduction trick or hadron-mass-ratio calculation, see the notebook's scope note),
    task space (m̃, x) — the most directly "LHC-relevant" physics of any track (the source paper computes
    actual meson/baryon mass ratios). Hamiltonian transcribed directly from the paper's Eqs. 3-6;
    Hermiticity-verified at N=2 and N=3, and the trickiest term (a 4-body color-exchange piece) separately
    cross-checked against an independent dense sigma+/sigma- matrix construction. **The cleanest result in
    this project so far, stronger even than Track A's L=2 case**: `qmaml` reaches the float64
    machine-precision floor (log-gap -17.411, ~0.00% relative error, matching the H2 molecule VQE
    calibration's own floor) while every classical scheme plateaus around log-gap -10 to -11 (0.02-0.04%
    error, visibly still noisy, not still descending). `zero`-init is an exact fixed point, matching every
    other Pauli-sum Hamiltonian in this project. This confirms a mechanistic prediction made before running
    the notebook: structurally this Hamiltonian is closer to Track A (gauge eliminated analytically, no
    penalty term) than Track B (gauge invariance penalty-enforced), and the result matches Track A's
    clean-win pattern rather than Track B's failure -- a second, independent (non-Abelian) gauge theory with
    gauge eliminated analytically again favoring `qmaml` cleanly, strengthening the project's existing
    "penalty-enforced gauge invariance, not gauge theories generically, is what qmaml struggles with"
    hypothesis. **3-seed replication confirms this robustly**: `qmaml` reaches ~1e-7 relative error at
    every seed, every classical scheme is 3-4 orders of magnitude worse at every seed, no seed reverses
    the ranking. **Qubit-count extensions, N=3 (6 qubits, depth 4, 250 iters) and N=4 (8 qubits, depth 6, 300 iters),
    3 seeds each, un-reduced Hamiltonian, Hermiticity-verified first** (`final_model/run_su2lgt_N4.py`,
    `run_verdict_changers_multiseed.py su2`; an earlier version of this entry quoted the N=3 numbers as
    percentages when they were fractions — corrected here). **N=3 is an "everyone fails" regime, not a
    tie**: every scheme is at 200-500% relative error at every seed (`qmaml` 198/257/504%, `gaussian`
    199/266/497%, `pi` 364/265/483%, `uniform` 400/252/494%, `zero` 810/515/1010%), i.e. nothing converges
    at the depth-4/250-iteration budget carried over from N=2 — the comparison is uninformative, not
    evidence init stops mattering. **N=4 is where `qmaml` becomes seed-unstable, not where it cleanly
    loses**: `pi` converges to 4.2/6.0/3.5% (mean 4.6%+/-1.1%, best at 2 of 3 seeds), while `qmaml` is
    23.3/2.9/450% (best at seed 1 by 2x over `pi`, catastrophic at seed 2; median 23.3% still trails `pi`).
    Seed 0 alone would have read as a clear loss and seed 1 alone as a win. Untested: whether N=4's bad
    seeds are pretraining failures fixable by a lower pretrain lr (as at Track A's L=4), and whether N=3 with
    a converging budget restores a margin. Not the source paper's own reduced N=4 hadron-mass-ratio circuit.
    **Follow-ups (`final_model/run_su2_followup.py`, 3 seeds each)**: (i) N=3 at depth 6 / 300 iters: every
    scheme now converges (the depth-4 result was a budget artifact) and `pi` is best at every seed
    (0.28%+/-0.07%), `qmaml` a stable second-tier scheme (2.02%+/-0.39%, 5.7x-8.9x behind `pi`; `uniform`
    11.8%+/-7.7%, `gaussian` 9.8%+/-8.4%). (ii) N=4 with pretrain lr 0.002 and 100 epochs: `qmaml` 312% / 2.9% /
    450% vs. 23.3% / 2.91% / 450.4% at the original setting — the Track A learning-rate fix does not help, and
    seeds 1 and 2 match the original to four digits despite very different pretraining, cause unknown
    (pretraining diagnostics not recorded; a same-basin explanation is untested). Net: `pi` is the reliable
    scheme at N=3/4 and the N=2 advantage does not extend.

14. **`13_Neutrino_QMAML.ipynb`** — Track E: collective (all-to-all) neutrino oscillations, N=4 neutrinos
    (4 qubits), task space (mixing angle θ, interaction strength μ). **Required a genuine reformulation,
    stated explicitly rather than glossed over**: the track's literature precedent (arXiv:2102.12556)
    studies real-time entanglement dynamics from a flavor eigenstate, not a ground-state problem, so this
    notebook asks a different (though physically motivated — the adiabatic/spectral-split literature on
    collective oscillations treats this Hamiltonian's ground state as physically meaningful) question:
    does `qmaml` help a VQE find this Hamiltonian's ground state. Construction verified against two exact
    limits, not assumed: at μ=0 the ground energy must equal −N·ω/2 for any mixing angle (confirmed to
    ~1e-16); at θ=0,μ=0 the ground state must be exactly |11...1⟩ (confirmed, overlap 1.000000). **`qmaml`
    wins clearly — the fourth clean win of five Hamiltonians tested** (final log-gap −14.346 vs. −9.0 to
    −11.4 for classical schemes), though unlike Tracks A/D none of the five schemes have converged by
    iteration 200 — a still-open speed gap, closer to the Heisenberg/molecule pattern than Track A/D's
    permanently-separated curves. **`zero`-init is *not* an exact fixed point here**, unlike every other
    Pauli-sum Hamiltonian tested in this project (Heisenberg, molecule, Schwinger, Z2 LGT, SU(2) LGT) — a
    second, independent data point (after Track F's SUSY QM) against the hypothesis that the fixed point
    tracks Pauli-sum-vs-Hermitian-matrix Hamiltonian type, since this Hamiltonian *is* a Pauli sum. Leading
    hypothesis (untested directly): every other Pauli-sum Hamiltonian here has a purely-Z diagonal/mass
    term, while this one mixes Z and X in its single-particle term whenever θ≠0. This is also the project's
    first genuinely non-local (all-to-all, no lattice structure) Hamiltonian, and locality does not appear
    to be the deciding factor for either the `qmaml` win or the `zero`-fixed-point question, at least on
    this one data point. **3-seed replication complicates the single-seed picture**: `qmaml` wins clearly
    at 2 of 3 seeds (2-3 orders of magnitude better) but is only second-best at the third, behind `pi` --
    consistent with none of the five schemes having converged within the 200-iteration budget, so
    seed-to-seed variation in descent progress can reorder the final ranking. **New qubit-count extension,
    N=6 (6 qubits, single seed, both exact limits re-verified)**: here the win is unambiguous and *larger*
    than at N=4 (`qmaml` 1.1e-6 vs. `pi` 1.5e-4, `uniform` 3.3e-4, `gaussian` 7.6e-4, `zero` 1.58%),
    suggesting the seed-1 ambiguity at N=4 is a slow-convergence artifact at that smaller size rather than
    a sign the advantage weakens with scale.

15. **`14_D8_LGT_QMAML.ipynb`** — Track G: non-Abelian D8 (dihedral group of order 8) lattice gauge
    theory (Gaz, Popov, Pardo, Lewenstein, Hauke & Zohar, arXiv:2501.17863), the one track this project's
    own tracks document flagged "not recommended for now" (qudit hardware, ansatz not clearly specified).
    That flag was checked, not assumed: the paper's own qubit-reduced ("matter removal") Hamiltonian, the
    version they actually run on trapped-ion qudits, could not be extracted with full confidence (several
    intermediate equations for their smallest N=4 example were only partially recoverable). This notebook
    instead implements the paper's ORIGINAL, fully-specified Hamiltonian (their Eqs. 1-24) directly with
    explicit Jordan-Wigner fermionic matter, N=2 (1 link, 7 qubits), trading qubit count for full
    verifiability. **Six construction checks, the most of any track**, including an independent
    representation-theory check (D²(g₁)D²(g₂)=D²(g₁·g₂) for all 64 element pairs) that **caught and fixed a
    real bug** (a matrix-multiplication order error producing the reversed group action) before the
    notebook was first run — the kind of error that produces a Hermitian, plausible-looking, but physically
    wrong Hamiltonian, and exactly why that check was included. **`qmaml` wins**, though more modestly than
    most tracks: final log-gap -1.246 (10.49% error) vs. gaussian -1.202 (10.97%) and uniform -1.123
    (11.86%) — a fairly tight cluster at the *final* value — with the clearer advantage in convergence
    *speed* (flattens by iteration ~30-40 vs. ~75-100 for uniform/gaussian). `pi` plateaus early (-0.34);
    `zero` is again an exact, unmoving fixed point despite this being a dense Hermitian-matrix Hamiltonian,
    not a Pauli sum — a third distinct `zero`-init outcome among this project's Hermitian-matrix
    Hamiltonians (no fixed point on the scalar field, a slow drift on SUSY QM, an exact fixed point here),
    reinforcing that no clean Pauli-sum-vs-Hermitian-matrix rule explains this pattern across any track
    tested. No plaquette term exists in a 1D chain at any N. **3-seed replication turns the "modest win"
    into a tie**: mean final relative error `qmaml` 10.8%±3.7% vs. `uniform` 10.7%±3.2% -- indistinguishable
    given the spread, with the two swapping places seed-to-seed. `gaussian` 12.1%±5.7% trails slightly;
    `pi` 21.4%±4.0% and `zero` 93.9%±1.2% (consistently, catastrophically worst) stay clearly behind at
    every seed. Honest finding: a tight three-way top cluster (qmaml/uniform/gaussian), not a clean qmaml
    win.

16. **`15_Nuclear_EFT_QMAML.ipynb`** — Track H: light nuclei (deuteron/triton/helium-3) in lattice
    pionless EFT (Cifci, Akkoyun & La Ronde, arXiv:2604.20908) — the best-precedented VQE-for-physics
    track in this project (lineage of the 2018 IBM deuteron result). N_SITES=2 for all three nuclei (8
    qubits: 4 proton + 4 neutron spin-orbitals). **First new ansatz family in this project**: a
    particle-number-conserving UCCSD circuit (PennyLane's built-in `qml.UCCSD`), not
    `StronglyEntanglingLayers` — proton and neutron excitations generated independently so no excitation
    ever crosses species. Construction verified: Hermiticity; N_p and N_n each exactly commute with H
    (no penalty term needed, unlike Track B's Gauss-law penalty, since particle number is a genuine
    symmetry of this Hamiltonian). **Unplanned physics sanity check**: helium-3 came out less bound than
    triton at matched couplings, matching the real experimental fact (driven by Coulomb repulsion between
    He-3's two protons) — this fell out of the construction, it wasn't designed in. **Every scheme
    converges to the identical final energy** (0.89% relative error, to 3 decimal places, for all six
    schemes tested) — a first in this project; UCCSD's structure (2-4 variational amplitudes vs. 48 angles
    for the generic ansatz) appears simple enough that init stops mattering for the final answer. The
    advantage is entirely in convergence *shape*: `qmaml` descends smoothly from iteration 0, while every
    classical scheme collapses onto one shared, strongly oscillatory trajectory (ringing for ~80
    iterations) before damping into the same final value. `qmaml` also modestly beats the source paper's
    own non-learned approach — a `warm_start_nearby` baseline (adapting from a converged nearby-task
    solution, approximating their "warm-started from a nearby statevector-simulator solution") starts and
    stays slightly behind `qmaml` throughout. **3-seed replication is the strongest confirmation in this
    project**: all five schemes agree on the final relative error to within 0.03% at every seed
    (0.885% / 1.409% / 2.025% for seeds 0/1/2, <0.001% spread across schemes within a seed) — the
    identical-final-value finding is a structural property of the particle-conserving UCCSD ansatz, not a
    single-seed coincidence. **Reproducibility oddity, flagged not hidden**: seed 2's pretraining took
    ~10.6 hours vs. 1010s/473s for seeds 0/1 on identical hardware/hyperparameters — a second instance of
    the large, unexplained per-seed slowdown already noted for notebook 10's quark-gluon run.
17. **`16_Yukawa_QMAML.ipynb`** — Track I: scalar Yukawa coupling, single-site (Kaldenbach, Heller, Alber
    & Stojanovic, arXiv:2211.02684), reformulated from the source paper's real-time-quench-dynamics
    question into a ground-state VQE target — a simpler reformulation than Track E's, since the same
    Hamiltonian (their Eqs. 6-7, quoted directly) already has a closed-form analytic cross-check available:
    fixing the fermion sector to vacuum reduces H to a displaced harmonic oscillator with known ground
    energy $E_0=-(\\eta/2)^2/m$. 4 qubits (2 fermion + 2 boson, Λ=4, matching the source paper's own "up to
    three bosons" showcase scale). Verified: numeric ground energy approaches the analytic formula within
    <1% at small $\\eta/2m$ (with the error's growth beyond that shown directly, not hidden); the ground
    state's measured fermion occupation is exactly 0 (vacuum sector really is the global minimum).
    **`qmaml` and `zero`-init tie for best**, both reaching the float64 precision floor (~0.00% error) —
    a first in this project, since `zero` was an exact fixed point on every Pauli-sum Hamiltonian tested
    and ranged from unstuck to slow-drift to exact-fixed-point on Hermitian-matrix ones elsewhere.
    Mechanistic explanation, not just observation: the task range was deliberately narrowed to keep the
    displacement small, so every ground state here is near-vacuum — and `zero`-init's circuit starts at
    exactly $|0000\\rangle$, already close for small displacements, plausibly explaining both why `zero` is
    competitive and why `qmaml` (trained on the same narrow task distribution) learns something similar.
    `pi`-init is a clear loser again, as on Track C's scalar field (the two Hermitian-matrix Hamiltonians
    where `pi` fails badly, despite strength on the Pauli-sum gauge-theory tracks). **3-seed replication
    confirms the tie**: `qmaml` (2.55-2.71e-7) and `zero` (2.24-2.99e-7) stay tied at the precision floor
    with overlapping ranges at every seed; `pi` is worst at every seed (0.55-2.42%); `uniform`/`gaussian`
    sit consistently 1-2 orders of magnitude behind `qmaml`/`zero` but far ahead of `pi`. The near-vacuum
    mechanism proposed above holds up seed-to-seed, not just at the one seed originally reported.

18. **`17_LMG_QMAML.ipynb`** — Track J: Lipkin-Meshkov-Glick model, flagged in this project's own tracks
    document as nuclear-structure-adjacent rather than core HEP, included for completeness. Precisely
    differentiated from Robin & Savage's "HL-VQE" (arXiv:2301.05976) — despite the similar name, HL-VQE
    iteratively co-optimizes an *effective Hamiltonian* via orbital rotation, not a neural-network-learned
    initialization; this notebook runs plain Q-MAML on their own un-rotated Hamiltonian (Eq. 1), $J=3/2$
    (2 qubits, 4 states — matching their own smallest/showcase scale). Construction verified via the exact
    su(2) algebra relations on the constructed generators and the analytically-known $V=0$ limit.
    **Genuine null result, not a close call**: all five schemes (`zero`/`pi`/`uniform`/`qmaml`/`gaussian`)
    reach the float64 precision floor, and the ranking among them is within run-to-run noise — the
    convergence plot shows all five trajectories running together in one noisy bundle for the entire
    150-iteration budget, never separating, unlike every other track in this project. Likely explanation:
    4 states on 2 qubits may be too small a Hilbert space for a bad initialization to get meaningfully
    stuck. The clear next step is a **larger $J$** (more qubits), not more seeds — this benchmark needs to
    get harder before an initialization-scheme comparison here is informative. **3-seed replication
    confirms the null result rather than resolving it**: every scheme at every seed lands in the
    6e-9-6e-8 range, tighter than the schemes' own seed-to-seed spread -- init scheme explains less
    variance than random seed does at this scale. **The larger-$J$ next step was tried and it does not fix
    the benchmark** (`final_model/run_lmg_larger_j.py`, $J{=}3.5$ -- $2J{+}1{=}8$ states, 3 qubits, no
    Hilbert-space padding needed, matching the original $J{=}1.5$ choice's own "no padding needed" reasoning;
    Hermiticity/su(2)-algebra/$V{=}0$-limit re-verified before running; 3 seeds): every scheme still lands
    at the float64 precision floor at every seed, no separation. This revises the diagnosis, not just the
    qubit count: doubling the Hilbert space (4 states -> 8 states) was expected to give a bad initialization
    room to get stuck, and it didn't, so "too small a Hilbert space" is not the operative explanation.
    Whether even larger $J$ would eventually differentiate schemes, or this task family is just easy for VQE
    regardless of size, is now the open question.

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
