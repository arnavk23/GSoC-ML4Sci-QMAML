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
   controlled experiment.
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
   rather than the mechanism itself failing to scale. `qmaml` still starts and stays ahead for most of the L=3
   trajectory, just not at the very end. A depth/budget re-tune per qubit count, a symmetry-preserving ansatz,
   and a second lattice gauge theory for cross-validation are noted as open follow-ups in its Findings cell.

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
