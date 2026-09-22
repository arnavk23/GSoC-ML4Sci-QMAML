# final_model

Q-MAML implementation and experiments for this project. Two kinds of tasks:

- **Classification**: HIGGS and quark-gluon jet tagging.
- **VQE**: ground-state search on nine HEP-and-adjacent Hamiltonians.

## Layout

```
config.py            shared hyperparameters
models/               Learner, PQC ansatz, CNN/tabular encoders
meta/                 training loops (paper-faithful Algorithm 1/2, and the prior GSoC variant)
data.py, higgs_data.py, higgs_tasks.py, quarkgluon_data.py, tasks.py
                      dataset loading and task sampling for classification
vqe_qmaml.py, vqe_heisenberg.py, vqe_molecule.py
                      VQE calibration against the original Q-MAML paper's settings
results/              saved metrics, plots, checkpoints (one subfolder per experiment)
```

## Hamiltonian tracks

Each track is one `hamiltonians_<name>.py` (Hamiltonian, exact diagonalization, task sampler,
verification) plus one `<name>_qmaml.py` (ansatz, classical baselines, Algorithm 1/2 pretrain and adapt).

| Track | Files | Hamiltonian |
|---|---|---|
| A | `hamiltonians_schwinger.py`, `schwinger_qmaml.py` | Lattice Schwinger model (1+1D QED) |
| B | `hamiltonians_z2lgt.py`, `z2lgt_qmaml.py` | Z2 lattice gauge theory |
| C | `hamiltonians_scalar.py`, `scalar_qmaml.py` | phi^4 scalar field theory |
| D | `hamiltonians_su2lgt.py`, `su2lgt_qmaml.py` | SU(2) lattice gauge theory |
| E | `hamiltonians_neutrino.py`, `neutrino_qmaml.py` | Collective neutrino oscillations (reformulated as a ground-state problem, see the module docstring) |
| F | `hamiltonians_susyqm.py`, `susyqm_qmaml.py` | Supersymmetric quantum mechanics |
| G | `hamiltonians_d8lgt.py`, `d8lgt_qmaml.py` | Non-Abelian D8 lattice gauge theory (original, un-reduced Hamiltonian, see the module docstring) |
| H | `hamiltonians_nuclear.py`, `nuclear_qmaml.py` | Light nuclei (deuteron/triton/helium-3), lattice pionless EFT, UCCSD ansatz |
| I | `hamiltonians_yukawa.py`, `yukawa_qmaml.py` | Scalar Yukawa coupling (fermion-boson), reformulated as a ground-state problem |
| J | `hamiltonians_lmg.py`, `lmg_qmaml.py` | Lipkin-Meshkov-Glick model (nuclear-structure adjacent, not core HEP) |

Every Hamiltonian module has a `verify_construction()` function. Run it before trusting results:

```bash
python hamiltonians_su2lgt.py
```

## Running an experiment

Each track's experiment lives in `research_notebooks/` (Hamiltonian verification, task sampling,
pretraining, adaptation comparison, findings). See `research_notebooks/README.md` for the full list and
results.

## Results

`results/<track>/` holds `adaptation_gaps.json` (raw curves), `adaptation_convergence.png`, and
`pretrain_learner.png` per track. Classification results are under `results/higgs*/` and
`results/quarkgluon/`.

## Multi-seed validation

`multiseed_validate.py` reuses each track's own verified `classical_init`/`pretrain_learner_<name>`/
`adapt_task_<name>` functions (no new modeling choices) to replicate a track at `n` seeds, and to run two
tracks (D, E) at one additional, verify-first qubit count:

```bash
python multiseed_validate.py --track su2lgt          # 3-seed replication (default seeds 0,1,2)
python multiseed_validate.py --track all             # all 7 tracks that were single-seed as of this writing
python multiseed_validate.py --track qubit_ext       # SU(2) LGT at N=3 (6 qubits), neutrino at N=6 (6 qubits)
```

Results land in `results/<track>/multiseed_results.json` ({scheme: [val_seed0, val_seed1, val_seed2]}) and
`results/<track>/adaptation_final_N*.json` for the qubit-count extensions. All ten tracks are now `n=3`-seed
replicated. Two tracks (F, G) revise from a single-seed "win" to a statistical tie with Uniform/Gaussian
once replicated; Track H (nuclear EFT, UCCSD) replicated *more* strongly than any other track (all five
schemes agree on the final energy to within 0.03% at every seed) but was by far the slowest to
run -- one of its three seeds took ~10.6h to pretrain alone, vs. minutes for every other track. See
`research_notebooks/README.md` and `paper/main.tex` \S7 for the full numbers.

## Further future-work scripts (second round)

Standalone scripts, each addressing one specific item from the paper's own Limitations/Conclusion TODO:

| Script | What it does | Headline result |
|---|---|---|
| `run_z2lgt_alt_ansatz.py` | Executes notebook 08's own never-run cells (19-24): gauge-invariant HVA + non-gauge-invariant control ansatz for Track B | **Track B's loss is an ansatz artifact, not a Q-MAML weakness**: qmaml is best at 3/3 seeds on the gauge-invariant ansatz (3x-10x ahead of the best classical scheme), only tied with uniform/gaussian on the control (`run_verdict_changers_multiseed.py z2` adds seeds 1,2) |
| `run_su2lgt_N4.py` | Track D at N=4 (8 qubits), un-reduced Hamiltonian, same method as the N=3 extension | At N=4 qmaml is **seed-unstable** (3 seeds: 23.3% / 2.9% / 450%) while pi is reliable (4.2 / 6.0 / 3.5%); at N=3 with the carried-over depth-4 budget nothing converges (200-500% for every scheme; see `run_su2_followup.py` for the depth-6 rerun). Seed 0 alone had looked like a clear loss (`run_verdict_changers_multiseed.py su2` adds seeds 1,2) |
| `run_susyqm_stratified.py` | Track F with a stratified (3/3/3) held-out split instead of the original random draw | Confirms the multi-seed finding a different way: gaussian edges out qmaml under a fair split |
| `run_lmg_larger_j.py` | Track J at J=3.5 (3 qubits, no Hilbert-space padding needed) | The suggested "larger J" fix doesn't work -- still no scheme separation, revises the diagnosis |
| `run_scalar_phase_multiseed_fixed.py` | Corrected re-run of notebook 09's phase-stratified multiseed cell, which had a real aggregation bug (mixed one seed-mean with twelve raw per-task values, mislabeled "3 seeds") | Clean 3-seed comparison, properly aggregated |
| `run_schwinger_l4_scan.py` | Track A: pretrain-epochs x pretrain-lr grid at L=4/depth=12, plus a depth=18 spot-check | Pretrain lr 0.005 stalls pretraining (fixed at 0.002); at depth 12 the adapted result stays mixed across seeds, but depth 18 gives qmaml the lead at every seed, even at the original lr 0.005 (`run_schwinger_l4_seeds.py` replicates at 3 seeds) -- **depth, not pretraining convergence, is the operative variable** |
| `run_schwinger_l4_seeds.py` | 3-seed replication of the depth-12 vs. depth-18 comparison at L=4 | Depth 18 wins at every seed (2.7-4.6% vs. pi's 8.7-12.4%); depth 12 stays mixed (18.4-21.2% vs. pi's 12.4-25.0%) |
| `run_su2_followup.py` | Track D follow-ups, 3 seeds: N=3 at depth 6/300 iters; N=4 with pretrain lr 0.002 / 100 epochs | N=3 converges and `pi` is best at every seed (qmaml second-tier); N=4's seed-unstable qmaml is unchanged by the lower lr (matches the original to 4 digits at 2 of 3 seeds), cause unknown at the time |
| `run_su2_n4_diagnostics.py` | Full pretraining-loss history and per-task init/final energies for SU(2) N=4, 3 seeds x 2 pretraining settings | Diagnoses the N=4 instability: seed 1 converges to a good solution, seed 2 stalls at a poor one (adaptation cannot move the point at all), seed 0 diverges at the original lr -- "pretraining converged" is not "pretraining worked", and it is visible in the training loss alone |
| `run_su2_n4_restarts.py` | 6 Learner-init restarts per data seed at the original N=4 setting, scored by held-out error vs. final training loss | A final loss near the mean ground energy is always followed by a low error (5/5 restarts, 2.0-3.4%); a stalled loss is a warning but not reliable (one seed's stalled restarts span 2.0-15.1%). Selecting the lowest-loss restart beats `pi` at both seeds tested |
| `run_fisher_info_probe.py`, `run_fisher_info_ext.py` | Quantum Fisher information / effective-dimension probe (PennyLane `metric_tensor`, block-diagonal approx) on the VQE-domain ansatz, qubits {4,6,8,10,12} | qmaml sits at 93-96% effective dimension vs. 68-74% for classical schemes, at every qubit count and the gap does not shrink -- quantitative content for "Q-MAML places the circuit differently, not that it resists plateaus" |
| `run_z2lgt_alt_ansatz.py` follow-up: `run_z2_fullbudget_and_anneal.py full` | Replicates the gauge-invariant vs. control ansatz comparison at the full 6-task/300-iteration budget (the headline runs use a reduced 4-task/200-iteration budget) | Holds: gauge-invariant HVA wins at all 3 seeds (2.5-6x ahead); the control ansatz is a loss at 2/3 seeds and a tie at the third (worse than the reduced-budget "tied" read) |
| `run_z2_fullbudget_and_anneal.py anneal` | Anneals the Gauss-law penalty V during pretraining (1->10 and 30->10 schedules) on the headline hardware-efficient ansatz | Roughly halves qmaml's mean error (49%->22-32%) but beats every classical scheme in only 1 of 6 seed/schedule cells |
| `run_higgs_extra_seeds.py` | Extends the HIGGS classification comparison from 4 to 7-8 seeds per qubit count | qmaml remains the highest mean validation accuracy at 4, 6 and 8 qubits; the margin against Gaussian is significant at every qubit count and against the best classical scheme at 4 and 6 qubits |

Two of these (`run_z2lgt_alt_ansatz.py`, `run_su2lgt_N4.py`) materially changed a track's headline verdict
as single-seed results, so `run_verdict_changers_multiseed.py` replicated both at 3 seeds: the Track B fix held at
every seed, while the Track D N=4 "reversal" turned out to be seed-dependent (it is why this replication was run).
