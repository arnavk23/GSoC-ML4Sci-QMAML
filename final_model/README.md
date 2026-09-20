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
schemes agree on the final energy to 4-5 significant figures at every seed) but was by far the slowest to
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
| `run_schwinger_l4_scan.py` | Track A: pretrain-epochs x pretrain-lr grid at L=4/depth=12, plus a depth=18 spot-check | **L=4 failure has two causes**: pretrain lr 0.005 stalls pretraining (fixed at 0.002), and depth 12 is too shallow; depth 18 + lr 0.002 gives qmaml 2.43% vs. pi 8.65% (single seed, reduced protocol) |
| `run_su2_followup.py` | Track D follow-ups, 3 seeds: N=3 at depth 6/300 iters; N=4 with pretrain lr 0.002 / 100 epochs | N=3 converges and `pi` is best at every seed (qmaml second-tier); N=4's seed-unstable qmaml is unchanged by the lower lr (matches the original to 4 digits at 2 of 3 seeds), cause unknown |
| `run_fisher_info_probe.py` | Quantum Fisher information / effective-dimension probe (PennyLane `metric_tensor`, block-diagonal approx) on the VQE-domain ansatz, qubits {4,6,8} | qmaml sits at 93-96% effective dimension vs. 68-73% for classical schemes, at every qubit count -- quantitative content for "landscape placement" |

Two of these (`run_z2lgt_alt_ansatz.py`, `run_su2lgt_N4.py`) materially changed a track's headline verdict
as single-seed results, so `run_verdict_changers_multiseed.py` replicated both at 3 seeds: the Track B fix held at
every seed, while the Track D N=4 "reversal" turned out to be seed-dependent (it is why this replication was run).
