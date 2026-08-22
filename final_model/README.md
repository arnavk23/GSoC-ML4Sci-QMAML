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
