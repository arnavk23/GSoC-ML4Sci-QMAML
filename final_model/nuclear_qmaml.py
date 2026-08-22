"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to the nuclear lattice EFT
Hamiltonians (hamiltonians_nuclear.py). Unlike every other track in this project, the ansatz here is a
particle-number-conserving Unitary Coupled-Cluster Singles-and-Doubles (UCCSD) circuit
(PennyLane's built-in `qml.UCCSD`), not `StronglyEntanglingLayers` -- the standard choice in the
nuclear/molecular VQE literature, and the "new circuit type" this track's own scope calls for. Proton and
neutron excitations are generated independently (`qml.qchem.excitations` on each species' own spin-orbital
block) and concatenated, so no excitation ever converts a proton mode into a neutron mode or vice versa --
exact particle-number conservation for each species by construction, not by penalty (contrast Track B's
Gauss-law penalty).

Task descriptor mixes a discrete choice (nucleus in {deuteron, triton, helium3}) with continuous
coefficients (t, C, D), one-hot + continuous, exactly Track F's task-space pattern.
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.learner import Learner
from hamiltonians_nuclear import (nuclear_hamiltonian, NUCLEI, NUM_QUBITS, N_PROTON_ORBITALS,
                                   N_NEUTRON_ORBITALS)

NUCLEUS_NAMES = tuple(NUCLEI.keys())
TASK_DIM = len(NUCLEUS_NAMES) + 3  # one-hot(nucleus) + (t, C, D)


def encode_task(nucleus: str, t: float, C: float, D: float) -> torch.Tensor:
    onehot = [1.0 if s == nucleus else 0.0 for s in NUCLEUS_NAMES]
    return torch.tensor(onehot + [t, C, D], dtype=torch.float64)


def _excitations_for(nucleus: str):
    """Singles/doubles excitation wires for the proton block and the neutron block, generated
    independently (so no excitation crosses species), then concatenated. Also returns the Hartree-Fock-
    like reference occupation (lowest orbitals of each species filled)."""
    n_p, n_n = NUCLEI[nucleus]
    s_p, d_p = qml.qchem.excitations(n_p, N_PROTON_ORBITALS)
    s_n, d_n = qml.qchem.excitations(n_n, N_NEUTRON_ORBITALS)
    s_n = [[w + N_PROTON_ORBITALS for w in exc] for exc in s_n]
    d_n = [[w + N_PROTON_ORBITALS for w in exc] for exc in d_n]
    singles = s_p + s_n
    doubles = d_p + d_n
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    init_state = np.zeros(NUM_QUBITS, dtype=int)
    init_state[:n_p] = 1                              # lowest n_p proton orbitals occupied
    init_state[N_PROTON_ORBITALS:N_PROTON_ORBITALS + n_n] = 1  # lowest n_n neutron orbitals occupied
    return s_wires, d_wires, init_state


def n_excitations_for(nucleus: str) -> int:
    s_wires, d_wires, _ = _excitations_for(nucleus)
    return len(s_wires) + len(d_wires)


def classical_init(shape, kind: str, depth: int) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros(shape, dtype=torch.float64)
    if kind == "pi":
        return torch.full(shape, float(np.pi), dtype=torch.float64)
    if kind == "uniform":
        alpha = 0.05
        return (torch.rand(shape, dtype=torch.float64) * 2 - 1) * alpha * np.pi
    if kind == "gaussian":
        S = 2  # per Q-MAML paper's convention (Zhang et al. 2022)
        gamma2 = 1.0 / (4 * S * (depth + 2))
        return torch.normal(0.0, gamma2 ** 0.5, size=shape, dtype=torch.float64)
    raise ValueError(kind)


class NuclearUCCSDPQC(nn.Module):
    """UCCSD ansatz PQC (particle-number-conserving) whose cost is
    <psi(theta)|H_nuclear(t,C,D)|psi(theta)>, measured via qml.Hermitian. One instance is built per
    nucleus (its excitation set and reference state are nucleus-specific)."""

    def __init__(self, nucleus: str, num_qubits: int = NUM_QUBITS, depth: int = 1):
        super().__init__()
        self.nucleus = nucleus
        self.num_qubits = num_qubits
        self.depth = depth
        self.s_wires, self.d_wires, self.init_state = _excitations_for(nucleus)
        self.n_excitations = len(self.s_wires) + len(self.d_wires)
        self.dev = qml.device("default.qubit", wires=num_qubits)

    def make_circuit(self, H_matrix: np.ndarray):
        obs = qml.Hermitian(H_matrix, wires=range(self.num_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(weights):
            qml.UCCSD(weights, wires=range(self.num_qubits), s_wires=self.s_wires, d_wires=self.d_wires,
                      init_state=self.init_state, n_repeats=self.depth)
            return qml.expval(obs)
        return circuit

    def cost(self, weights: torch.Tensor, H_matrix: np.ndarray) -> torch.Tensor:
        circuit = self.make_circuit(H_matrix)
        w = weights.reshape(self.depth, self.n_excitations)
        return circuit(w.to(torch.float64))


def pretrain_learner_nuclear(depth, train_tasks, epochs, lr, w0_scale=0.3, log_every=5):
    """Algorithm 1: no inner loop -- backprop straight through Learner(nucleus,t,C,D) -> PQC cost.
    One PQC (fixed excitation structure) and one shared Learner per nucleus type present in train_tasks,
    since each nucleus has a different number of UCCSD excitations (different weight-tensor shape)."""
    pqcs = {nucleus: NuclearUCCSDPQC(nucleus, depth=depth) for nucleus in NUCLEUS_NAMES}
    learners = {
        nucleus: Learner(TASK_DIM, (depth, pqcs[nucleus].n_excitations), hidden=256).double()
        for nucleus in NUCLEUS_NAMES
    }
    params = [p for learner in learners.values() for p in learner.parameters()]
    opt = torch.optim.Adam(params, lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for nucleus, t, C, D in train_tasks:
            H = nuclear_hamiltonian(nucleus, t, C, D)
            opt.zero_grad()
            task_vec = encode_task(nucleus, t, C, D)
            theta = w0_scale * learners[nucleus](task_vec)
            cost = pqcs[nucleus].cost(theta, H)
            cost.backward()
            gn = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None) ** 0.5
            epoch_grad.append(gn)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            epoch_loss += cost.item()
        loss_hist.append(epoch_loss / len(train_tasks))
        grad_hist.append(float(np.mean(epoch_grad)))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[Nuclear pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learners, pqcs, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_nuclear(pqc, theta0, H, iters, lr):
    """Algorithm 2: fine-tune UCCSD amplitudes only, starting from theta0."""
    theta = nn.Parameter(theta0.clone().detach().double(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    costs = []
    for _ in range(iters):
        opt.zero_grad()
        c = pqc.cost(theta, H)
        c.backward()
        opt.step()
        costs.append(c.item())
    return theta.detach(), costs


__all__ = ["encode_task", "classical_init", "NuclearUCCSDPQC", "pretrain_learner_nuclear",
           "adapt_task_nuclear", "n_excitations_for", "NUCLEUS_NAMES", "TASK_DIM"]
