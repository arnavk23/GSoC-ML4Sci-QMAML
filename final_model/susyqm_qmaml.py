"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to SUSY QM
(hamiltonians_susyqm.py). Mirrors scalar_qmaml.py's structure (H is a dense Hermitian matrix, measured
via qml.Hermitian, default.qubit + backprop rather than lightning/adjoint), with one addition specific
to this track: the task descriptor mixes a discrete choice (which superpotential) with continuous
coefficients (m, g, mu), encoded as a one-hot-plus-continuous 6-vector
[is_HO, is_AHO, is_DW, m, g, mu] fed to the Learner.
"""
import numpy as np
import torch
import torch.nn as nn

from models.learner import Learner
from hamiltonians_susyqm import susyqm_hamiltonian, param_shape, N_B_DEFAULT, N_FERMION_QUBITS, SUPERPOTENTIALS

NUM_QUBITS = N_B_DEFAULT + N_FERMION_QUBITS
TASK_DIM = len(SUPERPOTENTIALS) + 3  # one-hot(superpotential) + (m, g, mu)


def encode_task(superpotential: str, m: float, g: float, mu: float) -> torch.Tensor:
    onehot = [1.0 if s == superpotential else 0.0 for s in SUPERPOTENTIALS]
    return torch.tensor(onehot + [m, g, mu], dtype=torch.float64)


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
    if kind == "basis_paper":
        # The source paper's own hand-picked basis-state baseline: |1>|0...0> (fermion occupied) for
        # HO/AHO, |0>|0...0> (fermion vacuum) for DW, chosen by inspecting eigenvector overlap. We
        # approximate "start in basis state |b>" with a fixed *classical* weight vector, exactly as
        # Track B's gauss_vacuum baseline does: zero variational angles, paired with a state-prep flag
        # handled by the PQC class (see SusyQMPQC.cost's `fermion_occupied` kwarg).
        return torch.zeros(shape, dtype=torch.float64)
    raise ValueError(kind)


class SusyQMPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_SUSY(superpotential,m,g,mu)|psi(theta)>, measured via qml.Hermitian since H is a dense
    Fock-basis-times-fermion matrix, not a Pauli sum."""

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3):
        super().__init__()
        import pennylane as qml
        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self._diff_method = "backprop"

    def make_circuit(self, H_matrix: np.ndarray, fermion_occupied: bool = False):
        qml = self._qml
        obs = qml.Hermitian(H_matrix, wires=range(self.num_qubits))
        fermion_wire = self.num_qubits - 1  # last qubit = fermionic mode (see hamiltonians_susyqm.py)

        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            if fermion_occupied:
                qml.PauliX(fermion_wire)
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(obs)
        return circuit

    def cost(self, weights: torch.Tensor, H_matrix: np.ndarray, fermion_occupied: bool = False) -> torch.Tensor:
        circuit = self.make_circuit(H_matrix, fermion_occupied=fermion_occupied)
        return circuit(weights.to(torch.float64))


def pretrain_learner_susyqm(num_qubits, depth, train_tasks, epochs, lr, w0_scale=0.3, log_every=5):
    """Algorithm 1: no inner loop -- backprop straight through Learner(task) -> PQC cost.
    train_tasks: list of (superpotential, m, g, mu) tuples."""
    shape = param_shape(depth, num_qubits)
    learner = Learner(TASK_DIM, shape, hidden=256).double()
    pqc = SusyQMPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for sp, m, g, mu in train_tasks:
            H = susyqm_hamiltonian(sp, n_b=num_qubits - N_FERMION_QUBITS, m=m, g=g, mu=mu)
            opt.zero_grad()
            task_vec = encode_task(sp, m, g, mu)
            theta = w0_scale * learner(task_vec)
            cost = pqc.cost(theta, H)
            cost.backward()
            gn = sum(float(p.grad.detach().pow(2).sum()) for p in learner.parameters() if p.grad is not None) ** 0.5
            epoch_grad.append(gn)
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
            opt.step()
            epoch_loss += cost.item()
        loss_hist.append(epoch_loss / len(train_tasks))
        grad_hist.append(float(np.mean(epoch_grad)))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[SUSY-QM pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, pqc, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_susyqm(pqc, theta0, H, iters, lr, fermion_occupied: bool = False):
    """Algorithm 2: fine-tune PQC weights only, starting from theta0."""
    theta = nn.Parameter(theta0.clone().detach().double(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    costs = []
    for _ in range(iters):
        opt.zero_grad()
        c = pqc.cost(theta, H, fermion_occupied=fermion_occupied)
        c.backward()
        opt.step()
        costs.append(c.item())
    return theta.detach(), costs


__all__ = ["encode_task", "classical_init", "SusyQMPQC", "pretrain_learner_susyqm", "adapt_task_susyqm",
           "NUM_QUBITS", "TASK_DIM"]
