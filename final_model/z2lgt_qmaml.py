"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to the
single-plaquette Z2 lattice gauge theory (hamiltonians_z2lgt.py). Mirrors
schwinger_qmaml.py's structure exactly, with a 3D task vector (J, m, mu) in
place of Schwinger's (m0, g) and an 8-qubit Hamiltonian in place of 4/6.
"""
import numpy as np
import torch
import torch.nn as nn

from models.learner import Learner
from hamiltonians_z2lgt import z2_lgt_hamiltonian, param_shape, NUM_QUBITS


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


class Z2LGTPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_Z2LGT(J,m,mu)|psi(theta)>, measured directly via qml.expval(H).
    This is the same "naive" ansatz family Track A uses -- deliberately not the
    gauge-invariant Hamiltonian-variational ansatz from arXiv:2507.19203, so the
    Gauss-law penalty term in the Hamiltonian (not the ansatz) is what has to pull
    the optimizer toward the physical subspace. Swapping in a gauge-invariant
    ansatz is exactly the "physics-derived warm start" comparison flagged as
    future work in the companion notebook."""

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3, use_lightning: bool = True):
        super().__init__()
        import pennylane as qml
        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        diff_method = "parameter-shift"
        if use_lightning:
            try:
                self.dev = qml.device("lightning.qubit", wires=num_qubits)
                diff_method = "adjoint"
            except Exception:
                self.dev = qml.device("default.qubit", wires=num_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=num_qubits)
        self._diff_method = diff_method

    def make_circuit(self, H):
        qml = self._qml

        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(H)
        return circuit

    def cost(self, weights: torch.Tensor, H) -> torch.Tensor:
        circuit = self.make_circuit(H)
        return circuit(weights.to(torch.float64))


def pretrain_learner_z2lgt(num_qubits, depth, train_tasks, epochs, lr, w0_scale=0.3, log_every=5,
                            gauss_V=10.0):
    """Algorithm 1: no inner loop -- backprop straight through Learner(J,m,mu) -> PQC cost."""
    shape = param_shape(depth, num_qubits)
    learner = Learner(3, shape, hidden=256).double()  # task descriptor = (J, m, mu)
    pqc = Z2LGTPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for J, m, mu in train_tasks:
            H = z2_lgt_hamiltonian(J, m, mu, V=gauss_V)
            opt.zero_grad()
            task_vec = torch.tensor([J, m, mu], dtype=torch.float64)
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
            print(f"[Z2LGT pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, pqc, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_z2lgt(pqc, theta0, H, iters, lr):
    """Algorithm 2: fine-tune PQC weights only, starting from theta0."""
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


__all__ = ["classical_init", "Z2LGTPQC", "pretrain_learner_z2lgt", "adapt_task_z2lgt"]
