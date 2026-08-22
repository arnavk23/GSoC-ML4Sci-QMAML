"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to the Lipkin-Meshkov-Glick model
(hamiltonians_lmg.py). Mirrors scalar_qmaml.py / susyqm_qmaml.py's structure (H is a dense Hermitian
matrix in the J,M basis, not a Pauli sum; default.qubit + backprop). Does not reimplement HL-VQE's own
orbital-rotation learning step (see hamiltonians_lmg.py's docstring on the naming collision) -- only the
plain, un-rotated LMG Hamiltonian is used here, with this project's usual classical baselines
(zero/pi/uniform/Gaussian) as the comparison set. A literal numeric comparison against HL-VQE's own
reported energies is future work (see the companion notebook's Findings).
"""
import numpy as np
import torch
import torch.nn as nn

from models.learner import Learner
from hamiltonians_lmg import lmg_hamiltonian, param_shape, NUM_QUBITS, J_SPIN


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


class LMGPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_LMG(epsilon,V)|psi(theta)>, measured via qml.Hermitian."""

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3):
        super().__init__()
        import pennylane as qml
        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self._diff_method = "backprop"

    def make_circuit(self, H_matrix: np.ndarray):
        qml = self._qml
        obs = qml.Hermitian(H_matrix, wires=range(self.num_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(obs)
        return circuit

    def cost(self, weights: torch.Tensor, H_matrix: np.ndarray) -> torch.Tensor:
        circuit = self.make_circuit(H_matrix)
        return circuit(weights.to(torch.float64))


def pretrain_learner_lmg(num_qubits, depth, train_tasks, epochs, lr, w0_scale=0.3, log_every=5):
    """Algorithm 1: no inner loop -- backprop straight through Learner(epsilon,V) -> PQC cost."""
    shape = param_shape(depth, num_qubits)
    learner = Learner(2, shape, hidden=256).double()  # task descriptor = (epsilon, V)
    pqc = LMGPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for epsilon, V in train_tasks:
            H = lmg_hamiltonian(epsilon, V)
            opt.zero_grad()
            task_vec = torch.tensor([epsilon, V], dtype=torch.float64)
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
            print(f"[LMG pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, pqc, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_lmg(pqc, theta0, H, iters, lr):
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


__all__ = ["classical_init", "LMGPQC", "pretrain_learner_lmg", "adapt_task_lmg", "NUM_QUBITS"]
