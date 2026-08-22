"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to the collective neutrino
oscillation ground-state problem (hamiltonians_neutrino.py -- see its docstring for the reformulation
this track required). Mirrors schwinger_qmaml.py / z2lgt_qmaml.py / su2lgt_qmaml.py's structure: task
vector (theta, mu), lightning.qubit + adjoint differentiation (a genuine Pauli-sum Hamiltonian).
"""
import numpy as np
import torch
import torch.nn as nn

from models.learner import Learner
from hamiltonians_neutrino import neutrino_hamiltonian, param_shape, N_NEUTRINOS_DEFAULT

NUM_QUBITS = N_NEUTRINOS_DEFAULT


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


class NeutrinoPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_neutrino(theta_mix,mu)|psi(theta)>, measured directly via qml.expval(H).
    (Naming note: `theta` below is always the PQC's variational weight tensor, never the neutrino mixing
    angle -- the Hamiltonian's mixing angle is always spelled `theta_mix` at call sites in this module to
    avoid the name collision.)"""

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


def pretrain_learner_neutrino(num_qubits, depth, train_tasks, epochs, lr, w0_scale=0.3, log_every=5):
    """Algorithm 1: no inner loop -- backprop straight through Learner(theta_mix,mu) -> PQC cost."""
    shape = param_shape(depth, num_qubits)
    learner = Learner(2, shape, hidden=256).double()  # task descriptor = (theta_mix, mu)
    pqc = NeutrinoPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for theta_mix, mu in train_tasks:
            H = neutrino_hamiltonian(num_qubits, theta_mix, mu)
            opt.zero_grad()
            task_vec = torch.tensor([theta_mix, mu], dtype=torch.float64)
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
            print(f"[Neutrino pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, pqc, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_neutrino(pqc, theta0, H, iters, lr):
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


__all__ = ["classical_init", "NeutrinoPQC", "pretrain_learner_neutrino", "adapt_task_neutrino", "NUM_QUBITS"]
