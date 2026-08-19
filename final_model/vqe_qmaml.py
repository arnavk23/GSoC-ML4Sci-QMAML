"""
Q-MAML Algorithm 1 (pre-training) / Algorithm 2 (adaptation) applied to the
Heisenberg XYZ VQE task space (vqe_heisenberg.py) -- this reproduces the paper's
OWN experimental setting (Lee, Cho & Kim 2025), as a correctness check on our
reimplementation (meta/qmaml_paper_loops.py uses the same algorithm, applied
instead to HEP classification in research_notebooks/01_Higgs_*).
"""
import numpy as np
import torch
import torch.nn as nn

from models.learner import Learner
from vqe_heisenberg import HeisenbergPQC, param_shape


def classical_init(shape, kind: str, depth: int) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros(shape, dtype=torch.float64)
    if kind == "pi":
        return torch.full(shape, float(np.pi), dtype=torch.float64)
    if kind == "uniform":
        alpha = 0.05
        return (torch.rand(shape, dtype=torch.float64) * 2 - 1) * alpha * np.pi
    if kind == "gaussian":
        S = 2  # per paper's Heisenberg experiment setting
        gamma2 = 1.0 / (4 * S * (depth + 2))
        return torch.normal(0.0, gamma2 ** 0.5, size=shape, dtype=torch.float64)
    raise ValueError(kind)


def pretrain_learner_vqe(num_qubits, depth, train_tasks, epochs, lr, w0_scale=1.0, log_every=5):
    """Algorithm 1: no inner loop -- backprop straight through Learner(J) -> PQC cost(J)."""
    shape = param_shape(num_qubits, depth)
    learner = Learner(3, shape, hidden=256).double()
    pqc = HeisenbergPQC(num_qubits, depth)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_hist, grad_hist = [], []
    for epoch in range(epochs):
        epoch_loss, epoch_grad = 0.0, []
        for Jx, Jy, Jz in train_tasks:
            opt.zero_grad()
            emb = torch.tensor([Jx, Jy, Jz], dtype=torch.float64)
            theta = w0_scale * learner(emb)
            cost = pqc.cost(theta, Jx, Jy, Jz)
            cost.backward()
            gn = sum(float(p.grad.detach().pow(2).sum()) for p in learner.parameters() if p.grad is not None) ** 0.5
            epoch_grad.append(gn)
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
            opt.step()
            epoch_loss += cost.item()
        loss_hist.append(epoch_loss / len(train_tasks))
        grad_hist.append(float(np.mean(epoch_grad)))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[VQE pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, pqc, {"loss": loss_hist, "grad_norm": grad_hist}


def adapt_task_vqe(pqc: HeisenbergPQC, theta0: torch.Tensor, Jx, Jy, Jz, iters, lr):
    """Algorithm 2: fine-tune PQC weights only, starting from theta0."""
    theta = nn.Parameter(theta0.clone().detach().double(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    costs = []
    for _ in range(iters):
        opt.zero_grad()
        c = pqc.cost(theta, Jx, Jy, Jz)
        c.backward()
        opt.step()
        costs.append(c.item())
    return theta.detach(), costs


__all__ = ["classical_init", "pretrain_learner_vqe", "adapt_task_vqe"]
