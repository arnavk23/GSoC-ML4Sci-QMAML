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
from hamiltonians_z2lgt import (z2_lgt_hamiltonian, param_shape, NUM_QUBITS, MATTER, LINKS,
                                 M0, L01, M1, L12, M2, L23, M3, L30, vacuum_prep as _apply_vacuum_prep)


def classical_init(shape, kind: str, depth: int) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros(shape, dtype=torch.float64)
    if kind == "gauss_vacuum":
        # Physics-derived warm start (arXiv:2507.19203): the *state* (see
        # hamiltonians_z2lgt.vacuum_prep), not theta, carries the physical
        # information -- zero variational angles on top of it, exactly parallel
        # to how "zero" is zero angles on top of the trivial |0...0> state.
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
        # Separate device/diff-method for the vacuum_prep=True path: lightning+adjoint
        # fails to decompose this Hamiltonian's 3-body Gauss-penalty terms once a
        # Hadamard/PauliX precedes the ansatz (an observed lightning quirk, not a
        # physics issue) -- default.qubit+backprop is exact and, at 8 qubits, fast.
        self._dev_default = qml.device("default.qubit", wires=num_qubits)

    def make_circuit(self, H, vacuum_prep: bool = False):
        qml = self._qml
        dev = self._dev_default if vacuum_prep else self.dev
        diff_method = "backprop" if vacuum_prep else self._diff_method

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(weights):
            if vacuum_prep:
                _apply_vacuum_prep()
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(H)
        return circuit

    def cost(self, weights: torch.Tensor, H, vacuum_prep: bool = False) -> torch.Tensor:
        circuit = self.make_circuit(H, vacuum_prep=vacuum_prep)
        return circuit(weights.to(torch.float64))


class Z2LGTHVAPQC(nn.Module):
    """Gauge-invariant Hamiltonian-variational ansatz (arXiv:2507.19203). Each layer
    applies exp(-i*theta_k/2 * P_k) (via qml.PauliRot) for every individual Pauli
    term P_k in the *physical* Hamiltonian -- hopping XZX and YZY per bond, staggered
    mass Z per matter site, electric X per link, magnetic ZZZZ over the plaquette --
    but NOT the Gauss-law penalty term. Gauge invariance is therefore structural
    (every P_k is a term of the gauge-invariant physical Hamiltonian, hence commutes
    with every G_l) rather than penalty-enforced as in Z2LGTPQC. Always starts from
    the Gauss-law vacuum (vacuum_prep) -- there is no meaningful "non-vacuum" version
    of this ansatz, since its whole point is exact gauge invariance from the vacuum
    up.

    Parameter count: our open-boundary single-plaquette Hamiltonian (3 hopping bonds,
    the documented MVP simplification in hamiltonians_z2lgt.py, vs. the paper's 4
    periodic bonds) gives 3 XZX + 3 YZY + 4 Z + 4 X + 1 ZZZZ = 15 params/layer, exactly
    2 fewer than the paper's reported 17/layer for its full periodic geometry -- the
    missing bond's 2 Pauli terms, a useful consistency check on this construction."""

    N_PARAMS_PER_LAYER = 15
    _HOPPING_BONDS = [(M0, L01, M1), (M1, L12, M2), (M2, L23, M3)]

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3):
        super().__init__()
        import pennylane as qml
        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.shape = (depth, self.N_PARAMS_PER_LAYER)

    def make_circuit(self, H):
        qml = self._qml

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(weights):
            _apply_vacuum_prep()
            for layer in range(weights.shape[0]):
                idx = 0
                for (i, link, j) in self._HOPPING_BONDS:
                    qml.PauliRot(weights[layer, idx], "XZX", wires=[i, link, j]); idx += 1
                for (i, link, j) in self._HOPPING_BONDS:
                    qml.PauliRot(weights[layer, idx], "YZY", wires=[i, link, j]); idx += 1
                for site in MATTER:
                    qml.PauliRot(weights[layer, idx], "Z", wires=[site]); idx += 1
                for link in LINKS:
                    qml.PauliRot(weights[layer, idx], "X", wires=[link]); idx += 1
                qml.PauliRot(weights[layer, idx], "ZZZZ", wires=LINKS); idx += 1
            return qml.expval(H)
        return circuit

    def cost(self, weights: torch.Tensor, H, vacuum_prep: bool = None) -> torch.Tensor:
        circuit = self.make_circuit(H)
        return circuit(weights.to(torch.float64))


class Z2LGTHWEffZZPQC(nn.Module):
    """Hardware-efficient 'ZZ' ansatz, arXiv:2507.19203's other ready-made ansatz.
    Replaces the physical multi-body Pauli terms with generic multi-qubit Z-rotations
    (qml.MultiRZ = exp(-i*theta/2 * Z^{\\otimes k})) at the same qubit positions the
    Hamiltonian's 3-/4-body terms occupy (hopping-bond triples, the plaquette's four
    links), plus a single-qubit RX on every link qubit ("gauge field") and a free
    single-qubit rotation (qml.Rot, 3 params) on every matter qubit ("matter qubits
    have the freedom of X, Y, Z rotations", per the paper). NOT gauge-invariant by
    construction (unlike Z2LGTHVAPQC) -- "hardware-efficient" names a circuit that's
    easy to implement, not one that respects the symmetry structurally; only the
    shared vacuum_prep starting state is physical here.

    Parameter count: 3 hopping-triple ZZZ + 1 plaquette ZZZZ + 4 link RX + 4*3 matter
    Rot = 20 params/layer, one fewer triple (20 vs. the paper's 21) for the same
    open-boundary reason as Z2LGTHVAPQC's 15-vs-17."""

    N_PARAMS_PER_LAYER = 20
    _HOPPING_TRIPLES = [(M0, L01, M1), (M1, L12, M2), (M2, L23, M3)]

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3):
        super().__init__()
        import pennylane as qml
        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.shape = (depth, self.N_PARAMS_PER_LAYER)

    def make_circuit(self, H):
        qml = self._qml

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(weights):
            _apply_vacuum_prep()
            for layer in range(weights.shape[0]):
                idx = 0
                for triple in self._HOPPING_TRIPLES:
                    qml.MultiRZ(weights[layer, idx], wires=list(triple)); idx += 1
                qml.MultiRZ(weights[layer, idx], wires=LINKS); idx += 1
                for link in LINKS:
                    qml.RX(weights[layer, idx], wires=link); idx += 1
                for site in MATTER:
                    qml.Rot(weights[layer, idx], weights[layer, idx + 1], weights[layer, idx + 2], wires=site)
                    idx += 3
            return qml.expval(H)
        return circuit

    def cost(self, weights: torch.Tensor, H, vacuum_prep: bool = None) -> torch.Tensor:
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


def adapt_task_z2lgt(pqc, theta0, H, iters, lr, vacuum_prep: bool = False):
    """Algorithm 2: fine-tune PQC weights only, starting from theta0. `vacuum_prep`
    is only meaningful for Z2LGTPQC (the other ansatz classes ignore it -- they are
    always vacuum-prepped internally)."""
    theta = nn.Parameter(theta0.clone().detach().double(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    costs = []
    for _ in range(iters):
        opt.zero_grad()
        c = pqc.cost(theta, H, vacuum_prep=vacuum_prep)
        c.backward()
        opt.step()
        costs.append(c.item())
    return theta.detach(), costs


def pretrain_learner_z2lgt_ansatz(pqc, task_dim: int, train_tasks, epochs, lr, w0_scale=0.3,
                                   log_every=5, gauss_V=10.0, tag="ansatz"):
    """Algorithm 1, generalized to any ansatz object exposing `.shape` (weight tensor
    shape) and `.cost(weights, H)` -- i.e. Z2LGTHVAPQC or Z2LGTHWEffZZPQC in place of
    Z2LGTPQC. Same loop body as pretrain_learner_z2lgt, parameterized over the ansatz."""
    learner = Learner(task_dim, pqc.shape, hidden=256).double()
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
            print(f"[Z2LGT-{tag} pretrain] epoch {epoch+1}/{epochs}  meanE={loss_hist[-1]:.4f}  gradnorm={grad_hist[-1]:.4f}")
    return learner, {"loss": loss_hist, "grad_norm": grad_hist}


__all__ = ["classical_init", "Z2LGTPQC", "Z2LGTHVAPQC", "Z2LGTHWEffZZPQC",
           "pretrain_learner_z2lgt", "pretrain_learner_z2lgt_ansatz", "adapt_task_z2lgt"]
