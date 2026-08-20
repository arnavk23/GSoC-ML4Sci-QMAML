"""
Track B: Z2 lattice gauge theory with dynamical matter, single-plaquette instance
(the minimal unit of the two-leg-ladder geometry used in arXiv:2507.19203's
barren-plateau-free Z2 LGT study). This is a genuinely different HEP Hamiltonian
family from the Schwinger model (hamiltonians_schwinger.py): it has a real
plaquette/magnetic term (no plaquettes exist in 1+1D), a per-site local gauge
symmetry (Gauss's law) rather than the Schwinger model's single global U(1)
constraint eliminated analytically, and a 3D task-parameter space (J, m, mu)
instead of 2D (m0, g).

General Hamiltonian (arXiv:2507.19203, Eq. 1-2, general multi-plaquette form):
  H = Hg + Hm + V * sum_l (G_l - q_l)^2
  Hg = -mu * sum_{l,k} X_{l,l+k}  -  sum_p Z Z Z Z (plaquette)
  Hm = J * sum_{l,k} (phi_l^dag Z_{l,l+k} phi_{l+k} + h.c.) + m * sum_l (-1)^(l1+l2) phi_l^dag phi_l
  G_l = (-1)^(l1+l2) (-1)^{n_l} prod(X on links touching l)   [Gauss law generator]
The paper gives this general form for an arbitrary ladder but does not work out an
explicit single-plaquette example; the instantiation below is our own minimal
8-qubit realization of it (4 matter corners + 4 gauge links around one plaquette),
built directly from that general formula, not copied from a worked example.

Qubit layout (8 qubits, matter-link-matter-link around the loop, following the
paper's own "matter-link-matter-link" traversal convention):
  0=m0(BL) 1=l01 2=m1(BR) 3=l12 4=m2(TR) 5=l23 6=m3(TL) 7=l30
Staggering sign eta_l = (-1)^(l1+l2) on the corners (checkerboard): BL=+1, BR=-1,
TR=+1, TL=-1.

Matter hopping uses OPEN boundary conditions around the plaquette (3 hopping bonds:
m0-m1, m1-m2, m2-m3; the closing m3-m0 bond carries a gauge link -- contributing to
the electric, magnetic and Gauss-law terms -- but no matter-hopping term). This
sidesteps the Jordan-Wigner sign ambiguity of a genuinely periodic fermion loop
(a real implementation detail, not addressed in the cited paper's abstract-level
description) at the cost of one fewer kinetic bond; a documented MVP simplification,
analogous to hamiltonians_schwinger.py's single-flavor simplification of the
2-flavor Schwinger model. Nearest-neighbor JW hopping with no other matter qubits
in between reduces to the standard gauge-dressed form
  phi_i^dag Z_link phi_{i+1} + h.c. = (X_i Z_link X_{i+1} + Y_i Z_link Y_{i+1}) / 2
(no extra Jordan-Wigner string, since link qubits do not carry fermionic parity and
i, i+1 are adjacent matter sites in the linear JW order).

Gauss's law penalty term uses G_l^2 = I (every G_l is a product of commuting,
self-inverse Paulis) to expand (G_l - q_l)^2 = 2 - 2 q_l G_l exactly, avoiding any
approximation. The physical (charge-free, "vacuum") sector q_l=+1 for all four
sites is used throughout -- comparing charge sectors is future work (see the
companion notebook's Findings).
"""
from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg

NUM_QUBITS = 8
# qubit positions
M0, L01, M1, L12, M2, L23, M3, L30 = range(8)
MATTER = [M0, M1, M2, M3]
LINKS = [L01, L12, L23, L30]
ETA = {M0: 1.0, M1: -1.0, M2: 1.0, M3: -1.0}  # checkerboard staggering


def _hopping_terms(J: float):
    coeffs, ops = [], []
    for (i, link, j) in [(M0, L01, M1), (M1, L12, M2), (M2, L23, M3)]:
        coeffs += [J / 2.0, J / 2.0]
        ops += [
            qml.PauliX(i) @ qml.PauliZ(link) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliZ(link) @ qml.PauliY(j),
        ]
    return coeffs, ops


def _mass_terms(m: float):
    coeffs, ops = [], []
    for site in MATTER:
        coeffs.append((m / 2.0) * ETA[site])
        ops.append(qml.PauliZ(site))
    return coeffs, ops


def _electric_terms(mu: float):
    coeffs, ops = [], []
    for link in LINKS:
        coeffs.append(-mu)
        ops.append(qml.PauliX(link))
    return coeffs, ops


def _magnetic_term():
    return [-1.0], [qml.PauliZ(L01) @ qml.PauliZ(L12) @ qml.PauliZ(L23) @ qml.PauliZ(L30)]


def gauss_operator(site: int) -> qml.Hamiltonian:
    """G_l for the matter site `site`: eta_l * Z_site * (X on both links touching it)."""
    if site == M0:
        links = (L30, L01)
    elif site == M1:
        links = (L01, L12)
    elif site == M2:
        links = (L12, L23)
    elif site == M3:
        links = (L23, L30)
    else:
        raise ValueError(site)
    op = qml.PauliZ(site) @ qml.PauliX(links[0]) @ qml.PauliX(links[1])
    return qml.Hamiltonian([ETA[site]], [op])


def _gauss_penalty_terms(V: float, q: float = 1.0):
    """sum_l (G_l - q)^2 = sum_l (2 - 2*q*G_l), expanded exactly (G_l^2 = I)."""
    coeffs, ops = [4.0 * V * len(MATTER)], [qml.Identity(0)]
    for site in MATTER:
        G = gauss_operator(site)
        for c, op in zip(G.coeffs, G.ops):
            coeffs.append(-2.0 * V * q * float(c))
            ops.append(op)
    return coeffs, ops


def z2_lgt_hamiltonian(J: float, m: float, mu: float, V: float = 10.0, q: float = 1.0) -> qml.Hamiltonian:
    """Single-plaquette Z2 lattice gauge theory Hamiltonian, 8 qubits.
    Task parameters: hopping J, staggered mass m, electric coupling mu.
    V is a fixed (non-task) Gauss-law penalty strength; q is the fixed charge
    sector (vacuum, q=+1 at every site)."""
    coeffs, ops = [], []
    for c, o in [_hopping_terms(J), _mass_terms(m), _electric_terms(mu), _magnetic_term(),
                 _gauss_penalty_terms(V, q)]:
        coeffs += c
        ops += o
    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(H: qml.Hamiltonian, num_qubits: int = NUM_QUBITS) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_task_space(n_tasks: int, seed: int, J_range=(0.1, 2.0), m_range=(0.1, 2.0),
                       mu_range=(0.1, 2.0)) -> np.ndarray:
    """Task = (J, m, mu), uniformly sampled -- hopping, mass, electric coupling."""
    rng = np.random.default_rng(seed)
    J = rng.uniform(*J_range, size=n_tasks)
    m = rng.uniform(*m_range, size=n_tasks)
    mu = rng.uniform(*mu_range, size=n_tasks)
    return np.stack([J, m, mu], axis=1)


def param_shape(depth: int, num_qubits: int = NUM_QUBITS) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_gauge_invariance(J=0.7, m=1.1, mu=0.9, V=10.0, atol=1e-8) -> bool:
    """Numerically checks G_l^2 = I and [H, G_l] = 0 for every site -- the physical
    correctness condition (local gauge invariance) for this construction."""
    H = z2_lgt_hamiltonian(J, m, mu, V=V)
    Hd = H.sparse_matrix(wire_order=list(range(NUM_QUBITS))).toarray()
    ok = True
    for site in MATTER:
        G = gauss_operator(site)
        Gd = G.sparse_matrix(wire_order=list(range(NUM_QUBITS))).toarray()
        g2_err = np.max(np.abs(Gd @ Gd - np.eye(2 ** NUM_QUBITS)))
        comm_err = np.max(np.abs(Hd @ Gd - Gd @ Hd))
        passed = g2_err < atol and comm_err < atol
        ok = ok and passed
        print(f"  site {site}: G^2-I max err={g2_err:.2e}  [H,G] max err={comm_err:.2e}  {'OK' if passed else 'FAIL'}")
    return ok


__all__ = [
    "z2_lgt_hamiltonian", "gauss_operator", "exact_ground_energy", "sample_task_space",
    "param_shape", "verify_gauge_invariance", "NUM_QUBITS",
]

if __name__ == "__main__":
    print("Verifying gauge invariance ([H, G_l] = 0) at a few (J, m, mu) points:")
    all_ok = True
    for (J, m, mu) in [(0.7, 1.1, 0.9), (0.1, 2.0, 0.1), (2.0, 0.1, 2.0), (1.0, 1.0, 1.0)]:
        print(f" (J={J}, m={m}, mu={mu}):")
        all_ok = verify_gauge_invariance(J, m, mu) and all_ok
    print("ALL PASSED" if all_ok else "FAILED -- do not trust this Hamiltonian yet")
