"""
Track F: Supersymmetric quantum mechanics (SUSY QM), following the setup studied with variational
quantum algorithms in arXiv:2603.18749 ("Simulating SUSY QM with VQAs"). One bosonic mode (truncated
Fock space, cutoff Lambda = 2^N_b) coupled to one fermionic mode (single qubit, Jordan-Wigner trivial
for one mode), via a superpotential W(q):

  H = (1/2) * (p^2 + [W'(q)]^2) - (1/2) * W''(q) * [b_dagger, b]

With the qubit convention |0> = fermion vacuum, |1> = fermion occupied, n_f = b_dagger b = (I - Z)/2,
so [b_dagger, b] = 2 n_f - I = -Z, giving the qubit-friendly form used below:

  H = (1/2) * (p^2 + [W'(q)]^2) + (1/2) * W''(q) * Z_fermion

q, p are the standard truncated-Fock-space quadrature operators built from ladder operators a, a_dagger
(a|n> = sqrt(n)|n-1>, truncated to dimension Lambda): q = (a + a_dagger)/sqrt(2), p = i(a_dagger - a)/sqrt(2).

Three superpotentials (paper's Table 1, m=g=mu=1 by default -- these are our task parameters, not fixed):
  Harmonic oscillator (HO):    W(q) = (m/2) q^2
  Anharmonic oscillator (AHO): W(q) = (m/2) q^2 + (g/4) q^4
  Double well (DW):            W(q) = (m/2) q^2 + (g/3) q^3 - g*mu^2*q
(DW's cubic-term sign: the source paper's rendered equation lost a sign in our extraction -- we use the
sign that actually produces a genuine double-well potential V(q)=(1/2)[W'(q)]^2 with two local minima,
verified numerically below by verify_double_well(), rather than guessing; this is the standard Witten
SUSY-QM double-well convention (Cooper, Khare & Sukhatme's review), not our own invention.)

This is a genuinely different task-parameter *shape* from every other track in this project: the choice
of superpotential is a discrete label (3 options), not a continuous physical parameter -- only the
cutoff Lambda and the superpotential's own coefficients (m, g, mu) are continuous/tunable. Q-MAML's task
descriptor here is therefore (superpotential_id, m, g, mu) rather than a purely continuous physics vector,
which changes how the task distribution/generalization experiment needs to be designed (see the
companion notebook).
"""
from typing import Tuple

import numpy as np

N_B_DEFAULT = 3          # bosonic qubits -> Lambda = 8 (paper's Lambda=8 is the first cutoff where the
                          # DW ground state qualitatively resolves the second well)
N_FERMION_QUBITS = 1
SUPERPOTENTIALS = ("HO", "AHO", "DW")


def _ladder_operators(n_levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """a (annihilation), a_dagger, truncated to `n_levels` Fock states."""
    a = np.zeros((n_levels, n_levels), dtype=complex)
    for n in range(1, n_levels):
        a[n - 1, n] = np.sqrt(n)
    return a, a.conj().T


def _quadratures(n_levels: int) -> Tuple[np.ndarray, np.ndarray]:
    a, adag = _ladder_operators(n_levels)
    q = (a + adag) / np.sqrt(2.0)
    p = 1j * (adag - a) / np.sqrt(2.0)
    return q, p


def _w_prime_w_doubleprime(superpotential: str, q: np.ndarray, m: float, g: float, mu: float):
    """W'(q) and W''(q) as matrices (functions of the q operator), for the given superpotential."""
    I = np.eye(q.shape[0], dtype=complex)
    if superpotential == "HO":
        Wp = m * q
        Wpp = m * I
    elif superpotential == "AHO":
        Wp = m * q + g * (q @ q @ q)
        Wpp = m * I + 3.0 * g * (q @ q)
    elif superpotential == "DW":
        Wp = m * q + g * (q @ q) - g * (mu ** 2) * I
        Wpp = m * I + 2.0 * g * q
    else:
        raise ValueError(f"unknown superpotential {superpotential!r}, expected one of {SUPERPOTENTIALS}")
    return Wp, Wpp


def susyqm_hamiltonian(superpotential: str, n_b: int = N_B_DEFAULT, m: float = 1.0, g: float = 1.0,
                        mu: float = 1.0) -> np.ndarray:
    """Dense Hermitian matrix, dimension 2*Lambda (Lambda=2**n_b bosonic levels x 2 fermionic states).
    Task parameters: which superpotential (HO/AHO/DW), and its coefficients (m, g, mu; g and mu are
    inert for HO)."""
    n_levels = 2 ** n_b
    q, p = _quadratures(n_levels)
    Wp, Wpp = _w_prime_w_doubleprime(superpotential, q, m, g, mu)

    H_boson = 0.5 * (p @ p + Wp @ Wp)
    I_boson = np.eye(n_levels, dtype=complex)
    Z_fermion = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I_fermion = np.eye(2, dtype=complex)

    H = np.kron(H_boson, I_fermion) + 0.5 * np.kron(Wpp, Z_fermion)
    return H


def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(H)))


def sample_task_space(n_tasks: int, seed: int, superpotentials=SUPERPOTENTIALS,
                       m_range=(0.5, 1.5), g_range=(0.5, 1.5), mu_range=(0.5, 1.5)):
    """Task = (superpotential_id, m, g, mu). superpotential_id is a discrete index into
    `superpotentials`, sampled uniformly; (m, g, mu) uniform within range. Returns a list of
    (superpotential_str, m, g, mu) tuples (not a plain ndarray, since the first entry is categorical)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(superpotentials), size=n_tasks)
    m = rng.uniform(*m_range, size=n_tasks)
    g = rng.uniform(*g_range, size=n_tasks)
    mu = rng.uniform(*mu_range, size=n_tasks)
    return [(superpotentials[i], float(m[k]), float(g[k]), float(mu[k])) for k, i in enumerate(idx)]


def param_shape(depth: int, num_qubits: int) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_double_well(m: float = 1.0, g: float = 1.0, mu: float = 1.0, n_grid: int = 2001) -> bool:
    """V(q) = (1/2)[W'(q)]^2 (classical potential, continuum q, not the truncated operator) should have
    two local minima for the DW superpotential with this sign convention -- checks that directly on a
    dense grid, rather than assuming it."""
    qs = np.linspace(-4.0, 4.0, n_grid)
    Wp = m * qs + g * qs ** 2 - g * mu ** 2
    V = 0.5 * Wp ** 2
    # local minima: V[i] < V[i-1] and V[i] < V[i+1]
    minima = [i for i in range(1, n_grid - 1) if V[i] < V[i - 1] and V[i] < V[i + 1]]
    print(f"  DW classical potential: {len(minima)} local minima found at q={[round(qs[i], 3) for i in minima]}")
    return len(minima) >= 2


def verify_construction(n_b: int = N_B_DEFAULT, atol: float = 1e-8) -> bool:
    """(1) H Hermitian to machine precision for all three superpotentials. (2) Boson-truncation sanity:
    the HO case's boson-only block (fermion traced out at Z=+1) should reproduce the standard quantum
    harmonic oscillator spectrum n+1/2 on the low-lying (well below cutoff) levels -- same style of check
    as hamiltonians_scalar.py's QHO verification. (3) DW potential genuinely has two minima
    (verify_double_well)."""
    ok = True
    for sp in SUPERPOTENTIALS:
        H = susyqm_hamiltonian(sp, n_b=n_b)
        herm_err = np.max(np.abs(H - H.conj().T))
        print(f"  {sp}: Hermiticity max|H-H^dag|={herm_err:.2e}")
        ok = ok and herm_err < atol

    n_levels = 2 ** n_b
    q, p = _quadratures(n_levels)
    H_ho_boson = 0.5 * (p @ p + q @ q)  # m=1 HO boson-only block, no fermion coupling at Wpp fixed spin
    spectrum = np.sort(np.real(np.linalg.eigvalsh(H_ho_boson)))
    n_check = max(1, n_levels // 3)
    analytic = np.arange(n_check) + 0.5
    rel_err = np.max(np.abs(spectrum[:n_check] - analytic) / analytic)
    print(f"  HO boson-only spectrum vs. analytic n+1/2, lowest {n_check} levels: max rel err={rel_err:.2%}")
    ok = ok and rel_err < 0.05

    ok = ok and verify_double_well()
    return ok


__all__ = [
    "susyqm_hamiltonian", "exact_ground_energy", "sample_task_space", "param_shape",
    "verify_construction", "verify_double_well", "SUPERPOTENTIALS", "N_B_DEFAULT", "N_FERMION_QUBITS",
]

if __name__ == "__main__":
    print("Verifying SUSY QM Hamiltonian construction:")
    ok = verify_construction()
    print("ALL PASSED" if ok else "FAILED -- do not trust this Hamiltonian yet")
