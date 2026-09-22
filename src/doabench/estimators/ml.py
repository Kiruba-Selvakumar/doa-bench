"""Deterministic (conditional) maximum likelihood by grid search.

    θ̂ = argmax_θ tr{P_A(θ) R̂},    P_A = A (AᴴA)⁻¹ Aᴴ

The cost is evaluated without forming the N x N projector, using the cyclic property of
the trace: tr{A (AᴴA)⁻¹ Aᴴ R̂} = tr{(AᴴA)⁻¹ (Aᴴ R̂ A)}, an M x M problem.

Two details are load-bearing.

**Degenerate candidates.** det(AᴴA) = N² - |aᵢᴴaⱼ|² vanishes when two candidate steering
vectors are parallel, and round-off turns the resulting 0/0 into an arbitrary cost that can
win the argmax. The pair that actually does this on a 1° grid with d = λ/2 is not two
nearby angles but θ = -90° and +90°: their phases ±πn coincide modulo 2π, so
a(-90°) = a(+90°) and the array cannot tell the two endfire directions apart. With the
guard removed, ML picks that pair in many trials (RMSE ~60° on the reference data). An
angle-proximity guard would not catch it; the determinant guard does. Candidates with
det(AᴴA) below ``DET_TOL`` are skipped, and on the 1° grid that pair is the only one.

**Refinement.** A coarse grid of step h has an RMS quantisation floor of h/√12 (0.29° for
h = 1°), larger than MUSIC's error in benign conditions. Without refinement ML looks
*worse* than MUSIC at ρ = 0 (0.395° vs 0.197°), which inverts the headline result. A local
0.1° search around the coarse winner removes the floor for a few hundred evaluations.
"""

from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from doabench.arrays import SensorArray
from doabench.grids import window

DET_TOL = 1e-8


def ml_cost(covariance: NDArray[np.complex128], steering: NDArray[np.complex128]) -> float:
    """tr{P_A R̂} for one candidate steering matrix A, or -inf if AᴴA is near-singular."""
    gram = steering.conj().T @ steering
    if abs(np.linalg.det(gram)) < DET_TOL:
        return -np.inf
    projected = steering.conj().T @ covariance @ steering
    return float(np.real(np.trace(np.linalg.solve(gram, projected))))


def ml_loop(
    covariance: NDArray[np.complex128],
    array: SensorArray,
    n_sources: int,
    grid: NDArray[np.float64],
    refine_step: float = 0.1,
    refine_half_width: float = 1.0,
) -> tuple[NDArray[np.float64], int]:
    """Exhaustive search over all M-subsets of ``grid``, then local refinement.

    The honest O(G^M) implementation, and the readable definition of the estimator;
    ``ml_grid`` is tested against it. Returns the estimates (ascending) and the number of
    cost evaluations performed (degenerate candidates skipped by the guard are not
    counted). Set ``refine_step=0`` to disable refinement.
    """
    steering = array.steering(grid)
    best, winner, n_eval = -np.inf, None, 0

    for idx in combinations(range(grid.size), n_sources):
        cost = ml_cost(covariance, steering[:, idx])
        if cost == -np.inf:
            continue
        n_eval += 1
        if cost > best:
            best, winner = cost, grid[list(idx)]

    if winner is None:
        return np.full(n_sources, np.nan), n_eval

    if refine_step > 0:
        windows = [window(c, refine_half_width, refine_step) for c in winner]
        for cand in product(*windows):
            cost = ml_cost(covariance, array.steering(cand))
            if cost == -np.inf:
                continue
            n_eval += 1
            if cost > best:
                best, winner = cost, np.array(cand)

    return np.sort(winner), n_eval


def _pair_costs(
    covariance: NDArray[np.complex128],
    first: NDArray[np.complex128],
    second: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """tr{P_A R̂} for every pair A = [first[:, i], second[:, j]], shape (G1, G2).

    For A = [a b] with g = aᴴb and w_ab = aᴴR̂b (R̂ Hermitian, so w_ba is its conjugate):

        tr{(AᴴA)⁻¹ AᴴR̂A} = (‖b‖² w_aa + ‖a‖² w_bb - 2 Re(g · conj(w_ab))) / (‖a‖²‖b‖² - |g|²)

    Degenerate pairs (determinant below ``DET_TOL``) are set to -inf.
    """
    gram = first.conj().T @ second
    cross = first.conj().T @ covariance @ second
    norm1 = np.sum(np.abs(first) ** 2, axis=0)[:, None]
    norm2 = np.sum(np.abs(second) ** 2, axis=0)[None, :]
    w1 = np.real(np.einsum("ni,nm,mi->i", first.conj(), covariance, first))[:, None]
    w2 = np.real(np.einsum("ni,nm,mi->i", second.conj(), covariance, second))[None, :]

    det = norm1 * norm2 - np.abs(gram) ** 2
    num = norm2 * w1 + norm1 * w2 - 2 * np.real(gram * cross.conj())
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = num / det
    cost[det <= DET_TOL] = -np.inf
    return cost


def ml_grid(
    covariance: NDArray[np.complex128],
    array: SensorArray,
    n_sources: int,
    grid: NDArray[np.float64],
    refine_step: float = 0.1,
    refine_half_width: float = 1.0,
) -> NDArray[np.float64]:
    """Vectorised ``ml_loop`` for M = 2: same maximiser, all pairs in one matrix."""
    if n_sources != 2:
        raise NotImplementedError("ml_grid handles M = 2; use ml_loop for other M")

    steering = array.steering(grid)
    cost = _pair_costs(covariance, steering, steering)
    cost[np.tril_indices(grid.size)] = -np.inf  # keep i < j only
    i, j = np.unravel_index(np.argmax(cost), cost.shape)
    best, winner = cost[i, j], np.array([grid[i], grid[j]])
    if best == -np.inf:
        return np.full(2, np.nan)

    if refine_step > 0:
        g1 = window(winner[0], refine_half_width, refine_step)
        g2 = window(winner[1], refine_half_width, refine_step)
        refined = _pair_costs(covariance, array.steering(g1), array.steering(g2))
        a, b = np.unravel_index(np.argmax(refined), refined.shape)
        if refined[a, b] > best:
            winner = np.array([g1[a], g2[b]])

    return np.sort(winner)
