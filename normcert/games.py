"""Random zero-sum matrix game generation + Nash equilibrium structure via LP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

DIMS = (5, 20, 50)
DISTS = ("gaussian", "uniform", "pm1_sparse")
GENS = ("plain", "antisym")
SUPPORT_TOL = 1e-8


def make_game(dim: int, dist: str, gen: str, seed: int) -> np.ndarray:
    """Generate a dim x dim zero-sum payoff matrix (row player maximizes)."""
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        G = rng.normal(size=(dim, dim))
    elif dist == "uniform":
        G = rng.uniform(-1.0, 1.0, size=(dim, dim))
    elif dist == "pm1_sparse":
        signs = rng.choice([-1.0, 1.0], size=(dim, dim))
        mask = rng.random(size=(dim, dim)) < 0.3
        G = signs * mask
    else:
        raise ValueError(dist)
    if gen == "antisym":
        return (G - G.T) / np.sqrt(2.0)
    return G


@dataclass
class NashInfo:
    value: float
    x: np.ndarray
    y: np.ndarray
    supp_x: int
    supp_y: int
    fully_mixed: bool
    strict_comp_margin: float  # min slack of non-support actions (both sides)
    unique_likely: bool        # strict complementarity + balanced supports
    gap_check: float           # duality gap of computed (x, y), sanity


def solve_nash(A: np.ndarray) -> Optional[NashInfo]:
    """Solve zero-sum game by LP (row max, col min). Returns None on failure."""
    n1, n2 = A.shape
    # Row player: max v s.t. A^T x >= v*1, sum x = 1, x >= 0
    # vars: (x, v); minimize -v
    c = np.zeros(n1 + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((n2, 1))])  # v - A^T x <= 0
    b_ub = np.zeros(n2)
    A_eq = np.zeros((1, n1 + 1))
    A_eq[0, :n1] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n1 + [(None, None)]
    r1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method="highs")
    # Col player: min w s.t. A y <= w*1, sum y = 1, y >= 0
    c2 = np.zeros(n2 + 1)
    c2[-1] = 1.0
    A_ub2 = np.hstack([A, -np.ones((n1, 1))])   # A y - w <= 0
    b_ub2 = np.zeros(n1)
    A_eq2 = np.zeros((1, n2 + 1))
    A_eq2[0, :n2] = 1.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0.0, None)] * n2 + [(None, None)]
    r2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                 bounds=bounds2, method="highs")
    if not (r1.success and r2.success):
        return None
    x = np.maximum(r1.x[:n1], 0.0)
    x /= x.sum()
    y = np.maximum(r2.x[:n2], 0.0)
    y /= y.sum()
    v = float(r1.x[-1])
    Ay = A @ y
    xA = x @ A
    supp_x = x > SUPPORT_TOL
    supp_y = y > SUPPORT_TOL
    # strict complementarity margins: off-support actions strictly suboptimal
    mx = float(np.min(v - Ay[~supp_x])) if np.any(~supp_x) else np.inf
    my = float(np.min(xA[~supp_y] - v)) if np.any(~supp_y) else np.inf
    margin = min(mx, my)
    fully_mixed = bool(np.all(supp_x) and np.all(supp_y))
    unique_likely = bool((supp_x.sum() == supp_y.sum())
                         and (margin > 1e-7 or fully_mixed))
    gap = float(np.max(Ay) - np.min(xA))
    return NashInfo(v, x, y, int(supp_x.sum()), int(supp_y.sum()),
                    fully_mixed, margin, unique_likely, gap)


def all_configs(seeds_per_config: int = 12) -> List[Dict]:
    """Full instance grid: dims x dists x gens x seeds."""
    out = []
    uid = 0
    for dim in DIMS:
        for dist in DISTS:
            for gen in GENS:
                for k in range(seeds_per_config):
                    out.append({
                        "instance_id": f"d{dim}_{dist}_{gen}_s{k}",
                        "dim": dim, "dist": dist, "gen": gen,
                        "seed": 10_000 + 997 * uid,
                    })
                    uid += 1
    return out
