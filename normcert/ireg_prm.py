"""IREG-PRM+ (IR-PRM+ in the extra-gradient setup) for zero-sum matrix games.

Reference: Zhang, Anagnostides, Sandholm, "Scale-Invariant Regret Matching and
Online Learning with Optimal Convergence" (EC 2026, arXiv:2510.04407v3),
Algorithm 2 (IR-PRM+) + extra-gradient setup (Algorithm 5).

Convention: row player maximizes x^T A y, column player minimizes (i.e. its
utility vector for action j is -(A^T x)_j).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# gamma-shift: solve ||[v - gamma*1]_+||_2 = c  (Appendix B, Algorithm 3)
# ---------------------------------------------------------------------------

def _newton_refine(v: np.ndarray, c: float, g: float, iters: int = 50) -> float:
    """Newton-refine gamma on f(g) = ||[v-g]_+||^2 - c^2 (convex, decreasing).

    Computed directly on v, avoiding the catastrophic cancellation of the
    sorted closed form when c << ||v||. Start point must satisfy f(g) >= 0
    (approach from the left), guaranteed by g <= max(v) - c.
    """
    g = min(g, float(np.max(v)) - c)  # phi(max(v)-c) >= c
    c2 = c * c
    for _ in range(iters):
        w = np.maximum(v - g, 0.0)
        f = float(w @ w) - c2
        if abs(f) <= 1e-14 * c2:
            break
        fp = -2.0 * float(w.sum())
        if fp >= 0.0:
            break
        g_new = g - f / fp
        if not np.isfinite(g_new) or g_new == g:
            break
        g = g_new
    return g


def gamma_shift(v: np.ndarray, c: float) -> float:
    """Return gamma such that ||[v - gamma]_+||_2 = c (c > 0).

    phi(gamma) = ||[v - gamma]_+||_2 is continuous, non-increasing,
    -> +inf as gamma -> -inf, and = 0 for gamma >= max(v).
    O(n log n) piecewise closed form via sorting + Newton refinement.
    """
    if c <= 0.0:
        raise ValueError("c must be positive")
    vs = np.sort(v)[::-1]  # descending
    n = vs.shape[0]
    cs = np.cumsum(vs)
    cs2 = np.cumsum(vs * vs)
    c2 = c * c
    g_cand: Optional[float] = None
    # Active set = top-k entries, gamma in [vs[k], vs[k-1]) (vs[n] = -inf).
    for k in range(1, n + 1):
        s, s2 = cs[k - 1], cs2[k - 1]
        # k*g^2 - 2*s*g + (s2 - c^2) = 0 ; take smaller root (phi decreasing)
        disc = s * s - k * (s2 - c2)
        if disc < 0.0:
            continue
        g = (s - np.sqrt(disc)) / k
        hi = vs[k - 1]
        lo = vs[k] if k < n else -np.inf
        if lo - 1e-12 <= g <= hi + 1e-12:
            g_cand = float(g)
            break
    if g_cand is None:
        g_cand = float(vs[0]) - c  # safe left-side start
    return _newton_refine(v, c, g_cand)


def gamma_shift_batch(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized gamma_shift over batch: v (B, n), c (B,) all positive.

    Returns gamma (B,).
    """
    B, n = v.shape
    vs = -np.sort(-v, axis=1)  # descending along actions
    cs = np.cumsum(vs, axis=1)
    cs2 = np.cumsum(vs * vs, axis=1)
    ks = np.arange(1, n + 1, dtype=np.float64)[None, :]  # (1, n)
    c2 = (c * c)[:, None]
    disc = cs * cs - ks * (cs2 - c2)
    valid = disc >= 0.0
    g = np.where(valid, (cs - np.sqrt(np.maximum(disc, 0.0))) / ks, np.nan)
    hi = vs
    lo = np.concatenate([vs[:, 1:], np.full((B, 1), -np.inf)], axis=1)
    ok = valid & (g >= lo - 1e-12) & (g <= hi + 1e-12)
    # first valid k per row
    idx = np.argmax(ok, axis=1)
    found = ok[np.arange(B), idx]
    out = g[np.arange(B), idx]
    out = np.where(found, out, vs[:, 0] - c)
    # vectorized Newton refinement (see _newton_refine)
    out = np.minimum(out, vs[:, 0] - c)
    c2 = c * c
    for _ in range(50):
        w = np.maximum(v - out[:, None], 0.0)
        f = np.einsum('bi,bi->b', w, w) - c2
        fp = -2.0 * w.sum(axis=1)
        act = (np.abs(f) > 1e-14 * c2) & (fp < 0.0)
        if not np.any(act):
            break
        step = np.where(act, f / np.where(fp < 0.0, fp, -1.0), 0.0)
        new = out - step
        out = np.where(np.isfinite(new), new, out)
    return out


# ---------------------------------------------------------------------------
# IR-PRM+ regret minimizer (Algorithm 2)
# ---------------------------------------------------------------------------

class IRPRMPlus:
    """IR-PRM+ regret minimizer over the simplex (single instance)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.rtilde = np.zeros(n)          # r̃^(t), kept non-negative (RM+)
        self.xtilde = np.full(n, 1.0 / n)  # x̃^(t)
        self.r = np.zeros(n)               # r^(t) after prediction step
        self.x = np.full(n, 1.0 / n)       # x^(t)
        self.m = np.zeros(n)               # last prediction actually used

    def next_strategy(self, m: np.ndarray) -> np.ndarray:
        norm_rt = np.linalg.norm(self.rtilde)
        if norm_rt <= 0.0:
            # "no movement until nonzero utility"
            self.m = np.zeros(self.n)
            self.r = self.rtilde.copy()
            self.x = self.xtilde.copy()
            return self.x
        self.m = m
        v = self.rtilde + m
        gam = gamma_shift(v, norm_rt)
        # Algorithm 2 line 9: r^(t) = rtilde + m - gamma*1, UNCLIPPED.
        # (The pilot implementation clipped here; near a strict-complementarity
        #  fixed point the clipped variant emits off-support "dust" [g_i]_+
        #  while the paper's version freezes off-support coords exactly at 0.)
        self.r = v - gam
        pos = np.maximum(self.r, 0.0)
        s = pos.sum()
        if s <= 0.0:
            self.x = self.xtilde.copy()
        else:
            self.x = pos / s
        return self.x

    def observe_utility(self, u: np.ndarray) -> None:
        res = u - self.m
        g = res - float(res @ self.x)
        # Algorithm 2 line 13: rtilde^(t+1) = [r^(t) + g]_+ with unclipped r.
        self.rtilde = np.maximum(self.r + g, 0.0)
        s = self.rtilde.sum()
        if s <= 0.0:
            self.xtilde = self.x.copy()
        else:
            self.xtilde = self.rtilde / s

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.rtilde))


# ---------------------------------------------------------------------------
# Matrix-game runners
# ---------------------------------------------------------------------------

def duality_gap(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Nash/duality gap of (x, y): max_x' x'Ay - min_y' xAy'."""
    return float(np.max(A @ y) - np.min(x @ A))


@dataclass
class Trace:
    """Recorded time series for one run."""
    ts: np.ndarray
    norm1: np.ndarray
    norm2: np.ndarray
    gap_last: np.ndarray
    gap_best: np.ndarray
    gap_avg: np.ndarray
    cum_regret1: np.ndarray
    cum_regret2: np.ndarray


def run_ireg_matrix(A: np.ndarray, T: int, record_every: int = 50,
                    ) -> Trace:
    """Run IREG-PRM+ (extra-gradient setup) on zero-sum matrix game A."""
    n1, n2 = A.shape
    p1 = IRPRMPlus(n1)
    p2 = IRPRMPlus(n2)
    ts: List[int] = []
    norm1: List[float] = []
    norm2: List[float] = []
    gap_last: List[float] = []
    gap_best: List[float] = []
    gap_avg: List[float] = []
    cr1: List[float] = []
    cr2: List[float] = []
    best = np.inf
    xsum = np.zeros(n1)
    ysum = np.zeros(n2)
    # external regret bookkeeping: Reg = max_a sum_t u_a - sum_t <u, x>
    cum_u1 = np.zeros(n1)
    cum_u2 = np.zeros(n2)
    cum_val1 = 0.0
    cum_val2 = 0.0
    for t in range(1, T + 1):
        # prediction at pre-iterates (extra gradient: m = F(x̃))
        m1 = A @ p2.xtilde
        m2 = -(p1.xtilde @ A)
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        u1 = A @ y
        u2 = -(x @ A)
        p1.observe_utility(u1)
        p2.observe_utility(u2)
        cum_u1 += u1
        cum_u2 += u2
        cum_val1 += float(u1 @ x)
        cum_val2 += float(u2 @ y)
        xsum += x
        ysum += y
        if t % record_every == 0 or t == 1:
            g = duality_gap(A, x, y)
            best = min(best, g)
            ts.append(t)
            norm1.append(p1.norm)
            norm2.append(p2.norm)
            gap_last.append(g)
            gap_best.append(best)
            gap_avg.append(duality_gap(A, xsum / t, ysum / t))
            cr1.append(float(np.max(cum_u1) - cum_val1))
            cr2.append(float(np.max(cum_u2) - cum_val2))
    return Trace(np.array(ts), np.array(norm1), np.array(norm2),
                 np.array(gap_last), np.array(gap_best), np.array(gap_avg),
                 np.array(cr1), np.array(cr2))


# ---------------------------------------------------------------------------
# Batched runner: B games of identical shape, vectorized over the batch
# ---------------------------------------------------------------------------

@dataclass
class BatchTrace:
    ts: np.ndarray          # (K,)
    norm1: np.ndarray       # (K, B)
    norm2: np.ndarray       # (K, B)
    gap_last: np.ndarray    # (K, B)
    gap_best: np.ndarray    # (K, B)
    gap_avg: np.ndarray     # (K, B)
    cum_regret1: np.ndarray # (K, B)
    cum_regret2: np.ndarray # (K, B)


class _BatchIRPRM:
    """Batch of IR-PRM+ instances: rtilde (B, n)."""

    def __init__(self, B: int, n: int) -> None:
        self.B, self.n = B, n
        self.rtilde = np.zeros((B, n))
        self.xtilde = np.full((B, n), 1.0 / n)
        self.r = np.zeros((B, n))
        self.x = np.full((B, n), 1.0 / n)
        self.m = np.zeros((B, n))

    def next_strategy(self, m: np.ndarray) -> np.ndarray:
        norm_rt = np.linalg.norm(self.rtilde, axis=1)  # (B,)
        active = norm_rt > 0.0
        self.m = np.where(active[:, None], m, 0.0)
        v = self.rtilde + self.m
        gam = np.zeros(self.B)
        if np.any(active):
            gam[active] = gamma_shift_batch(v[active], norm_rt[active])
        # Algorithm 2 line 9: unclipped r (see single-instance comment).
        r = v - gam[:, None]
        r[~active] = self.rtilde[~active]
        self.r = r
        pos = np.maximum(r, 0.0)
        s = pos.sum(axis=1)
        ok = s > 0.0
        x = np.where(ok[:, None], pos / np.where(ok, s, 1.0)[:, None],
                     self.xtilde)
        x[~active] = self.xtilde[~active]
        self.x = x
        return x

    def observe_utility(self, u: np.ndarray) -> float:
        res = u - self.m
        inner = np.einsum('bi,bi->b', res, self.x)
        g = res - inner[:, None]
        self.rtilde = np.maximum(self.r + g, 0.0)
        s = self.rtilde.sum(axis=1)
        ok = s > 0.0
        self.xtilde = np.where(ok[:, None],
                               self.rtilde / np.where(ok, s, 1.0)[:, None],
                               self.x)
        return float((g * g).sum())  # sum ||g||^2 (cf. matrix cum_g2)

    @property
    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.rtilde, axis=1)


def make_record_schedule(T: int, n_points: int = 400) -> np.ndarray:
    """Log-spaced recording iterations in [1, T] (unique ints, includes T)."""
    pts = np.unique(np.round(np.logspace(0.0, np.log10(T), n_points))
                    .astype(np.int64))
    return pts[(pts >= 1) & (pts <= T)]


def run_ireg_matrix_batch(As: np.ndarray, T: int, record_every: int = 50,
                          record_at: Optional[np.ndarray] = None,
                          ) -> BatchTrace:
    """Run IREG-PRM+ on a batch of games As (B, n1, n2).

    If record_at is given (sorted iteration numbers), it overrides
    record_every.
    """
    B, n1, n2 = As.shape
    p1 = _BatchIRPRM(B, n1)
    p2 = _BatchIRPRM(B, n2)
    AsT = np.transpose(As, (0, 2, 1)).copy()
    ts: List[int] = []
    rec: Dict[str, List[np.ndarray]] = {k: [] for k in
        ("norm1", "norm2", "gap_last", "gap_best", "gap_avg", "cr1", "cr2")}
    best = np.full(B, np.inf)
    xsum = np.zeros((B, n1))
    ysum = np.zeros((B, n2))
    cum_u1 = np.zeros((B, n1))
    cum_u2 = np.zeros((B, n2))
    cum_val1 = np.zeros(B)
    cum_val2 = np.zeros(B)

    def _gap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Ay = np.einsum('bij,bj->bi', As, y)
        xA = np.einsum('bij,bi->bj', As, x)
        return Ay.max(axis=1) - xA.min(axis=1)

    if record_at is not None:
        rec_set = set(int(v) for v in record_at)
    else:
        rec_set = None

    for t in range(1, T + 1):
        m1 = np.einsum('bij,bj->bi', As, p2.xtilde)
        m2 = -np.einsum('bji,bj->bi', As, p1.xtilde)
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        u1 = np.einsum('bij,bj->bi', As, y)
        u2 = -np.einsum('bij,bi->bj', As, x)
        p1.observe_utility(u1)
        p2.observe_utility(u2)
        cum_u1 += u1
        cum_u2 += u2
        cum_val1 += np.einsum('bi,bi->b', u1, x)
        cum_val2 += np.einsum('bi,bi->b', u2, y)
        xsum += x
        ysum += y
        do_rec = (t in rec_set) if rec_set is not None \
            else (t % record_every == 0 or t == 1)
        if do_rec:
            g = _gap(x, y)
            best = np.minimum(best, g)
            ts.append(t)
            rec["norm1"].append(p1.norm)
            rec["norm2"].append(p2.norm)
            rec["gap_last"].append(g)
            rec["gap_best"].append(best.copy())
            rec["gap_avg"].append(_gap(xsum / t, ysum / t))
            rec["cr1"].append(cum_u1.max(axis=1) - cum_val1)
            rec["cr2"].append(cum_u2.max(axis=1) - cum_val2)
    return BatchTrace(np.array(ts),
                      np.stack(rec["norm1"]), np.stack(rec["norm2"]),
                      np.stack(rec["gap_last"]), np.stack(rec["gap_best"]),
                      np.stack(rec["gap_avg"]),
                      np.stack(rec["cr1"]), np.stack(rec["cr2"]))
