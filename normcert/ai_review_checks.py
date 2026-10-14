"""Two follow-up checks on the 216-instance testbed.

Check A (the 32 no-snapshot instances).  Each of them reports a gap of exactly
0.0 in double precision before the snapshot trigger fires.  Here we ask
whether that is the exact one-round convergence predicted for pure saddles
(m = 1, Section 4.2) or a rounding artifact:
  * does the game have a pure saddle point (an entry that is a column maximum
    and a row minimum)?
  * re-running in double with per-iteration recording, at which round do the
    two main iterates become exact one-hot vectors on a pure saddle?
  * re-run every instance in 60-digit arithmetic (200 rounds) and report the
    gap at the round where double reports 0.0, and the first round at which
    the 60-digit gap is exactly zero.

Check B (the 8 degenerate-kernel instances of degenerate_rows.csv).  Theorem 2
gives gap -> 0 without uniqueness; point convergence of the iterate is open
there.  We re-run each instance, store the state at geometric checkpoints and
report the distance of (x, y) and of z = (r1, r2) to the final state.

Usage:  python3 ai_review_checks.py [A|B|AB]
Output: results/ai_review_checkA.csv, results/ai_review_checkB.csv
"""
from __future__ import annotations

import csv
import sys

import mpmath as mp
import numpy as np

from games import make_game
from ireg_prm import IRPRMPlus, duality_gap, set_seed

ROWS = "results/n1n2_rows.csv"
DEG = "results/degenerate_rows.csv"


def load_rows():
    return {r["instance_id"]: r for r in csv.DictReader(open(ROWS))}


def pure_saddles(A):
    colmax, rowmin = A.max(0), A.min(1)
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] == colmax[j] and A[i, j] == rowmin[i]]


def run_double(A, T):
    """Per-iteration run; returns first round with gap == 0.0 exactly, first
    round with one-hot (x, y) on a pure saddle, and the final (x, y)."""
    n1, n2 = A.shape
    p1, p2 = IRPRMPlus(n1), IRPRMPlus(n2)
    sad = set(pure_saddles(A))
    t_zero = t_onehot = -1
    for t in range(1, T + 1):
        x = p1.next_strategy(A @ p2.xtilde)
        y = p2.next_strategy(-(p1.xtilde @ A))
        p1.observe_utility(A @ y)
        p2.observe_utility(-(x @ A))
        g = duality_gap(A, x, y)
        if t_zero < 0 and g == 0.0:
            t_zero = t
        if t_onehot < 0 and np.count_nonzero(x) == 1 and np.count_nonzero(y) == 1:
            if (int(np.argmax(x)), int(np.argmax(y))) in sad:
                t_onehot = t
        if t_zero > 0 and t_onehot > 0:
            break
    return t_zero, t_onehot, x, y


# ---------------------------------------------------------------- mpmath ---

def _gamma_shift_mp(v, c):
    """Solve ||[v - g]_+||_2 = c by bisection in mpmath (phi is decreasing)."""
    lo, hi = min(v) - c, max(v)
    for _ in range(400):
        mid = (lo + hi) / 2
        w = [max(vi - mid, mp.mpf(0)) for vi in v]
        phi = mp.sqrt(mp.fsum(wi * wi for wi in w))
        if phi > c:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class _MPPlayer:
    def __init__(self, n):
        self.n = n
        self.rt = [mp.mpf(0)] * n
        self.xt = [mp.mpf(1) / n] * n
        self.r = list(self.rt)
        self.x = list(self.xt)
        self.m = [mp.mpf(0)] * n

    def next_strategy(self, m):
        nrm = mp.sqrt(mp.fsum(r * r for r in self.rt))
        if nrm == 0:
            self.m = [mp.mpf(0)] * self.n
            self.r = list(self.rt)
            self.x = list(self.xt)
            return self.x
        self.m = m
        v = [a + b for a, b in zip(self.rt, m)]
        gam = _gamma_shift_mp(v, nrm)
        self.r = [vi - gam for vi in v]
        pos = [max(ri, mp.mpf(0)) for ri in self.r]
        s = mp.fsum(pos)
        self.x = [p / s for p in pos] if s > 0 else list(self.xt)
        return self.x

    def observe(self, u):
        res = [a - b for a, b in zip(u, self.m)]
        inner = mp.fsum(a * b for a, b in zip(res, self.x))
        g = [a - inner for a in res]
        self.rt = [max(a + b, mp.mpf(0)) for a, b in zip(self.r, g)]
        s = mp.fsum(self.rt)
        self.xt = [a / s for a in self.rt] if s > 0 else list(self.x)


def run_mp(A, T, dps=60):
    mp.mp.dps = dps
    n1, n2 = A.shape
    Am = [[mp.mpf(float(A[i, j])) for j in range(n2)] for i in range(n1)]
    p1, p2 = _MPPlayer(n1), _MPPlayer(n2)
    gaps = []
    for t in range(1, T + 1):
        m1 = [mp.fsum(Am[i][j] * p2.xt[j] for j in range(n2)) for i in range(n1)]
        m2 = [-mp.fsum(Am[i][j] * p1.xt[i] for i in range(n1)) for j in range(n2)]
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        u1 = [mp.fsum(Am[i][j] * y[j] for j in range(n2)) for i in range(n1)]
        u2 = [-mp.fsum(Am[i][j] * x[i] for i in range(n1)) for j in range(n2)]
        p1.observe(u1)
        p2.observe(u2)
        gaps.append(max(u1) - min(-u for u in u2))
    return gaps, x, y


def check_a():
    rows = load_rows()
    nos = [r for r in rows.values() if r["snap_t"] == "-1"]
    out = []
    for r in sorted(nos, key=lambda r: r["instance_id"]):
        set_seed(int(r["seed"]))
        A = make_game(int(r["dim"]), r["dist"], r["gen"], int(r["seed"]))
        sad = pure_saddles(A)
        t_zero, t_onehot, x, y = run_double(A, 2000)
        rec = dict(instance_id=r["instance_id"], dim=r["dim"], dist=r["dist"],
                   gen=r["gen"], supp_x_lp=r["supp_x_lp"], supp_y_lp=r["supp_y_lp"],
                   n_pure_saddles=len(sad), t_gap_zero_double=t_zero,
                   t_onehot_on_saddle=t_onehot, mp_gap_at_t_zero="",
                   mp_gap_at_200="", mp_t_gap_zero=-1)
        gaps, _, _ = run_mp(A, 200)
        tz = t_zero if 0 < t_zero <= 200 else 200
        rec["mp_gap_at_t_zero"] = mp.nstr(gaps[tz - 1], 6)
        rec["mp_gap_at_200"] = mp.nstr(gaps[-1], 6)
        rec["mp_t_gap_zero"] = next((i + 1 for i, g in enumerate(gaps) if g == 0), -1)
        out.append(rec)
        print(rec)
    with open("results/ai_review_checkA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n_sad = sum(1 for o in out if o["n_pure_saddles"] > 0)
    n_hit = sum(1 for o in out if o["t_onehot_on_saddle"] > 0)
    print(f"no-snapshot: {len(out)}  with pure saddle: {n_sad}  "
          f"one-hot on saddle in double: {n_hit}")


def check_b():
    rows = load_rows()
    deg = [r for r in csv.DictReader(open(DEG)) if r["status"] == "deg_kernel"]
    out = []
    for r in deg:
        q = rows[r["instance_id"]]
        T = int(q["T"])
        set_seed(int(q["seed"]))
        A = make_game(int(q["dim"]), q["dist"], q["gen"], int(q["seed"]))
        n1, n2 = A.shape
        p1, p2 = IRPRMPlus(n1), IRPRMPlus(n2)
        cps = sorted(set(int(T / 2 ** k) for k in range(0, 8)) | {T})
        snap = {}
        for t in range(1, T + 1):
            x = p1.next_strategy(A @ p2.xtilde)
            y = p2.next_strategy(-(p1.xtilde @ A))
            p1.observe_utility(A @ y)
            p2.observe_utility(-(x @ A))
            if t in cps:
                snap[t] = (x.copy(), y.copy(), p1.rtilde.copy(), p2.rtilde.copy(),
                           duality_gap(A, x, y))
        xT, yT, r1T, r2T, gT = snap[T]
        for t in cps:
            x, y, r1, r2, g = snap[t]
            rec = dict(instance_id=r["instance_id"], n_ker=r["n_ker"], T=T, t=t,
                       gap=f"{g:.3e}",
                       dist_xy_to_final=f"{np.hypot(np.linalg.norm(x - xT), np.linalg.norm(y - yT)):.3e}",
                       dist_z_to_final=f"{np.hypot(np.linalg.norm(r1 - r1T), np.linalg.norm(r2 - r2T)):.3e}",
                       norm_z=f"{np.hypot(np.linalg.norm(r1), np.linalg.norm(r2)):.6f}")
            out.append(rec)
            print(rec)
    with open("results/ai_review_checkB.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "A" in which:
        check_a()
    if "B" in which:
        check_b()
