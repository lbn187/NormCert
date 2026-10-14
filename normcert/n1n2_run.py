"""N1/N2/N3 theorem-guided numerics for the norm-saturation certificate.

N1: active-set (support) freezing of rtilde, clip-loss decay, gamma -> game
    value (fixed-point Proposition P1 in the paper).
N2: spectral prediction -- finite-difference Jacobian of the one-step map Phi
    at a late snapshot; predicted linear rate = log(rho2) vs measured semilog
    slope of gap_last.
N3: N_inf distribution / effective step eta_inf = 1/N_inf.

Run:  python3 n1n2_run.py [seeds_per_config] [nproc]
Outputs: results/n1n2_rows.csv, results/traces/<iid>.npz
"""
from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

from games import all_configs, make_game, solve_nash
from ireg_prm import IRPRMPlus, gamma_shift, set_seed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRACES = os.path.join(RESULTS, "traces")
SUPP_TAU = 1e-6          # support = rtilde > SUPP_TAU * N(t)
SNAP_REL = 1e-4          # snapshot when gap < SNAP_REL * gap(t=1)
GAP_CHECK_EVERY = 10
FD_EPS_REL = 1e-7
UNIT_TOL = 1e-5          # |lambda| > 1 - UNIT_TOL counts as unit eigenvalue


def t_horizon(dim: int) -> int:
    """Longer horizon for slow d50 instances (gate2b: t_conv up to 8.1e5)."""
    return 1_000_000 if dim >= 50 else 200_000


# ---------------------------------------------------------------------------
# one-step map Phi on state z = (rtilde1, rtilde2), pure function
# ---------------------------------------------------------------------------

def _mk_player(r: np.ndarray) -> IRPRMPlus:
    p = IRPRMPlus(r.shape[0])
    p.rtilde = r.copy()
    s = float(r.sum())
    p.xtilde = (r / s) if s > 0.0 else np.full(r.shape[0], 1.0 / r.shape[0])
    return p


def phi(A: np.ndarray, z: np.ndarray) -> np.ndarray:
    """One full IREG-PRM+ iteration as a map on z = concat(r1, r2)."""
    n1 = A.shape[0]
    p1 = _mk_player(z[:n1])
    p2 = _mk_player(z[n1:])
    m1 = A @ p2.xtilde
    m2 = -(p1.xtilde @ A)
    x = p1.next_strategy(m1)
    y = p2.next_strategy(m2)
    p1.observe_utility(A @ y)
    p2.observe_utility(-(x @ A))
    return np.concatenate([p1.rtilde, p2.rtilde])


def fd_jacobian(A: np.ndarray, z0: np.ndarray, eps_rel: float = FD_EPS_REL,
                ) -> Tuple[np.ndarray, float]:
    """Finite-difference Jacobian of phi at z0.

    Central differences for clearly-positive coords, forward for ~0 coords
    (negative rtilde is not a reachable state). Returns (J, consistency):
    consistency = median relative column diff between eps and 10*eps on a
    probe subset (large value => z0 sits on a nonsmooth piece boundary).
    """
    n = z0.size
    scale = float(np.max(z0))
    eps = eps_rel * max(scale, 1e-12)
    J = np.zeros((n, n))
    f0 = phi(A, z0)
    dust = 10.0 * eps
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = eps
        if z0[j] > dust:
            J[:, j] = (phi(A, z0 + ej) - phi(A, z0 - ej)) / (2.0 * eps)
        else:
            J[:, j] = (phi(A, z0 + ej) - f0) / eps
    # consistency probe on up to 6 random-ish columns
    rng = np.random.default_rng(0)
    probe = rng.choice(n, size=min(6, n), replace=False)
    rel: List[float] = []
    for j in probe:
        ej = np.zeros(n)
        ej[j] = 10.0 * eps
        if z0[j] > dust:
            col = (phi(A, z0 + ej) - phi(A, z0 - ej)) / (20.0 * eps)
        else:
            col = (phi(A, z0 + ej) - f0) / (10.0 * eps)
        denom = max(float(np.linalg.norm(J[:, j])), 1e-300)
        rel.append(float(np.linalg.norm(col - J[:, j])) / denom)
    return J, float(np.median(rel))


def spectrum_summary(J: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Return (top |eig| desc, rho2 = largest below unit band, n_unit)."""
    ev = np.abs(np.linalg.eigvals(J))
    ev_sorted = np.sort(ev)[::-1]
    n_unit = int(np.sum(ev_sorted > 1.0 - UNIT_TOL))
    below = ev_sorted[ev_sorted <= 1.0 - UNIT_TOL]
    rho2 = float(below[0]) if below.size else float("nan")
    return ev_sorted[:12], rho2, n_unit


# ---------------------------------------------------------------------------
# instrumented run (single instance)
# ---------------------------------------------------------------------------

def run_instance(cfg: Dict) -> Optional[Dict]:
    set_seed(cfg["seed"])
    A = make_game(cfg["dim"], cfg["dist"], cfg["gen"], cfg["seed"])
    nash = solve_nash(A)
    if nash is None:
        return None
    n1, n2 = A.shape
    T = t_horizon(cfg["dim"])
    rec_every = max(10, T // 4000)
    p1, p2 = IRPRMPlus(n1), IRPRMPlus(n2)

    ts: List[int] = []
    gaps: List[float] = []
    norms1: List[float] = []
    norms2: List[float] = []
    closs: List[float] = []      # cumulative clip loss (both players)
    cg2: List[float] = []        # cumulative ||g||^2 (both players)
    cg2_supp: List[float] = []   # cumulative ON-SUPPORT energy sum_{j in S} g_j^2
    gam_err: List[float] = []    # |gamma1 - v*| at record times
    cum_loss = 0.0
    cum_g2 = 0.0
    cum_g2_supp = 0.0
    pat1 = np.zeros(n1, dtype=bool)
    pat2 = np.zeros(n2, dtype=bool)
    t_freeze = 0                 # last support-pattern change (either player)
    n_changes = 0
    epat1 = np.zeros(n1, dtype=bool)
    epat2 = np.zeros(n2, dtype=bool)
    t_freeze_exact = 0           # last EXACT-zero-pattern change (rtilde > 0)
    n_changes_exact = 0
    snap_z: Optional[np.ndarray] = None
    snap_t = -1
    snap_gap = float("nan")
    g0 = float("nan")

    for t in range(1, T + 1):
        m1 = A @ p2.xtilde
        m2 = -(p1.xtilde @ A)
        want_rec = (t % rec_every == 0) or (t == 1)
        if want_rec and p1.norm > 0.0:
            gam1 = gamma_shift(p1.rtilde + m1, p1.norm)
        else:
            gam1 = float("nan")
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        u1 = A @ y
        u2 = -(x @ A)
        # instrumented: exact truncation loss D(t) = N(t)^2 + ||g||^2
        # - N(t+1)^2 (anchor notes ANCHOR_THEORY.md section 3), per player;
        # on-support energy sum_{j in S} g_j^2 (Lemma lem:ndot-identity),
        # computed from the pre-update support S = {j : rtilde_j > 0}, exactly
        # the theorem's Delta N_tot but free of float64 accumulated-cumulant
        # cancellation that plagues diff(norm^2) once norms reach ~1e1.
        for p, u in ((p1, u1), (p2, u2)):
            res = u - p.m
            g = res - float(res @ p.x)
            n_before2 = p.norm ** 2
            g2 = float(g @ g)
            supp = p.rtilde > 0.0
            g_supp2 = float((g * supp) @ (g * supp))
            p.observe_utility(u)
            cum_g2 += g2
            cum_g2_supp += g_supp2
            cum_loss += n_before2 + g2 - p.norm ** 2
        # support pattern freeze tracking
        np1 = p1.rtilde > SUPP_TAU * max(p1.norm, 1e-300)
        np2 = p2.rtilde > SUPP_TAU * max(p2.norm, 1e-300)
        if not (np.array_equal(np1, pat1) and np.array_equal(np2, pat2)):
            t_freeze = t
            n_changes += 1
            pat1, pat2 = np1, np2
        ep1 = p1.rtilde > 0.0
        ep2 = p2.rtilde > 0.0
        if not (np.array_equal(ep1, epat1) and np.array_equal(ep2, epat2)):
            t_freeze_exact = t
            n_changes_exact += 1
            epat1, epat2 = ep1, ep2
        # gap checks: snapshot + records
        if t == 1 or t % GAP_CHECK_EVERY == 0 or want_rec:
            gap = float(np.max(A @ y) - np.min(x @ A))
            if t == 1:
                g0 = gap
            if snap_z is None and t >= 200 and gap < SNAP_REL * g0 \
                    and gap > 0.0:
                snap_z = np.concatenate([p1.rtilde, p2.rtilde])
                snap_t = t
                snap_gap = gap
            if want_rec:
                ts.append(t)
                gaps.append(gap)
                norms1.append(p1.norm)
                norms2.append(p2.norm)
                closs.append(cum_loss)
                cg2.append(cum_g2)
                cg2_supp.append(cum_g2_supp)
                gam_err.append(abs(gam1 - nash.value))

    # N2: FD Jacobian at snapshot
    rho2, n_unit, fd_cons = float("nan"), -1, float("nan")
    ev_top = np.full(12, np.nan)
    if snap_z is not None:
        J, fd_cons = fd_jacobian(A, snap_z)
        ev_top, rho2, n_unit = spectrum_summary(J)

    # support match vs LP equilibrium (final pattern)
    supp_lp1 = nash.x > 1e-8
    supp_lp2 = nash.y > 1e-8
    match1 = bool(np.array_equal(pat1, supp_lp1))
    match2 = bool(np.array_equal(pat2, supp_lp2))

    iid = cfg["instance_id"]
    np.savez_compressed(
        os.path.join(TRACES, f"{iid}.npz"),
        ts=np.array(ts), gaps=np.array(gaps),
        norms1=np.array(norms1), norms2=np.array(norms2),
        closs=np.array(closs), cg2=np.array(cg2),
        cg2_supp=np.array(cg2_supp),
        gam_err=np.array(gam_err), ev_top=ev_top,
        snap_t=snap_t, snap_gap=snap_gap)
    return {
        "instance_id": iid, "dim": cfg["dim"], "dist": cfg["dist"],
        "gen": cfg["gen"], "seed": cfg["seed"], "T": T,
        "supp_x_lp": int(supp_lp1.sum()), "supp_y_lp": int(supp_lp2.sum()),
        "fully_mixed": nash.fully_mixed, "margin": nash.strict_comp_margin,
        "t_freeze": t_freeze, "n_pattern_changes": n_changes,
        "t_freeze_exact": t_freeze_exact,
        "n_pattern_changes_exact": n_changes_exact,
        "supp_match1": match1, "supp_match2": match2,
        "N_inf1": p1.norm, "N_inf2": p2.norm,
        "gamma_err_final": gam_err[-1] if gam_err else float("nan"),
        "gap_final": gaps[-1] if gaps else float("nan"), "gap0": g0,
        "snap_t": snap_t, "snap_gap": snap_gap,
        "rho2": rho2, "n_unit_eigs": n_unit, "fd_consistency": fd_cons,
        "slope_pred": float(np.log(rho2)) if np.isfinite(rho2) else
        float("nan"),
    }


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 56
    os.makedirs(TRACES, exist_ok=True)
    cfgs = all_configs(seeds)
    print(f"[n1n2] {len(cfgs)} instances, nproc={nproc}", flush=True)
    with Pool(nproc) as pool:
        rows = [r for r in pool.imap_unordered(run_instance, cfgs)
                if r is not None]
    rows.sort(key=lambda r: str(r["instance_id"]))
    out = os.path.join(RESULTS, "n1n2_rows.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[n1n2] wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
