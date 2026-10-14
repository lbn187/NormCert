"""Closed-form Jacobian spectrum (T1' in the paper) vs the FD Jacobian.

Two modes:
  lp        -- build Q from the LP equilibrium supports;
               mismatch expected when the run converges to a different NE.
  converged -- build Q from the ACTUAL converged state: exact-zero support
               pattern of rtilde, xi/psi = normalized rtilde, c = ||rtilde||_1.
               This is the honest test of the J = I + L + L^2 formula.

Run:  python3 nb_closedform.py [n_instances] [mode] [nproc]
"""
from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np

from games import make_game, solve_nash
from ireg_prm import IRPRMPlus
from n1n2_run import fd_jacobian, spectrum_summary, RESULTS

SUPPORT_TOL = 1e-8


def mu_from_blocks(At: np.ndarray, xi: np.ndarray, psi: np.ndarray,
                   c1: float, c2: float) -> np.ndarray:
    """|mu| sorted desc from J = I + L + L^2, general (possibly nonsym) Q."""
    k1, k2 = At.shape
    P_xi = np.eye(k1) - np.outer(xi, np.ones(k1))
    P_psi = np.eye(k2) - np.outer(psi, np.ones(k2))
    U = At @ P_psi
    V = At.T @ P_xi
    Q = (U @ V) / (c1 * c2)
    q = np.linalg.eigvals(Q).astype(complex)
    # drop kernel modes (B psi = 0, B^T xi = 0): q = 0 <=> mu = 1 are the
    # fixed-manifold (scale) directions, excluded from the rate
    q = q[np.abs(q) > 1e-9 * max(float(np.max(np.abs(q))), 1e-300)]
    lam = np.concatenate([1j * np.sqrt(q), -1j * np.sqrt(q)])
    mu = 1.0 + lam + lam * lam
    return np.sort(np.abs(mu))[::-1]


def run_state(A: np.ndarray, T: int) -> np.ndarray:
    """Run IREG-PRM+ for T iters, return final state z = (r1, r2)."""
    n1, n2 = A.shape
    p1, p2 = IRPRMPlus(n1), IRPRMPlus(n2)
    for _ in range(T):
        m1 = A @ p2.xtilde
        m2 = -(p1.xtilde @ A)
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        p1.observe_utility(A @ y)
        p2.observe_utility(-(x @ A))
    return np.concatenate([p1.rtilde, p2.rtilde])


def one_instance(job: Dict) -> Optional[Dict]:
    r, mode = job["row"], job["mode"]
    A = make_game(int(r["dim"]), r["dist"], r["gen"], int(r["seed"]))
    n1 = A.shape[0]
    z = run_state(A, T=int(r["snap_t"]) * 4 + 2000)
    r1, r2 = z[:n1], z[n1:]
    c1, c2 = float(r1.sum()), float(r2.sum())
    if mode == "lp":
        nash = solve_nash(A)
        s1 = nash.x > SUPPORT_TOL
        s2 = nash.y > SUPPORT_TOL
        xi = nash.x[s1]
        psi = nash.y[s2]
    else:  # converged: exact-zero pattern (L3: off-support frozen at 0)
        s1 = r1 > 0.0
        s2 = r2 > 0.0
        xi = r1[s1] / c1
        psi = r2[s2] / c2
    At = A[np.ix_(s1, s2)]
    mu = mu_from_blocks(At, xi, psi, c1, c2)
    rho_cf = float(mu[0])
    # FD Jacobian at the same state, not a stale snapshot, for a
    # like-for-like comparison, plus the stored snapshot rho2 from the csv
    J, fd_cons = fd_jacobian(A, z)
    _, rho_fd_here, n_unit = spectrum_summary(J)
    rho_fd_csv = float(r["rho2"])
    err = abs(rho_cf - rho_fd_here) / max(rho_fd_here, 1e-300)
    err_csv = abs(rho_cf - rho_fd_csv) / max(rho_fd_csv, 1e-300)
    return {
        "instance_id": r["instance_id"], "mode": mode,
        "supp": f"{int(s1.sum())}x{int(s2.sum())}",
        "rho_fd_here": rho_fd_here, "rho_fd_csv": rho_fd_csv,
        "rho_cf": rho_cf, "rel_err": err, "rel_err_csv": err_csv,
        "n_unit": n_unit, "fd_cons": fd_cons,
    }


def main() -> None:
    n_take = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    mode = sys.argv[2] if len(sys.argv) > 2 else "converged"
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    assert mode in ("lp", "converged")
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if np.isfinite(float(r["rho2"]))
                and float(r["gap_final"]) < 1e-10 * float(r["gap0"])]
    rng = np.random.default_rng(7)
    take = list(rng.choice(rows, size=min(n_take, len(rows)), replace=False))
    jobs = [{"row": r, "mode": mode} for r in take]
    with Pool(min(nproc, len(jobs))) as pool:
        recs = [x for x in pool.imap_unordered(one_instance, jobs)
                if x is not None]
    recs.sort(key=lambda x: x["rel_err"])
    print(f"mode={mode}  n={len(recs)}")
    print(f"{'instance':34s} {'supp':>7s} {'rho2_FD':>9s} {'rho2_CF':>9s}"
          f" {'rel_err':>8s} {'n_unit':>6s}")
    for x in recs:
        print(f"{x['instance_id']:34s} {x['supp']:>7s} {x['rho_fd_here']:9.5f}"
              f" {x['rho_cf']:9.5f} {x['rel_err']:8.2%} {x['n_unit']:6d}")
    errs = np.array([x["rel_err"] for x in recs])
    print(f"\nmedian rel err: {np.median(errs):.3%}  "
          f"p90: {np.percentile(errs, 90):.3%}  max: {np.max(errs):.3%}")
    out = os.path.join(RESULTS, f"closedform_{mode}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
