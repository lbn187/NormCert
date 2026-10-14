"""Falsifiable prediction check (G2(b) in the paper):

Converged trajectories should satisfy eta_inf * sigma_max(B) <= 1, where
eta_inf = 1/sqrt(c1*c2) at the converged state and B = U = At @ P_psi built
from the actual converged support/strategies.  A converged instance with
eta*sigma_max clearly > 1 falsifies the self-stabilization picture (or flags
non-strict-complementarity / exceptional-set membership).

Also records the margin distribution q_max = (eta*sigma_max)^2 vs the q=1
boundary (GAP-3: nonhyperbolic boundary).

Run:  python3 eta_sigma_check.py [n_instances] [nproc]
"""
from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np

from games import make_game
from n1n2_run import RESULTS
from nb_closedform import run_state


def one(job: Dict) -> Optional[Dict]:
    r = job["row"]
    A = make_game(int(r["dim"]), r["dist"], r["gen"], int(r["seed"]))
    n1 = A.shape[0]
    z = run_state(A, T=int(r["snap_t"]) * 4 + 2000)
    r1, r2 = z[:n1], z[n1:]
    c1, c2 = float(r1.sum()), float(r2.sum())
    s1 = r1 > 0.0
    s2 = r2 > 0.0
    xi = r1[s1] / c1
    psi = r2[s2] / c2
    At = A[np.ix_(s1, s2)]
    U = At @ (np.eye(int(s2.sum())) - np.outer(psi, np.ones(int(s2.sum()))))
    sig = np.linalg.svd(U, compute_uv=False)
    eta = 1.0 / np.sqrt(c1 * c2)
    es = float(eta * sig[0])
    return {"instance_id": r["instance_id"], "dim": r["dim"],
            "eta_sigma_max": es, "q_max": es * es,
            "n_inf1": float(np.linalg.norm(r1)),
            "n_inf2": float(np.linalg.norm(r2))}


def main() -> None:
    n_take = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if np.isfinite(float(r["rho2"]))
                and float(r["gap_final"]) < 1e-10 * float(r["gap0"])]
    rng = np.random.default_rng(11)
    take = list(rng.choice(rows, size=min(n_take, len(rows)), replace=False))
    with Pool(min(nproc, len(take))) as pool:
        recs = [x for x in pool.imap_unordered(one, [{"row": r} for r in take])
                if x is not None]
    es = np.array([x["eta_sigma_max"] for x in recs])
    n_viol = int(np.sum(es > 1.0))
    print(f"n={len(recs)} converged instances")
    print(f"eta*sigma_max: median {np.median(es):.4f}  "
          f"p90 {np.percentile(es, 90):.4f}  max {np.max(es):.4f}")
    print(f"violations (>1): {n_viol}")
    print(f"q_max distance to boundary 1: median {np.median(1 - es**2):.4f} "
          f" min {np.min(1 - es**2):.4f}")
    recs.sort(key=lambda x: -x["eta_sigma_max"])
    for x in recs[:8]:
        print(f"  {x['instance_id']:34s} eta*sig={x['eta_sigma_max']:.4f}")
    out = os.path.join(RESULTS, "eta_sigma_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
