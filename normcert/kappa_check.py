"""Empirical measurement of the ratio-certificate constant kappa (R1, iv).

Theory (07_certificate.tex, Theorem E / thm:certificate): in the frozen
local regime, for all t2 >= t1,

    kappa^-1 sqrt(Ndot_tot(t2)/Ndot_tot(t1))
        <= gap(t2)/gap(t1)
        <= kappa sqrt(Ndot_tot(t2)/Ndot_tot(t1)),

    kappa = (L_gap / c_-) * sqrt(n_+ / n_-).

This script measures, per theta-usable instance (same 129-instance window
logic as theta_fix.py):

  * kappa_emp = max over window pairs (t1, t2) of
        max(ratio_dev, 1/ratio_dev),
    ratio_dev = [gap(t2)/gap(t1)] / sqrt(Ndot_tot(t2)/Ndot_tot(t1)).
    Equivalently kappa_emp = max_t s(t) / min_t s(t) with
    s(t) = gap(t)/sqrt(Ndot_tot(t)) (all pairs reduce to the extremes).
    Two variants: exact Ndot (telescoped N^2 increments, cg2 - closs,
    representability-guarded) and the cancellation-free proxy
    Delta(sum ||g||^2) used by theta_fix.py.

  * kappa_theory assembled per instance from the converged state
    (nb_closedform.py "converged"-mode reconstruction: exact-zero support,
    xi/psi = normalized rtilde, c_i = ||rtilde_i||_1, B = Atil - v* 11^T):
      - c_-  = min(xi_min, psi_min) / (2 sqrt(m)) * c_T          (lem:gap-sharp)
        c_T  = sigmin+ eta_-^2 / sqrt(2 sigmax^2 eta_+^4 + 1/a_-^2 + 1/b_-^2)
      - n_-  = 1/2 eta_-^4 (sigmin+)^4,  n_+ = 2 eta_+^4 sigmax^4 (lem:ndot-theta)
      - L_gap: the paper leaves the Lipschitz constant non-closed-form
        (lem:gap-lip); we instantiate its tight first-order value
        L_gap = sqrt(2) * max_k sigma_k * sigmax(M_k(a_-, b_-))
        from the same block map T_{a,b} (lem:tmap) that yields c_T.  The
        true global Lipschitz constant dominates this value, so the
        assembled kappa_theory is a LOWER bound on the paper's kappa and
        the test kappa_emp <= kappa_theory is conservative (harder to pass).
    Two eta-ranges: "win" uses the (a, b) range the trajectory actually
    occupies in the measurement window (a_t = N_1(t)/||xi||_2, exact to
    O(||w||^2)); "reg" uses the paper's regime rectangle
    a in [c1/2, 3c1/2], b in [c2/2, 3c2/2] (eta_+/eta_- = 3, worst-case).

Run:  OMP_NUM_THREADS=1 python3 kappa_check.py [nproc]
Reads  results/theta_rows.csv, results/n1n2_rows.csv, results/traces/*.npz
Writes results/kappa_check.csv, results/kappa_summary.txt
Does not modify any other script.
"""
from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

from games import make_game
from ireg_prm import IRPRMPlus, duality_gap

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
TRACES = os.path.join(RESULTS, "traces")

EPS = np.finfo(float).eps
REPR_GUARD = 64.0        # same guard as theta_fix.py
GAP_FLOOR = 1e-13        # same absolute gap floor as theta_fix.py
MIN_PTS = 6              # same minimum surviving increments as theta_fix.py
KER_TOL = 1e-8           # relative kernel threshold on singular values


# ---------------------------------------------------------------------------
# empirical side: kappa_emp from the trace, theta_fix window logic
# ---------------------------------------------------------------------------

def emp_side(iid: str, snap_t: int) -> Optional[Dict]:
    """Window quantities + empirical kappa for one instance."""
    d = np.load(os.path.join(TRACES, f"{iid}.npz"))
    ts, gaps, cg2, closs = d["ts"], d["gaps"], d["cg2"], d["closs"]
    n1s, n2s = d["norms1"], d["norms2"]
    # on-support cumulative energy (Lemma lem:ndot-identity), only in the
    # nan fallback keeps the script runnable on traces without this field.
    cg2_supp = d["cg2_supp"] if "cg2_supp" in d.files \
        else np.full(ts.size, np.nan)
    keep = ts >= snap_t
    ts, gaps, cg2, closs = ts[keep], gaps[keep], cg2[keep], closs[keep]
    cg2_supp = cg2_supp[keep]
    n1s, n2s = n1s[keep], n2s[keep]
    if ts.size < MIN_PTS + 1:
        return None
    dt = np.diff(ts).astype(float)
    dn2 = np.diff(n1s ** 2 + n2s ** 2)      # telescoped Delta N^2 (proxy):
                                            # cancellation-free, THE usable rate
    dg2 = np.diff(cg2)                      # Delta cum||g||^2: additive-cancel
                                            # after norm saturation, not usable
    dgs2 = np.diff(cg2_supp)                # on-support energy increments:
                                            # THE theorem's exact Delta N_tot,
                                            # cancellation-free (PRIMARY)
    gm = np.sqrt(gaps[:-1] * np.maximum(gaps[1:], 0.0))
    cum = cg2[1:]
    ok = ((dg2 > REPR_GUARD * EPS * cum) & (gm > GAP_FLOOR)
          & (gaps[:-1] > 0) & (gaps[1:] > 0))
    okn = ((dn2 > REPR_GUARD * EPS * cum) & (gm > GAP_FLOOR)
           & (gaps[:-1] > 0) & (gaps[1:] > 0))
    oks = ((np.isfinite(dgs2))
           & (dgs2 > REPR_GUARD * EPS * np.maximum(cg2_supp[1:], 1e-300))
           & (gm > GAP_FLOOR) & (gaps[:-1] > 0) & (gaps[1:] > 0))

    def kap(mask: np.ndarray, num: np.ndarray) -> Tuple[float, int]:
        if int(mask.sum()) < MIN_PTS:
            return float("nan"), int(mask.sum())
        s = gm[mask] / np.sqrt(num[mask] / dt[mask])
        return float(np.max(s) / np.min(s)), int(mask.sum())

    kap_n, npts_n = kap(okn, dn2)
    kap_g, npts_g = kap(ok, dg2)
    kap_s, npts_s = kap(oks, dgs2)
    return {
        "kap_emp_ndot": kap_n, "n_pts_ndot": npts_n,
        "kap_emp_g2": kap_g, "n_pts_g2": npts_g,
        "kap_emp_onsupp": kap_s, "n_pts_onsupp": npts_s,
        "norms1_win": n1s, "norms2_win": n2s,
    }


# ---------------------------------------------------------------------------
# theory side: rerun to convergence, rebuild B, assemble kappa
# ---------------------------------------------------------------------------

def run_state(A: np.ndarray, T: int) -> np.ndarray:
    """Run IREG-PRM+ for T iters, return final state z = (r1, r2).

    Same loop as nb_closedform.run_state (not imported to keep this file
    self-contained without touching other scripts).
    """
    n1, n2 = A.shape
    p1, p2 = IRPRMPlus(n1), IRPRMPlus(n2)
    for _ in range(T):
        m1 = A @ p2.xtilde
        m2 = -(p1.xtilde @ A)
        x = p1.next_strategy(m1)
        y = p2.next_strategy(m2)
        p1.observe_utility(A @ y)
        p2.observe_utility(-(x @ A))
    return np.concatenate([p1.rtilde, p2.rtilde]), p1.x, p2.x


def lgap_first_order(sig: np.ndarray, a: float, b: float) -> float:
    """sqrt(2) * sigmax(T_{a,b}) via the 2x2 blocks M_k (lem:tmap)."""
    smax = 0.0
    for sk in sig:
        Mk = np.array([[-sk / (a * b), 1.0 / b],
                       [1.0 / a, sk / (a * b)]])
        smax = max(smax, sk * float(np.linalg.svd(Mk, compute_uv=False)[0]))
    return float(np.sqrt(2.0) * smax)


def assemble_kappa(sig_min: float, sig_max: float, sig: np.ndarray,
                   xi_min: float, psi_min: float, m: int,
                   eta_lo: float, eta_hi: float,
                   a_lo: float, b_lo: float) -> Dict:
    """kappa = (L_gap / c_-) sqrt(n_+ / n_-) from the paper's formulas."""
    c_t = (sig_min * eta_lo ** 2
           / np.sqrt(2.0 * sig_max ** 2 * eta_hi ** 4
                     + 1.0 / a_lo ** 2 + 1.0 / b_lo ** 2))
    c_minus = min(xi_min, psi_min) / (2.0 * np.sqrt(m)) * c_t
    n_minus = 0.5 * eta_lo ** 4 * sig_min ** 4
    n_plus = 2.0 * eta_hi ** 4 * sig_max ** 4
    l_gap = lgap_first_order(sig, a_lo, b_lo)
    kappa = l_gap / c_minus * np.sqrt(n_plus / n_minus)
    return {"c_T": float(c_t), "c_minus": float(c_minus),
            "n_minus": float(n_minus), "n_plus": float(n_plus),
            "L_gap": float(l_gap), "kappa": float(kappa)}


def one_instance(row: Dict) -> Dict:
    iid = row["instance_id"]
    # frozen-regime window (R2 review fix): same max(snap, freeze) start as
    # theta_fix.py, so the Lemma ndot-identity hypothesis holds on-window.
    snap_t = max(int(row["snap_t"]),
                 int(float(row.get("t_freeze_exact", 0) or 0)))
    out: Dict = {"instance_id": iid, "dim": row["dim"],
                 "dist": row["dist"], "gen": row["gen"]}
    emp = emp_side(iid, snap_t)
    if emp is None:
        out["status"] = "no_window"
        return out
    n1s, n2s = emp.pop("norms1_win"), emp.pop("norms2_win")
    out.update(emp)

    # converged-state reconstruction (nb_closedform converged-mode recipe);
    # if the short rerun leaves the kernel of B unresolved (n_ker = 0 can
    # only be a convergence artifact: B psi = 0 exactly at any fixed
    # point), retry once with the full original horizon T.
    A = make_game(int(row["dim"]), row["dist"], row["gen"], int(row["seed"]))
    t_full = int(row["T"])
    for t_end in (min(t_full, snap_t * 4 + 2000), t_full):
        z, x_fin, y_fin = run_state(A, t_end)
        n1 = A.shape[0]
        r1, r2 = z[:n1], z[n1:]
        c1, c2 = float(r1.sum()), float(r2.sum())
        s1, s2 = r1 > 0.0, r2 > 0.0
        m1s, m2s = int(s1.sum()), int(s2.sum())
        out["supp"] = f"{m1s}x{m2s}"
        out["t_end"] = t_end
        out["gap_end"] = duality_gap(A, x_fin, y_fin)  # rerun quality
        if m1s != m2s or min(m1s, m2s) < 2:
            out["status"] = "degenerate_support"
            return out
        m = m1s
        xi = r1[s1] / c1
        psi = r2[s2] / c2
        At = A[np.ix_(s1, s2)]
        v_star = float(xi @ At @ psi)
        B = At - v_star * np.ones((m, m))
        sv = np.linalg.svd(B, compute_uv=False)
        n_ker = int(np.sum(sv <= KER_TOL * sv[0]))
        out["n_ker"] = n_ker
        if n_ker >= 1 or t_end == t_full:
            break
    if n_ker != 1:
        out["status"] = "degenerate_kernel"
        return out
    sig = sv[sv > KER_TOL * sv[0]]
    sig_max, sig_min = float(sig[0]), float(sig[-1])
    out["sig_max"], out["sig_min_plus"] = sig_max, sig_min
    out["xi_min"], out["psi_min"] = float(xi.min()), float(psi.min())
    out["m"] = m
    out["eta_inf"] = 1.0 / np.sqrt(c1 * c2)

    # realized (a, b) range on the measurement window: a_t = N1(t)/||xi||_2
    a_t = n1s / float(np.linalg.norm(xi))
    b_t = n2s / float(np.linalg.norm(psi))
    a_lo, a_hi = float(a_t.min()), float(a_t.max())
    b_lo, b_hi = float(b_t.min()), float(b_t.max())
    eta_lo = 1.0 / np.sqrt(a_hi * b_hi)
    eta_hi = 1.0 / np.sqrt(a_lo * b_lo)
    out["eta_ratio_win"] = eta_hi / eta_lo

    win = assemble_kappa(sig_min, sig_max, sig, out["xi_min"],
                         out["psi_min"], m, eta_lo, eta_hi, a_lo, b_lo)
    out.update({f"{k}_win": v for k, v in win.items()})
    # paper regime rectangle: a in [c1/2, 3c1/2], b in [c2/2, 3c2/2]
    reg = assemble_kappa(sig_min, sig_max, sig, out["xi_min"],
                         out["psi_min"], m,
                         eta_lo=1.0 / np.sqrt(2.25 * c1 * c2),
                         eta_hi=2.0 / np.sqrt(c1 * c2),
                         a_lo=c1 / 2.0, b_lo=c2 / 2.0)
    out["kappa_reg"] = reg["kappa"]

    for tag in ("ndot", "g2", "onsupp"):
        ke = out.get(f"kap_emp_{tag}", float("nan"))
        out[f"hold_{tag}_win"] = bool(np.isfinite(ke)
                                      and ke <= win["kappa"])
        out[f"slack_{tag}_win"] = (win["kappa"] / ke
                                   if np.isfinite(ke) else float("nan"))
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# driver + summary
# ---------------------------------------------------------------------------

def pctl(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q))


def main() -> None:
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with open(os.path.join(RESULTS, "theta_rows.csv")) as f:
        theta_ids = [r["instance_id"] for r in csv.DictReader(f)]
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        nrows = {r["instance_id"]: r for r in csv.DictReader(f)}
    jobs = [nrows[i] for i in theta_ids]
    with Pool(nproc) as pool:
        recs: List[Dict] = list(pool.imap_unordered(one_instance, jobs))
    recs.sort(key=lambda r: r["instance_id"])

    fields: List[str] = []
    for r in recs:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(RESULTS, "kappa_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(recs)

    ok = [r for r in recs if r.get("status") == "ok"]
    excl = [r for r in recs if r.get("status") != "ok"]
    lines: List[str] = [
        "kappa ratio-certificate empirical check "
        "(window = theta_fix.py 129-instance logic; "
        "converged rebuild = nb_closedform converged mode, "
        "T_end = min(T, 4*snap_t + 2000))",
        f"instances: {len(recs)} theta-usable; {len(ok)} with assembled "
        f"kappa_theory; {len(excl)} excluded:",
    ]
    for r in excl:
        lines.append(f"  EXCLUDED {r['instance_id']}: {r['status']} "
                     f"(supp={r.get('supp', '?')}, "
                     f"n_ker={r.get('n_ker', '?')})")

    def dist_line(tag: str, vals: List[float]) -> None:
        v = np.array([x for x in vals if np.isfinite(x)])
        if v.size == 0:
            lines.append(f"{tag}: none finite")
            return
        lines.append(f"{tag}: n={v.size} median={np.median(v):.4g} "
                     f"p90={pctl(v, 90):.4g} max={np.max(v):.4g} "
                     f"min={np.min(v):.4g}")

    lines.append("")
    lines.append("== kappa_emp (exact Ndot_tot = telescoped N^2 increments,"
                 " repr-guarded) ==")
    dist_line("kappa_emp_ndot", [r["kap_emp_ndot"] for r in ok])
    ndot_ok = [r for r in ok if np.isfinite(r["kap_emp_ndot"])]
    lines.append(f"instances with >= {MIN_PTS} guarded exact-Ndot "
                 f"increments: {len(ndot_ok)}/{len(ok)}")
    lines.append("== kappa_emp (cancellation-free proxy "
                 "Delta sum||g||^2, theta_fix numerator) ==")
    dist_line("kappa_emp_g2", [r["kap_emp_g2"] for r in ok])
    lines.append("== kappa_emp (PRIMARY: on-support energy increments, "
                 "exact Delta N_tot) ==")
    dist_line("kappa_emp_onsupp", [r["kap_emp_onsupp"] for r in ok])

    lines.append("")
    lines.append("== kappa_theory (assembled, realized-window eta range; "
                 "lower bound on the paper's kappa) ==")
    dist_line("kappa_theory_win", [r["kappa_win"] for r in ok])
    dist_line("kappa_theory_reg (paper regime rectangle)",
              [r["kappa_reg"] for r in ok])
    dist_line("eta_+/eta_- realized in window",
              [r["eta_ratio_win"] for r in ok])

    lines.append("")
    lines.append("== verdict: kappa_emp <= kappa_theory "
                 "(theory predicts 100%) ==")
    onsupp_ok = [r for r in ok if np.isfinite(r["kap_emp_onsupp"])]
    for tag, sub in (("ndot", ndot_ok),
                     ("g2", [r for r in ok
                             if np.isfinite(r["kap_emp_g2"])]),
                     ("onsupp", onsupp_ok)):
        holds = [r for r in sub if r[f"hold_{tag}_win"]]
        lines.append(f"{tag}: {len(holds)}/{len(sub)} hold "
                     f"({100.0 * len(holds) / max(len(sub), 1):.1f}%)")
        for r in sub:
            if not r[f"hold_{tag}_win"]:
                lines.append(
                    f"  VIOLATION {r['instance_id']}: "
                    f"kappa_emp_{tag}={r[f'kap_emp_{tag}']:.4g} > "
                    f"kappa_theory_win={r['kappa_win']:.4g}")

    lines.append("")
    lines.append("== conservatism: slack = kappa_theory_win / kappa_emp ==")
    dist_line("slack_ndot", [r["slack_ndot_win"] for r in ndot_ok])
    dist_line("slack_g2", [r["slack_g2_win"] for r in ok])
    dist_line("slack_onsupp", [r["slack_onsupp_win"] for r in onsupp_ok])

    lines.append("")
    lines.append("== rerun quality (gap at converged rebuild state) ==")
    dist_line("gap_end", [r["gap_end"] for r in recs if "gap_end" in r])

    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "kappa_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
