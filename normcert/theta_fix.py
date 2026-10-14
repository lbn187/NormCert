"""Fixed theta-exponent measurement for the T2'-ratio certificate.

Theory (T2' in the paper): in the local linear regime,
Ndot := N(t+1)^2 - N(t)^2 (and per-step ||g||^2) scale like gap^2,
i.e. log Ndot = 2 log gap + const  =>  theta = 2.

Naive whole-run measurement fails: differencing N^2 across log-spaced checkpoints mixed
regimes and hit float cancellation (dN^2 < 1e-16 unrepresentable once
gap ~ 1e-12).  Fix here:
  * window = records at t >= snap_t (gap < 1e-4 * gap0, linear regime only);
  * numerator = increments of cum ||g||^2 (pure sum of positives, no
    cancellation) with an explicit representability guard;
  * two estimators per instance:
      theta_xy    = OLS slope of log(dG2/dt) vs log(gap_mid)
      theta_ratio = slope_t(log dG2/dt) / slope_t(log gap)
    (the ratio estimator averages complex-spectrum oscillation out of
    both axes before dividing).

Run:  python3 theta_fix.py            # uses results/traces/*.npz
Writes results/theta_rows.csv, results/theta_summary.txt, figs/theta_hist.png
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
TRACES = os.path.join(RESULTS, "traces")
FIGS = os.path.join(BASE, "figs")

EPS = np.finfo(float).eps
REPR_GUARD = 64.0          # dG2 > REPR_GUARD * eps * cum  to survive float
GAP_FLOOR = 1e-13          # absolute gap floor (payoffs are O(1))
MIN_PTS = 6


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple:
    """OLS slope, intercept, R^2."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ybar = float(np.mean(y))
    ss_tot = float(np.sum((y - ybar) ** 2))
    ss_res = float(res[0]) if res.size else float(np.sum((A @ coef - y) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def theta_instance(row: Dict) -> Optional[Dict]:
    iid = row["instance_id"]
    snap_t = int(row["snap_t"])
    if snap_t < 0:
        return None
    # frozen-regime window (R2 review fix): Lemma ndot-identity requires the
    # frozen support, so the window must start no earlier than the last
    # exact-zero-pattern change, not just the snapshot.
    t_freeze = int(float(row.get("t_freeze_exact", 0) or 0))
    snap_t = max(snap_t, t_freeze)
    d = np.load(os.path.join(TRACES, f"{iid}.npz"))
    ts, gaps, cg2 = d["ts"], d["gaps"], d["cg2"]
    closs = d["closs"]
    # on-support cumulative energy (Lemma lem:ndot-identity):
    # sum_{j in S} g_j^2 accumulated per step, THE theorem's exact Delta N_tot
    # but computed directly from g (small magnitudes, no float64
    # accumulated-cumulant cancellation).  Only present in the newest rerun;
    # fall back to nan on older traces so the script stays runnable.
    if "cg2_supp" in d.files:
        cg2_supp = d["cg2_supp"]
    else:
        cg2_supp = np.full(ts.size, np.nan)
    # linear-regime window
    keep = ts >= snap_t
    ts, gaps, cg2, closs = ts[keep], gaps[keep], cg2[keep], closs[keep]
    cg2_supp = cg2_supp[keep]
    if ts.size < MIN_PTS + 1:
        return None
    dt = np.diff(ts).astype(float)
    dg2 = np.diff(cg2)
    dn2 = np.diff(cg2 - closs)          # telescoped Ndot (noisier: closs
    #                                     accumulates per-step cancellation)
    dgs2 = np.diff(cg2_supp)            # on-support energy increments:
    #                                     exact Delta N_tot, cancellation-free
    gm = np.sqrt(gaps[:-1] * np.maximum(gaps[1:], 0.0))
    cum = cg2[1:]
    ok = ((dg2 > REPR_GUARD * EPS * cum) & (gm > GAP_FLOOR)
          & (gaps[:-1] > 0) & (gaps[1:] > 0))
    if int(ok.sum()) < MIN_PTS:
        return None
    lx = np.log(gm[ok])
    ly = np.log(dg2[ok] / dt[ok])
    lt = (ts[:-1][ok] + ts[1:][ok]).astype(float) / 2.0
    # estimator 1: direct log-log slope
    th_xy, _, r2_xy = fit_line(lx, ly)
    # estimator 2: ratio of time-slopes (oscillation-robust)
    s_gap, _, r2_gap = fit_line(lt, lx)
    s_g2, _, r2_g2 = fit_line(lt, ly)
    th_ratio = s_g2 / s_gap if abs(s_gap) > 0 else float("nan")
    # secondary: same on exact Ndot increments where representable
    okn = ok & (dn2 > REPR_GUARD * EPS * cum)
    th_n = float("nan")
    if int(okn.sum()) >= MIN_PTS:
        th_n, _, _ = fit_line(np.log(np.sqrt(gaps[:-1][okn] * gaps[1:][okn])),
                              np.log(dn2[okn] / dt[okn]))
    # PRIMARY: same slope-two law on the on-support energy, i.e. on
    # the theorem's exact Delta N_tot.  Its guard is relative to the on-support
    # cumulative (small, no ~1e1 large-cumulant floor), so late-regime signal
    # survives where dg2/dn2 hit the float noise floor.
    ok_supp = ((np.isfinite(dgs2))
               & (dgs2 > REPR_GUARD * EPS * np.maximum(cg2_supp[1:], 1e-300))
               & (gm > GAP_FLOOR) & (gaps[:-1] > 0) & (gaps[1:] > 0))
    th_ons, r2_ons = float("nan"), float("nan")
    n_ons = 0
    if int(ok_supp.sum()) >= MIN_PTS:
        th_ons, _, r2_ons = fit_line(np.log(gm[ok_supp]),
                                     np.log(dgs2[ok_supp] / dt[ok_supp]))
        n_ons = int(ok_supp.sum())
    return {
        "instance_id": iid, "dim": row["dim"], "dist": row["dist"],
        "gen": row["gen"], "n_pts": int(ok.sum()),
        "theta_xy": th_xy, "r2_xy": r2_xy,
        "theta_ratio": th_ratio, "r2_gap_t": r2_gap, "r2_g2_t": r2_g2,
        "theta_ndot": th_n,
        "theta_onsupp": th_ons, "r2_onsupp": r2_ons, "n_pts_onsupp": n_ons,
        "gap_span_dec": float(np.log10(np.max(gm[ok]) / np.min(gm[ok]))),
    }


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        rows = list(csv.DictReader(f))
    out: List[Dict] = []
    for r in rows:
        rec = theta_instance(r)
        if rec is not None:
            out.append(rec)
    with open(os.path.join(RESULTS, "theta_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    lines: List[str] = []

    def rep(tag: str, vals: np.ndarray, r2: Optional[np.ndarray] = None,
            r2_min: float = 0.0) -> None:
        v = vals[np.isfinite(vals)]
        if r2 is not None:
            m = np.isfinite(vals) & np.isfinite(r2) & (r2 >= r2_min)
            v = vals[m]
        if v.size == 0:
            lines.append(f"{tag}: no usable instances")
            return
        frac = float(np.mean((v > 1.8) & (v < 2.2)))
        lines.append(
            f"{tag}: n={v.size} median={np.median(v):.3f} "
            f"[p10 {np.percentile(v, 10):.3f}, p90 {np.percentile(v, 90):.3f}]"
            f" frac_in_[1.8,2.2]={frac:.1%}")

    th_xy = np.array([o["theta_xy"] for o in out])
    r2_xy = np.array([o["r2_xy"] for o in out])
    th_ra = np.array([o["theta_ratio"] for o in out])
    r2g = np.array([o["r2_gap_t"] for o in out])
    th_n = np.array([o["theta_ndot"] for o in out])
    th_os = np.array([o["theta_onsupp"] for o in out])
    r2_os = np.array([o["r2_onsupp"] for o in out])
    span = np.array([o["gap_span_dec"] for o in out])

    lines.append(f"usable instances: {len(out)}/{len(rows)} "
                 f"(window: t>=snap_t, dG2 repr-guard, gap>{GAP_FLOOR:g})")
    lines.append(f"gap decades in window: median {np.median(span):.2f}, "
                 f"p10 {np.percentile(span, 10):.2f}")
    rep("theta_xy   (all)          ", th_xy)
    rep("theta_xy   (R2>=0.8)      ", th_xy, r2_xy, 0.8)
    rep("theta_ratio(all)          ", th_ra)
    rep("theta_ratio(R2_gap>=0.8)  ", th_ra, r2g, 0.8)
    rep("theta_Ndot (exact, repr.) ", th_n)
    # PRIMARY: on-support energy = the theorem's exact Delta N_tot.
    # n_ons counts instances whose on-support increments pass the guard
    # (fewer than theta_xy: guard is relative to the small on-support
    # cumulative, so it is tighter against noise but exact).
    n_os = np.array([o["n_pts_onsupp"] for o in out], dtype=float)
    os_ok = np.isfinite(th_os) & (n_os >= MIN_PTS)
    lines.append(f"theta_onsupp usable: {int(os_ok.sum())}/{len(out)} "
                 f"(on-support energy, exact Delta N_tot)")
    rep("theta_onsupp(all)         ", th_os[os_ok])
    rep("theta_onsupp(R2>=0.8)     ", th_os[os_ok], r2_os[os_ok], 0.8)

    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "theta_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m = np.isfinite(th_xy) & np.isfinite(r2_xy) & (r2_xy >= 0.8)
    axes[0].hist(th_xy[m], bins=30)
    axes[0].axvline(2.0, color="r", ls="--", label="theory 2")
    axes[0].set_title(f"theta_xy, R2>=0.8 (n={int(m.sum())})")
    axes[0].set_xlabel("theta")
    axes[0].legend()
    m2 = np.isfinite(th_ra) & np.isfinite(r2g) & (r2g >= 0.8)
    axes[1].hist(th_ra[m2], bins=30)
    axes[1].axvline(2.0, color="r", ls="--")
    axes[1].set_title(f"theta_ratio, R2_gap>=0.8 (n={int(m2.sum())})")
    axes[1].set_xlabel("theta")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "theta_hist.png"), dpi=110)
    print(f"wrote figs/theta_hist.png")


if __name__ == "__main__":
    main()
