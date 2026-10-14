"""Aggregate N1/N2/N3 results: freeze stats, clip decay, spectral prediction.

Run after n1n2_run.py:  python3 n_analyze.py
Outputs: results/n_analysis.txt, figs/n2_spectral.png, figs/n1_clip.png
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRACES = os.path.join(RESULTS, "traces")
FIGS = os.path.join(HERE, "figs")
FIT_R2_MIN = 0.9


def load_rows() -> List[Dict]:
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        return list(csv.DictReader(f))


def fit_measured_slope(iid: str, snap_gap: float,
                       ) -> Tuple[float, float, int]:
    """Semilog fit of gap(t) on the exponential segment after the snapshot.

    Window: gap in [1e3 * floor, 0.5 * snap_gap]. Returns (slope, R2, npts).
    """
    d = np.load(os.path.join(TRACES, f"{iid}.npz"))
    ts, gaps = d["ts"].astype(float), d["gaps"]
    pos = gaps > 0.0
    ts, gaps = ts[pos], gaps[pos]
    if gaps.size < 10 or not np.isfinite(snap_gap):
        return float("nan"), float("nan"), 0
    floor = float(gaps.min())
    lo, hi = 1e3 * floor, 0.5 * snap_gap
    if hi <= lo:  # never reached float floor: fit tail below snap level
        lo = floor * 1.5
    sel = (gaps > lo) & (gaps < hi)
    if sel.sum() < 5:
        return float("nan"), float("nan"), int(sel.sum())
    x, y = ts[sel], np.log(gaps[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(res[0]) / ss_tot if res.size and ss_tot > 0 else 0.0
    return float(coef[0]), r2, int(sel.sum())


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    rows = load_rows()
    lines: List[str] = [f"total instances: {len(rows)}"]

    # ---- N1: freeze + support match + gamma -> v* -------------------------
    tf = np.array([float(r["t_freeze"]) for r in rows])
    T = np.array([float(r["T"]) for r in rows])
    match = np.array([(r["supp_match1"] == "True") and
                      (r["supp_match2"] == "True") for r in rows])
    gerr = np.array([float(r["gamma_err_final"]) for r in rows])
    ninf = np.array([max(float(r["N_inf1"]), float(r["N_inf2"]))
                     for r in rows])
    tfe = np.array([float(r["t_freeze_exact"]) for r in rows])
    lines += [
        "", "== N1: active-set freezing ==",
        f"t_freeze/T median={np.median(tf / T):.4g} "
        f"p90={np.quantile(tf / T, 0.9):.4g} max={np.max(tf / T):.4g}",
        f"frozen before T/2: {np.mean(tf < 0.5 * T):.1%}",
        f"EXACT-zero pattern freeze (L3 hard prediction): "
        f"t/T median={np.median(tfe / T):.4g} "
        f"p90={np.quantile(tfe / T, 0.9):.4g}; "
        f"frozen before T/2: {np.mean(tfe < 0.5 * T):.1%}",
        f"final pattern == LP support (both players): {np.mean(match):.1%}",
        f"|gamma - v*| final: median={np.median(gerr):.3g} "
        f"p90={np.quantile(gerr, 0.9):.3g}",
        "", "== N3: norm saturation levels ==",
        f"N_inf: median={np.median(ninf):.3g} max={np.max(ninf):.3g} "
        f"(eta_inf = 1/N_inf all positive by construction)",
    ]

    # clip-loss decay + T2'-ratio exponent theta (log Ndot vs log gap)
    ratios: List[float] = []
    thetas: List[float] = []
    for r in rows:
        d = np.load(os.path.join(TRACES, f"{r['instance_id']}.npz"))
        cl, cg, ts = d["closs"], d["cg2"], d["ts"]
        k = ts.size // 2
        dcl = cl[-1] - cl[k]
        dcg = cg[-1] - cg[k]
        ratios.append(dcl / dcg if dcg > 0 else float("nan"))
        # theta fit: Ndot(t) ~ increments of N^2 = cg - closs cumulative;
        # use windowed increments of (cg - closs) vs gap at window centers.
        n2 = np.array(d["norms1"]) ** 2 + np.array(d["norms2"]) ** 2
        gaps = d["gaps"]
        dn2 = np.diff(n2)
        dts = np.diff(ts).astype(float)
        gm = gaps[1:]
        ok = (dn2 > 0) & (gm > 0) & (dts > 0)
        if ok.sum() >= 8:
            lx = np.log(gm[ok])
            ly = np.log(dn2[ok] / dts[ok])
            A2 = np.vstack([lx, np.ones_like(lx)]).T
            coef, _, _, _ = np.linalg.lstsq(A2, ly, rcond=None)
            thetas.append(float(coef[0]))
    ra = np.array(ratios)
    ra = ra[np.isfinite(ra)]
    th = np.array(thetas)
    lines += [
        "", "== N1b: late-half clip-loss ratio (loss/||g||^2) ==",
        f"median={np.median(ra):.3g} p90={np.quantile(ra, 0.9):.3g} "
        f"max={np.max(ra):.3g}",
        "", "== T2'-ratio exponent: log Ndot vs log gap, THEORY = 2 ==",
        f"theta: median={np.median(th):.3f} p10={np.quantile(th, 0.1):.3f} "
        f"p90={np.quantile(th, 0.9):.3f} (n={th.size})",
    ]

    # ---- N2: spectral prediction vs measured slope ------------------------
    pred, meas, dims, r2s, cons = [], [], [], [], []
    n_nosnap = n_badfit = 0
    for r in rows:
        sp = float(r["slope_pred"])
        if not np.isfinite(sp):
            n_nosnap += 1
            continue
        sm, r2, _ = fit_measured_slope(r["instance_id"],
                                       float(r["snap_gap"]))
        if not np.isfinite(sm) or r2 < FIT_R2_MIN or sm >= 0:
            n_badfit += 1
            continue
        pred.append(sp)
        meas.append(sm)
        dims.append(int(r["dim"]))
        r2s.append(r2)
        cons.append(float(r["fd_consistency"]))
    p, m = np.array(pred), np.array(meas)
    nu = np.array([int(r["n_unit_eigs"]) for r in rows
                   if np.isfinite(float(r["slope_pred"]))])
    lines += [
        "", "== N2: spectral prediction (slope_pred = log rho2) ==",
        f"usable pairs: {p.size} (no-snapshot: {n_nosnap}, "
        f"bad fit/positive slope: {n_badfit})",
    ]
    if p.size >= 5:
        lp, lm = np.log(-p), np.log(-m)
        corr = float(np.corrcoef(lp, lm)[0, 1])
        ratio = p / m
        lines += [
            f"corr(log(-slope_pred), log(-slope_meas)) = {corr:.4f}"
            f"   [HARD TARGET > 0.9]",
            f"pred/meas ratio: median={np.median(ratio):.3f} "
            f"p10={np.quantile(ratio, 0.1):.3f} "
            f"p90={np.quantile(ratio, 0.9):.3f}",
            f"unit eigenvalues per instance: median={np.median(nu):.0f} "
            f"min={nu.min()} max={nu.max()}  (expect >= 2: scale dirs)",
            f"FD consistency (median rel col diff): "
            f"median={np.median(cons):.2g} p90={np.quantile(cons, 0.9):.2g}",
        ]
        fig, ax = plt.subplots(figsize=(6, 6))
        for d_, c in ((5, "tab:blue"), (20, "tab:orange"), (50, "tab:red")):
            s = np.array(dims) == d_
            ax.loglog(-m[s], -p[s], "o", ms=4, alpha=0.6, color=c,
                      label=f"dim={d_}")
        lim = [min((-m).min(), (-p).min()) * 0.5,
               max((-m).max(), (-p).max()) * 2]
        ax.plot(lim, lim, "k--", lw=1, label="y=x")
        ax.set_xlabel("measured rate  -slope(log gap)/iter")
        ax.set_ylabel("predicted rate  -log rho2(J)")
        ax.set_title(f"N2 spectral prediction, corr={corr:.3f}, "
                     f"n={p.size}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "n2_spectral.png"), dpi=130)

    # clip ratio figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(np.maximum(ra, 1e-18)), bins=40)
    ax.set_xlabel("log10 late-half clip-loss / ||g||^2")
    ax.set_ylabel("#instances")
    ax.set_title("N1b: permanent-clipping escape check")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "n1_clip.png"), dpi=130)

    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "n_analysis.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
