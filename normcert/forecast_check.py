"""Point-forecast demonstration of the ratio certificate (verification 7).

Theory (07_certificate.tex:201-210, Theorem thm:certificate): in the frozen
local regime the certificate is the forecast formula

    gap(t2) = gap(t1) * sqrt(Ndot_tot(t2)/Ndot_tot(t1)) * (factor in [k^-1, k]).

This script makes that forward reading explicit: calibrate the gap at one
time t1 and forecast the gap at every later guarded interval, then compare
against the realized gap.  The multiplicative deviation D(t1,t2) is bounded
by kappa; we report

  * fwd_max(t1)  = max_{t2>t1} max(D, 1/D)   -- forward subset of kappa,
  * kappa_emp_consistent                       -- all-window-pairs max, the
                                                kappa_check.py definition
                                                (max_t s(t)/min_t s(t),
                                                 s(t)=gm(t)/sqrt(dG2(t)/dt)),
  * the identity check  fwd_max(t1) <= kappa_emp_consistent  (any violation
    is a bug, not a finding).

Numerics follow theta_fix.py / kappa_check.py: window t >= snap_t; primary
numerator = increments of cum ||g||^2 (cancellation-free; equals Ndot by
the identity Ndot_i = sum_{S_i} g_j^2), proxy = telescoped norm^2
increments where representable.

Run:  python3 forecast_check.py [--theta-csv results/theta_rows.csv]
                                [--pilot <pilot certificate dir>]
                                [--nproc 10]
Writes results/forecast_rows.csv, results/forecast_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
TRACES = os.path.join(RESULTS, "traces")

EPS = np.finfo(float).eps
REPR_GUARD = 64.0        # same guard as theta_fix.py / kappa_check.py
GAP_FLOOR = 1e-13
MIN_PTS = 6


# ---------------------------------------------------------------------------
# window + guarded increments (theta_fix / kappa_check 口径)
# ---------------------------------------------------------------------------

def window(iid: str, snap_t: int) -> Optional[Dict]:
    d = np.load(os.path.join(TRACES, f"{iid}.npz"))
    ts, gaps, cg2, closs = d["ts"], d["gaps"], d["cg2"], d["closs"]
    n1s, n2s = d["norms1"], d["norms2"]
    cg2_supp = d["cg2_supp"] if "cg2_supp" in d.files \
        else np.full(ts.size, np.nan)
    keep = ts >= snap_t
    ts, gaps = ts[keep], gaps[keep]
    cg2, closs, n1s, n2s = cg2[keep], closs[keep], n1s[keep], n2s[keep]
    cg2_supp = cg2_supp[keep]
    if ts.size < MIN_PTS + 1:
        return None
    dt = np.diff(ts).astype(float)
    dn2 = np.diff(n1s ** 2 + n2s ** 2)       # telescoped Delta N^2 (proxy):
                                             # cancellation-free on the recorded
                                             # (log-spaced) grid, THE usable rate
    dg2 = np.diff(cg2)                       # Delta cum||g||^2: after norm
                                             # saturation the per-step g^2 is
                                             # < eps*cum, so cg2 stops growing
                                             # and dg2 is not usable
    dgs2 = np.diff(cg2_supp)                 # on-support energy increments:
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
    return {"ts": ts, "gm": gm, "dg2": dg2, "dn2": dn2, "dgs2": dgs2,
            "dt": dt, "ok": ok, "okn": okn, "oks": oks}


def _inc(w: Dict, proxy: str) -> tuple:
    if proxy == "g2":
        return w["ok"], w["dg2"]
    if proxy == "onsupp":
        return w["oks"], w["dgs2"]
    return w["okn"], w["dn2"]


def forecast_from_window(w: Dict, calib_idx: int, proxy: str) -> Dict:
    """Calibrate at one guarded interval, forecast all later ones.

    Returns fwd_max, median |log D|, median |D-1|, n_forward.
    D(i2) = gm(calib) * sqrt(inc(i2)/inc(calib)) / gm(i2).
    """
    ok, inc_raw = _inc(w, proxy)
    inc = inc_raw / w["dt"]                  # per-step rate (Ndot)
    idx = np.where(ok)[0]
    if calib_idx not in idx:
        return {"fwd_max": float("nan"), "med_log": float("nan"),
                "med_rel": float("nan"), "n_fwd": 0}
    i1 = calib_idx
    later = idx[idx > i1]
    if later.size == 0:
        return {"fwd_max": float("nan"), "med_log": float("nan"),
                "med_rel": float("nan"), "n_fwd": 0}
    g1, i1v = w["gm"][i1], inc[i1]
    D = g1 * np.sqrt(inc[later] / i1v) / w["gm"][later]
    return {"fwd_max": float(np.max(np.maximum(D, 1.0 / D))),
            "med_log": float(np.median(np.abs(np.log(D)))),
            "med_rel": float(np.median(np.abs(D - 1.0))),
            "n_fwd": int(later.size)}


def kappa_consistent(w: Dict, proxy: str) -> Tuple[float, int]:
    ok, inc_raw = _inc(w, proxy)
    if int(ok.sum()) < MIN_PTS:
        return float("nan"), int(ok.sum())
    s = w["gm"][ok] / np.sqrt((inc_raw[ok] / w["dt"][ok]))
    return float(np.max(s) / np.min(s)), int(ok.sum())


# ---------------------------------------------------------------------------
# per-instance row
# ---------------------------------------------------------------------------

def forecast_instance(row: Dict) -> Optional[Dict]:
    iid = row["instance_id"]
    snap_t = int(row["snap_t"])
    if snap_t < 0:
        return None
    # frozen-regime window (R2 review fix): same max(snap, freeze) start as
    # theta_fix.py, so the Lemma ndot-identity hypothesis holds on-window.
    snap_t = max(snap_t, int(float(row.get("t_freeze_exact", 0) or 0)))
    w = window(iid, snap_t)
    if w is None:
        return None
    # PRIMARY: on-support energy increments = the theorem's exact
    # Delta N_tot, computed directly from g (small magnitudes, no float64
    # accumulated-cumulant cancellation, no norm-saturation noise floor).
    # Fallbacks for older traces / degenerate windows: g2 (cum ||g||^2
    # increments), then n2 (telescoped norm^2 increments).  All three are
    # reported so the choice is transparent.
    ok_os = np.where(w["oks"])[0]
    ok_g2 = np.where(w["ok"])[0]
    ok_n2 = np.where(w["okn"])[0]
    if ok_os.size < MIN_PTS and ok_g2.size < MIN_PTS \
            and ok_n2.size < MIN_PTS:
        return None
    if ok_os.size >= MIN_PTS:
        prim = "onsupp"
    elif ok_g2.size >= MIN_PTS:
        prim = "g2"
    else:
        prim = "n2"
    idx = {"onsupp": ok_os, "g2": ok_g2, "n2": ok_n2}[prim]
    i1_main = int(idx[0])                        # earliest regime entry
    i1_13 = int(idx[len(idx) // 3]) if len(idx) >= 3 else i1_main
    i1_12 = int(idx[len(idx) // 2]) if len(idx) >= 2 else i1_main
    f_main = forecast_from_window(w, i1_main, prim)
    f_13 = forecast_from_window(w, i1_13, prim)
    f_12 = forecast_from_window(w, i1_12, prim)
    f_main_n2 = forecast_from_window(w, i1_main, "n2")
    f_main_os = forecast_from_window(w, i1_main, "onsupp")
    fwd = [x for x in (f_main["fwd_max"], f_13["fwd_max"], f_12["fwd_max"])
           if np.isfinite(x)]
    sens = float(max(fwd) / min(fwd)) if fwd else float("nan")
    kap, nkap = kappa_consistent(w, prim)
    kapn, nkapn = kappa_consistent(w, "n2")
    kapos, nkapos = kappa_consistent(w, "onsupp")
    # norm saturation: late/early median rate ratio on the primary grid
    inc2 = {"onsupp": w["dgs2"], "g2": w["dg2"], "n2": w["dn2"]}[prim] \
        / w["dt"]
    if idx.size >= 3:
        m = max(1, len(idx) // 3)
        norm_sat = float(np.median(inc2[idx[2*m:]]) /
                         max(np.median(inc2[idx[:m]]), 1e-300))
    else:
        norm_sat = float("nan")
    t_free = float(row["t_freeze_exact"])
    post_freeze = bool(np.isfinite(t_free) and snap_t >= t_free)
    gm = w["gm"][w["okn"]]
    return {
        "instance_id": iid, "dim": row["dim"], "dist": row["dist"],
        "gen": row["gen"],
        "n_pts": int(idx.size),
        "fwd_max_snap": f_main["fwd_max"], "fwd_max_13": f_13["fwd_max"],
        "fwd_max_12": f_12["fwd_max"], "sens": sens,
        "med_log_err": f_main["med_log"], "med_rel_err": f_main["med_rel"],
        "n_fwd": f_main["n_fwd"],
        "fwd_max_snap_n2": f_main_n2["fwd_max"],
        "fwd_max_snap_onsupp": f_main_os["fwd_max"],
        "kappa_emp_consistent": kap, "n_pts_kappa": nkap,
        "kappa_emp_n2": kapn, "n_pts_kappa_n2": nkapn,
        "kappa_emp_onsupp": kapos, "n_pts_kappa_onsupp": nkapos,
        "rate": prim,
        "post_freeze": post_freeze, "norm_sat": norm_sat,
        "gap_span_dec": float(np.log10(np.max(gm) / np.min(gm))),
        "gap_final": row["gap_final"], "snap_t": snap_t,
        "t_freeze_exact": t_free, "margin": row["margin"],
        "n_unit_eigs": row["n_unit_eigs"],
    }



def pilot_rows(pilot_dir: str, nproc: int) -> List[Dict]:
    """Optional sanity pass on pilot gate1_traces_d*.npz (records x instances
    layout, no cg2): window from first record with gap < 1e-3 * gap0, proxy =
    norm^2 increments."""
    import glob
    cap = 0.001
    rows: List[Dict] = []

    def one(iid: str, ts, gaps, n1s, n2s) -> Optional[Dict]:
        g0 = gaps[0]
        i0 = int(np.where(gaps < cap * g0)[0][0]) if (gaps < cap * g0).any() \
            else 0
        keep = np.arange(ts.size) >= i0
        ts, gaps = ts[keep], gaps[keep]
        n1s, n2s = n1s[keep], n2s[keep]
        if ts.size < MIN_PTS + 1:
            return None
        dn2 = np.diff(n1s ** 2 + n2s ** 2)
        gm = np.sqrt(gaps[:-1] * np.maximum(gaps[1:], 0.0))
        cum = (n1s ** 2 + n2s ** 2)[1:]
        ok = ((dn2 > REPR_GUARD * EPS * cum) & (gm > GAP_FLOOR)
              & (gaps[:-1] > 0) & (gaps[1:] > 0))
        if int(ok.sum()) < MIN_PTS:
            return None
        idx = np.where(ok)[0]
        i1 = int(idx[0])
        later = idx[idx > i1]
        if later.size == 0:
            return None
        dt = np.diff(ts).astype(float)
        r2 = dn2 / dt
        D = gm[i1] * np.sqrt(r2[later] / r2[i1]) / gm[later]
        s = gm[ok] / np.sqrt(r2[ok])
        return {"instance_id": f"pilot_{iid}", "dim": iid, "dist": "pilot",
                "gen": "pilot", "n_pts": int(ok.sum()),
                "fwd_max_snap": float(np.max(np.maximum(D, 1 / D))),
                "fwd_max_13": float("nan"), "fwd_max_12": float("nan"),
                "sens": float("nan"), "med_log_err": float("nan"),
                "med_rel_err": float("nan"), "n_fwd": int(later.size),
                "fwd_max_snap_n2": float("nan"),
                "kappa_emp_consistent": float(np.max(s) / np.min(s)),
                "n_pts_kappa": int(ok.sum()),
                "kappa_emp_n2": float("nan"), "n_pts_kappa_n2": 0,
                "rate": "pilot", "post_freeze": True,
                "gap_span_dec": float("nan"), "gap_final": float(gaps[-1]),
                "snap_t": int(ts[0]), "t_freeze_exact": float("nan"),
                "margin": float("nan"), "n_unit_eigs": -1}

    for path in sorted(glob.glob(os.path.join(pilot_dir, "results",
                                              "gate1_traces_*.npz"))):
        d = np.load(path)
        ts, ids = d["ts"], d["instance_ids"]
        gl, n1, n2 = d["gap_last"], d["norm1"], d["norm2"]
        for j in range(len(ids)):
            r = one(ids[j], ts, gl[:, j], n1[:, j], n2[:, j])
            if r is not None:
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _summary(rows: List[Dict]) -> str:
    fwd = np.array([r["fwd_max_snap"] for r in rows])
    fwdn = np.array([r["fwd_max_snap_n2"] for r in rows])
    fwdo = np.array([r["fwd_max_snap_onsupp"] for r in rows])
    sens = np.array([r["sens"] for r in rows])
    kap = np.array([r["kappa_emp_consistent"] for r in rows])
    kapo = np.array([r["kappa_emp_onsupp"] for r in rows])
    rate = np.array([r["rate"] for r in rows])
    pf = np.array([r["post_freeze"] for r in rows], dtype=bool)
    li = []
    li.append(f"instances forecastable: {len(rows)}")
    if len(rows):
        li.append(f"primary rate used: onsupp={100*float(np.mean(rate=='onsupp')):.1f}%  "
                  f"g2={100*float(np.mean(rate=='g2')):.1f}%  "
                  f"n2={100*float(np.mean(rate=='n2')):.1f}%")
        li.append(f"fwd_max (onsupp primary, snap calib): median "
                  f"{np.median(fwdo):.3f}  p90 {np.percentile(fwdo, 90):.3f}  "
                  f"max {np.max(fwdo):.3f}")
        li.append(f"fwd_max < 2: {100 * float(np.mean(fwdo < 2)):.1f}%  "
                  f"< 5: {100 * float(np.mean(fwdo < 5)):.1f}%")
        li.append(f"fwd_max (g2 fallback, snap calib): median "
                  f"{np.median(fwd):.3f}  p90 {np.percentile(fwd, 90):.3f}  "
                  f"max {np.max(fwd):.3f}")
        li.append(f"calibration sensitivity: median "
                  f"{np.median(sens):.3f}")
        li.append(f"kappa_emp_consistent (onsupp): median {np.median(kapo):.3f}  "
                  f"max {np.max(kapo):.3f}")
        li.append(f"kappa_emp_consistent (g2): median {np.median(kap):.3f}  "
                  f"max {np.max(kap):.3f}")
        bad = [r["instance_id"] for r in rows
               if r["fwd_max_snap"] > r["kappa_emp_consistent"] + 1e-9]
        li.append(f"identity fwd_max <= kappa_emp: "
                  f"{'OK' if not bad else 'VIOLATED ' + str(bad)}")
        li.append(f"post_freeze (snap_t >= t_freeze_exact): "
                  f"{100 * float(np.mean(pf)):.1f}%")
        li.append(f"n2 proxy (telescoped Delta norm^2) fwd_max: median "
                  f"{np.median(fwdn):.3f}  p90 {np.percentile(fwdn, 90):.1f}  "
                  f"max {np.nanmax(fwdn):.1f}")
        fin = np.isfinite(fwdn)
        li.append(f"n2 proxy < 2 (finite only): "
                  f"{100 * float(np.mean(fwdn[fin] < 2)):.1f}%")
        big = [r["instance_id"] for r in rows
               if np.isfinite(r["fwd_max_snap_n2"])
               and r["fwd_max_snap_n2"] > 100]
        li.append(f"n2 proxy diverges (>100, norm^2 noise floor): "
                  f"n={len(big)}  {big[:6]}")
        ns = np.array([r["norm_sat"] for r in rows])
        for thr, tag in ((0.1, "norm_sat<0.1 (exponential-type)"),
                         (1.0, "0.1<=norm_sat<1.0"),
                         (None, "norm_sat>=1.0 (constant-rate)")):
            if thr is None:
                m = ns >= 1.0
            else:
                m = ns < thr
            if m.any():
                sub = fwdo[m]
                sub = sub[np.isfinite(sub)]
                if sub.size:
                    li.append(f"{tag}: n={int(m.sum())} "
                              f"fwd_max med={np.median(sub):.3f} "
                              f"max={np.max(sub):.1f}")
        li.append(f"gap span (decades, median): "
                  f"{np.median([r['gap_span_dec'] for r in rows]):.1f}")
    return "\n".join(li)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-csv", default=os.path.join(RESULTS,
                                                        "theta_rows.csv"))
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--nproc", type=int, default=10)
    args = ap.parse_args()

    if args.pilot:
        rows = pilot_rows(args.pilot, args.nproc)
        out = os.path.join(RESULTS, "forecast_pilot_rows.csv")
    else:
        pool_file = args.theta_csv
        if os.path.exists(pool_file):
            with open(pool_file) as f:
                pool = set(r["instance_id"]
                           for r in csv.DictReader(f))
        else:
            pool = None
        with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
            all_rows = list(csv.DictReader(f))
        rows = []
        for row in all_rows:
            if pool is not None and row["instance_id"] not in pool:
                continue
            r = forecast_instance(row)
            if r is not None:
                rows.append(r)
        out = os.path.join(RESULTS, "forecast_rows.csv")

    rows.sort(key=lambda r: r["instance_id"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out.replace("rows.csv", "summary.txt"), "w") as f:
        f.write(_summary(rows) + "\n")
    print(_summary(rows))


if __name__ == "__main__":
    sys.exit(main())
