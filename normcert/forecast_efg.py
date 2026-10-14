"""EFG point forecast (Verification 8): free progress monitoring in
extensive-form games (Kuhn / Leduc).

Uses the same sandwich as the matrix games: on a window t >= snap_t the
per-step cumulative gradient-squared rate inc = d(cum_g2)/dt predicts
  gap(t2) ~= gap(t1) * sqrt(inc(t2)/inc(t1)),
with multiplicative deviation D(t1,t2) guaranteed in [kappa^-1, kappa]
in the matrix setting (thm:certificate).  We record the empirical sandwich
quantities kappa_emp_consistent and fwd_max exactly as forecast_check.py
does for matrices, and we also report the on-line coupling corr(log inc,
log gap) for cross-checking against pilot Gate 3 (Kuhn 0.996 / Leduc 0.990).

NashConv (duality gap) is computed ONLY at the sparse record times; the
certificate interpolates the gap between two BR passes for free.  The BR
pass cost is measured (cost_br / cost_iter) rather than assumed.

Guards match theta_fix / forecast_check: a delta of cum_g2 is kept only if
d(cum_g2) > 64 * eps * cum_g2, and the window must keep >= 6 such deltas.

Run:  python3 forecast_efg.py --T-kuhn 100000 --T-leduc 50000
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np

from efg import build_kuhn, build_leduc, run_ireg_pcfr, nash_gap

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
EPS = np.finfo(float).eps
REPR_GUARD = 64.0
MIN_PTS = 6
SNAP_REL = 1e-4
SNAP_MIN_T = 10


def record_schedule(T: int) -> np.ndarray:
    """Log block up to T plus a linear densification on the tail so that
    the window (t >= snap_t) has >= ~200 roughly uniform points."""
    if T < 10:
        return np.array([1])
    logp = np.unique(np.round(np.logspace(0.0, np.log10(T), 250))
                     .astype(np.int64))
    logp = logp[(logp >= 1) & (logp <= T)]
    # early linear block: the fast-converging EFGs (Kuhn in particular)
    # decay to the floor within the first few hundred iterations, so the
    # log block alone leaves too few points in that segment
    early_end = min(2000, T)
    step_e = max(1, int(early_end / 200))
    early = np.arange(1, early_end, step_e)
    tail_start = int(T // 16)
    step = max(1, int((T - tail_start) / 400))
    tail = np.arange(tail_start, T + 1, step)
    return np.unique(np.concatenate([logp, early, tail]))


def per_step_rate(ts: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """inc[i] = (cum[i+1]-cum[i]) / (ts[i+1]-ts[i]), per-iteration rate."""
    d = np.diff(cum)
    dt = np.diff(ts).astype(float)
    return d / np.maximum(dt, 1.0)


GAP_FLOOR = 1e-13


def window(ts: np.ndarray, gap: np.ndarray, cum: np.ndarray
           ) -> Dict[str, np.ndarray]:
    """Window = the block where the gap is still meaningful: all points above
    the float floor after a short warmup, so that the certificate's
    interpolation region is covered (for the fast-converging EFGs this is
    the early exponential segment, unlike the matrix games where SNAP_REL
    targets the asymptotic tail).  ok mask = guard on d(cum) with >= MIN_PTS
    survivors."""
    g0 = gap[0]
    cand = np.where((ts >= SNAP_MIN_T) & (gap > GAP_FLOOR * max(g0, 1.0)))[0]
    i0 = int(cand[0]) if len(cand) else -1
    if i0 < 0:
        return {"ts": ts, "gap": gap, "cum": cum, "ok": np.zeros(len(ts),
                                                                 bool),
                "i0": -1}
    w_ts = ts[i0:]
    w_gap = gap[i0:]
    w_cum = cum[i0:]
    inc = per_step_rate(w_ts, w_cum)
    ok = np.zeros(len(w_ts), bool)
    total = float(w_cum[-1])
    for i, d in enumerate(np.diff(w_cum)):
        if d > REPR_GUARD * EPS * total and inc[i] > 0.0:
            ok[i + 1] = True
    ok[0] = False
    if int(ok.sum()) < MIN_PTS:
        return {"ts": w_ts, "gap": w_gap, "cum": w_cum, "ok": ok, "i0": i0}
    return {"ts": w_ts, "gap": w_gap, "cum": w_cum, "ok": ok, "i0": i0}


def kappa_consistent(w: Dict[str, np.ndarray]) -> Tuple[float, int]:
    """kappa_emp_consistent = max_t s(t)/min_t s(t), s = gm/sqrt(rate)."""
    ts = w["ts"]
    inc = per_step_rate(ts, w["cum"])
    gm = w["gap"]
    ok = np.where(w["ok"])[0]
    if len(ok) < MIN_PTS:
        return float("nan"), 0
    s = gm[ok] / np.sqrt(np.maximum(inc[ok - 1], 1e-300))
    return float(np.max(s) / np.min(s)), len(ok)


def forecast(w: Dict[str, np.ndarray], i1: int
             ) -> Dict[str, float]:
    """Point forecast from calibration point i1 to all later points."""
    ts, gap, cum = w["ts"], w["gap"], w["cum"]
    inc = per_step_rate(ts, cum)
    idx = np.where(w["ok"])[0]
    idx = idx[idx > i1]
    base = float(inc[i1 - 1]) if i1 >= 1 else float(inc[0])
    if base <= 0.0 or len(idx) < 1:
        return {"fwd_max": float("nan"), "n_fwd": 0,
                "med_abs_log_err": float("nan")}
    D = gap[i1] * np.sqrt(inc[idx - 1] / base) / gap[idx]
    fwd = np.maximum(D, 1.0 / D)
    return {"fwd_max": float(np.max(fwd)), "n_fwd": len(idx),
            "med_abs_log_err": float(np.median(np.abs(np.log(D))))}


def coupling(ts: np.ndarray, gap: np.ndarray, cum: np.ndarray
             ) -> Tuple[float, float]:
    """corr(log rate, log gap) for rate = d(cum)/dt and for norm^2 diff."""
    m = (gap[1:] > 0) & (ts[1:] >= 1)
    lr = np.log(per_step_rate(ts, cum))
    lg = np.log(gap[1:])
    fin = np.isfinite(lr) & np.isfinite(lg) & m
    c_cum = float(np.corrcoef(lr[fin], lg[fin])[0, 1]) if fin.sum() > 3 \
        else float("nan")
    return c_cum


def nash_cost(g, T: int) -> Dict[str, float]:
    """Measure BR-pass vs iteration cost on a short warm run."""
    from ireg_prm import make_record_schedule
    rec = make_record_schedule(T, n_points=20)
    t0 = time.time()
    run_ireg_pcfr(g, T, rec)
    iter_s = (time.time() - t0) / T
    x = [np.full((g.n_hands, len(nd.actions)), 1.0 / len(nd.actions))
         for nd in g.player_nodes]
    t0 = time.time()
    nash_gap(g, x)
    br_s = (time.time() - t0)
    return {"iter_s": iter_s, "br_s": br_s,
            "cost_br_over_iter": br_s / max(iter_s, 1e-300)}


def run_one(g, T: int, name: str) -> Dict[str, float]:
    rec = record_schedule(T)
    t0 = time.time()
    tr = run_ireg_pcfr(g, T, rec)
    wall = time.time() - t0
    n2tot = tr.norm1 ** 2 + tr.norm2 ** 2   # recorded-norm proxy (numerator
                                            # of the sandwich when cum_g2 is
                                            # below float resolution; its
                                            # rate then sits at the norm^2
                                            # noise floor, cf. matrices)
    # Primary rate = exact cum ||g||^2 increments (the theorem's Ndot_tot).
    def stats(cum: np.ndarray) -> Dict[str, float]:
        """kappa + point forecast on one rate grid; returns nan if the guard
        leaves fewer than MIN_PTS points."""
        w = window(tr.ts, tr.gap_last, cum)
        kap, npts = kappa_consistent(w)
        ok_idx = np.where(w["ok"])[0]
        i1s = []
        if len(ok_idx) >= 1:
            i1s.append(int(ok_idx[0]))
            for frac in (1, 2):
                j = int(ok_idx[len(ok_idx) * frac // 3])
                if j > i1s[-1]:
                    i1s.append(j)
        fwd = [forecast(w, i1) for i1 in i1s]
        return {"w": w, "kap": kap, "npts": npts,
                "fwd0": fwd[0]["fwd_max"] if fwd else float("nan"),
                "fwd13": fwd[1]["fwd_max"] if len(fwd) > 1 else float("nan"),
                "fwd12": fwd[2]["fwd_max"] if len(fwd) > 2 else float("nan"),
                "med_log_err": fwd[0]["med_abs_log_err"] if fwd
                else float("nan")}
    # exact cum ||g||^2 increments are the theorem's Ndot_tot and the
    # headline where they survive; the recorded-norm proxy n2tot = norm1^2 +
    # norm2^2 is reported alongside (its noise-floor caveat is documented
    # for the matrix games).  With the floor-based window the exact rate
    # survives on the fast EFGs too (Kuhn ~54 points), unlike the SNAP_REL
    # tail window used before.
    sg = stats(tr.cum_g2)
    sn = stats(n2tot)
    c_cum = coupling(tr.ts, tr.gap_last, tr.cum_g2)
    c_n2 = coupling(tr.ts, tr.gap_last, n2tot) if sn["w"]["i0"] >= 0 \
        else float("nan")
    prim = "g2" if sg["npts"] >= MIN_PTS else "n2"
    s = sg if prim == "g2" else sn
    cost = nash_cost(g, 2000)
    # BR total passes: every iteration would cost T passes; schedule does
    # len(rec) passes.
    br_pass = len(rec)
    snap_t = int(s["w"]["ts"][0]) if s["w"]["i0"] >= 0 else -1
    res = {
        "game": name, "T": int(T), "n_infosets": len(tr.infoset_meta),
        "snap_t": snap_t, "wall_s": wall,
        "kappa_emp": s["kap"], "n_pts_kappa": s["npts"],
        "kappa_emp_g2": sg["kap"], "kappa_emp_n2": sn["kap"],
        "n_pts_g2": sg["npts"], "n_pts_n2": sn["npts"],
        "fwd0": s["fwd0"], "fwd13": s["fwd13"], "fwd12": s["fwd12"],
        "fwd0_g2": sg["fwd0"], "fwd0_n2": sn["fwd0"],
        "med_log_err": s["med_log_err"],
        "rate": prim,
        "coupling_cg2": c_cum, "coupling_n2": c_n2,
        "iter_s": cost["iter_s"], "br_s": cost["br_s"],
        "cost_br_over_iter": cost["cost_br_over_iter"],
        "br_pass": br_pass, "gap_final": float(tr.gap_last[-1]),
    }
    np.savez(os.path.join(RESULTS, f"forecast_efg_{name}.npz"),
             ts=tr.ts, gap_last=tr.gap_last, cum_g2=tr.cum_g2,
             norm1=tr.norm1, norm2=tr.norm2, n2tot=n2tot)
    return res


def pilot_coupling(npz_path: str) -> float:
    d = np.load(npz_path)
    ts, gap = d["ts"], d["gap_last"]
    n1, n2 = d["norm1"], d["norm2"]
    n2tot = n1 ** 2 + n2 ** 2
    m = gap > 0
    lr = np.log(per_step_rate(ts, n2tot))
    lg = np.log(gap[1:])
    fin = np.isfinite(lr) & np.isfinite(lg)
    return float(np.corrcoef(lr[fin], lg[fin])[0, 1]) if fin.sum() > 3 \
        else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-kuhn", type=int, default=100000)
    ap.add_argument("--T-leduc", type=int, default=50000)
    ap.add_argument("--pilot-dir", default=None,
                    help="optional dir of pilot traces for a cross-check "
                         "(columns filled with nan when absent)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    pilot = {}
    if args.pilot_dir and os.path.isdir(args.pilot_dir):
        for name in ("kuhn", "leduc"):
            p = os.path.join(args.pilot_dir, f"gate3_{name}.npz")
            if os.path.exists(p):
                pilot[name] = p
    lines = []
    if args.T_kuhn > 0:
        rk = run_one(build_kuhn(), args.T_kuhn, "kuhn")
        rk["pilot_coupling"] = pilot_coupling(pilot["kuhn"]) if "kuhn" in pilot \
            else float("nan")
        lines.append(rk)
    if args.T_leduc > 0:
        rl = run_one(build_leduc(), args.T_leduc, "leduc")
        rl["pilot_coupling"] = pilot_coupling(pilot["leduc"]) if "leduc" in pilot \
            else float("nan")
        lines.append(rl)
    hdr = sorted(lines[0].keys())
    with open(os.path.join(RESULTS, "forecast_efg.csv"), "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in lines:
            f.write(",".join(str(r.get(k, "")) for k in hdr) + "\n")
    for r in lines:
        print(" | ".join(f"{k}={r[k]:.4f}" if isinstance(r[k], float)
                         else f"{k}={r[k]}" for k in hdr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
