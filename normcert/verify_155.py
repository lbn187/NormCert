#!/usr/bin/env python3
"""Independent re-derivation of the 216 = 155 + 32 + 29 accounting.

This is a second implementation of the usable-instance filter of
``n_analyze.py``, written from the released traces alone (no import of
``n_analyze``), so that the counts reported in Appendix D can be checked
without trusting the analysis script.

Inputs (released):
  results/n1n2_rows.csv   columns ``instance_id``, ``slope_pred``,
                          ``snap_gap``
  results/traces/*.npz    keys ``ts`` (recorded times), ``gaps`` (Nash gap)

Filter (as described in the paper):
  * an instance is *snapshot-eligible* when ``slope_pred`` is finite; the
    snapshot trigger never fired for the others (no-snapshot group);
  * on the post-snapshot positive gaps, fit ``log gap(t)`` by least squares
    on the window ``gap in [1e3 * floor, 0.5 * snap_gap]``, where ``floor``
    is the smallest positive gap (if that window is empty, fall back to
    ``[1.5 * floor, 0.5 * snap_gap]``);
  * the instance is *usable* when the window holds at least 5 points, the
    fit has ``R^2 >= 0.9``, and the fitted slope is negative.

Expected output: 155 usable, 32 no-snapshot, 29 non-fittable.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRACES = os.path.join(RESULTS, "traces")

R2_MIN = 0.9
MIN_PTS = 5


def fit_slope(iid: str, snap_gap: float):
    """Return (slope, r2, npts) on the post-snapshot exponential window."""
    path = os.path.join(TRACES, f"{iid}.npz")
    if not os.path.exists(path):
        return float("nan"), float("nan"), 0
    with np.load(path) as d:
        ts = d["ts"].astype(float)
        gaps = d["gaps"].astype(float)
    keep = gaps > 0.0
    ts, gaps = ts[keep], gaps[keep]
    if ts.size < 10 or not np.isfinite(snap_gap):
        return float("nan"), float("nan"), 0
    floor = float(gaps.min())
    lo, hi = 1e3 * floor, 0.5 * snap_gap
    if hi <= lo:
        lo = 1.5 * floor
    sel = (gaps > lo) & (gaps < hi)
    if sel.sum() < MIN_PTS:
        return float("nan"), float("nan"), int(sel.sum())
    x, y = ts[sel], np.log(gaps[sel])
    design = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), r2, int(sel.sum())


def main() -> int:
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        rows = list(csv.DictReader(f))
    usable, no_snap, no_fit = [], [], []
    for r in rows:
        pred = float(r["slope_pred"])
        if not np.isfinite(pred):
            no_snap.append(r["instance_id"])
            continue
        slope, r2, _ = fit_slope(r["instance_id"], float(r["snap_gap"]))
        if not np.isfinite(slope) or r2 < R2_MIN or slope >= 0.0:
            no_fit.append(r["instance_id"])
            continue
        usable.append(r["instance_id"])
    print(f"total instances: {len(rows)}")
    print(f"usable:        {len(usable)}")
    print(f"no-snapshot:   {len(no_snap)}")
    print(f"non-fittable:  {len(no_fit)}")
    ok = (len(usable), len(no_snap), len(no_fit)) == (155, 32, 29)
    print("accounting 216 = 155 + 32 + 29:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
