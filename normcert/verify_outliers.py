#!/usr/bin/env python3
"""Freeze-time split of the 155 usable instances (Figure 1, left).

The snapshot time is chosen from the positive-gap trigger; the exact support
freeze time ``t_freeze_exact`` is recorded separately.  When the snapshot
precedes the exact freeze, the finite-difference Jacobian is taken off the
frozen face, so the predicted rate can sit off the ``y = x`` line.

This script reports
  * how many usable instances have ``snap_t < t_freeze_exact``,
  * the log--log correlation between predicted and measured slopes on all
    usable instances and with the pre-freeze ones removed.

It reuses the filter of ``verify_155.py`` (an independent implementation of
the ``n_analyze.py`` usable-pair selection).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

from verify_155 import R2_MIN, fit_slope

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main() -> int:
    with open(os.path.join(RESULTS, "n1n2_rows.csv")) as f:
        rows = list(csv.DictReader(f))
    pred, meas, pre_flag, pre_ids = [], [], [], []
    for r in rows:
        sp = float(r["slope_pred"])
        if not np.isfinite(sp):
            continue
        sm, r2, _ = fit_slope(r["instance_id"], float(r["snap_gap"]))
        if not np.isfinite(sm) or r2 < R2_MIN or sm >= 0.0:
            continue
        pred.append(sp)
        meas.append(sm)
        is_pre = float(r["snap_t"]) < float(r["t_freeze_exact"])
        pre_flag.append(is_pre)
        if is_pre:
            pre_ids.append(r["instance_id"])
    p, m = np.array(pred), np.array(meas)
    pre = np.array(pre_flag, dtype=bool)
    keep = ~pre
    print(f"usable instances:            {p.size}")
    print(f"snap_t < t_freeze_exact:     {int(pre.sum())}")
    for iid in pre_ids:
        print(f"  {iid}")
    print(f"corr (all):                  "
          f"{float(np.corrcoef(np.log(-p), np.log(-m))[0, 1]):.4f}")
    print(f"corr (pre-freeze excluded, n={int(keep.sum())}): "
          f"{float(np.corrcoef(np.log(-p[keep]), np.log(-m[keep]))[0, 1]):.4f}")
    print(f"median ratio (all):          {float(np.median(p / m)):.4f}")
    print(f"median ratio (excluded):     {float(np.median(p[keep] / m[keep])):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
