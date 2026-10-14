"""Wall-clock cost of NashConv (best-response) vs one IREG-PCFR iteration.

The paper motivates the free certificate by the cost of exploitability
evaluation; this script measures that cost, per game (Kuhn, Leduc):

  * t_iter     -- median wall-clock of ONE algorithm iteration (the exact
                  loop body of efg.run_ireg_pcfr: two profile passes,
                  next_strategy, observe_utility, averaging);
  * t_nashconv -- median wall-clock of ONE NashConv evaluation
                  (efg.nash_gap = two best-response tree traversals);
  * ratio      -- t_nashconv / t_iter.

Protocol: warm up the state with W iterations so strategies are generic
(not the uniform init), then time R repetitions of each unit with
time.perf_counter, report median and IQR.  Single process, single thread.

Run:  OMP_NUM_THREADS=1 python3 efg_timing.py
Writes results/efg_timing.txt
"""
from __future__ import annotations

import os
import time
from typing import List

import numpy as np

from efg import (EFG, _BatchIRPRM, build_kuhn, build_leduc, nash_gap,
                 profile_pass)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

WARMUP = 2000        # iterations to reach a generic mid-run state
REPS = 400           # timed repetitions per unit
SEED = 0


def one_game(g: EFG, label: str, lines: List[str]) -> None:
    np.random.seed(SEED)
    minimizers = [_BatchIRPRM(g.n_hands, len(nd.actions))
                  for nd in g.player_nodes]
    avg = [np.zeros((g.n_hands, len(nd.actions))) for nd in g.player_nodes]

    def iterate() -> None:
        stilde = [mz.xtilde for mz in minimizers]
        _, m_u, _ = profile_pass(g, stilde)
        x = [mz.next_strategy(m_u[i]) for i, mz in enumerate(minimizers)]
        _, u, reach = profile_pass(g, x)
        for i, mz in enumerate(minimizers):
            mz.observe_utility(u[i])
            avg[i] += reach[i][:, None] * x[i]

    for _ in range(WARMUP):
        iterate()
    x_now = [mz.xtilde for mz in minimizers]

    t_iter = np.empty(REPS)
    for r in range(REPS):
        t0 = time.perf_counter()
        iterate()
        t_iter[r] = time.perf_counter() - t0

    t_nc = np.empty(REPS)
    for r in range(REPS):
        t0 = time.perf_counter()
        nash_gap(g, x_now)
        t_nc[r] = time.perf_counter() - t0

    mi, mn = float(np.median(t_iter)), float(np.median(t_nc))
    lines.append(
        f"{label}: t_iter median {mi*1e3:.3f} ms "
        f"[IQR {np.percentile(t_iter,25)*1e3:.3f}, "
        f"{np.percentile(t_iter,75)*1e3:.3f}]  "
        f"t_nashconv median {mn*1e3:.3f} ms "
        f"[IQR {np.percentile(t_nc,25)*1e3:.3f}, "
        f"{np.percentile(t_nc,75)*1e3:.3f}]  "
        f"ratio nashconv/iter = {mn/mi:.3f}")


def main() -> None:
    lines: List[str] = [
        f"EFG wall-clock: one NashConv evaluation vs one iteration "
        f"(warmup {WARMUP}, {REPS} reps, single thread)"]
    for g, label in ((build_kuhn(), "Kuhn"), (build_leduc(), "Leduc")):
        one_game(g, label, lines)
    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "efg_timing.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
