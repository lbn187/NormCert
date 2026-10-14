# Norm saturation and last-iterate convergence in regret matching

Source code for the paper *Norm Saturation Explains Last-Iterate Convergence in
Regret Matching* (ICLR 2027 submission).

The code implements the scale-invariant regret-matching algorithm IREG-PRM+
(extra-gradient setup of Zhang et al., *Scale-Invariant Regret Matching and
Online Learning with Optimal Convergence*, EC 2026, arXiv:2510.04407,
Algorithm 2) and reproduces every experimental result of the paper:

- **5 predictions on 216 random matrix games** (support freezing; the
  last-iterate rate equals the local spectrum of the frozen-face Jacobian;
  the certificate exponent $\theta = 2$; self-stabilization
  $\eta_\infty \sigma_{\max}(B) \le 1$ together with the ratio certificate and
  the point forecast; free progress monitoring in extensive-form games);
- **Kuhn and Leduc poker** experiments (point forecast and free monitoring,
  Table `tab:numerics` in the paper).

Everything runs on CPU with `numpy`/`scipy` (HiGHS LP solver, bundled since
scipy 1.9) and `matplotlib`.

## Layout

```
normcert/
├── ireg_prm.py            # IREG-PRM+ for zero-sum matrix games
├── games.py               # random zero-sum matrix games + LP equilibrium
├── efg.py                 # self-contained Kuhn/Leduc poker + IR-PCFR+ solver
├── n1n2_run.py            # 216-instance testbed (seeds 0..11, d{5,20,50})
├── n_analyze.py           # support freezing + local-rate verification
├── verify_155.py          # independent re-derivation of 216 = 155 + 32 + 29
├── verify_outliers.py     # freeze-time split of the 155 (pre-freeze snapshots)
├── ai_review_checks.py    # 32 finite-step convergers in 60-digit arithmetic; drift check on 8 degenerate instances
├── theta_fix.py           # certificate exponent theta = 2
├── eta_sigma_check.py     # eta_inf * sigma_max(B) <= 1 self-stabilization
├── kappa_check.py         # ratio certificate (kappa_emp, forecast bound)
├── nb_closedform.py       # closed-form Jacobian spectrum vs FD Jacobian
├── forecast_check.py      # point forecast on the theta-usable instances
├── forecast_efg.py        # Kuhn/Leduc point forecast + monitoring cost
├── efg_timing.py          # wall-clock: one NashConv vs one iteration (Kuhn, Leduc)
├── paper_figs.py          # the three figures of the paper
└── results/               # pre-generated data (traces + tables)
```

The `results/` directory ships the pre-generated output of step 1 (216
trajectories, one `.npz` per instance) and every downstream table, so the
analysis and figures can be reproduced directly without rerunning the costly
first step.

## Dependencies

- Python 3.9+
- `numpy`
- `scipy` (>= 1.9 for the HiGHS LP solver)
- `matplotlib`

```
pip install numpy scipy matplotlib
```

## Quick reproduction (minutes, using the shipped data)

From inside `normcert/`, run the analysis scripts on the pre-generated data in
dependency order:

```bash
cd normcert
python3 n_analyze.py          # support freezing + local rate
python3 verify_155.py         # independent re-derivation of the counts
python3 verify_outliers.py    # freeze-time split of the 155
python3 theta_fix.py          # theta = 2
python3 eta_sigma_check.py n_take=184   # self-stabilization; 184 = paper's
                                        # "all converged instances, no sampling"
python3 kappa_check.py        # ratio certificate
python3 nb_closedform.py      # closed-form spectrum
python3 forecast_check.py --theta-csv results/theta_rows.csv
python3 forecast_efg.py --T-kuhn 100000 --T-leduc 50000   # ~1 min Leduc
python3 paper_figs.py         # writes the three paper figures to figs/
```

Each script writes its summary table to `results/` and prints the headline
numbers (e.g. predicted/measured rate median ratio `1.000`, `theta` median
`1.994`/`2.003`, zero violations of the stability boundary, `kappa_emp`
median `1.63`).

## Full reproduction from scratch (hours)

Step 1 regenerates all 216 trajectories (the only expensive step; `T=10^6`
for `n=50`, else `2*10^5`):

```bash
cd normcert
python3 n1n2_run.py 12 56    # [seeds=12] [nproc]; reduce nproc on shared nodes
```

> **Caution:** step 1 writes `results/traces/` and `results/n1n2_rows.csv`,
> overwriting the shipped pre-generated data. Back up `results/` first, or run
> step 1 in a fresh copy of the directory if you only want to verify the
> pipeline.

Then run the analysis steps above. A quick smoke test that verifies the
pipeline end-to-end on 18 instances:

```bash
python3 n1n2_run.py 1 4      # 3 dims x 3 distributions x 2 generators x 1 seed
```

## Correspondence with the paper

| Script | Paper content |
|--------|---------------|
| `n1n2_run.py` + `n_analyze.py` | 216-instance testbed; exact support freezing; predicted vs measured local rate (Fig. 1, left) |
| `verify_155.py` + `verify_outliers.py` | independent re-derivation of the 216 = 155 + 32 + 29 accounting and of the freeze-time split of the 155 |
| `nb_closedform.py` | closed-form spectrum equals the FD Jacobian spectrum (32 deterministic instances) |
| `theta_fix.py` | certificate exponent $\theta = 2$ (Fig. 2) |
| `eta_sigma_check.py` | $\eta_\infty \sigma_{\max}(B) \le 1$ on all 184 converged instances, no sampling (Fig. 1, right) |
| `kappa_check.py` + `forecast_check.py` | ratio certificate $\kappa_{\mathrm{emp}}$ and the point forecast |
| `forecast_efg.py` | Kuhn/Leduc free progress monitoring and best-response-budget reduction |
| `efg_timing.py` | NashConv/iteration wall-clock ratio (0.076 Kuhn, 0.087 Leduc; `results/efg_timing.txt`) |
| `paper_figs.py` | the three paper figures |

All proofs appear in the paper's appendix; the exact definitions of every
measured quantity are in the paper's experimental-protocol appendix
(`app:experiments`).

## License

MIT. See `LICENSE`.
