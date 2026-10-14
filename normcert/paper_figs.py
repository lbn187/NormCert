"""Paper-grade figures for mypaper/EC27_NormCert from numerics/results/.

Redraws the paper figures (each saved as .pdf and .png, 300 dpi):
  theta_hist    -- theta_xy / theta_ratio histograms, theory line theta = 2
                   (fig:theta;  data: results/theta_rows.csv)
  n2_spectral   -- predicted linear rate (closed-form/FD spectrum) vs
                   measured semilog rate, log-log + y = x
                   (fig:spectral;  data: results/n1n2_rows.csv + traces)
  eta_sigma     -- eta_inf * sigma_max(B) distribution of the 48 converged
                   instances, boundary at 1 (fig:etasigma;
                   data: results/eta_sigma_check.csv)
  fig1_local    -- n2_spectral and eta_sigma side by side at the 5.5 in
                   text width with one font scale (main-text Figure 1)

All annotated numbers are computed from the data at run time (never
hardcoded).  The measured-rate fit reuses fit_measured_slope from
n_analyze.py unchanged, so the scatter reproduces the published
n_analysis.txt aggregation exactly.

Run:  python3 paper_figs.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from n_analyze import FIT_R2_MIN, fit_measured_slope  # noqa: E402

RESULTS = os.path.join(BASE, "results")
OUT = os.environ.get("NORMCERT_FIG_OUT") or os.path.normpath(os.path.join(
    BASE, "..", "..", "..", "mypaper", "EC27_NormCert", "figures"))

# Okabe-Ito colorblind-safe palette (fixed assignment order)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMILLION = "#D55E00"
C_GRAY = "#444444"
DIM_COLORS = {5: C_BLUE, 20: C_ORANGE, 50: C_GREEN}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# rc overrides for figures drawn at the final physical width (5.5 in ICLR
# text width, included at width=\textwidth): fonts then print at ~9 pt.
SMALL_RC = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
}


def read_csv(name: str) -> List[Dict[str, str]]:
    """Load a results CSV as a list of dict rows."""
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, stem: str) -> None:
    """Save figure as 300-dpi pdf + png with tight bbox."""
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: theta histograms (fig:theta)
# ---------------------------------------------------------------------------

def fig_theta() -> None:
    rows = read_csv("theta_rows.csv")
    th_os = np.array([float(r["theta_onsupp"]) for r in rows])
    th_xy = np.array([float(r["theta_xy"]) for r in rows])
    th_os = th_os[np.isfinite(th_os)]
    th_xy = th_xy[np.isfinite(th_xy)]

    # sized to the full ICLR text width (5.5 in) so fonts render at ~9 pt
    with plt.rc_context(SMALL_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
        panels: Tuple[Tuple[np.ndarray, str, str], ...] = (
            (th_os, r"$\hat\theta_{\mathrm{supp}}$"
                    r" (on-support energy $= \dot N_{\mathrm{tot}}$)", C_BLUE),
            (th_xy, r"$\hat\theta_{xy}$ (full update energy)", C_ORANGE),
        )
        lo = min(float(th_os.min()), float(th_xy.min()))
        hi = max(float(th_os.max()), float(th_xy.max()))
        bins = np.linspace(min(lo, 1.75), max(hi, 2.25), 33)
        for ax, (vals, xlabel, color) in zip(axes, panels):
            ax.hist(vals, bins=bins, color=color, edgecolor="white",
                    linewidth=0.4, zorder=3)
            ax.axvline(2.0, color=C_GRAY, ls="--", lw=1.2, zorder=4)
            med = float(np.median(vals))
            frac = float(np.mean((vals > 1.8) & (vals < 2.2)))
            ax.annotate(
                f"$n={vals.size}$\nmedian $= {med:.3f}$\n"
                f"${100 * frac:.1f}\\%\\in[1.8,2.2]$",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right",
                va="top", fontsize=SMALL_RC["legend.fontsize"],
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="#cccccc", alpha=0.9))
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("instances")
        # headroom so the tallest bar never touches the theory label
        axes[0].set_ylim(top=1.18 * axes[0].get_ylim()[1])
        axes[1].annotate("theory $\\theta=2$\n(dashed)", xy=(0.03, 0.95),
                         xycoords="axes fraction", ha="left", va="top",
                         fontsize=SMALL_RC["legend.fontsize"], color=C_GRAY)
        fig.tight_layout()
        save(fig, "theta_hist")
    logger.info("  theta_onsupp median=%.4f  theta_xy median=%.4f",
                float(np.median(th_os)), float(np.median(th_xy)))


# ---------------------------------------------------------------------------
# Figure 2: spectral prediction vs measured rate (fig:spectral)
# ---------------------------------------------------------------------------

def collect_spectral() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Reproduce the n_analyze.py usable-pair selection exactly.

    Returns (pred, meas, dims, n_nosnap, n_badfit); pred/meas are the
    (negative) semilog slopes: slope_pred = log rho2, slope_meas from the
    exponential-segment fit.
    """
    rows = read_csv("n1n2_rows.csv")
    pred: List[float] = []
    meas: List[float] = []
    dims: List[int] = []
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
    return (np.array(pred), np.array(meas), np.array(dims),
            n_nosnap, n_badfit)


def draw_spectral(ax: plt.Axes, fs: float, stats_box: bool = True) -> None:
    """Predicted vs measured rate scatter on a given axes (fs = annotation pt)."""
    pred, meas, dims, n_nosnap, n_badfit = collect_spectral()
    corr = float(np.corrcoef(np.log(-pred), np.log(-meas))[0, 1])
    med_ratio = float(np.median(pred / meas))
    logger.info("  spectral: n=%d (no-snapshot %d, bad-fit %d) "
                "corr=%.4f median ratio=%.3f",
                pred.size, n_nosnap, n_badfit, corr, med_ratio)

    lim_lo = 0.6 * min(float((-meas).min()), float((-pred).min()))
    lim_hi = 1.8 * max(float((-meas).max()), float((-pred).max()))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", lw=1.2,
            color=C_GRAY, zorder=2, label=r"$y=x$")
    for d in (5, 20, 50):
        s = dims == d
        ax.scatter(-meas[s], -pred[s], s=22, alpha=0.75,
                   color=DIM_COLORS[d], edgecolors="white",
                   linewidths=0.4, zorder=3, label=f"$n={d}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"measured rate $-\,\mathrm{d}\log(\mathrm{gap})"
                  r"/\mathrm{d}t$")
    ax.set_ylabel(r"predicted rate $-\log\rho^*$")
    # legend upper-left and statistics lower-right: both corners lie off
    # the y = x diagonal that carries the data
    if stats_box:
      ax.annotate(
        f"$n={pred.size}$, corr(log, log) $= {corr:.3f}$\n"
        f"median pred/meas $= {med_ratio:.3f}$",
        xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=0.88 * fs, bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="#cccccc", alpha=0.9))
    if stats_box:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="#cccccc")
    else:
        # short panel: a legend inside would sit on the y = x diagonal,
        # so place it as one row above the axes
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                  frameon=False, columnspacing=0.9, handletextpad=0.3,
                  fontsize=0.92 * fs)


def fig_spectral() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    draw_spectral(ax, 10)
    fig.tight_layout()
    save(fig, "n2_spectral")


# ---------------------------------------------------------------------------
# Figure 3: eta_inf * sigma_max(B) distribution (fig:etasigma)
# ---------------------------------------------------------------------------

def draw_eta_sigma(ax: plt.Axes, fs: float) -> None:
    """eta_inf*sigma_max(B) histogram with the stability boundary at 1."""
    rows = read_csv("eta_sigma_check.csv")
    es = np.array([float(r["eta_sigma_max"]) for r in rows])
    es = es[np.isfinite(es)]
    n_viol = int(np.sum(es > 1.0))
    es_max = float(es.max())
    q_dist_min = float(np.min(1.0 - es ** 2))
    logger.info("  eta_sigma: n=%d violations=%d max=%.4f "
                "min q-dist=%.4f", es.size, n_viol, es_max, q_dist_min)

    # last bin edge pinned at the boundary: all mass renders strictly inside
    bins = np.linspace(min(float(es.min()), 0.3), 1.0, 29)
    ax.hist(es, bins=bins, color=C_BLUE, edgecolor="white",
            linewidth=0.4, zorder=3)
    # rug: individual instances, highlighting the near-boundary cluster
    ax.plot(es, np.full_like(es, -0.28), "|", color=C_BLUE, ms=8,
            alpha=0.8, clip_on=False, zorder=4)
    ax.axvline(1.0, color=C_VERMILLION, ls="--", lw=1.6, zorder=5)
    # both labels sit strictly left of the dashed boundary (ha="right")
    ax.annotate("stability boundary\n"
                r"$\eta_\infty\sigma_{\max}(B)=1$",
                xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-5, -2), textcoords="offset points",
                ha="right", va="top", fontsize=fs, color=C_VERMILLION,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.9))
    # headroom above the tallest bar for the boundary label
    ax.set_ylim(top=1.32 * ax.get_ylim()[1])
    ymax = ax.get_ylim()[1]
    # "max" box in the empty region left of the main mass, arrow to the bar
    ax.annotate(f"max $= {es_max:.4f}$\n({n_viol} violations)",
                xy=(es_max, 0.42 * ymax), xytext=(0.64, 0.66 * ymax),
                textcoords="data",
                fontsize=fs, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.0))
    ax.set_xlabel(r"$\eta_\infty\,\sigma_{\max}(B)$ at the converged state")
    ax.set_ylabel("instances")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)


def fig_eta_sigma() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    draw_eta_sigma(ax, 10)
    fig.tight_layout()
    save(fig, "eta_sigma")


# ---------------------------------------------------------------------------
# Combined main-text figure: both panels at the final 5.5 in text width
# ---------------------------------------------------------------------------

def fig_main_panel() -> None:
    """Single two-panel figure (fig:spectral / fig:etasigma) with one font
    scale, so the two panels no longer need separate rescaling in LaTeX."""
    with plt.rc_context(SMALL_RC):
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(5.5, 1.8),
            gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.32})
        # statistics are in the caption/table; the box would sit on the data
        draw_spectral(ax_l, SMALL_RC["legend.fontsize"], stats_box=False)
        draw_eta_sigma(ax_r, SMALL_RC["legend.fontsize"])
        fig.tight_layout()
        save(fig, "fig1_local")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fig_theta()
    fig_spectral()
    fig_eta_sigma()
    fig_main_panel()
    logger.info("all figures written to %s", OUT)


if __name__ == "__main__":
    main()
