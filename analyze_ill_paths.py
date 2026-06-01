"""
Diagnose runaway ("ill") trajectories in a generated factor-path file.

Given an .npy of shape (N, F, T) — N paths, F factors, T days — this script:
  1. plots the cumulative-return trajectories,
  2. flags ill paths (per-path peak |x| >> the bulk) and reports their fraction,
  3. drops them, re-plots, and prints higher-order moments before vs after.

A single runaway path is enough to blow up kurtosis (a 4th-power statistic), so
the point is to show how few "ill" samples there are and how the moments recover.

Usage:
    python analyze_ill_paths.py model/temporal_cond/samples/path_cond_another_ill.npy
    python analyze_ill_paths.py <file.npy> --thresh-mult 10 --out assets/
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FACTORS = ["market", "growth", "momentum", "quality", "size", "value", "volatility"]


def find_ill(paths, thresh_mult):
    """A path is ill if its peak |return| exceeds thresh_mult * median peak across paths.
    Returns (ill_mask, peak_per_path, threshold)."""
    peak   = np.abs(paths).reshape(len(paths), -1).max(axis=1)   # (N,) per-path peak |x|
    thresh = thresh_mult * np.median(peak)
    return peak > thresh, peak, thresh


def plot_trajectories(paths, factors, title, out_png):
    """Grid of cumulative-return paths, one subplot per factor (grey = all paths, black = mean)."""
    N, F, T = paths.shape
    cum  = np.cumsum(paths, axis=2)                              # (N, F, T)
    days = np.arange(T)

    fig, axes = plt.subplots(F, 1, figsize=(9, 1.4 * F), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(days, cum[:, j].T, color="C0", lw=0.4, alpha=0.12)   # all N paths
        ax.plot(days, cum[:, j].mean(0), color="k", lw=1.6)
        ax.axhline(0, color="r", ls="--", lw=0.6)
        ax.set_ylabel(factors[j], fontsize=8)
    axes[-1].set_xlabel("day")
    fig.suptitle(f"{title}  (N={N})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png}")


def moments(paths, factors):
    """Per-factor (std, skew, kurtosis) of the per-path cumulative log-return.

    Matches `_aggregate(op=1)` in factor_evaluation.ipynb: compound each path over
    time as sum_t log(1 + r_t), giving one value per (path, factor), then take moments
    over the N paths. NB: a return <= -1 (a >100% loss, e.g. a runaway path) makes
    log1p NaN, so the raw ill sample reports NaN until those paths are dropped.
    """
    agg = np.log1p(paths).sum(axis=2)                            # (N, F) cumulative log-return
    return {f: (agg[:, j].std(), stats.skew(agg[:, j]), stats.kurtosis(agg[:, j]))
            for j, f in enumerate(factors)}


def print_moments(before, after, factors):
    print(f"\n{'factor':12s} {'std (before→after)':>26s} {'kurtosis (before→after)':>30s}")
    for f in factors:
        sb, _, kb = before[f]
        sa, _, ka = after[f]
        print(f"{f:12s} {sb:11.4f} → {sa:<11.4f} {kb:14.1f} → {ka:<14.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy", help="generated paths, shape (N, F, T)")
    ap.add_argument("--thresh-mult", type=float, default=10.0,
                    help="ill if per-path peak |x| > mult * median peak (default 10)")
    ap.add_argument("--out", default=".", help="directory for output figures")
    args = ap.parse_args()

    paths = np.load(args.npy)                                    # (N, F, T)
    assert paths.ndim == 3, f"expected (N, F, T), got {paths.shape}"
    N, F, T = paths.shape
    factors = FACTORS if F == len(FACTORS) else [f"f{j}" for j in range(F)]
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.npy))[0]
    print(f"loaded {args.npy}  shape={paths.shape}  global max|x|={np.abs(paths).max():.3f}")

    ill_mask, peak, thresh = find_ill(paths, args.thresh_mult)
    n_ill = int(ill_mask.sum())
    print(f"\nill threshold (peak |x| > {args.thresh_mult:g} * median): {thresh:.4f}")
    print(f"ill paths: {n_ill}/{N}  ({100 * n_ill / N:.2f}%)")
    if n_ill:
        worst = np.argsort(peak)[::-1][:n_ill]
        print(f"ill path ids & peaks: " +
              ", ".join(f"{int(i)}={peak[i]:.2f}" for i in worst))

    clean = paths[~ill_mask]

    plot_trajectories(paths, factors, f"{stem}: all paths", f"{args.out}/{stem}_all.png")
    plot_trajectories(clean, factors, f"{stem}: ill removed", f"{args.out}/{stem}_clean.png")

    print_moments(moments(paths, factors), moments(clean, factors), factors)


if __name__ == "__main__":
    main()
