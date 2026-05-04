"""
Compute a daily stress score for the full 25y factor return series.

Economic-meaning version (Option A):
    Each factor is assigned a sign reflecting its role in a risk-off / stress regime.
    Factors are then z-scored on the full sample and combined with those signs.

    Sign convention (higher score => more stressed market):
      market     : -1  (broad market sells off in stress)
      momentum   : -1  (momentum crashes during sharp reversals)
      size       : -1  (small caps underperform in stress, SMB negative)
      volatility : -1  (high-vol stocks underperform, low-vol "defensive" wins)
      quality    : +1  (flight to quality)
      value      :  0  (sign is regime-dependent; left out of the score)
      growth     :  0  (sign is regime-dependent; left out of the score)

The resulting StressScore is itself standardised so percentiles map directly
onto an interpretable "how-stressed-was-this-day" axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRESS_WEIGHTS: dict[str, float] = {
    "market":     -1.0,
    "momentum":   -1.0,
    "size":       -1.0,
    "volatility": -1.0,
    "quality":    +1.0,
    "value":       0.0,
    "growth":      0.0,
}


def compute_stress_score(F: pd.DataFrame, weights: dict[str, float] = STRESS_WEIGHTS) -> pd.DataFrame:
    """
    :param F:       (T, K) factor returns indexed by date; must contain the columns in `weights`
    :param weights: factor -> sign/weight mapping (economic-meaning version)
    :return:        DataFrame indexed by date with columns:
                      - z_<factor> for every active factor
                      - raw_score          (sum of signed z-scores, before final standardisation)
                      - stress_score       (raw_score standardised to mean 0, std 1)
                      - stress_percentile  (rank in [0, 1])
    """
    active = {f: w for f, w in weights.items() if w != 0.0 and f in F.columns}
    if not active:
        raise ValueError("No active factors found in F for the given weights.")

    Z = pd.DataFrame(index=F.index)
    for f in active:
        x = F[f].astype(float)
        Z[f"z_{f}"] = (x - x.mean()) / x.std(ddof=0)

    raw = sum(w * Z[f"z_{f}"] for f, w in active.items())
    score = (raw - raw.mean()) / raw.std(ddof=0)

    out = Z.copy()
    out["raw_score"] = raw
    out["stress_score"] = score
    out["stress_percentile"] = score.rank(pct=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute economic-meaning stress score over full 25y factor series.")
    parser.add_argument("--factors", default="model/full/factors.csv",
                        help="Path to factors.csv produced by factor_model.py (full25y model).")
    parser.add_argument("--out", default="model/full/stress_score.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    F = pd.read_csv(args.factors, index_col=0, parse_dates=True)
    if "alpha" in F.columns:
        F = F.drop(columns=["alpha"])

    out = compute_stress_score(F)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)

    s = out["stress_score"]
    qs = s.quantile([0.50, 0.90, 0.95, 0.99])
    top = out["stress_score"].nlargest(10)

    print(f"Wrote {len(out)} rows -> {args.out}")
    print(f"stress_score: mean={s.mean():.3f}  std={s.std():.3f}  "
          f"min={s.min():.3f}  max={s.max():.3f}")
    print("quantiles  q50/q90/q95/q99: "
          + "  ".join(f"{q:.3f}" for q in qs.values))
    print("top-10 most stressed days:")
    print(top.to_string())


if __name__ == "__main__":
    main()
