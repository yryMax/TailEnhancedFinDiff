"""
Data-scaling ablation — expanding-window walk-forward (one-off, throwaway).

Fold k (k = 1..20):  train on 2001..(2004+k)  ->  OOS year = 2005+k.
  - fold 1 : train 2001-2005 -> OOS 2006
  - fold 20: train 2001-2024 -> OOS 2025

Per fold: slice the big 25y factor csv, retrain the DLPM from scratch (reusing the
existing factor_diffusion_train.py via subprocess), sample 4096, and compare 5 moment
metrics against THAT fold's own OOS year (no Resample/Gaussian baseline).

Outputs:
  ablations/data_scaling_results.csv   (the table)
  ablations/data_scaling.png           (5 metrics vs train-years, one figure)

Run:
  python ablations/data_scaling.py --fold 1   # single fold smoke test (prints, no save)
  python ablations/data_scaling.py            # all 20 folds -> table + figure
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# --- make repo root the cwd so all relative paths + the train subprocess resolve ---
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

from factor_diffusion_train import load_cfg          # noqa: E402
from metrics.statistic import compute_cov            # noqa: E402

FULL_CSV = "model/full/factors.csv"   # 2001-2025 factor returns (date index, alpha + 7 factors)
BASE_EXP = "stable"                   # copy this cfg (levy_alpha, use_L_noise, epochs, ...)
EXP      = "data_scaling"             # scratch experiment dir: model/data_scaling/
EXP_DIR  = f"model/{EXP}"
N_GEN    = 4096
RESULTS  = "ablations/data_scaling_results.csv"
FIGURE   = "ablations/data_scaling.png"

METRICS = ["mean_mae", "var_mae", "skew_mae", "kurt_mae", "cov_fro"]


def evaluate(gen: np.ndarray, oos: np.ndarray) -> dict:
    """5 metrics: marginal moment MAEs (across factors) + relative covariance Frobenius."""
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    return {
        "mean_mae": mae(gen.mean(0),            oos.mean(0)),
        "var_mae":  mae(gen.var(0),             oos.var(0)),
        "skew_mae": mae(skew(gen, axis=0),      skew(oos, axis=0)),
        "kurt_mae": mae(kurtosis(gen, axis=0),  kurtosis(oos, axis=0)),
        "cov_fro":  float(compute_cov(gen, oos)),
    }


def run_fold(k: int, F: pd.DataFrame, base_cfg: dict, factors: list[str]) -> dict:
    train_end, test_yr = 2004 + k, 2005 + k
    train_df = F[F.index.year <= train_end]
    oos      = F[F.index.year == test_yr][factors].values
    tag      = f"fold{k:02d}"
    os.makedirs(f"{EXP_DIR}/checkpoints", exist_ok=True)

    # (a) write the fold's training csv + cfg, then reuse the existing training script.
    #     load_data() refits the StandardScaler on this csv => scaler is correct for the fold.
    train_df.to_csv(f"{EXP_DIR}/factors_{tag}.csv")
    cfg = {**base_cfg, "data_file": f"factors_{tag}.csv", "ckpt_name": f"ds_{tag}"}
    with open(f"{EXP_DIR}/cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\n=== fold {k:02d}: train 2001-{train_end} ({k + 4}y, {len(train_df)} rows) "
          f"-> OOS {test_yr} ({len(oos)} rows) ===")
    subprocess.run([sys.executable, "factor_diffusion_train.py", EXP], check=True)

    # (b) sample 4096 (already inverse-transformed to raw factor-return units).
    from scenario_generator import DiffusionSampler
    gen = DiffusionSampler(f"{EXP_DIR}/checkpoints/ds_{tag}.pt").generate(N_GEN)

    # (c) metrics vs this fold's own OOS.
    row = {"fold": k, "train_years": k + 4, "test_year": test_yr,
           "n_train": len(train_df), **evaluate(gen, oos)}
    print("  " + "  ".join(f"{m}={row[m]:.4g}" for m in METRICS))
    return row


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.6))
    for ax, m in zip(axes, METRICS):
        ax.plot(df["train_years"], df[m], "o-", color="#2ca02c")
        ax.set_title(m, fontweight="bold")
        ax.set_xlabel("train years")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Data-Scaling Ablation — expanding walk-forward (gen vs own OOS)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=130)
    print(f"saved {FIGURE}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None,
                   help="run a single fold 1..20 (smoke test; prints, no save)")
    args = p.parse_args()

    F = pd.read_csv(FULL_CSV, index_col=0, parse_dates=True)
    base_cfg = load_cfg(BASE_EXP)
    factors  = base_cfg["factors"]
    print(f"base cfg: levy_alpha={base_cfg.get('levy_alpha')} "
          f"use_L_noise={base_cfg.get('use_L_noise')} epochs={base_cfg.get('epochs')} "
          f"factors={factors}")

    if args.fold is not None:
        row = run_fold(args.fold, F, base_cfg, factors)
        print("\n" + pd.DataFrame([row]).to_string(index=False))
        return

    rows = []
    for k in range(1, 21):
        rows.append(run_fold(k, F, base_cfg, factors))
        pd.DataFrame(rows).to_csv(RESULTS, index=False)   # checkpoint after each fold
    df = pd.DataFrame(rows)
    print(f"\nsaved {RESULTS}")
    print(df.to_string(index=False))
    plot(df)


if __name__ == "__main__":
    main()
