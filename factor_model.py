from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES: list[str] = ["growth", "momentum", "quality", "size", "value", "volatility"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(*paths: str) -> pd.DataFrame:
    """Concatenate one or more parquet files into a single DataFrame."""
    return pd.concat([pd.read_parquet(p) for p in paths], axis=0)


def _pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (R, chars) where R is the returns pivot and chars is per-factor."""
    R = df.pivot_table(index="date", columns="csecid", values="returns")
    chars = {f: df.pivot_table(index="date", columns="csecid", values=f) for f in FEATURES}
    return R, chars


# ---------------------------------------------------------------------------
# Factor return construction
# ---------------------------------------------------------------------------

def build_regression_factors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional OLS each period: R_t = B_{t-1} * f_t + eps

    Returns
    -------
    R : (T, S) stock returns pivot
    F : (T-1, K+1) factor returns, columns = ['alpha', 'market', *FACTORS]
        alpha column is a constant 1.0 (intercept of the second-stage OLS in fit_beta)
    """
    R, chars = _pivot(df)
    dates = sorted(R.index.unique())

    rows, valid_dates = [], []
    for i in range(1, len(dates)):
        date, prev_date = dates[i], dates[i - 1]

        R_t = R.loc[date].sort_index()
        B_cols = [chars[f].loc[prev_date].sort_index() for f in FEATURES]
        B_df = pd.concat(B_cols, axis=1)

        mask = R_t.notna() & B_df.notna().all(axis=1)
        R_clean = R_t[mask].values
        B_clean = np.column_stack([np.ones(mask.sum()), B_df[mask].values])

        f_t, _, _, _ = np.linalg.lstsq(B_clean, R_clean, rcond=None)
        rows.append(f_t)
        valid_dates.append(prev_date)

    F = pd.DataFrame(rows, index=valid_dates, columns=["market"] + FEATURES)
    F.insert(0, "alpha", 1.0)   # constant intercept column for fit_beta
    return R, F


def build_portsort_factors(df: pd.DataFrame, N_QUANTILES = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Portfolio-sort: long–short spread (Q_J – Q_1) per factor each period.

    Returns
    -------
    R : (T, S) stock returns pivot
    F : (T-1, K) factor returns, columns = FACTORS
        No alpha column — port-sort factors are zero-investment spreads.
    """
    R, chars = _pivot(df)
    dates = sorted(R.index.unique())

    spreads: dict[str, list] = {f: [] for f in FEATURES}
    valid_dates = []

    for i in range(1, len(dates)):
        date, prev_date = dates[i], dates[i - 1]
        R_t = R.loc[date]

        for f in FEATURES:
            char_prev = chars[f].loc[prev_date]
            mask = R_t.notna() & char_prev.notna()
            r, c = R_t[mask], char_prev[mask]
            labels = pd.qcut(c, q=N_QUANTILES, labels=False, duplicates="drop")
            port_ret = r.groupby(labels).mean().reindex(range(N_QUANTILES))
            spreads[f].append(port_ret.iloc[-1] - port_ret.iloc[0])

        valid_dates.append(prev_date)

    F = pd.DataFrame(spreads, index=valid_dates)
    return R, F


# ---------------------------------------------------------------------------
# Factor model dataclass
# ---------------------------------------------------------------------------

@dataclass
class FactorModel:
    """Fitted factor model: R ≈ F @ beta + eps."""

    # factor returns used to fit (T, K)
    F: pd.DataFrame

    # OLS coefficients (K, S)
    beta: np.ndarray

    # per-stock residual std (S,)
    res_std: np.ndarray

    # time-series residuals (T, S), may contain NaN for missing observations
    residuals: np.ndarray

    # factor type tag, either "regression" or "portsort"
    factor_type: str = "regression"

    # factor names used to build F
    features: list = field(default_factory=list)

    # parquet path(s) used to fit the model
    data_source: list = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, prefix: str) -> None:
        """
        Write two files:
          {prefix}_factors.csv   — factor return time series (human-readable)
          {prefix}_model.npz     — beta, res_std, residuals + scalar config
        """
        os.makedirs(prefix, exist_ok=True)

        self.F.to_csv(f"{prefix}/test_factors.csv")
        print(f"Saved → {prefix}/test_factors.csv")
        np.savez(
            f"{prefix}/model.npz",
            beta=self.beta,
            res_std=self.res_std,
            residuals=self.residuals,
            factor_type=np.array(self.factor_type),
            factor_columns=np.array(self.F.columns.tolist()),
            factor_index=np.array(self.F.index.astype(str).tolist()),
            features=np.array(self.features),
            data_source=np.array(self.data_source),
        )
        print(f"Saved → {prefix}/factors.csv + {prefix}/model.npz")

    @classmethod
    def load(cls, prefix: str) -> "FactorModel":
        """Load a model previously saved with .save()."""
        F = pd.read_csv(f"{prefix}/factors.csv", index_col=0, parse_dates=True)

        npz = np.load(f"{prefix}/model.npz", allow_pickle=True)
        return cls(
            F=F,
            beta=npz["beta"],
            res_std=npz["res_std"],
            residuals=npz["residuals"],
            factor_type=str(npz["factor_type"]),
            features=npz["features"].tolist(),
            data_source=npz["data_source"].tolist(),
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_beta(F: pd.DataFrame, R: pd.DataFrame) -> FactorModel:
    """
    Fit per-stock OLS betas: r_s = F @ beta_s + eps_s

    F is aligned to dates (T, K); R is stock returns (T', S).
    The function handles the next-date shift: F at t predicts R at t+1.

    Returns a FactorModel with beta (K, S), res_std (S,), residuals (T, S).
    """
    date_all = sorted(R.index.unique())
    next_date = {date_all[i]: date_all[i + 1] for i in range(len(date_all) - 1)}

    valid_F_dates = [d for d in F.index if d in next_date]
    F_aligned = F.loc[valid_F_dates]
    R_aligned = R.loc[[next_date[d] for d in valid_F_dates]]
    R_aligned.index = F_aligned.index

    K, S = F_aligned.shape[1], R_aligned.shape[1]
    beta = np.full((K, S), np.nan)

    for i in range(S):
        r = R_aligned.iloc[:, i]
        mask = r.notna()
        if mask.sum() == 0:
            continue
        beta[:, i] = np.linalg.lstsq(
            F_aligned.values[mask], r.values[mask], rcond=None
        )[0]

    R_hat = F_aligned.values @ beta                    # (T, S)
    residuals = R_aligned.values - R_hat               # (T, S)
    res_std = np.sqrt(np.nanmean(residuals ** 2, axis=0))  # (S,)

    factor_type = "portsort" if "alpha" not in F.columns else "regression"
    return FactorModel(F=F_aligned, beta=beta, res_std=res_std,
                       residuals=residuals, factor_type=factor_type)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_returns(model: FactorModel, fs: np.ndarray) -> np.ndarray:
    """
    Reconstruct stock returns from factor samples.

    Parameters
    ----------
    model : FactorModel
    fs    : (N, K) array of factor samples — same column order as model.F

    Returns
    -------
    R_gen : (N, S) reconstructed stock returns
    """
    N, S = fs.shape[0], model.beta.shape[1]
    systematic = fs @ model.beta                                         # (N, S)
    idiosyncratic = np.random.normal(0, 1, size=(N, S)) * model.res_std # (N, S)
    return systematic + idiosyncratic


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def save_model(model: FactorModel, prefix: str) -> None:
    model.save(prefix)

def load_model(prefix: str) -> FactorModel:
    return FactorModel.load(prefix)

if __name__ == '__main__':
    df   = load_parquet("data/test1y.parquet")
    R, F = build_regression_factors(df)
    model = fit_beta(F, R)
    model.save("model/regression")