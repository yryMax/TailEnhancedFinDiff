from __future__ import annotations

import os
from dataclasses import dataclass, field
import yaml
import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t

with open("cfg.yaml") as f:
    _cfg = yaml.safe_load(f)["train"]

FEATURES: list[str] = [f for f in _cfg["factor_names"] if f != "market"]


def _pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Pivot panel data into wide-format matrices. Dates are sorted by pivot; output may contain NaN.
    :param df:    panel DataFrame with columns [date, csecid, returns, *FEATURES]
    :return:      R     : (T, S) stock returns pivot
                  chars : dict mapping each feature name to its (T, S) characteristic pivot
    """
    R = df.pivot_table(index="date", columns="csecid", values="returns")
    chars = {f: df.pivot_table(index="date", columns="csecid", values=f) for f in FEATURES}
    return R, chars

def build_regression_factors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional OLS each period: R_t = B_{t-1} * f_t + eps
    :param df: the pivot return value on sectional stocks
    :return:   R : (T, S) stock (backward) returns pivot
    F : (T-1, K+1) factor returns, columns = ['alpha', 'market', *FACTORS]
        alpha column is a constant 1.0 (intercept of the second-stage OLS in fit_beta)
        we assume market is a factor return and the corresponding factor loader is 1
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
    Portfolio-sort factor construction: long–short spread (top quantile – bottom quantile) per factor each period.
    :param df:          panel DataFrame with columns [date, csecid, returns, *FEATURES]
    :param N_QUANTILES: number of quantile buckets for sorting
    :return:            R : (T, S) stock returns pivot
                        F : (T-1, K) factor returns, columns = FEATURES
                            no alpha column — port-sort factors are zero-investment spreads
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

@dataclass
class FactorModel:
    """Fitted factor model: R ≈ F @ beta + eps."""

    # factor returns used to fit (T, K)
    F: pd.DataFrame

    # OLS coefficients (K, S)
    beta: np.ndarray

    # per-stock residual std (S,)
    res_std: np.ndarray

    # degrees of freedom fitted from pooled standardised residuals
    res_df: float

    # time-series residuals (T, S), may contain NaN for missing observations
    residuals: np.ndarray

    # factor type tag, either "regression" or "portsort"
    factor_type: str = "regression"

    # factor names used to build F
    features: list = field(default_factory=list)

    # parquet path(s) used to fit the model
    data_source: str = ""

    def save(self, prefix: str) -> None:
        """
        Persist the fitted model to disk.
        :param prefix: output directory; two files are written:
                       {prefix}/factors.csv — factor return time series (human-readable)
                       {prefix}/model.npz   — beta, res_std, residuals and scalar metadata
        """
        os.makedirs(prefix, exist_ok=True)

        self.F.to_csv(f"{prefix}/factors.csv")
        np.savez(
            f"{prefix}/model.npz",
            beta=self.beta,
            res_std=self.res_std,
            res_df=np.array(self.res_df),
            residuals=self.residuals,
            factor_type=np.array(self.factor_type),
            factor_columns=np.array(self.F.columns.tolist()),
            factor_index=np.array(self.F.index.astype(str).tolist()),
            features=np.array(self.features),
            data_source=np.array(self.data_source)
        )

        print(f"Model saved to {prefix}/factors.csv and {prefix}/model.npz")

    @classmethod
    def load(cls, prefix: str) -> FactorModel:
        """
        Load a model previously saved with .save().
        :param prefix: directory containing factors.csv and model.npz
        :return:  reconstructed FactorModel instance
        """
        F = pd.read_csv(f"{prefix}/factors.csv", index_col=0, parse_dates=True)

        npz = np.load(f"{prefix}/model.npz", allow_pickle=True)
        return cls(
            F=F,
            beta=npz["beta"],
            res_std=npz["res_std"],
            res_df=float(npz["res_df"]),
            residuals=npz["residuals"],
            factor_type=str(npz["factor_type"]),
            features=npz["features"].tolist(),
            data_source=npz["data_source"].tolist(),
        )

def fit_beta(F: pd.DataFrame, R: pd.DataFrame, path: str) -> FactorModel:
    """
    Fit per-stock OLS betas: r_s = F @ beta_s + eps_s.
    Handles the next-date shift: factor F at t predicts return R at t+1.
    Residual distribution is fitted to a Student-t to capture fat tails, the df id get from
    experimental data.
    :param F:    (T, K) factor returns
    :param R:    (T, S) stock returns pivot
    :param path: data source path stored in the returned model for traceability
    :return:     FactorModel with beta (K, S), res_std (S,), residuals (T, S)
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

    R_hat = F_aligned.values @ beta
    residuals = R_aligned.values - R_hat
    res_std = np.sqrt(np.nanmean(residuals ** 2, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        std_res = residuals / np.where(res_std > 0, res_std, np.nan)
    flat = std_res.ravel()
    flat = flat[np.isfinite(flat)]
    df_fit, _, _ = scipy_t.fit(flat, floc=0)
    print(f"Fitted residual t-distribution: df = {df_fit:.3f}")

    factor_type = "portsort" if "alpha" not in F.columns else "regression"
    return FactorModel(F=F_aligned, beta=beta, res_std=res_std, res_df=df_fit,
                       residuals=residuals, factor_type=factor_type, data_source=path)


def reconstruct_returns(model: FactorModel, fs: np.ndarray) -> np.ndarray:
    """
    Reconstruct stock returns from factor samples via R = F @ beta + idiosyncratic noise.
    Idiosyncratic noise is drawn from a scaled Student-t fitted to model residuals.
    :param model: fitted FactorModel
    :param fs:    (N, K) factor samples — column order must match model.F
    :return:      (N, S) reconstructed stock returns
    """
    N, S = fs.shape[0], model.beta.shape[1]
    systematic = fs @ model.beta
    noise = np.random.standard_t(model.res_df, size=(N, S)) * np.sqrt((model.res_df - 2) / model.res_df)
    idiosyncratic = noise * model.res_std
    return systematic + idiosyncratic


def save_model(model: FactorModel, prefix: str) -> None:
    model.save(prefix)

def load_model(prefix: str) -> FactorModel:
    return FactorModel.load(prefix)


def get_factor_model(path: str) -> FactorModel:
    df   = pd.read_parquet(path)
    R, F = build_regression_factors(df)
    model = fit_beta(F, R, path)
    return model


if __name__ == '__main__':
    get_factor_model("data/train24y.parquet").save("model/regression")
    get_factor_model("data/test1y.parquet").save("model/regression/test")
