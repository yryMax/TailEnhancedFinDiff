from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
import yaml
import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t


def _pivot(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Pivot panel data into wide-format matrices. Dates are sorted by pivot; output may contain NaN.
    :param df:       panel DataFrame with columns [date, csecid, returns, *features]
    :param features: factor names to pivot
    :return:      R     : (T, S) stock returns pivot
                  chars : dict mapping each feature name to its (T, S) characteristic pivot
    """
    R = df.pivot_table(index="date", columns="csecid", values="returns")
    chars = {f: df.pivot_table(index="date", columns="csecid", values=f) for f in features}
    return R, chars

def build_regression_factors(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional OLS each period: R_t = B_{t-1} * f_t + eps
    :param df:       the pivot return value on sectional stocks
    :param features: factor names
    :return:   R : (T, S) stock (backward) returns pivot
    F : (T-1, K+1) factor returns, columns = ['alpha', 'market', *FACTORS]
        alpha column is a constant 1.0 (intercept of the second-stage OLS in fit_beta)
        we assume market is a factor return and the corresponding factor loader is 1
    """


    R, chars = _pivot(df, features)
    dates = sorted(R.index.unique())

    rows, valid_dates = [], []
    for i in range(1, len(dates)):
        date, prev_date = dates[i], dates[i - 1]

        R_t = R.loc[date].sort_index()
        B_cols = [chars[f].loc[prev_date].sort_index() for f in features]
        B_df = pd.concat(B_cols, axis=1)

        mask = R_t.notna() & B_df.notna().all(axis=1)
        R_clean = R_t[mask].values
        B_clean = np.column_stack([np.ones(mask.sum()), B_df[mask].values])

        f_t, _, _, _ = np.linalg.lstsq(B_clean, R_clean, rcond=None)
        rows.append(f_t)
        valid_dates.append(prev_date)

    F = pd.DataFrame(rows, index=valid_dates, columns=["market"] + features)
    F.insert(0, "alpha", 1.0)   # constant intercept column for fit_beta
    return R, F


def build_portsort_factors(df: pd.DataFrame, features: list[str], N_QUANTILES = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Portfolio-sort factor construction: long–short spread (top quantile – bottom quantile) per factor each period.
    :param df:          panel DataFrame with columns [date, csecid, returns, *features]
    :param features:    factor names
    :param N_QUANTILES: number of quantile buckets for sorting
    :return:            R : (T, S) stock returns pivot
                        F : (T-1, K) factor returns, columns = features
                            no alpha column — port-sort factors are zero-investment spreads
    """
    R, chars = _pivot(df, features)
    dates = sorted(R.index.unique())

    spreads: dict[str, list] = {f: [] for f in features}
    valid_dates = []

    for i in range(1, len(dates)):
        date, prev_date = dates[i], dates[i - 1]
        R_t = R.loc[date]

        for f in features:
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

    # per-stock Student-t degrees of freedom (S,); np.inf marks Gaussian fallback
    res_df: np.ndarray

    # time-series residuals (T, S), may contain NaN for missing observations
    residuals: np.ndarray

    # factor type tag, either "regression" or "portsort"
    factor_type: str = "regression"

    # factor names used to build F
    features: list = field(default_factory=list)

    # parquet path(s) used to fit the model
    data_source: str = ""

    # stock IDs aligned to axis-1 (S) of beta/res_std/residuals; same order as R's pivot columns
    csecids: np.ndarray = field(default_factory=lambda: np.array([]))

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
            res_df=np.asarray(self.res_df),
            residuals=self.residuals,
            factor_type=np.array(self.factor_type),
            factor_columns=np.array(self.F.columns.tolist()),
            factor_index=np.array(self.F.index.astype(str).tolist()),
            features=np.array(self.features),
            data_source=np.array(self.data_source),
            csecids=np.asarray(self.csecids),
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
            res_df=np.asarray(npz["res_df"]),
            residuals=npz["residuals"],
            factor_type=str(npz["factor_type"]),
            features=npz["features"].tolist(),
            data_source=npz["data_source"].tolist(),
            csecids=npz["csecids"] if "csecids" in npz.files else np.array([]),
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

    # Per-stock Student-t fit on standardised residuals; np.inf means Gaussian fallback.
    res_df = np.full(S, np.inf)
    for s in range(S):
        col = residuals[:, s]
        col = col[np.isfinite(col)]
        if len(col) < 30 or not (res_std[s] > 0):
            continue
        try:
            df_s, _, _ = scipy_t.fit(col / res_std[s], floc=0)
        except Exception:
            continue
        if 2.1 <= df_s < 30:
            res_df[s] = df_s

    factor_type = "portsort" if "alpha" not in F.columns else "regression"
    return FactorModel(F=F_aligned, beta=beta, res_std=res_std, res_df=res_df,
                       residuals=residuals, factor_type=factor_type, data_source=path,
                       csecids=R_aligned.columns.to_numpy())


def reconstruct_returns(model: FactorModel, fs: np.ndarray,
                        return_components: bool = False):
    """
    Reconstruct stock returns from factor samples via R = F @ beta + idiosyncratic noise.
    Idiosyncratic noise is drawn from a scaled Student-t fitted to model residuals.
    :param model: fitted FactorModel
    :param fs:    (N, K) factor samples — column order must match model.F
    :param return_components: if True, return (R, systematic, idiosyncratic).
    :return:      (N, S) reconstructed stock returns, or tuple of three (N, S) arrays.
    """
    N, S = fs.shape[0], model.beta.shape[1]
    systematic = fs @ model.beta

    # Vectorized scaled Student-t via t = z / sqrt(chi2/df). For non-finite df,
    # use a large df (effectively standard normal).
    res_df  = np.asarray(model.res_df, dtype=np.float64)
    df_use  = np.where(np.isfinite(res_df), res_df, 1e6)           # (S,)
    z       = np.random.standard_normal((N, S))
    chi2    = np.random.chisquare(df_use, size=(N, S))
    noise   = z / np.sqrt(chi2 / df_use) * np.sqrt((df_use - 2) / df_use)

    idiosyncratic = noise * model.res_std
    total = systematic + idiosyncratic
    if return_components:
        return total, systematic, idiosyncratic
    return total


def _path_dates(dates, T):
    """Validate/normalize a length-T date array; None -> all-NaT. No bdate_range
    synthesis — callers must pass the real trading-day calendar to stay aligned with
    the OOS ground truth (bdate_range silently includes market holidays)."""
    if dates is None:
        return np.full(T, np.datetime64("NaT"), dtype="datetime64[ns]")
    dates = pd.DatetimeIndex(dates).values
    if len(dates) != T:
        raise ValueError(f"dates length {len(dates)} != path length T={T}")
    return dates


def save_reconstructed_stocks(R: np.ndarray, csecids, out_path: str,
                              dates=None) -> None:
    """
    Long-format parquet of reconstructed stock returns (parquet, not CSV: the dense
    N*S*T table is huge — parquet is ~10x smaller and far faster to read/write).
    :param R:       (N, S, T) reconstructed stock returns
    :param csecids: length-S array of stock IDs (axis-1 of R)
    :param dates:   length-T array of the actual calendar dates of the path (real trading
                    days, not synthesized); None -> NaT for all rows
    Columns: path_id, csecid, date, returns
    """
    N, S, T = R.shape
    dates = _path_dates(dates, T)
    pd.DataFrame({
        "path_id": np.repeat(np.arange(N), S * T),
        "csecid":  np.tile(np.repeat(np.asarray(csecids), T), N),
        "date":    np.tile(dates, N * S),
        "returns": R.reshape(-1),
    }).to_parquet(out_path, index=False)


def save_reconstructed_factors(F: np.ndarray, factor_names, out_path: str,
                               dates=None) -> None:
    """
    Long-format parquet of generated factor returns.
    :param F:            (N, K, T) factor returns
    :param factor_names: length-K list of factor names (axis-1 of F)
    :param dates:        length-T array of the actual calendar dates of the path (real
                         trading days, not synthesized); None -> NaT
    Columns: path_id, factor_name, date, returns.
    """
    N, K, T = F.shape
    dates = _path_dates(dates, T)
    pd.DataFrame({
        "path_id":     np.repeat(np.arange(N), K * T),
        "factor_name": np.tile(np.repeat(np.asarray(factor_names), T), N),
        "date":        np.tile(dates, N * K),
        "returns":     F.reshape(-1),
    }).to_parquet(out_path, index=False)


def save_model(model: FactorModel, prefix: str) -> None:
    model.save(prefix)

def load_model(prefix: str) -> FactorModel:
    return FactorModel.load(prefix)


def get_factor_model(path: str, features: list[str]) -> FactorModel:
    df   = pd.read_parquet(path)
    R, F = build_regression_factors(df, features)
    model = fit_beta(F, R, path)
    return model

def dump_to_achievement(model: FactorModel, fs_path: str, out_dir: str = "achievement",
                        dates=None) -> tuple[str, str]:
    """
    End-to-end: load generated factor paths, reconstruct stock returns, and dump both
    factor and stock returns as long-format parquet files under `out_dir`.

    :param model:      fitted FactorModel (must carry .csecids; refit if loaded from an
                       old checkpoint that predates the csecids field)
    :param fs_path:    .npy of generated factor paths, shape (N, F, T) or (N, F) for a
                       single cross-section. F is the non-alpha factor count and must
                       equal len(model.F.columns) - 1, ordered as model.F.columns[1:].
    :param out_dir:    output directory (created if missing)
    :param dates:      length-T array of the path's real calendar dates; None -> NaT dates
    :return:           (stocks_path, factors_path); file stems taken from fs_path
    """

    fs = np.load(fs_path)
    if fs.ndim == 2:                       # (N, F) single cross-section -> (N, F, 1)
        fs = fs[:, :, None]
    N, F, T = fs.shape

    factor_names = list(model.F.columns[1:])   # drop the 'alpha' intercept column

    alpha   = np.ones((N, 1, T), dtype=fs.dtype)
    fs_full = np.concatenate([alpha, fs], axis=1)                 # (N, F+1, T)
    fs_flat = fs_full.transpose(0, 2, 1).reshape(N * T, F + 1)    # (N*T, F+1)
    R_flat  = reconstruct_returns(model, fs_flat)                 # (N*T, S)
    S       = R_flat.shape[1]
    R       = R_flat.reshape(N, T, S).transpose(0, 2, 1)         # (N, S, T)

    os.makedirs(out_dir, exist_ok=True)
    stem         = os.path.splitext(os.path.basename(fs_path))[0]
    stocks_path  = f"{out_dir}/{stem}_stocks.parquet"
    factors_path = f"{out_dir}/{stem}_factors.parquet"

    save_reconstructed_stocks(R, model.csecids, stocks_path, dates)
    save_reconstructed_factors(fs, factor_names, factors_path, dates)
    print(f"Dumped {R.shape} stocks -> {stocks_path}\n"
          f"Dumped {fs.shape} factors -> {factors_path}")
    return stocks_path, factors_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; reads model/<exp_name>/cfg.yaml")
    args = parser.parse_args()

    prefix = f"model/{args.exp_name}"
    with open(f"{prefix}/cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    features   = cfg["characteristics"]
    train_path = cfg["train_path"]
    test_path  = cfg["test_path"]

    # stock return to factor return
    #get_factor_model(train_path, features).save(prefix)
    #get_factor_model(test_path,  features).save(f"{prefix}/test")

    model = load_model(prefix)
    dump_to_achievement(model, "model/temporal_cond/samples/path_cond_another_500x22.npy")