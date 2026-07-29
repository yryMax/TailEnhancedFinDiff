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

    # csecid column order of beta/residuals (S,); needed to map (T,S) matrices back to a panel
    stock_ids: list = field(default_factory=list)

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
            res_df=np.asarray(self.res_df),
            residuals=self.residuals,
            factor_type=np.array(self.factor_type),
            factor_columns=np.array(self.F.columns.tolist()),
            factor_index=np.array(self.F.index.astype(str).tolist()),
            features=np.array(self.features),
            stock_ids=np.array(self.stock_ids),
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
        # stock_ids added later; fall back to empty for models saved before it existed
        stock_ids = npz["stock_ids"].tolist() if "stock_ids" in npz.files else []
        return cls(
            F=F,
            beta=npz["beta"],
            res_std=npz["res_std"],
            res_df=np.asarray(npz["res_df"]),
            residuals=npz["residuals"],
            factor_type=str(npz["factor_type"]),
            features=npz["features"].tolist(),
            stock_ids=stock_ids,
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
                       residuals=residuals, factor_type=factor_type,
                       stock_ids=list(R_aligned.columns), data_source=path)


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

    res_df = np.asarray(model.res_df)
    noise = np.empty((N, S))
    for s in range(S):
        df_s = float(res_df[s])
        if np.isfinite(df_s):
            noise[:, s] = np.random.standard_t(df_s, size=N) * np.sqrt((df_s - 2) / df_s)
        else:
            noise[:, s] = np.random.randn(N)

    idiosyncratic = noise * model.res_std
    return systematic + idiosyncratic


def returns_to_panel(model: FactorModel, generated_returns: np.ndarray, *,
                     factor_dates=None, source_df=None, shift=True,
                     only_filled=False) -> pd.DataFrame:
    """
    Inverse of get_factor_model's return extraction: write a (T, S) generated-return matrix back
    into the original long-panel schema [date, csecid, returns, *features], overwriting ONLY the
    `returns` column. date / csecid / characteristics are kept exactly as in the source panel.

    :param model:             fitted model; supplies stock_ids (csecid column order of beta) and,
                              by default, the source parquet (model.data_source).
    :param generated_returns: (T, S) counterfactual stock returns, e.g. cf_returns. Columns must
                              follow model.stock_ids; rows follow `factor_dates`.
    :param factor_dates:      (T,) factor dates the rows correspond to. Defaults to model.F.index.
    :param source_df:         panel to copy structure/features from. Defaults to reading
                              model.data_source. Must contain columns [date, csecid, returns, ...].
    :param shift:             if True, row at factor date t is written at the NEXT panel date
                              (mirrors fit_beta: factor_t predicts return_{t+1}). Set False to
                              write at factor_dates directly.
    :param only_filled:       if True, return just the overwritten (date, csecid) rows; otherwise
                              return the full panel with the rest of `returns` untouched.
    :return: long-panel DataFrame, same columns as source_df.
    """
    if source_df is None:
        source_df = pd.read_parquet(model.data_source)
    if factor_dates is None:
        factor_dates = list(model.F.index)

    stock_ids = list(model.stock_ids) if len(model.stock_ids) else sorted(source_df["csecid"].unique())
    generated_returns = np.asarray(generated_returns)
    if generated_returns.shape[-1] != len(stock_ids):
        raise ValueError(f"generated_returns last dim {generated_returns.shape[-1]} != "
                         f"S={len(stock_ids)}; columns must follow model.stock_ids")

    # Simple mode: rows don't align to factor dates -> just map columns to csecid and
    # return a long [csecid, returns] df (one row per generated cell), no date schema.
    if generated_returns.shape[0] != len(factor_dates):
        return (pd.DataFrame(generated_returns, columns=pd.Index(stock_ids, name="csecid"))
                .stack().rename("returns").reset_index(level="csecid")
                .reset_index(drop=True)[["csecid", "returns"]])

    if shift:
        dates_sorted = sorted(source_df["date"].unique())
        nxt = {dates_sorted[i]: dates_sorted[i + 1] for i in range(len(dates_sorted) - 1)}
        row_dates = [nxt[d] for d in factor_dates]   # KeyError if a factor date is the last panel date
    else:
        row_dates = list(factor_dates)

    gen_long = (pd.DataFrame(generated_returns,
                             index=pd.Index(row_dates, name="date"),
                             columns=pd.Index(stock_ids, name="csecid"))
                .stack().rename("_returns_gen").reset_index())

    out = source_df.merge(gen_long, on=["date", "csecid"], how="left")
    filled = out["_returns_gen"].notna()
    out["returns"] = out["_returns_gen"].where(filled, out["returns"])
    out = out.drop(columns="_returns_gen")
    return out[filled.values].reset_index(drop=True) if only_filled else out


def save_model(model: FactorModel, prefix: str) -> None:
    model.save(prefix)

def load_model(prefix: str) -> FactorModel:
    return FactorModel.load(prefix)

def re_normalizing(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Sector-neutralize characteristics: cross-sectionally z-score each feature within
    every (date, GICS sector) group, so a characteristic carries no sector-level mean
    or scale and the fitted factors are not contaminated by static sector tilts.

    :param df:       long panel with columns [date, csecid, returns, *features, gics, ...]
    :param features: characteristic columns to renormalize
    :return:         copy of df with each feature replaced by its per-(date, sector) z-score;
                     all other columns (including original row order and index) are preserved
    """
    out = df.copy()
    grp = out.groupby(["date", "gics"], dropna=False)   # keep rows with missing gics as their own bucket
    for f in features:
        mean = grp[f].transform("mean")
        std  = grp[f].transform("std")           # NaN for singleton groups
        out[f] = (out[f] - mean) / std.where(std > 0, 1.0)   # guard zero/degenerate std
    return out

def get_factor_model(path: str, features: list[str]) -> FactorModel:
    df   = pd.read_parquet(path)
    processed_df = re_normalizing(df, features)
    R, F = build_regression_factors(processed_df, features)
    model = fit_beta(F, R, path)
    return model


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

    print(f"Factor model on {prefix}  (train={train_path}, test={test_path})")
    get_factor_model(train_path, features).save(prefix)
    get_factor_model(test_path,  features).save(f"{prefix}/test")
