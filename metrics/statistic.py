import numpy as np
from scipy.stats import skew, kurtosis, wasserstein_distance
from .registry import MetricRegistry

registry = MetricRegistry()


# ============== Base Functions ==============

def var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Value at Risk: alpha-quantile of returns (left tail)."""
    return float(np.percentile(returns, alpha * 100))


def es(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR): mean of returns below VaR."""
    v = var(returns, alpha)
    return float(np.mean(returns[returns <= v]))


def _valid_mask(Y: np.ndarray, min_samples: int = 100) -> np.ndarray:
    """Returns bool mask of assets with >= min_samples valid (non-NaN) rows."""
    return np.sum(~np.isnan(Y), axis=0) >= min_samples


def _agg(relative: np.ndarray) -> dict:
    """Aggregate per-asset relative errors, ignoring NaN and inf."""
    finite = relative[np.isfinite(relative)]
    return {'mean': float(np.mean(finite)), 'std': float(np.std(finite))}


def _nancovariance(X: np.ndarray) -> np.ndarray:
    """Pairwise complete observations covariance matrix for data with NaN.
    For each (i,j) pair, uses only rows where both assets are non-NaN.
    """
    means = np.nanmean(X, axis=0)
    Xc = np.where(~np.isnan(X), X - means, 0.0)
    mask = (~np.isnan(X)).astype(np.float64)
    cov_sum = Xc.T @ Xc           # [p, p]
    count = mask.T @ mask          # [p, p]: number of valid pairs per (i,j)
    return cov_sum / np.maximum(count - 1, 1)


def _compute_es_per_asset(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Helper: compute ES for each asset, ignoring NaN."""
    def _es_nan(col):
        col = col[~np.isnan(col)]
        return es(col, alpha) if len(col) > 0 else np.nan
    return np.array([_es_nan(data[:, i]) for i in range(data.shape[1])])


# ============== Registered Metrics ==============

@registry.register('W1')
def compute_wasserstein1(X: np.ndarray, Y: np.ndarray) -> float:
    """Per-asset marginal Wasserstein-1 distance, averaged across valid assets.
    For each asset, uses rows where both X and Y are non-NaN.
    """
    mask = _valid_mask(Y)
    X_v, Y_v = X[:, mask].astype(np.float64), Y[:, mask].astype(np.float64)
    n_assets = X_v.shape[1]
    distances = []
    for j in range(n_assets):
        x_col = X_v[:, j][~np.isnan(X_v[:, j])]
        y_col = Y_v[:, j][~np.isnan(Y_v[:, j])]
        if len(x_col) >= 2 and len(y_col) >= 2:
            distances.append(wasserstein_distance(x_col, y_col))
    return float(np.mean(distances)) if distances else np.nan


@registry.register('Cov')
def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices.
    Uses pairwise complete observations for Y to handle NaN.
    X is subsetted to the same valid assets as Y.
    """
    mask = _valid_mask(Y)
    X_v, Y_v = X[:, mask], Y[:, mask]
    cov_X = _nancovariance(X_v)
    cov_Y = _nancovariance(Y_v)
    return float(np.linalg.norm(cov_X - cov_Y, 'fro') / np.linalg.norm(cov_Y, 'fro'))


@registry.register('ES')
def compute_es(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """ES per asset (5th percentile tail mean), mean±std across assets."""
    mask = _valid_mask(Y)
    es_vals = _compute_es_per_asset(X[:, mask], alpha)
    finite = es_vals[np.isfinite(es_vals)]
    return {'mean': float(np.mean(finite)), 'std': float(np.std(finite))}


@registry.register('Mean')
def compute_mean(X: np.ndarray, Y: np.ndarray) -> dict:
    """Mean per asset, mean±std across assets."""
    mask = _valid_mask(Y)
    vals = np.nanmean(X[:, mask], axis=0)
    return {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}


@registry.register('Std')
def compute_std(X: np.ndarray, Y: np.ndarray) -> dict:
    """Std per asset, mean±std across assets."""
    mask = _valid_mask(Y)
    vals = np.nanstd(X[:, mask], axis=0)
    return {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}


@registry.register('Skew')
def compute_skew(X: np.ndarray, Y: np.ndarray) -> dict:
    """Skew per asset, mean±std across assets."""
    mask = _valid_mask(Y)
    vals = skew(X[:, mask], axis=0, nan_policy='omit')
    return {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}


@registry.register('Kurt')
def compute_kurt(X: np.ndarray, Y: np.ndarray) -> dict:
    """Kurt per asset, mean±std across assets."""
    mask = _valid_mask(Y)
    vals = kurtosis(X[:, mask], axis=0, nan_policy='omit')
    return {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}
