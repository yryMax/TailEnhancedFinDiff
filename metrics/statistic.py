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


def _valid_mask(Y: np.ndarray, min_samples: int = 30) -> np.ndarray:
    """Returns bool mask of assets with >= min_samples valid (non-NaN) rows."""
    return np.sum(~np.isnan(Y), axis=0) >= min_samples


def _agg(relative: np.ndarray) -> dict:
    """Aggregate per-asset relative errors, ignoring NaN and inf."""
    finite = relative[np.isfinite(relative)]
    return {'mean': float(np.mean(finite)), 'std': float(np.std(finite))}


def _compute_es_per_asset(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Helper: compute ES for each asset, ignoring NaN."""
    def _es_nan(col):
        col = col[~np.isnan(col)]
        return es(col, alpha) if len(col) > 0 else np.nan
    return np.array([_es_nan(data[:, i]) for i in range(data.shape[1])])


# ============== Registered Metrics ==============

@registry.register('W1')
def compute_wasserstein1(X: np.ndarray, Y: np.ndarray, n_projections: int = 500, seed: int = 42) -> float:
    """Sliced Wasserstein-1 distance between two multivariate distributions."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    X_proj = X @ directions.T  # [n, n_projections]
    Y_proj = Y @ directions.T  # [m, n_projections]
    return float(np.mean([wasserstein_distance(X_proj[:, i], Y_proj[:, i]) for i in range(n_projections)]))


@registry.register('Cov')
def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices."""
    cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
    if cov_X.ndim == 0:
        cov_X, cov_Y = np.array([[cov_X]]), np.array([[cov_Y]])
    return np.linalg.norm(cov_X - cov_Y, 'fro') / (np.linalg.norm(cov_Y, 'fro'))


@registry.register('ES')
def compute_es(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Relative ES error: (X - Y) / Y per asset."""
    mask = _valid_mask(Y)
    es_X, es_Y = _compute_es_per_asset(X[:, mask], alpha), _compute_es_per_asset(Y[:, mask], alpha)
    return _agg((es_X - es_Y) / np.abs(es_Y))


@registry.register('Mean')
def compute_mean(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative mean error: (X - Y) / Y per asset."""
    mask = _valid_mask(Y)
    X, Y = X[:, mask], Y[:, mask]
    return _agg((np.nanmean(X, axis=0) - np.nanmean(Y, axis=0)) / np.abs(np.nanmean(Y, axis=0)))


@registry.register('Std')
def compute_std(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative std error: (X - Y) / Y per asset."""
    mask = _valid_mask(Y)
    X, Y = X[:, mask], Y[:, mask]
    std_X, std_Y = np.nanstd(X, axis=0), np.nanstd(Y, axis=0)
    return _agg((std_X - std_Y) / std_Y)


@registry.register('Skew')
def compute_skew(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative skew error: (X - Y) / Y per asset."""
    mask = _valid_mask(Y)
    X, Y = X[:, mask], Y[:, mask]
    skew_X, skew_Y = skew(X, axis=0, nan_policy='omit'), skew(Y, axis=0, nan_policy='omit')
    return _agg((skew_X - skew_Y) / np.abs(skew_Y))


@registry.register('Kurt')
def compute_kurt(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative kurt error: (X - Y) / Y per asset."""
    mask = _valid_mask(Y)
    X, Y = X[:, mask], Y[:, mask]
    kurt_X, kurt_Y = kurtosis(X, axis=0, nan_policy='omit'), kurtosis(Y, axis=0, nan_policy='omit')
    return _agg((kurt_X - kurt_Y) / np.abs(kurt_Y))
