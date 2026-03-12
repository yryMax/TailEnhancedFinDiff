import numpy as np
from scipy.stats import skew, kurtosis, wasserstein_distance
from .registry import MetricRegistry

registry = MetricRegistry()


# ============== Helpers ==============

def _pool(X: np.ndarray) -> np.ndarray:
    """[M, N, D] → [M*D, N]: pool scenarios and days, keep asset axis."""
    M, N, D = X.shape
    return X.transpose(0, 2, 1).reshape(M * D, N)


def _cumsum(X: np.ndarray) -> np.ndarray:
    """[M, N, D] → [M, N]: cumulative return over D days per scenario."""
    return X.sum(axis=2)


# ============== Registered Metrics ==============

@registry.register('W1')
def compute_w1(X: np.ndarray, Y: np.ndarray) -> float:
    """1D Wasserstein-1 on completely flattened data (cross-sectional, no temporal aggregation)."""
    return float(wasserstein_distance(X.flatten(), Y.flatten()))


@registry.register('Cov')
def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between correlation matrices.
    Pool [M, N, D] → [M*D, N] (following assessment_ref.py) to get full-rank estimates.
    """
    X_pool, Y_pool = _pool(X), _pool(Y)
    corr_X = np.corrcoef(X_pool.T)
    corr_Y = np.corrcoef(Y_pool.T)
    return float(np.linalg.norm(corr_X - corr_Y, 'fro') / np.linalg.norm(corr_Y, 'fro'))


@registry.register('ES')
def compute_es(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Relative D-day cumulative ES error per asset.
    Sums returns over D days → [M, N], then computes ES on the tail.
    """
    cum_X, cum_Y = _cumsum(X), _cumsum(Y)
    N = X.shape[1]

    def _es(data, n):
        col = data[:, n]
        threshold = np.percentile(col, alpha * 100)
        return float(np.mean(col[col <= threshold]))

    es_X = np.array([_es(cum_X, n) for n in range(N)])
    es_Y = np.array([_es(cum_Y, n) for n in range(N)])
    relative = (es_X - es_Y) / (np.abs(es_Y))
    return {'mean': float(np.mean(relative)), 'std': float(np.std(relative))}


@registry.register('Mean')
def compute_mean(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative mean error per asset. Pool [M, N, D] → [M*D, N]."""
    X_pool, Y_pool = _pool(X), _pool(Y)
    mean_X, mean_Y = np.mean(X_pool, axis=0), np.mean(Y_pool, axis=0)
    relative = (mean_X - mean_Y) / (np.abs(mean_Y))
    return {'mean': float(np.mean(relative)), 'std': float(np.std(relative))}


@registry.register('Std')
def compute_std(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative std error per asset. Pool [M, N, D] → [M*D, N]."""
    X_pool, Y_pool = _pool(X), _pool(Y)
    std_X, std_Y = np.std(X_pool, axis=0), np.std(Y_pool, axis=0)
    relative = (std_X - std_Y) / (np.abs(std_Y))
    return {'mean': float(np.mean(relative)), 'std': float(np.std(relative))}


@registry.register('Skew')
def compute_skew(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative skew error per asset. Pool [M, N, D] → [M*D, N]."""
    X_pool, Y_pool = _pool(X), _pool(Y)
    skew_X, skew_Y = skew(X_pool, axis=0), skew(Y_pool, axis=0)
    relative = (skew_X - skew_Y) / (np.abs(skew_Y))
    return {'mean': float(np.mean(relative)), 'std': float(np.std(relative))}


@registry.register('Kurt')
def compute_kurt(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative kurt error per asset. Pool [M, N, D] → [M*D, N]."""
    X_pool, Y_pool = _pool(X), _pool(Y)
    kurt_X, kurt_Y = kurtosis(X_pool, axis=0), kurtosis(Y_pool, axis=0)
    relative = (kurt_X - kurt_Y) / (np.abs(kurt_Y))
    return {'mean': float(np.mean(relative)), 'std': float(np.std(relative))}


# ============== Temporal Stylized Fact Helpers ==============

def _acf_mean(X: np.ndarray, nlags: int) -> float:
    """Mean ACF of X [M, N, D] across M scenarios, N assets, lags 1..nlags."""
    X_c = X - X.mean(axis=2, keepdims=True)
    var = (X_c ** 2).mean(axis=2)  # [M, N]
    acf_vals = []
    for k in range(1, nlags + 1):
        cov_k = (X_c[:, :, k:] * X_c[:, :, :-k]).mean(axis=2)  # [M, N]
        acf_vals.append((cov_k / (var)).mean())
    return float(np.mean(acf_vals))


def _leverage_mean(X: np.ndarray, k_max: int) -> float:
    """Mean leverage effect of X [M, N, D] across M, N, lags 1..k_max.
    L_k = (E[r_t * r_{t+k}^2] - E[r_t]*E[r_t^2]) / E[r_t^2]^2
    """
    mean_r  = X.mean(axis=2)       # [M, N]
    mean_r2 = (X ** 2).mean(axis=2)  # [M, N]
    denom   = mean_r2 ** 2 + 1e-8
    lev_vals = []
    for k in range(1, min(k_max + 1, X.shape[2])):
        cross = (X[:, :, :-k] * X[:, :, k:] ** 2).mean(axis=2)  # [M, N]
        lev_vals.append(((cross - mean_r * mean_r2) / denom).mean())
    return float(np.mean(lev_vals))


def _coarse_fine_mean(X: np.ndarray, tau: int, k_max: int) -> float:
    """Mean coarse-fine correlation of X [M, N, D] across M, N, lags 1..k_max.
    Coarse: |rolling_sum(r, tau)|, Fine: rolling_sum(|r|, tau).
    Corr(coarse[t+k], fine[t]) averaged over lags.
    """
    D = X.shape[2]
    if D < tau + k_max:
        return 0.0
    coarse = np.stack([np.abs(X[:, :, t:t + tau].sum(axis=2)) for t in range(D - tau + 1)], axis=2)
    fine   = np.stack([np.abs(X[:, :, t:t + tau]).sum(axis=2)  for t in range(D - tau + 1)], axis=2)
    T_cf = coarse.shape[2]
    corrs = []
    for k in range(1, k_max + 1):
        if T_cf <= k:
            break
        A, B = coarse[:, :, k:], fine[:, :, :-k]
        A_c = A - A.mean(axis=2, keepdims=True)
        B_c = B - B.mean(axis=2, keepdims=True)
        num   = (A_c * B_c).mean(axis=2)
        denom = np.sqrt((A_c ** 2).mean(axis=2) * (B_c ** 2).mean(axis=2)) + 1e-8
        corrs.append((num / denom).mean())
    return float(np.mean(corrs)) if corrs else 0.0


# ============== Temporal Stylized Facts (difference vs reference) ==============

def _rel(gen: float, ref: float) -> float:
    return (gen - ref) / (abs(ref))


@registry.register('LinUnpred')
def compute_lin_unpred(X: np.ndarray, Y: np.ndarray, nlags: int = 10) -> float:
    """Linear unpredictability: mean ACF of returns (lags 1-10).
    Reported as (generated - reference) / |reference|. Closer to 0 is better.
    """
    return _rel(_acf_mean(X, nlags), _acf_mean(Y, nlags))


@registry.register('VolClust')
def compute_vol_clust(X: np.ndarray, Y: np.ndarray, nlags: int = 10) -> float:
    """Volatility clustering: mean ACF of |returns| (lags 1-10).
    Reported as (generated - reference) / |reference|. Closer to 0 is better.
    """
    return _rel(_acf_mean(np.abs(X), nlags), _acf_mean(np.abs(Y), nlags))


@registry.register('CoarseFine')
def compute_coarse_fine(X: np.ndarray, Y: np.ndarray, tau: int = 5, k_max: int = 10) -> float:
    """Coarse-fine volatility: corr(coarse_vol[t+k], fine_vol[t]), tau=5, lags 1-10.
    Reported as (generated - reference) / |reference|. Closer to 0 is better.
    """
    return _rel(_coarse_fine_mean(X, tau, k_max), _coarse_fine_mean(Y, tau, k_max))


@registry.register('Leverage')
def compute_leverage(X: np.ndarray, Y: np.ndarray, k_max: int = 10) -> float:
    """Leverage effect: mean (E[r_t * r_{t+k}^2] - E[r_t]*E[r_t^2]) / E[r_t^2]^2, lags 1-10.
    Reported as (generated - reference) / |reference|. Closer to 0 is better.
    """
    return _rel(_leverage_mean(X, k_max), _leverage_mean(Y, k_max))
