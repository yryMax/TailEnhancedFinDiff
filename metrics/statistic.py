import numpy as np
from scipy.stats import skew, kurtosis
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


def _compute_es_per_asset(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Helper: compute ES for each asset."""
    return np.array([es(data[:, i], alpha) for i in range(data.shape[1])])


# ============== Registered Metrics ==============

@registry.register('MMD')
def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Compute MMD between two distributions."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    if sigma is None:
        XY = np.vstack([X[:1000], Y[:1000]])
        dists = np.sqrt(((XY[:, None] - XY[None, :]) ** 2).sum(-1))
        sigma = np.median(dists[dists > 0]) + 1e-10

    def rbf(A, B):
        AA, BB = (A ** 2).sum(1, keepdims=True), (B ** 2).sum(1, keepdims=True)
        return np.exp(-(AA + BB.T - 2 * A @ B.T) / (2 * sigma ** 2))

    K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    n, m = len(X), len(Y)
    mmd_sq = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) + \
             (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) - \
             2 * K_XY.sum() / (n * m)
    return np.sqrt(max(0, mmd_sq))


@registry.register('Cov')
def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices."""
    cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
    if cov_X.ndim == 0:
        cov_X, cov_Y = np.array([[cov_X]]), np.array([[cov_Y]])
    return np.linalg.norm(cov_X - cov_Y, 'fro') / (np.linalg.norm(cov_Y, 'fro') + 1e-10)


@registry.register('ES')
def compute_es(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Relative ES error: (X - Y) / Y per asset."""
    es_X, es_Y = _compute_es_per_asset(X, alpha), _compute_es_per_asset(Y, alpha)
    relative = (es_X - es_Y) / (np.abs(es_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@registry.register('Mean')
def compute_mean(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative mean error: (X - Y) / Y per asset."""
    mean_X, mean_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    relative = (mean_X - mean_Y) / (np.abs(mean_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@registry.register('Std')
def compute_std(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative std error: (X - Y) / Y per asset."""
    std_X, std_Y = np.std(X, axis=0), np.std(Y, axis=0)
    relative = (std_X - std_Y) / (np.abs(std_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@registry.register('Skew')
def compute_skew(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative skew error: (X - Y) / Y per asset."""
    skew_X, skew_Y = skew(X, axis=0), skew(Y, axis=0)
    relative = (skew_X - skew_Y) / (np.abs(skew_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@registry.register('Kurt')
def compute_kurt(X: np.ndarray, Y: np.ndarray) -> dict:
    """Relative kurt error: (X - Y) / Y per asset."""
    kurt_X, kurt_Y = kurtosis(X, axis=0), kurtosis(Y, axis=0)
    relative = (kurt_X - kurt_Y) / (np.abs(kurt_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}
