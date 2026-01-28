import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Compute MMD between two distributions. Input: [n, assets], [m, assets]"""
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
    mmd = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) + \
          (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) - \
          2 * K_XY.sum() / (n * m)
    return max(0, mmd)


def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices."""
    cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
    if cov_X.ndim == 0:
        cov_X, cov_Y = np.array([[cov_X]]), np.array([[cov_Y]])
    return np.linalg.norm(cov_X - cov_Y, 'fro') / (np.linalg.norm(cov_Y, 'fro') + 1e-10)


def compute_es(data: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Compute ES per asset, return mean and std across assets."""
    d = data.shape[1]
    es_per_asset = np.zeros(d)
    for i in range(d):
        col = data[:, i]
        var = np.percentile(col, alpha * 100)
        es_per_asset[i] = np.mean(col[col <= var])
    return {'mean': np.mean(es_per_asset), 'std': np.std(es_per_asset)}




def compute_stats(data: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute Mean, Std, Skew, Kurt per asset, return mean and std across assets."""
    per_asset = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'skew': skew(data, axis=0),
        'kurt': kurtosis(data, axis=0),
    }
    return {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in per_asset.items()}


def evaluate(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Evaluate generated (X) vs training (Y) data.
    Input: X [n, assets], Y [m, assets]
    Returns dict with all metrics for both X and comparison metrics.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Input must be 2D [samples, assets]. Got X: {X.ndim}D, Y: {Y.ndim}D")

    stats_X, stats_Y = compute_stats(X), compute_stats(Y)
    es_X, es_Y = compute_es(X, alpha), compute_es(Y, alpha)

    results = {
        'mmd': compute_mmd(X, Y),
        'cov_error': compute_cov(X, Y),
        'es_X': es_X, 'es_Y': es_Y,
        'stats_X': stats_X, 'stats_Y': stats_Y,
    }

    print(f"Generated: {X.shape}, Training: {Y.shape}")
    print("=" * 60)
    print(f"{'Metric':<12} {'Gen (mean±std)':>22} {'Train (mean±std)':>22}")
    print("-" * 60)
    print(f"{'MMD':<12} {results['mmd']:>22.6f} {'-':>22}")
    print(f"{'Cov Error':<12} {results['cov_error']:>22.6f} {'-':>22}")
    print(f"{'ES (5%)':<12} {es_X['mean']:>10.4f}±{es_X['std']:<9.4f} {es_Y['mean']:>10.4f}±{es_Y['std']:<9.4f}")
    print(f"{'Mean':<12} {stats_X['mean']['mean']:>10.4f}±{stats_X['mean']['std']:<9.4f} {stats_Y['mean']['mean']:>10.4f}±{stats_Y['mean']['std']:<9.4f}")
    print(f"{'Std':<12} {stats_X['std']['mean']:>10.4f}±{stats_X['std']['std']:<9.4f} {stats_Y['std']['mean']:>10.4f}±{stats_Y['std']['std']:<9.4f}")
    print(f"{'Skew':<12} {stats_X['skew']['mean']:>10.4f}±{stats_X['skew']['std']:<9.4f} {stats_Y['skew']['mean']:>10.4f}±{stats_Y['skew']['std']:<9.4f}")
    print(f"{'Kurt':<12} {stats_X['kurt']['mean']:>10.4f}±{stats_X['kurt']['std']:<9.4f} {stats_Y['kurt']['mean']:>10.4f}±{stats_Y['kurt']['std']:<9.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True)
    parser.add_argument('--training', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    X, Y = np.load(args.generated), np.load(args.training)
    evaluate(X, Y, args.alpha)
