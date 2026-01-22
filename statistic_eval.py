import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from typing import Dict, Optional


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Compute MMD between two distributions. Lower is better."""
    X = X.reshape(X.shape[0], -1).astype(np.float64)
    Y = Y.reshape(Y.shape[0], -1).astype(np.float64)

    if sigma is None:
        XY = np.vstack([X[:1000], Y[:1000]])
        dists = np.sqrt(((XY[:, None] - XY[None, :]) ** 2).sum(-1))
        sigma = np.median(dists[dists > 0]) + 1e-10

    def rbf(A, B):
        AA = (A ** 2).sum(1, keepdims=True)
        BB = (B ** 2).sum(1, keepdims=True)
        D = AA + BB.T - 2 * A @ B.T
        return np.exp(-D / (2 * sigma ** 2))

    K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    n, m = len(X), len(Y)
    mmd = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) + \
          (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) - \
          2 * K_XY.sum() / (n * m)
    return max(0, mmd)


def compute_cov_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices."""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
    if cov_X.ndim == 0:
        cov_X, cov_Y = np.array([[cov_X]]), np.array([[cov_Y]])
    return np.linalg.norm(cov_X - cov_Y, 'fro') / (np.linalg.norm(cov_Y, 'fro') + 1e-10)


def compute_marginal_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Compute marginal distribution statistics."""
    X_flat, Y_flat = X.flatten(), Y.flatten()
    return {
        'mean_X': np.mean(X_flat), 'mean_Y': np.mean(Y_flat),
        'std_X': np.std(X_flat), 'std_Y': np.std(Y_flat),
        'skew_X': skew(X_flat), 'skew_Y': skew(Y_flat),
        'kurtosis_X': kurtosis(X_flat), 'kurtosis_Y': kurtosis(Y_flat),
    }


def compute_acf(data: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Compute average ACF across samples and assets. Input: [N, assets, time]"""
    N, A, T = data.shape
    acf_list = []
    for lag in range(1, min(max_lag + 1, T)):
        corrs = []
        for i in range(N):
            for a in range(A):
                ts = data[i, a, :]
                if np.std(ts) > 1e-10:
                    corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)
        acf_list.append(np.mean(corrs) if corrs else 0.0)
    return np.array(acf_list)


def compute_acf_metric(X: np.ndarray, Y: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
    """Compute ACF comparison. Returns mean ACF and ratio (Gen/Train)."""
    acf_X = compute_acf(X, max_lag)
    acf_Y = compute_acf(Y, max_lag)
    mean_acf_X = np.mean(np.abs(acf_X))
    mean_acf_Y = np.mean(np.abs(acf_Y))
    return {
        'acf_X': mean_acf_X,
        'acf_Y': mean_acf_Y,
        'acf_ratio': mean_acf_X / (mean_acf_Y + 1e-10),
    }


def compute_tail_metric(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Compute left tail Expected Shortfall (ES/CVaR)."""
    X_flat, Y_flat = X.flatten(), Y.flatten()
    var_X = np.percentile(X_flat, alpha * 100)
    var_Y = np.percentile(Y_flat, alpha * 100)
    es_X = np.mean(X_flat[X_flat <= var_X])
    es_Y = np.mean(Y_flat[Y_flat <= var_Y])
    return {
        'es_X': es_X, 'es_Y': es_Y,
        'es_error': abs(es_X - es_Y) / (abs(es_Y) + 1e-10),
    }


def plot_comparison(X: np.ndarray, Y: np.ndarray, save_path: Optional[str] = None, show: bool = False):
    """Plot distribution comparison with tail focus."""
    X_flat, Y_flat = X.flatten(), Y.flatten()
    _, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150)

    axes[0].hist(X_flat, bins=50, alpha=0.6, label='Generated', density=True)
    axes[0].hist(Y_flat, bins=50, alpha=0.6, label='Training', density=True)
    axes[0].set_title('Distribution')
    axes[0].legend()

    q = np.linspace(0, 100, 100)
    axes[1].scatter(np.percentile(Y_flat, q), np.percentile(X_flat, q), s=10, alpha=0.6)
    lim = [min(X_flat.min(), Y_flat.min()), max(X_flat.max(), Y_flat.max())]
    axes[1].plot(lim, lim, 'r--')
    axes[1].set_xlabel('Training')
    axes[1].set_ylabel('Generated')
    axes[1].set_title('Q-Q Plot')

    for data, label in [(X_flat, 'Generated'), (Y_flat, 'Training')]:
        left_tail = -data[data < 0]
        if len(left_tail) > 0:
            sorted_tail = np.sort(left_tail)[::-1]
            ccdf = np.arange(1, len(sorted_tail) + 1) / len(sorted_tail)
            axes[2].loglog(sorted_tail, ccdf, 'o', alpha=0.15, markersize=1, label=label)
    axes[2].set_xlabel('|Negative Value|')
    axes[2].set_ylabel('P(X < -x)')
    axes[2].set_title('Left Tail CCDF')
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def evaluate(generated_path: str, training_path: str, tail_alpha: float = 0.05,
             plot: bool = True, save_path: Optional[str] = None) -> Dict:
    """
    Evaluate generated samples against training data.
    Input must be 3D: [N, assets, time]
    """
    X, Y = np.load(generated_path), np.load(training_path)

    # Check 3D
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Input must be 3D [N, assets, time]. Got X: {X.ndim}D, Y: {Y.ndim}D")

    print(f"Generated: {X.shape}, Training: {Y.shape}")
    print("=" * 50)

    results = {}

    mmd = compute_mmd(X, Y)
    results['mmd'] = mmd
    print(f"MMD: {mmd:.6f}")

    cov_err = compute_cov_error(X, Y)
    results['cov_error'] = cov_err
    print(f"Cov Error: {cov_err:.6f}")

    tail = compute_tail_metric(X, Y, tail_alpha)
    results.update(tail)
    print(f"ES ({int(tail_alpha*100)}%) - Gen: {tail['es_X']:.4f}, Train: {tail['es_Y']:.4f}, Error: {tail['es_error']:.4f}")

    acf = compute_acf_metric(X, Y)
    results.update(acf)
    print(f"ACF - Gen: {acf['acf_X']:.4f}, Train: {acf['acf_Y']:.4f}, Ratio: {acf['acf_ratio']:.4f}")

    marginal = compute_marginal_stats(X, Y)
    results.update(marginal)
    print(f"Mean  - Gen: {marginal['mean_X']:.4f}, Train: {marginal['mean_Y']:.4f}")
    print(f"Std   - Gen: {marginal['std_X']:.4f}, Train: {marginal['std_Y']:.4f}")
    print(f"Skew  - Gen: {marginal['skew_X']:.4f}, Train: {marginal['skew_Y']:.4f}")
    print(f"Kurt  - Gen: {marginal['kurtosis_X']:.4f}, Train: {marginal['kurtosis_Y']:.4f}")

    if plot:
        plot_comparison(X, Y, save_path)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True)
    parser.add_argument('--training', type=str, required=True)
    parser.add_argument('--tail_alpha', type=float, default=0.05)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.generated, args.training, args.tail_alpha, save_path=args.save_path)
