import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field


class MetricRegistry:
    """Registry for evaluation metrics."""
    _metrics: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, compare: bool = True):
        """
        Decorator to register a metric.
        compare=True: metric needs both X and Y (e.g., MMD)
        compare=False: metric computed on single dataset (e.g., mean)
        """
        def decorator(func: Callable):
            cls._metrics[name] = {'func': func, 'compare': compare}
            return func
        return decorator

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        return cls._metrics


# ============== Register Metrics ==============

@MetricRegistry.register('MMD', compare=True)
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


@MetricRegistry.register('Cov', compare=True)
def compute_cov(X: np.ndarray, Y: np.ndarray) -> float:
    """Relative Frobenius norm error between covariance matrices."""
    cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
    if cov_X.ndim == 0:
        cov_X, cov_Y = np.array([[cov_X]]), np.array([[cov_Y]])
    return np.linalg.norm(cov_X - cov_Y, 'fro') / (np.linalg.norm(cov_Y, 'fro') + 1e-10)


def _compute_es_per_asset(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Helper: compute ES for each asset."""
    d = data.shape[1]
    es = np.zeros(d)
    for i in range(d):
        col = data[:, i]
        var = np.percentile(col, alpha * 100)
        es[i] = np.mean(col[col <= var])
    return es


@MetricRegistry.register('ES', compare=True)
def compute_es(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Relative ES error: (X - Y) / Y per asset."""
    es_X, es_Y = _compute_es_per_asset(X, alpha), _compute_es_per_asset(Y, alpha)
    relative = (es_X - es_Y) / (np.abs(es_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@MetricRegistry.register('Mean', compare=True)
def compute_mean(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Relative mean error: (X - Y) / Y per asset."""
    mean_X, mean_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    relative = (mean_X - mean_Y) / (np.abs(mean_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@MetricRegistry.register('Std', compare=True)
def compute_std(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Relative std error: (X - Y) / Y per asset."""
    std_X, std_Y = np.std(X, axis=0), np.std(Y, axis=0)
    relative = (std_X - std_Y) / (np.abs(std_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@MetricRegistry.register('Skew', compare=True)
def compute_skew(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Relative skew error: (X - Y) / Y per asset."""
    skew_X, skew_Y = skew(X, axis=0), skew(Y, axis=0)
    relative = (skew_X - skew_Y) / (np.abs(skew_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


@MetricRegistry.register('Kurt', compare=True)
def compute_kurt(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Relative kurt error: (X - Y) / Y per asset."""
    kurt_X, kurt_Y = kurtosis(X, axis=0), kurtosis(Y, axis=0)
    relative = (kurt_X - kurt_Y) / (np.abs(kurt_Y) + 1e-10)
    return {'mean': np.mean(relative), 'std': np.std(relative)}


# ============== Result Classes ==============

@dataclass
class EvalResult:
    """Single evaluation result for one generated dataset."""
    name: str
    shape: tuple
    metrics: Dict[str, Any] = field(default_factory=dict)

    def _format_value(self, key: str, value: Any) -> str:
        """Format metric value for display."""
        if isinstance(value, dict) and 'mean' in value:
            return f"{value['mean']:.2f}±{value['std']:.2f}"
        elif isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def to_dict(self) -> Dict[str, str]:
        """Convert metrics to formatted strings."""
        return {k: self._format_value(k, v) for k, v in self.metrics.items()}


@dataclass
class EvalResultCollection:
    """Collection of evaluation results for multiple generated datasets."""
    training_shape: tuple = None
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)

    def append(self, result: EvalResult):
        self.results.append(result)

    def _format_value(self, value: Any, mean_prec: int = 2, std_prec: int = 2) -> str:
        if isinstance(value, dict) and 'mean' in value:
            return f"{value['mean']:.{mean_prec}f}±{value['std']:.{std_prec}f}"
        elif isinstance(value, (int, float, np.floating)):
            return f"{float(value):.{mean_prec}f}"
        return str(value)

    def to_console(self):
        """Print results to console."""
        print(f"Training: {self.training_shape}")
        for r in self.results:
            print(f"{r.name}: {r.shape}")
        print("=" * 70)

        # Header
        header = ["Metric"] + [r.name for r in self.results]
        print(f"{header[0]:<10}", end="")
        for h in header[1:]:
            print(f"{h:>20}", end="")
        print()
        print("-" * 70)

        # Get all metric names
        all_metrics = list(MetricRegistry.get_all().keys())
        for metric_name in all_metrics:
            print(f"{metric_name:<10}", end="")
            for r in self.results:
                if metric_name in r.metrics:
                    print(f"{self._format_value(r.metrics[metric_name]):>20}", end="")
                else:
                    print(f"{'-':>20}", end="")
            print()

    def to_markdown(self) -> str:
        """Generate markdown table (rows=methods, cols=metrics)."""
        lines = []
        all_metrics = list(MetricRegistry.get_all().keys())

        # Header: Method | MMD | Cov | ES | Mean | Std | Skew | Kurt
        header = ["Method"] + all_metrics
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Rows for each generated dataset
        for r in self.results:
            row = [r.name]
            for metric_name in all_metrics:
                if metric_name in r.metrics:
                    row.append(self._format_value(r.metrics[metric_name], mean_prec=4, std_prec=2))
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


# ============== Evaluator ==============

class Evaluator:
    """Evaluator with one training set and multiple generated sets."""

    def __init__(self, Y: np.ndarray):
        """Initialize with training data."""
        if Y.ndim != 2:
            raise ValueError(f"Training data must be 2D [samples, assets]. Got {Y.ndim}D")
        self.Y = Y
        self.results = EvalResultCollection(training_shape=Y.shape)

    def add(self, X: np.ndarray, name: str) -> 'Evaluator':
        """Add a generated dataset and compute metrics (all relative to training)."""
        if X.ndim != 2:
            raise ValueError(f"Generated data must be 2D [samples, assets]. Got {X.ndim}D")
        if X.shape[1] != self.Y.shape[1]:
            raise ValueError(f"Asset dimension mismatch: X has {X.shape[1]}, Y has {self.Y.shape[1]}")

        result = EvalResult(name=name, shape=X.shape)
        for metric_name, info in MetricRegistry.get_all().items():
            result.metrics[metric_name] = info['func'](X, self.Y)

        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        """Return the result collection."""
        return self.results


# ============== Main ==============

if __name__ == "__main__":
    training = np.load("data/stocknet_81.npy")
    factor_dm = np.load("samples/factorDM2048.npy")
    bootstrap = np.load("samples/SB2048.npy")

    results = (Evaluator(training)
        .add(factor_dm, "FactorDM")
        .add(bootstrap, "Stationary Bootstrap")
        .report())

    results.to_console()
    print("\n" + "=" * 70 + "\n")
    print(results.to_markdown())
