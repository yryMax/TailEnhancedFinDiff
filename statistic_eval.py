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


@MetricRegistry.register('ES', compare=False)
def compute_es(data: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Compute ES per asset, return mean and std across assets."""
    d = data.shape[1]
    es_per_asset = np.zeros(d)
    for i in range(d):
        col = data[:, i]
        var = np.percentile(col, alpha * 100)
        es_per_asset[i] = np.mean(col[col <= var])
    return {'mean': np.mean(es_per_asset), 'std': np.std(es_per_asset)}


@MetricRegistry.register('Mean', compare=False)
def compute_mean(data: np.ndarray) -> Dict[str, float]:
    """Compute mean per asset."""
    per_asset = np.mean(data, axis=0)
    return {'mean': np.mean(per_asset), 'std': np.std(per_asset)}


@MetricRegistry.register('Std', compare=False)
def compute_std(data: np.ndarray) -> Dict[str, float]:
    """Compute std per asset."""
    per_asset = np.std(data, axis=0)
    return {'mean': np.mean(per_asset), 'std': np.std(per_asset)}


@MetricRegistry.register('Skew', compare=False)
def compute_skew(data: np.ndarray) -> Dict[str, float]:
    """Compute skewness per asset."""
    per_asset = skew(data, axis=0)
    return {'mean': np.mean(per_asset), 'std': np.std(per_asset)}


@MetricRegistry.register('Kurt', compare=False)
def compute_kurt(data: np.ndarray) -> Dict[str, float]:
    """Compute kurtosis per asset."""
    per_asset = kurtosis(data, axis=0)
    return {'mean': np.mean(per_asset), 'std': np.std(per_asset)}


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

    def _format_value(self, value: Any) -> str:
        if isinstance(value, dict) and 'mean' in value:
            return f"{value['mean']:.2f}±{value['std']:.2f}"
        elif isinstance(value, (int, float, np.floating)):
            return f"{float(value):.2f}"
        return str(value)

    def to_console(self):
        """Print results to console."""
        print(f"Training: {self.training_shape}")
        for r in self.results:
            print(f"{r.name}: {r.shape}")
        print("=" * 70)

        # Header
        header = ["Metric", "Training"] + [r.name for r in self.results]
        print(f"{header[0]:<10}", end="")
        for h in header[1:]:
            print(f"{h:>20}", end="")
        print()
        print("-" * 70)

        # Get all metric names
        all_metrics = list(MetricRegistry.get_all().keys())
        for metric_name in all_metrics:
            print(f"{metric_name:<10}", end="")

            # Training value
            if metric_name in self.training_metrics:
                print(f"{self._format_value(self.training_metrics[metric_name]):>20}", end="")
            else:
                print(f"{'-':>20}", end="")

            # Generated values
            for r in self.results:
                if metric_name in r.metrics:
                    print(f"{self._format_value(r.metrics[metric_name]):>20}", end="")
                else:
                    print(f"{'-':>20}", end="")
            print()

    def to_markdown(self) -> str:
        """Generate markdown table."""
        lines = []

        # Header
        header = ["Metric", "Training"] + [r.name for r in self.results]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Rows
        all_metrics = list(MetricRegistry.get_all().keys())
        for metric_name in all_metrics:
            row = [metric_name]

            # Training value
            if metric_name in self.training_metrics:
                row.append(self._format_value(self.training_metrics[metric_name]))
            else:
                row.append("-")

            # Generated values
            for r in self.results:
                if metric_name in r.metrics:
                    row.append(self._format_value(r.metrics[metric_name]))
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
        self._compute_training_metrics()

    def _compute_training_metrics(self):
        """Compute metrics for training data (non-compare metrics only)."""
        for name, info in MetricRegistry.get_all().items():
            if not info['compare']:
                self.results.training_metrics[name] = info['func'](self.Y)

    def add(self, X: np.ndarray, name: str) -> 'Evaluator':
        """Add a generated dataset and compute metrics."""
        if X.ndim != 2:
            raise ValueError(f"Generated data must be 2D [samples, assets]. Got {X.ndim}D")
        if X.shape[1] != self.Y.shape[1]:
            raise ValueError(f"Asset dimension mismatch: X has {X.shape[1]}, Y has {self.Y.shape[1]}")

        result = EvalResult(name=name, shape=X.shape)

        for metric_name, info in MetricRegistry.get_all().items():
            if info['compare']:
                result.metrics[metric_name] = info['func'](X, self.Y)
            else:
                result.metrics[metric_name] = info['func'](X)

        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        """Return the result collection."""
        return self.results


# ============== Main ==============

if __name__ == "__main__":
    training = np.load("data/stocknet_81.npy")
    factor_dm = np.load("samples/factorDM128.npy")
    bootstrap = np.load("samples/SB128.npy")

    results = (Evaluator(training)
        .add(factor_dm, "FactorDM")
        .add(bootstrap, "Bootstrap")
        .report())

    results.to_console()
    print("\n" + "=" * 70 + "\n")
    print(results.to_markdown())
