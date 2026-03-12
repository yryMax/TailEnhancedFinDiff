import numpy as np
from metrics.statistic import registry
from result import EvalResult, EvalResultCollection


class Evaluator:
    """Evaluator for distribution similarity and marginal statistics."""

    def __init__(self, Y: np.ndarray, metrics: list = None):
        if Y.ndim != 2:
            raise ValueError(f"Training data must be 2D [samples, assets]. Got {Y.ndim}D")
        self.Y = Y
        self.metrics = metrics
        self.results = EvalResultCollection(training_shape=Y.shape, training_data=Y)

    def add(self, X: np.ndarray, name: str) -> 'Evaluator':
        if X.ndim != 2:
            raise ValueError(f"Generated data must be 2D [samples, assets]. Got {X.ndim}D")
        if X.shape[1] != self.Y.shape[1]:
            raise ValueError(f"Asset dimension mismatch: X has {X.shape[1]}, Y has {self.Y.shape[1]}")

        all_metrics = registry.get_all()
        selected = {k: v for k, v in all_metrics.items() if self.metrics is None or k in self.metrics}

        result = EvalResult(name=name, data=X)
        for metric_name, func in selected.items():
            result.metrics[metric_name] = func(X, self.Y)

        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        return self.results


if __name__ == "__main__":
    training = np.load("data/ROBECO.npy")
    psm = np.load("data/PSM2048.npy")
    bootstrap = np.load("data/SB2048.npy")

    results = (Evaluator(training)
        .add(psm, "parametric simulation model")
        .add(bootstrap, "Stationary Bootstrap")
        .report())

    print(results.to_console())
    #results.plot_distribution(save_path="assets/distribution.png")
