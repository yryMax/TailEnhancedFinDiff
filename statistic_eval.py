import numpy as np
from metrics.statistic import registry
from result import EvalResult, EvalResultCollection


class Evaluator:
    """Evaluator for distribution similarity and marginal statistics."""

    def __init__(self, Y: np.ndarray):
        if Y.ndim != 2:
            raise ValueError(f"Training data must be 2D [samples, assets]. Got {Y.ndim}D")
        self.Y = Y
        self.results = EvalResultCollection(training_shape=Y.shape, training_data=Y)

    def add(self, X: np.ndarray, name: str) -> 'Evaluator':
        if X.ndim != 2:
            raise ValueError(f"Generated data must be 2D [samples, assets]. Got {X.ndim}D")
        if X.shape[1] != self.Y.shape[1]:
            raise ValueError(f"Asset dimension mismatch: X has {X.shape[1]}, Y has {self.Y.shape[1]}")

        result = EvalResult(name=name, shape=X.shape, data=X)
        for metric_name, info in registry.get_all().items():
            result.metrics[metric_name] = info['func'](X, self.Y)

        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        return self.results


if __name__ == "__main__":
    training = np.load("data/stocknet_81.npy")
    factor_dm = np.load("samples/factorDM2048.npy")
    bootstrap = np.load("samples/SB2048.npy")

    results = (Evaluator(training)
        .add(factor_dm, "FactorDM")
        .add(bootstrap, "Stationary Bootstrap")
        .report())

    print(results.to_console())
    #results.plot_distribution(save_path="assets/distribution.png")
