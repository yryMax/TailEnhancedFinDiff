import numpy as np
from metrics.statistic_temporal import registry
from result import EvalResult, EvalResultCollection


class TemporalEvaluator:
    """Evaluator for temporal scenario data [M, N, D]."""

    def __init__(self, Y: np.ndarray, metrics: list = None):
        if Y.ndim != 3:
            raise ValueError(f"Training data must be 3D [windows, assets, days]. Got {Y.ndim}D")
        self.Y = Y
        self.metrics = metrics

        all_metrics = registry.get_all()
        selected = {k: v for k, v in all_metrics.items() if metrics is None or k in metrics}
        gt_result = EvalResult(name='GT')
        for metric_name, func in selected.items():
            gt_result.metrics[metric_name] = func(Y, Y)

        self.results = EvalResultCollection(training_shape=Y.shape, gt_result=gt_result)

    def add(self, X: np.ndarray, name: str) -> 'TemporalEvaluator':
        if X.ndim != 3:
            raise ValueError(f"Generated data must be 3D [samples, assets, days]. Got {X.ndim}D")
        if X.shape[1] != self.Y.shape[1]:
            raise ValueError(f"Asset dimension mismatch: X has {X.shape[1]}, Y has {self.Y.shape[1]}")
        if X.shape[2] != self.Y.shape[2]:
            raise ValueError(f"Day dimension mismatch: X has {X.shape[2]}, Y has {self.Y.shape[2]}")

        all_metrics = registry.get_all()
        selected = {k: v for k, v in all_metrics.items() if self.metrics is None or k in self.metrics}

        result = EvalResult(name=name)
        for metric_name, func in selected.items():
            result.metrics[metric_name] = func(X, self.Y)

        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        return self.results


if __name__ == "__main__":
    training = np.load("data/stocknet_81_windows.npy")   # [1236, 81, 22]
    sb       = np.load("samples/SB2048_T22.npy")          # [2048, 81, 22]
    macrovae = np.load("samples/sampled_stocks_2048_v4.npy")  # [2048, 81, 22]

    results = (TemporalEvaluator(training)
        .add(sb,       "Stationary Bootstrap")
        .add(macrovae, "MacroVAE")
        .report())

    results.to_console()
    print(results.to_markdown())
