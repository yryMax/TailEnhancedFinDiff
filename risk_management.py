import numpy as np
from scipy.optimize import minimize
from metrics.risk import registry
from result import EvalResult, EvalResultCollection


class RiskEvaluator:
    """Evaluator for portfolio-level risk metrics."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.results = EvalResultCollection()

    def add(self, scenarios: np.ndarray, name: str) -> 'RiskEvaluator':
        r = scenarios @ self.weights
        result = EvalResult(name=name, shape=scenarios.shape)
        for metric_name, info in registry.get_all().items():
            result.metrics[metric_name] = info['func'](r)
        self.results.append(result)
        return self

    def report(self) -> EvalResultCollection:
        return self.results


def minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Solve min w'Σw s.t. Σw_i=1, w_i>=0 (long-only)."""
    d = cov.shape[0]
    w0 = np.ones(d) / d
    # Scale covariance for numerical stability (daily returns have tiny values)
    scale = 1.0 / np.median(np.diag(cov))
    cov_s = cov * scale
    res = minimize(
        lambda w: w @ cov_s @ w,
        w0,
        jac=lambda w: 2 * cov_s @ w,
        method='SLSQP',
        bounds=[(0, 1)] * d,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
    )
    return res.x


def run_evaluation(training, factor_dm, bootstrap, weights):
    """Run risk evaluation for a given portfolio and print results."""
    results = (RiskEvaluator(weights)
        .add(training, "Historical (GT)")
        .add(factor_dm, "FactorDM")
        .add(bootstrap, "Stationary Bootstrap")
        .report())
    results.to_console()
    return results


if __name__ == "__main__":
    training = np.load("data/stocknet_81.npy")
    factor_dm = np.load("samples/factorDM2048.npy")
    bootstrap = np.load("samples/SB2048.npy")

    d = training.shape[1]

    # Equal-weight
    w_eq = np.ones(d) / d
    run_evaluation(training, factor_dm, bootstrap, w_eq)

    # Minimum Variance
    cov = np.cov(training.T)
    w_mv = minimum_variance_weights(cov)
    run_evaluation(training, factor_dm, bootstrap, w_mv)
