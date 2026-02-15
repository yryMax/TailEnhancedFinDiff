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


class TemporalRiskEvaluator(RiskEvaluator):
    """Evaluator for multi-day trajectory risk metrics."""

    def __init__(self, weights: np.ndarray, window: int):
        super().__init__(weights)
        self.window = window

    def add(self, trajectory: np.ndarray, name: str) -> 'TemporalRiskEvaluator':
        r = rolling_cumulative_returns(trajectory, self.weights, self.window)
        result = EvalResult(name=name, shape=trajectory.shape)
        for metric_name, info in registry.get_all().items():
            result.metrics[metric_name] = info['func'](r)
        self.results.append(result)
        return self


def rolling_cumulative_returns(trajectory: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    """Compute cumulative portfolio returns over rolling windows.
    trajectory: [T, d], weights: [d] → [T-window+1,] cumulative returns.
    """
    r_p = trajectory @ weights
    # vectorized: rolling product of (1 + r) then subtract 1
    log_r = np.log1p(r_p)
    cum_log = np.cumsum(log_r)
    cum_log = np.insert(cum_log, 0, 0.0)
    window_log = cum_log[window:] - cum_log[:-window]
    return np.expm1(window_log)


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
    cov = np.cov(training.T)

    w_eq = np.ones(d) / d
    w_mv = minimum_variance_weights(cov)
    weights_list = [("Equal-Weight", w_eq), ("Minimum Variance", w_mv)]

    # === Cross-Sectional ===
    for w_name, w in weights_list:
        run_evaluation(training, factor_dm, bootstrap, w)

    # === Temporal (SB only) ===
    sb_traj = np.load("samples/SB4096.npy")
    windows = [5, 10, 20]

    for w_name, w in weights_list:
        for T in windows:
            results = (TemporalRiskEvaluator(w, T)
                .add(training, "Historical (GT)")
                .add(sb_traj, "Stationary Bootstrap")
                .report())
            print(results.to_markdown())
