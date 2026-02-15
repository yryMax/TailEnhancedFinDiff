import numpy as np
from .registry import MetricRegistry
from .statistic import var, es

registry = MetricRegistry()


@registry.register('VaR_5%', compare=False)
def compute_var(r: np.ndarray) -> float:
    return var(r, 0.05)


@registry.register('ES_5%', compare=False)
def compute_es_portfolio(r: np.ndarray) -> float:
    return es(r, 0.05)


@registry.register('Vol', compare=False)
def compute_vol(r: np.ndarray) -> float:
    return float(np.std(r))


@registry.register('MaxLoss', compare=False)
def compute_max_loss(r: np.ndarray) -> float:
    return float(np.min(r))
