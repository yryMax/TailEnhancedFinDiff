from typing import Dict, Any, Callable


class MetricRegistry:
    """Instance-based registry for evaluation metrics."""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, compare: bool = True):
        """
        Decorator to register a metric.
        compare=True: metric needs both X and Y (e.g., MMD)
        compare=False: metric only needs X (e.g., VaR)
        """
        def decorator(func: Callable):
            self._metrics[name] = {'func': func, 'compare': compare}
            return func
        return decorator

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return self._metrics
