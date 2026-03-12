from typing import Dict, Callable


class MetricRegistry:
    """Instance-based registry for evaluation metrics."""

    def __init__(self):
        self._metrics: Dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register a metric function."""
        def decorator(func: Callable):
            self._metrics[name] = func
            return func
        return decorator

    def get_all(self) -> Dict[str, Callable]:
        return self._metrics
