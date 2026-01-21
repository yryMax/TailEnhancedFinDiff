"""
Base Parser Abstract Class
"""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BaseParser(ABC):
    """
    Abstract base class for dataset parsers.
    All dataset parsers should inherit from this class.
    """

    @abstractmethod
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load raw data from files.

        Returns:
            Dict mapping stock symbol to its return series (numpy array).
            Each array should be sorted by date in ascending order.
        """
        pass

    @abstractmethod
    def get_returns(self) -> Dict[str, np.ndarray]:
        """
        Get return series for all stocks.

        Returns:
            Dict mapping stock symbol to its return series.
        """
        pass

    @abstractmethod
    def interpolate_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values in the data.

        Args:
            data: Input array that may contain NaN values.

        Returns:
            Array with missing values interpolated.
        """
        pass
