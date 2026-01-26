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
    def get_scenario(self, output_path: str = None):
        """
        Load data and return as numpy array, align the temporal dimension as preferred.
        Args:
            output_path: If provided, save the numpy array to this path.
        Returns:
            Numpy array of shape (num_stocks, num_days) with return series.
        """
        pass
