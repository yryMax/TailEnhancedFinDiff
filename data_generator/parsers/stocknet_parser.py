"""
Stocknet Dataset Parser
"""
import os
from typing import Dict
import numpy as np
from .base_parser import BaseParser


class StocknetParser(BaseParser):
    """
    Parser for Stocknet dataset.

    Data format (preprocessed price txt files):
    date, movement_percent, open, high, low, close, volume
    (tab-separated, dates in descending order)
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the preprocessed price data directory.
        """
        self.data_path = data_path
        self._returns: Dict[str, np.ndarray] = {}
        self._dates: Dict[str, np.ndarray] = {}

    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load all stock data from txt files.

        Returns:
            Dict mapping stock symbol to its return series (sorted by date ascending).
        """
        if self._returns:
            return self._returns

        txt_files = [f for f in os.listdir(self.data_path) if f.endswith('.txt')]

        for filename in txt_files:
            symbol = filename.replace('.txt', '')
            filepath = os.path.join(self.data_path, filename)

            dates = []
            returns = []

            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        date = parts[0]
                        movement_percent = float(parts[1])
                        dates.append(date)
                        returns.append(movement_percent)

            # Data is in descending order, reverse to ascending
            dates = dates[::-1]
            returns = returns[::-1]

            self._dates[symbol] = np.array(dates)
            self._returns[symbol] = np.array(returns)

        return self._returns

    def get_returns(self) -> Dict[str, np.ndarray]:
        """
        Get return series for all stocks.

        Returns:
            Dict mapping stock symbol to its return series.
        """
        if not self._returns:
            self.load_data()
        return self._returns

    def get_dates(self, symbol: str) -> np.ndarray:
        """
        Get date series for a specific stock.

        Args:
            symbol: Stock symbol.

        Returns:
            Array of dates.
        """
        if not self._dates:
            self.load_data()
        return self._dates.get(symbol, np.array([]))

    def interpolate_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values using linear interpolation.

        Args:
            data: Input array that may contain NaN values.

        Returns:
            Array with missing values interpolated.
        """
        if not np.any(np.isnan(data)):
            return data

        result = data.copy()
        nan_mask = np.isnan(result)

        if np.all(nan_mask):
            # All values are NaN, fill with zeros
            return np.zeros_like(result)

        # Get indices of valid and invalid values
        valid_indices = np.where(~nan_mask)[0]
        invalid_indices = np.where(nan_mask)[0]

        # Linear interpolation
        result[invalid_indices] = np.interp(
            invalid_indices,
            valid_indices,
            result[valid_indices]
        )

        return result

    def get_stock_symbols(self) -> list:
        """
        Get list of all stock symbols.

        Returns:
            List of stock symbols.
        """
        if not self._returns:
            self.load_data()
        return sorted(list(self._returns.keys()))
