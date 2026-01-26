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

    def __init__(self, data_path: str, excluded_stocks: list = None):
        self.data_path = data_path
        self.excluded_stocks = set(excluded_stocks) if excluded_stocks else set()
        self._returns: Dict[str, np.ndarray] = {}
        self._dates: Dict[str, np.ndarray] = {}

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load all stock data from txt files."""
        if self._returns:
            return self._returns

        for filename in os.listdir(self.data_path):
            if not filename.endswith('.txt'):
                continue
            symbol = filename.replace('.txt', '')
            if symbol in self.excluded_stocks:
                continue

            dates, returns = [], []
            with open(os.path.join(self.data_path, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        dates.append(parts[0])
                        returns.append(float(parts[1]))

            # Reverse from descending to ascending order
            self._dates[symbol] = np.array(dates[::-1])
            self._returns[symbol] = np.array(returns[::-1])

        return self._returns

    def get_scenario(self, output_path: str = None) -> np.ndarray:
        """
        Get aligned scenario matrix for all stocks.

        Args:
            output_path: If provided, save the scenario matrix as .npy file.

        Returns:
            np.ndarray of shape (n_stocks, n_timesteps)
        """
        if not self._returns:
            self.load_data()

        # 1. Build unified timeline (union of all dates)
        all_dates = set()
        for dates in self._dates.values():
            all_dates.update(dates.tolist())
        timeline = np.array(sorted(all_dates))
        n_time = len(timeline)

        # 2. Build matrix with NaN for missing values
        symbols = sorted(self._returns.keys())
        scenario = np.full((len(symbols), n_time), np.nan)

        date_to_idx = {d: i for i, d in enumerate(timeline)}
        for i, symbol in enumerate(symbols):
            for j, date in enumerate(self._dates[symbol]):
                scenario[i, date_to_idx[date]] = self._returns[symbol][j]

        # 3. Interpolate missing values per row
        for i in range(len(symbols)):
            row = scenario[i]
            valid = ~np.isnan(row)
            if valid.any() and not valid.all():
                scenario[i] = np.interp(np.arange(n_time), np.where(valid)[0], row[valid])

        if output_path is not None:
            np.save(output_path, scenario)

        return scenario

    def get_returns(self) -> Dict[str, np.ndarray]:
        if not self._returns:
            self.load_data()
        return self._returns

    def get_stock_symbols(self) -> list:
        if not self._returns:
            self.load_data()
        return sorted(self._returns.keys())
