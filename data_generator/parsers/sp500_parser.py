"""
SP500 Dataset Parser
"""
import os
from typing import Dict
import numpy as np
import pandas as pd
from .base_parser import BaseParser


class SP500Parser(BaseParser):
    """
    Parser for SP500 dataset.

    Data format (single CSV file):
    date, open, high, low, close, volume, Name
    """

    def __init__(self, data_path: str, min_trading_days: int = 1259, excluded_stocks: list = None):
        """
        Args:
            data_path: Path to SP500.csv file.
            min_trading_days: Minimum trading days required (default 1259 for complete data).
            excluded_stocks: List of stock symbols to exclude.
        """
        self.data_path = data_path
        self.min_trading_days = min_trading_days
        self.excluded_stocks = set(excluded_stocks) if excluded_stocks else set()
        self._returns: Dict[str, np.ndarray] = {}
        self._dates: Dict[str, np.ndarray] = {}

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load stock data from SP500.csv and compute daily returns from close prices."""
        if self._returns:
            return self._returns

        # Read CSV
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Name', 'date'])

        # Group by stock
        for symbol, group in df.groupby('Name'):
            if symbol in self.excluded_stocks:
                continue

            # Filter stocks with incomplete data
            if len(group) < self.min_trading_days:
                continue

            # Calculate daily return: (close_t - close_{t-1}) / close_{t-1} * 100
            close_prices = group['close'].values
            returns = np.diff(close_prices) / close_prices[:-1] * 100

            dates = group['date'].dt.strftime('%Y-%m-%d').values[1:]  # Skip first date (no return)

            self._dates[symbol] = np.array(dates)
            self._returns[symbol] = returns

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
