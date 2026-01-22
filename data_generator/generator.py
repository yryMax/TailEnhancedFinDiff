"""
Data Generator - Main logic for generating scenario matrices
"""
import os
import numpy as np
from typing import Dict, List, Tuple
from parsers.base_parser import BaseParser


class DataGenerator:
    """
    Generate scenario matrices from stock return data using sliding window.
    """

    def __init__(
        self,
        parser: BaseParser,
        scenario_size: int,
        t: int,
        window_step: int
    ):
        """
        Args:
            parser: Dataset parser instance.
            scenario_size: Number of stocks in one scenario.
            t: Number of trading days per stock.
            window_step: Sliding window step size (days).
        """
        self.parser = parser
        self.scenario_size = scenario_size
        self.t = t
        self.window_step = window_step

    def _align_stock_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Align all stock data to common dates and create a matrix.

        Returns:
            Tuple of (aligned data matrix [num_days, num_stocks], list of stock symbols)
        """
        returns_dict = self.parser.get_returns()
        symbols = sorted(returns_dict.keys())

        if len(symbols) > self.scenario_size:
            print(f"Found {len(symbols)} stocks, selecting first {self.scenario_size}")
            symbols = symbols[:self.scenario_size]
        elif len(symbols) < self.scenario_size:
            raise ValueError(
                f"Not enough stocks. Found {len(symbols)}, need {self.scenario_size}"
            )

        # Find the minimum length (common trading days)
        min_length = min(len(returns_dict[s]) for s in symbols)
        print(f"Minimum data length: {min_length} days")

        # Create aligned matrix [num_days, num_stocks]
        # Take the last min_length days from each stock (most recent data aligned)
        aligned_data = np.zeros((min_length, len(symbols)))

        for i, symbol in enumerate(symbols):
            data = returns_dict[symbol]
            # Take last min_length entries to align recent data
            aligned_data[:, i] = data[-min_length:]

        # Handle any missing values
        for i in range(aligned_data.shape[1]):
            if np.any(np.isnan(aligned_data[:, i])):
                aligned_data[:, i] = self.parser.interpolate_missing(aligned_data[:, i])

        return aligned_data, symbols

    def generate(self) -> np.ndarray:
        """
        Generate scenario matrix using sliding window.

        Returns:
            Numpy array of shape [N, scenario_size, t] where N is determined
            by the sliding window.
        """
        aligned_data, symbols = self._align_stock_data()
        num_days, num_stocks = aligned_data.shape

        print(f"Aligned data shape: {aligned_data.shape}")
        print(f"Number of stocks: {num_stocks}")

        # Calculate number of scenarios using sliding window
        # Starting positions: 0, window_step, 2*window_step, ...
        # Each window needs t days
        scenarios = []

        start = 0
        while start + self.t <= num_days:
            # Extract [t, num_stocks] window
            window = aligned_data[start:start + self.t, :]
            scenarios.append(window.T)  # Transpose to [num_stocks, t]
            start += self.window_step

        if len(scenarios) == 0:
            raise ValueError(
                f"Not enough data to generate scenarios. "
                f"Need at least {self.t} days, got {num_days}"
            )

        # Stack to get [N, scenario_size, t]
        result = np.stack(scenarios, axis=0)
        print(f"Generated {len(scenarios)} scenarios")
        print(f"Output shape: {result.shape}")

        return result

    def save(self, data: np.ndarray, output_path: str, filename: str = "scenarios.npy"):
        """
        Save generated data to npy file.

        Args:
            data: Generated scenario matrix.
            output_path: Directory to save the file.
            filename: Name of the output file.
        """
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)
        np.save(filepath, data)
        print(f"Saved to {filepath}")
