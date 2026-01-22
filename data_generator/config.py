"""
Data Generator Configuration
"""

# Stocknet dataset configuration
STOCKNET_CONFIG = {
    "SCENARIO_SIZE": 80,  # Number of stocks in one scenario
    "t": 32,              # Number of trading days per stock
    "WINDOW_STEP": 1,     # Sliding window step size (days)
    "DATA_PATH": "stocknet-dataset/price/preprocessed/",
    "OUTPUT_PATH": "../data",
    "EXCLUDED_STOCKS": ["GMRE", "AGFS", "BABA", "ABBV", "PTR", "REX", "SNP", "BRK-A"],
}

# Default configuration (can be changed based on dataset)
DEFAULT_CONFIG = STOCKNET_CONFIG
