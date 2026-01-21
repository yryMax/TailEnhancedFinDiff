"""
Data Generator Configuration
"""

# Stocknet dataset configuration
STOCKNET_CONFIG = {
    "SCENARIO_SIZE": 88,  # Number of stocks in one scenario
    "t": 60,              # Number of trading days per stock
    "WINDOW_STEP": 30,    # Sliding window step size (days)
    "DATA_PATH": "stocknet-dataset/price/preprocessed/",
    "OUTPUT_PATH": "output/",
}

# Default configuration (can be changed based on dataset)
DEFAULT_CONFIG = STOCKNET_CONFIG
