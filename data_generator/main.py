"""
Data Generator - Main entry point
"""
import os
import argparse
import numpy as np

from config import STOCKNET_CONFIG
from parsers import StocknetParser
from generator import DataGenerator


def generate_stocknet():
    """Generate scenario data from Stocknet dataset."""
    config = STOCKNET_CONFIG

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, config["DATA_PATH"])
    output_path = os.path.join(script_dir, config["OUTPUT_PATH"])

    print("=" * 50)
    print("Stocknet Data Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  SCENARIO_SIZE: {config['SCENARIO_SIZE']}")
    print(f"  t (days per stock): {config['t']}")
    print(f"  Window step: {config['WINDOW_STEP']}")
    print(f"  Excluded stocks: {config.get('EXCLUDED_STOCKS', [])}")
    print(f"  Data path: {data_path}")
    print(f"  Output path: {output_path}")
    print("=" * 50)

    # Create parser
    parser = StocknetParser(data_path, excluded_stocks=config.get("EXCLUDED_STOCKS", []))

    # Load data
    print("\nLoading data...")
    returns = parser.get_returns()
    print(f"Loaded {len(returns)} stocks")

    # Create generator
    generator = DataGenerator(
        parser=parser,
        scenario_size=config["SCENARIO_SIZE"],
        t=config["t"],
        window_step=config["WINDOW_STEP"]
    )

    # Generate scenarios
    print("\nGenerating scenarios...")
    scenarios = generator.generate()

    # Save results
    print("\nSaving results...")
    generator.save(scenarios, output_path, "stocknet_scenarios.npy")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Output shape: {scenarios.shape}")
    print(f"  N (number of scenarios): {scenarios.shape[0]}")
    print(f"  SCENARIO_SIZE: {scenarios.shape[1]}")
    print(f"  t: {scenarios.shape[2]}")
    print(f"  Data range: [{scenarios.min():.6f}, {scenarios.max():.6f}]")
    print(f"  Mean return: {scenarios.mean():.6f}")
    print(f"  Std return: {scenarios.std():.6f}")
    print("=" * 50)

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Data Generator for Stock Scenarios")
    parser.add_argument(
        "--dataset",
        type=str,
        default="stocknet",
        choices=["stocknet"],
        help="Dataset to process (default: stocknet)"
    )

    args = parser.parse_args()

    if args.dataset == "stocknet":
        generate_stocknet()


if __name__ == "__main__":
    main()
