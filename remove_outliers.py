"""
Remove extreme outliers by trimming the smallest and largest X%.

Usage examples:
  - Per-column trimming 1% (default):
      python remove_outliers.py --input model/regression/factors.csv \
                               --output model/regression/factors_trimmed.csv \
                               --percent 1

  - Specify columns and a different trim percent:
      python remove_outliers.py --input data.csv --percent 2 --columns col1 col2 col3

New
  - Outputs the dropped (extreme) row dates in chronological order when a date column is present.
    Use --date-column to specify the date column name (defaults to the first column).

Notes:
  - By default, only numeric columns are considered when computing percentiles.
  - Rows are dropped if ANY selected numeric column is outside its [p, 1-p] quantile.
  - Set --method global to compute thresholds on the stacked numeric values across all selected columns.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _derive_output_path(input_path: str, suffix: str = "_trimmed") -> str:
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".csv"
    return f"{base}{suffix}{ext}"


def trim_outliers_df(
    df: pd.DataFrame,
    percent: float = 1.0,
    columns: Optional[Iterable[str]] = None,
    method: str = "per-column",
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove rows outside the [p, 1-p] range, either per column or globally.

    - per-column: keep rows where each selected column lies within its own
                  [p, 1-p] quantile range.
    - global:     compute thresholds across the stacked selected columns, and
                  keep rows only if all selected columns of that row lie within
                  [p, 1-p] computed globally.

    Returns (trimmed_df, stats_dict).
    """
    if not (0 <= percent < 50):
        raise ValueError("percent must be in [0, 50)")

    # Determine which columns to consider
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in columns if c in df.columns]
        missing = set(columns) - set(num_cols)
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {sorted(missing)}")

    if not num_cols:
        raise ValueError("No numeric columns to process. Specify --columns if needed.")

    p = percent / 100.0
    original_rows = len(df)

    if method == "per-column":
        q_low = df[num_cols].quantile(p)
        q_high = df[num_cols].quantile(1 - p)
        within_low = df[num_cols].ge(q_low)
        within_high = df[num_cols].le(q_high)
        mask = (within_low & within_high).all(axis=1)
    elif method == "global":
        stacked = df[num_cols].stack(dropna=True)
        low = stacked.quantile(p)
        high = stacked.quantile(1 - p)
        mask = df[num_cols].apply(lambda col: col.between(low, high, inclusive="both"))
        mask = mask.all(axis=1)
    else:
        raise ValueError("method must be 'per-column' or 'global'")

    trimmed = df[mask].copy()

    stats = {
        "original_rows": int(original_rows),
        "kept_rows": int(len(trimmed)),
        "dropped_rows": int(original_rows - len(trimmed)),
        "trim_percent_each_side": float(percent),
        "method": method,
        "columns_used": list(num_cols),
        # Expose the boolean mask to allow callers to identify dropped rows/dates
        "_mask": mask,
    }
    return trimmed, stats


def main():
    parser = argparse.ArgumentParser(description="Trim extreme outliers from a CSV.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default=None, help="Output CSV file path")
    parser.add_argument("--percent", type=float, default=1.0, help="Trim percent on each side (default: 1.0)")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to consider (default: all numeric columns)",
    )
    parser.add_argument(
        "--method",
        choices=["per-column", "global"],
        default="per-column",
        help="How to compute thresholds (default: per-column)",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help=(
            "Name of the date column to report dropped dates. "
            "Defaults to the first column if not provided."
        ),
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output or _derive_output_path(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    trimmed, stats = trim_outliers_df(
        df,
        percent=args.percent,
        columns=args.columns,
        method=args.method,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    trimmed.to_csv(output_path, index=False)

    print(
        "Outlier trimming complete:\n"
        f"  Input:   {input_path}\n"
        f"  Output:  {output_path}\n"
        f"  Method:  {stats['method']}\n"
        f"  Columns: {', '.join(stats['columns_used'])}\n"
        f"  Rows:    {stats['kept_rows']} kept / {stats['original_rows']} original"
        f"  (dropped {stats['dropped_rows']}, {args.percent}% each side)"
    )

    # Report dropped dates in chronological order if a date column is available
    try:
        date_col = args.date_column if args.date_column is not None else df.columns[0]
        if date_col not in df.columns:
            # Fallback to first column if specified one is missing
            date_col = df.columns[0]

        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = stats.get("_mask")
        if mask is not None and dates.notna().any():
            dropped_dates = dates[~mask]
            # Keep only valid dates, drop NaT, sort ascending, unique preserve order
            dropped_dates = dropped_dates.dropna().sort_values()
            # Use date only (YYYY-MM-DD)
            dropped_str = dropped_dates.dt.strftime("%Y-%m-%d").tolist()
            if len(dropped_str) > 0:
                print("\nDropped (extreme) dates in chronological order:")
                for d in dropped_str:
                    print(f"  {d}")
            else:
                print("\nNo dropped dates to report (no rows dropped or no valid date column).")
        else:
            print("\nSkipped dropped-date report (no date column or mask unavailable).")
    except Exception as e:
        # Do not fail the main trimming workflow because of reporting
        print(f"\nWarning: failed to report dropped dates: {e}")


if __name__ == "__main__":
    main()
