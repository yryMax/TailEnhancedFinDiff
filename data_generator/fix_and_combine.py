import os
import pandas as pd
import glob

output_dir = 'SP500'

# Read all individual CSV files and combine
all_dfs = []
files = glob.glob(os.path.join(output_dir, '*_data.csv'))

for file in files:
    try:
        # Read the file
        df = pd.read_csv(file)

        # Check if it has the weird yfinance format (Ticker row)
        if 'Ticker' in df.columns or (len(df.columns) > 0 and str(df.iloc[0, 0]) == 'Ticker'):
            # Skip the first two rows (header info) and re-read
            df = pd.read_csv(file, skiprows=2)
            if 'Date' not in df.columns and df.columns[0] != 'Date':
                df = pd.read_csv(file, skiprows=3, header=None)
                df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Name']

        # Parse Date and ensure it's timezone-naive
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
            df = df.set_index('Date')
        elif df.index.name == 'Date' or (hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype)):
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            df.index.name = 'Date'

        # Remove any rows that are all NaN (except Name)
        value_cols = [c for c in df.columns if c != 'Name']
        if value_cols:
            df = df.dropna(subset=value_cols, how='all')

        if len(df) > 0:
            all_dfs.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

# Combine all data
if all_dfs:
    combined = pd.concat(all_dfs)
    combined = combined.sort_index()
    combined.to_csv(os.path.join(output_dir, 'SP500_combined.csv'))

    # Print summary
    print(f"Combined {len(all_dfs)} stock files")
    print(f"Total rows: {len(combined)}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")
    print(f"Unique tickers: {combined['Name'].nunique()}")
    print(f"\nSaved to {os.path.join(output_dir, 'SP500_combined.csv')}")
