"""
Minimal Massive (Polygon.io) 1-minute bar puller — verification + small pull.

Massive.com is the rebrand of Polygon.io; the REST host api.polygon.io still works.
Auth: set your key in the environment before running:

    export MASSIVE_API_KEY=your_key_here      # (POLYGON_API_KEY also accepted)

Quick verification (1 ticker, 1 day, prints a few 10:00-12:00 bars):

    python massiveDS/pull_minute.py --verify

Small pull (a handful of tickers, a short range -> parquet):

    python massiveDS/pull_minute.py --tickers AAPL MSFT NVDA \
        --from 2024-01-02 --to 2024-01-31 --out massiveDS/sample_minute.parquet

Notes
-----
* Endpoint: /v2/aggs/ticker/{T}/range/1/minute/{from}/{to}
* Returns up to 50_000 bars/request; we follow `next_url` pagination so any
  range works regardless of the cap.
* Bars come back in UTC ms. We convert to America/New_York and keep only the
  intraday window (default 10:00-12:00 ET) — the slice you want for the thesis.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.polygon.io")
API_KEY = "dWrHYjK0nHW0gXoraLoK3YLnGScQwVJb"

# Column rename: Polygon's terse keys -> readable names.
_COLS = {"t": "ts", "o": "open", "h": "high", "l": "low",
         "c": "close", "v": "volume", "n": "trades", "vw": "vwap"}


def fetch_minute_bars(ticker: str, start: str, end: str,
                      session: requests.Session | None = None) -> pd.DataFrame:
    """Pull every 1-minute bar for `ticker` in [start, end] (inclusive, YYYY-MM-DD)."""
    if not API_KEY:
        sys.exit("ERROR: set MASSIVE_API_KEY (or POLYGON_API_KEY) in your environment.")
    sess = session or requests.Session()

    url = (f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000")
    rows: list[dict] = []
    while url:
        # apiKey appended here so it's also attached to paginated next_url calls.
        sep = "&" if "?" in url else "?"
        resp = sess.get(f"{url}{sep}apiKey={API_KEY}", timeout=30)
        if resp.status_code == 429:           # rate limited -> back off and retry
            time.sleep(2)
            continue
        if resp.status_code == 403:           # plan doesn't cover this data
            msg = resp.json().get("message", resp.text)
            sys.exit(f"NOT_AUTHORIZED ({ticker}): {msg}\n"
                     f"-> Historical minute bars need a paid plan (Starter $29/mo). "
                     f"Upgrade, then re-run; the same key works.")
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("results", []) or [])
        url = payload.get("next_url")          # None when no more pages

    if not rows:
        return pd.DataFrame(columns=["ticker", "ts", *list(_COLS.values())[1:]])

    df = pd.DataFrame(rows).rename(columns=_COLS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df.insert(0, "ticker", ticker)
    keep = ["ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "trades"]
    return df[[c for c in keep if c in df.columns]]


def clip_window(df: pd.DataFrame, start_hhmm="10:00", end_hhmm="12:00") -> pd.DataFrame:
    """Keep only bars whose ET time falls in [start_hhmm, end_hhmm)."""
    t = df["ts"].dt.time
    lo = pd.to_datetime(start_hhmm).time()
    hi = pd.to_datetime(end_hhmm).time()
    return df[(t >= lo) & (t < hi)].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true",
                   help="Pull AAPL for one day and print a few bars to confirm access.")
    p.add_argument("--tickers", nargs="+", default=["AAPL"])
    p.add_argument("--from", dest="start", default="2024-01-02")
    p.add_argument("--to", dest="end", default="2024-01-02")
    p.add_argument("--window", nargs=2, metavar=("START", "END"), default=["10:00", "12:00"])
    p.add_argument("--out", default=None, help="Parquet path; if omitted, just prints a summary.")
    args = p.parse_args()

    if args.verify:
        args.tickers, args.start, args.end, args.out = ["AAPL"], "2024-01-02", "2024-01-02", None

    sess = requests.Session()
    frames = []
    for tk in args.tickers:
        raw = fetch_minute_bars(tk, args.start, args.end, sess)
        clipped = clip_window(raw, *args.window)
        print(f"{tk}: {len(raw):>6} raw bars -> {len(clipped):>5} in "
              f"{args.window[0]}-{args.window[1]} ET")
        frames.append(clipped)
        time.sleep(0.05)  # be polite even with unlimited calls

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if args.verify or args.out is None:
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print("\nFirst rows:")
            print(out.head(8).to_string(index=False))
        print(f"\nTotal rows: {len(out)}  tickers: {out['ticker'].nunique() if len(out) else 0}")

    if args.out:
        out.to_parquet(args.out, index=False)
        print(f"\nSaved {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()