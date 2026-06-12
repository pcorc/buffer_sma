"""
All-Funds Latest-Day Downloader
================================

Downloads the most recent day of outcome-period data for EVERY buffer fund on
the ECR website (all series — F, G, D, S, Q…), not just the F-series that
input_data/data.csv tracks.

How it works (reuses the two existing, verified modules):
  1. export_latest_ecr.fetch_main_table_excel() — pulls the strategy-filtered
     fund list the website uses (Strategy=BUF/CBUF/DBUF/MBUF), giving the full
     ticker universe plus Strategy Type / Series metadata.
  2. update_input_data.fetch_nsid() + fetch_period_excel() — for each ticker,
     downloads the fund's CURRENT outcome-period Excel from First Trust's API
     (same per-fund endpoint that feeds data.csv).
  3. Keeps each fund's most recent row (or the row for --date if given) and
     writes one combined file in the data.csv column layout (percent units),
     plus Strategy Type / Series columns for identification.

Usage:
    python download_all_funds_latest.py                    # latest available day
    python download_all_funds_latest.py --date 2026-06-12  # a specific day
    python download_all_funds_latest.py --full-period      # all days of each
                                                           # fund's current period

Output: output/all_buffer_funds_<date>.xlsx and .csv

Note: First Trust publishes end-of-day data with roughly a one-trading-day
lag, so "latest available" is usually yesterday's close. The script prints
exactly which date each fund returned.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from export_latest_ecr import fetch_main_table_excel, parse_main_table
from update_input_data import (
    CSV_COLUMNS,
    fetch_nsid,
    fetch_period_excel,
    parse_period_excel,
    format_date_mdy,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(
        description="Download the latest day of data for all buffer funds.")
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD",
                        help="take this exact date instead of each fund's latest")
    parser.add_argument("--full-period", action="store_true",
                        help="keep every day of each fund's current outcome "
                             "period instead of only the most recent row")
    args = parser.parse_args()
    target_date = pd.Timestamp(args.date) if args.date else None

    print("=" * 80)
    print("DOWNLOADING LATEST DATA FOR ALL BUFFER FUNDS")
    print("=" * 80)

    funds = parse_main_table(fetch_main_table_excel())
    print(f"Fund universe from the website's fund list: {len(funds)} tickers\n")

    frames = []
    failures = []
    for i, meta in enumerate(funds.itertuples(index=False), start=1):
        ticker = meta.ticker
        try:
            nsid = fetch_nsid(ticker)
            period = parse_period_excel(fetch_period_excel(nsid), ticker)

            if args.full_period:
                rows = period
            elif target_date is not None:
                rows = period[period["Date"] == target_date]
                if rows.empty:
                    print(f"  [{i:2}/{len(funds)}] {ticker}: no row for "
                          f"{target_date.date()} (period has "
                          f"{period['Date'].min().date()} → {period['Date'].max().date()})")
                    failures.append(ticker)
                    continue
            else:
                rows = period[period["Date"] == period["Date"].max()]

            rows = rows.copy()
            rows["Strategy Type"] = meta.strategy_type
            rows["Series"] = meta.series
            frames.append(rows)
            print(f"  [{i:2}/{len(funds)}] {ticker}: "
                  f"{', '.join(d.strftime('%Y-%m-%d') for d in rows['Date'].unique()[:1])}"
                  f"{' (+' + str(len(rows) - 1) + ' more days)' if len(rows) > 1 else ''}")
            time.sleep(0.5)
        except Exception as err:  # noqa: BLE001 - keep going, report at the end
            print(f"  [{i:2}/{len(funds)}] {ticker}: ❌ {err}")
            failures.append(ticker)

    if not frames:
        raise SystemExit("❌ Nothing downloaded.")

    out = pd.concat(frames, ignore_index=True)
    columns = ["Date", "Fund", "Strategy Type", "Series"] + CSV_COLUMNS[2:]
    out = out[columns].sort_values(["Date", "Fund"]).reset_index(drop=True)

    latest = out["Date"].max()
    date_counts = out.groupby(out["Date"].dt.date)["Fund"].nunique()
    print("\n" + "-" * 80)
    print("Funds per date in the result:")
    for d, n in date_counts.items():
        print(f"  {d}: {n} funds")

    out["Date"] = out["Date"].map(format_date_mdy)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (target_date or latest).strftime("%Y-%m-%d") + ("_full" if args.full_period else "")
    xlsx_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.xlsx"
    csv_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.csv"
    out.to_excel(xlsx_path, index=False, sheet_name="All Buffer Funds")
    out.to_csv(csv_path, index=False)

    print(f"\n✅ {len(out)} rows / {out['Fund'].nunique()} funds")
    print(f"   → {xlsx_path.relative_to(PROJECT_ROOT)}")
    print(f"   → {csv_path.relative_to(PROJECT_ROOT)}")
    if failures:
        print(f"⚠️  No data for: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
