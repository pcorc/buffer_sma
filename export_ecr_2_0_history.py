"""
ECR 2.0 FULL-HISTORY Exporter — non-F (or any) funds since inception
====================================================================

Walks every outcome period of each fund back to its inception and stacks all
daily rows into one CSV in the ECR 2.0 format (same columns/methodology as
export_ecr_2_0.py). Default universe: every fund whose ticker does NOT start
with 'F' (i.e. excludes the F-series).

How the walk-back works: First Trust's per-fund API returns the CURRENT
outcome period at .../nsid/{nsid}, and any completed period at
.../nsid/{nsid}/asofdate/{period_end}. Each period's start date equals the
previous period's end date, so we fetch the current period, then repeatedly
fetch the period ending on the current period's start date, until no earlier
period exists (inception reached).

Usage:
    python export_ecr_2_0_history.py                 # all non-F funds, full history
    python export_ecr_2_0_history.py --include-f     # every fund
    python export_ecr_2_0_history.py --tickers DAUG GAUG   # specific funds

Output: output/ecr_2_0_history_nonF_<asof>.csv
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from export_latest_ecr import fetch_main_table_excel, parse_main_table
from update_input_data import (
    fetch_nsid, fetch_period_excel, parse_period_excel, _fetch_previous_period,
)
from export_ecr_2_0 import transform

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MAX_PERIODS = 30  # safety cap; no target-outcome fund has anywhere near this many


def fetch_full_history(ticker):
    """All daily rows for a fund across every outcome period back to inception."""
    nsid = fetch_nsid(ticker)
    frames = []
    current = parse_period_excel(fetch_period_excel(nsid), ticker)
    frames.append(current)
    period_start = current["Date"].min()

    for _ in range(MAX_PERIODS):
        prev = _fetch_previous_period(ticker, nsid, period_start)
        if prev is None:
            break
        new_start = prev["Date"].min()
        if new_start >= period_start:   # no progress -> inception reached
            break
        frames.append(prev)
        period_start = new_start
        time.sleep(0.4)

    hist = pd.concat(frames, ignore_index=True)
    # Roll days appear in two periods (old end = new day-1): keep the day-1 row.
    hist = (hist.sort_values(["Date", "Remaining Outcome Days"])
                .drop_duplicates("Date", keep="last"))
    return hist


def main():
    parser = argparse.ArgumentParser(description="Full-history ECR 2.0 export (non-F by default).")
    parser.add_argument("--include-f", action="store_true", help="include F-series funds too")
    parser.add_argument("--tickers", nargs="+", help="restrict to these tickers")
    args = parser.parse_args()

    print("=" * 80)
    print("ECR 2.0 FULL-HISTORY EXPORT" + ("" if args.include_f else " — NON-F FUNDS"))
    print("=" * 80)

    funds = parse_main_table(fetch_main_table_excel())
    meta = funds.set_index("ticker")[["strategy_type", "series"]].to_dict("index")
    tickers = list(funds["ticker"])
    if args.tickers:
        tickers = [t for t in tickers if t in set(args.tickers)]
    elif not args.include_f:
        tickers = [t for t in tickers if not t.upper().startswith("F")]
    print(f"Universe: {len(tickers)} funds")
    print(f"  {', '.join(tickers)}\n")

    frames, failures = [], []
    pending = list(tickers)
    for attempt in range(3):          # self-healing: retry transient blips
        if not pending:
            break
        if attempt > 0:
            print(f"\n  retry round {attempt}: {len(pending)} fund(s) — {', '.join(pending)}")
            time.sleep(5)
        errored = []
        for i, ticker in enumerate(pending, start=1):
            try:
                hist = fetch_full_history(ticker)
                hist["Strategy Type"] = meta.get(ticker, {}).get("strategy_type")
                hist["Series"] = meta.get(ticker, {}).get("series")
                frames.append(hist)
                print(f"  [{i:2}/{len(pending)}] {ticker}: {len(hist)} rows "
                      f"({hist['Date'].min().date()} → {hist['Date'].max().date()})", flush=True)
                time.sleep(0.6)
            except Exception as err:  # noqa: BLE001
                print(f"  [{i:2}/{len(pending)}] {ticker}: ❌ {err}", flush=True)
                errored.append(ticker)
        pending = errored
    failures = pending

    if not frames:
        raise SystemExit("❌ Nothing downloaded.")

    raw = pd.concat(frames, ignore_index=True)
    final = transform(raw).sort_values(["ticker", "date"]).reset_index(drop=True)
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")

    print("\n" + "-" * 80)
    print(f"Stacked {len(final)} rows | {final['ticker'].nunique()} funds | "
          f"{final['date'].min()} → {final['date'].max()} | null ecr_2_0: {int(final['ecr_2_0'].isna().sum())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asof = datetime.now().strftime("%Y-%m-%d")
    scope = "allfunds" if args.include_f else "nonF"
    csv_path = OUTPUT_DIR / f"ecr_2_0_history_{scope}_{asof}.csv"
    final.to_csv(csv_path, index=False)
    print(f"\n✅ {csv_path}")
    if failures:
        print(f"⚠️  Failed after retries: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
