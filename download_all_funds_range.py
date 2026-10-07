"""
All-Funds Date-Range Exporter
=============================

Stacks every website metric for every buffer fund across a date range into
one Excel (and CSV) file — one row per (fund, trading day).

"Website metrics" = the full per-fund daily dataset First Trust publishes
(the data.csv column set, in percent units) PLUS the website's computed
Enhanced Cost Ratio and its components (DBB Score, Buffer Integrity, Cap
Integrity, Time Scaling, ECR), using the exact calculator from
cost-ratio-enhanced (src/lib/calculator/enhanced-cost-ratio.ts, ported in
export_latest_ecr.py).

How it works (reuses verified modules):
  - export_latest_ecr.parse_main_table -> fund universe + Strategy Type / Series
  - update_input_data.fetch_fund_history -> each fund's daily outcome-period
    rows, walking back across any outcome-period roll inside the window so
    there are no gaps (e.g. the September series rolled on 2026-09-18).
  - rows are filtered to [start, end] and stacked.

ECR note: the per-fund feed is in PERCENT units; the ECR formulas expect
decimals, so the net columns are divided by 100 before the ECR calc (exactly
as backtesting/data_pipeline.py does).

Usage:
    python download_all_funds_range.py --start 2026-09-21 --end 2026-10-06
    python download_all_funds_range.py --start 2026-09-21 --end 2026-10-06 --no-ecr

Output: output/all_buffer_funds_<start>_to_<end>.xlsx and .csv
"""

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

from export_latest_ecr import (
    fetch_main_table_excel,
    parse_main_table,
    compute_all_components,
    calc_enhanced_cost_ratio,
)
from update_input_data import CSV_COLUMNS, fetch_fund_history, format_date_mdy

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Per-fund percent column -> decimal value feeding the ECR calculator.
ECR_INPUTS = {
    "downside_before_buffer_net": "Downside Before Buffer Net (%)",
    "remaining_buffer_net": "Remaining Buffer Net (%)",
    "buffer_net": "Original Buffer Net (%)",
    "remaining_cap_net": "Remaining Cap Net (%)",
    "fund_cap_net": "Original Cap Net (%)",
}
ECR_OUTPUT_COLUMNS = ["DBB Score", "Buffer Integrity", "Cap Integrity",
                      "Time Scaling", "ECR"]


def _nan_to_none(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def compute_ecr_columns(row):
    """Compute DBB/Buffer/Cap/TimeScaling/ECR for one per-fund (percent) row."""
    inputs = {
        key: (None if pd.isna(row[col]) else float(row[col]) / 100.0)
        for key, col in ECR_INPUTS.items()
    }
    inputs["remaining_outcome_days"] = _nan_to_none(row["Remaining Outcome Days"])
    comp = compute_all_components(inputs)
    ecr = calc_enhanced_cost_ratio(
        comp["dbb_score"], comp["buffer_integrity"],
        comp["cap_integrity"], comp["time_scaling"])
    return pd.Series({
        "DBB Score": comp["dbb_score"],
        "Buffer Integrity": comp["buffer_integrity"],
        "Cap Integrity": comp["cap_integrity"],
        "Time Scaling": comp["time_scaling"],
        "ECR": ecr,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Stack all website metrics for all funds across a date range.")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--no-ecr", action="store_true",
                        help="omit the computed ECR columns (raw metrics only)")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise SystemExit("❌ --end is before --start")

    print("=" * 80)
    print(f"ALL-FUNDS METRICS — {start.date()} → {end.date()}")
    print("=" * 80)

    funds = parse_main_table(fetch_main_table_excel())
    meta = funds.set_index("ticker")[["strategy_type", "series"]].to_dict("index")
    print(f"Fund universe: {len(funds)} tickers\n")

    frames = []
    failures = []
    for i, ticker in enumerate(funds["ticker"], start=1):
        try:
            # last_known = start - 1 day makes fetch_fund_history walk back
            # across any roll until the period covering `start` is included.
            hist = fetch_fund_history(ticker, start - pd.Timedelta(days=1))
            hist = (hist.sort_values(["Date", "Remaining Outcome Days"])
                        .drop_duplicates("Date", keep="last"))  # roll day -> new period row
            win = hist[(hist["Date"] >= start) & (hist["Date"] <= end)].copy()
            if win.empty:
                print(f"  [{i:2}/{len(funds)}] {ticker}: no rows in window")
                failures.append(ticker)
                continue
            win["Strategy Type"] = meta.get(ticker, {}).get("strategy_type")
            win["Series"] = meta.get(ticker, {}).get("series")
            frames.append(win)
            print(f"  [{i:2}/{len(funds)}] {ticker}: {len(win)} days "
                  f"({win['Date'].min().date()} → {win['Date'].max().date()})")
            time.sleep(0.4)
        except Exception as err:  # noqa: BLE001 - keep going, report at the end
            print(f"  [{i:2}/{len(funds)}] {ticker}: ❌ {err}")
            failures.append(ticker)

    if not frames:
        raise SystemExit("❌ Nothing downloaded.")

    out = pd.concat(frames, ignore_index=True)
    if not args.no_ecr:
        out[ECR_OUTPUT_COLUMNS] = out.apply(compute_ecr_columns, axis=1)

    lead = ["Date", "Fund", "Strategy Type", "Series"]
    body = CSV_COLUMNS[2:]
    extras = ECR_OUTPUT_COLUMNS if not args.no_ecr else []
    out = out[lead + body + extras].sort_values(["Date", "Fund"]).reset_index(drop=True)

    trading_days = out["Date"].nunique()
    print("\n" + "-" * 80)
    print(f"Stacked {len(out)} rows | {out['Fund'].nunique()} funds | "
          f"{trading_days} trading days ({out['Date'].min().date()} → {out['Date'].max().date()})")
    per_day = out.groupby(out["Date"].dt.date)["Fund"].nunique()
    if per_day.nunique() == 1:
        print(f"  every trading day has {per_day.iloc[0]} funds ✓")
    else:
        print("  ⚠️ uneven funds-per-day:", per_day.to_dict())

    out["Date"] = out["Date"].map(format_date_mdy)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start:%Y-%m-%d}_to_{end:%Y-%m-%d}"
    xlsx_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.xlsx"
    csv_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.csv"
    out.to_excel(xlsx_path, index=False, sheet_name=f"{start:%m-%d} to {end:%m-%d}")
    out.to_csv(csv_path, index=False)

    print(f"\n✅ {xlsx_path.relative_to(PROJECT_ROOT)}")
    print(f"   {csv_path.relative_to(PROJECT_ROOT)}")
    if failures:
        print(f"⚠️  No data for: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
