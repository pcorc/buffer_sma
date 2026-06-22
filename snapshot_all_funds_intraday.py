"""
All-Funds INTRADAY Snapshot
===========================

Captures a live intraday snapshot of EVERY buffer fund (all ~67 tickers,
every series) from First Trust's main fund-list Excel — the same feed the
ECR website scrapes 3x/day (8am/12pm/4pm ET).

Why a separate script from download_all_funds_latest.py:
  - download_all_funds_latest.py uses the PER-FUND route (each fund's
    outcome-period Excel via its nsid). That route is END-OF-DAY only and
    breaks for funds that are mid-roll (no nsid during the transition).
  - This script uses the MAIN-TABLE route (export_latest_ecr.parse_main_table),
    which carries live current values for all funds, needs no nsid, and so
    works intraday AND through outcome-period rolls.

Trade-off: the main-table feed exposes the NET metrics (which is what ECR
uses) plus fund/reference values and the period's original cap/buffer, but
NOT the gross (non-net) columns or option-payoff columns. Those are written
blank so the layout still lines up with the per-fund snapshots.

Units: the main table reports decimals (0.1573 = 15.73%); this script
multiplies the percentage columns by 100 so the output matches the percent
convention of data.csv and the prior all_buffer_funds_* snapshots.

Usage:
    python snapshot_all_funds_intraday.py                 # capture now, all funds
    python snapshot_all_funds_intraday.py --include-ecr   # also add the ECR column

Output: output/all_buffer_funds_<date>_intraday_<HHMM>ET.xlsx and .csv
"""

import argparse
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from export_latest_ecr import (
    fetch_main_table_excel,
    parse_main_table,
    compute_all_components,
    calc_enhanced_cost_ratio,
)
from update_input_data import CSV_COLUMNS, format_date_mdy

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Eastern time = UTC-4 during EDT (First Trust's schedule is ET).
ET_OFFSET = timedelta(hours=-4)

# Main-table field -> data.csv column. Percentage fields are scaled x100 below.
PERCENT_FIELDS = {
    "fund_return": "Fund Return (%)",
    "reference_asset_return": "Reference Asset Return (%)",
    "remaining_cap_net": "Remaining Cap Net (%)",
    "remaining_buffer_net": "Remaining Buffer Net (%)",
    "downside_before_buffer_net": "Downside Before Buffer Net (%)",
    "fund_cap_net": "Original Cap Net (%)",
    "buffer_net": "Original Buffer Net (%)",
}
PASSTHROUGH_FIELDS = {
    "fund_value": "Fund Value (USD)",
    "reference_asset_value": "Reference Asset Value (USD)",
    "remaining_outcome_days": "Remaining Outcome Days",
}
# Columns in the data.csv layout the main-table feed does not provide.
UNAVAILABLE_INTRADAY = [
    "Remaining Cap (%)",
    "Reference Asset Return to Realize Cap (%)",
    "Remaining Buffer (%)",
    "Downside Before Buffer (%)",
    "Reference Asset to Buffer End (%)",
    "Unrealized Option Payoff (%)",
    "Unrealized Option Payoff Net (%)",
]


def main():
    parser = argparse.ArgumentParser(
        description="Capture a live intraday snapshot of all buffer funds.")
    parser.add_argument("--include-ecr", action="store_true",
                        help="add the Enhanced Cost Ratio column (website metric)")
    args = parser.parse_args()

    now_et = datetime.now(timezone.utc) + ET_OFFSET
    print("=" * 80)
    print(f"INTRADAY SNAPSHOT — ALL BUFFER FUNDS — {now_et:%Y-%m-%d %H:%M} ET")
    print("=" * 80)

    funds = parse_main_table(fetch_main_table_excel())
    print(f"  captured {len(funds)} funds from the live fund-list feed")

    out = pd.DataFrame()
    out["Date"] = [format_date_mdy(now_et)] * len(funds)
    out["Captured At (ET)"] = now_et.strftime("%Y-%m-%d %H:%M")
    out["Fund"] = funds["ticker"].values
    out["Strategy Type"] = funds["strategy_type"].values
    out["Series"] = funds["series"].values

    for field, column in PASSTHROUGH_FIELDS.items():
        out[column] = funds[field].values
    for field, column in PERCENT_FIELDS.items():
        out[column] = (funds[field] * 100).values
    for column in UNAVAILABLE_INTRADAY:
        out[column] = pd.NA

    if args.include_ecr:
        ecrs = []
        for rec in funds.to_dict("records"):
            for key, value in rec.items():
                if isinstance(value, float) and math.isnan(value):
                    rec[key] = None
            comp = compute_all_components(rec)
            ecrs.append(calc_enhanced_cost_ratio(
                comp["dbb_score"], comp["buffer_integrity"],
                comp["cap_integrity"], comp["time_scaling"]))
        out["ECR"] = ecrs

    # Order: identifiers, then the data.csv columns, then any extras (ECR)
    lead = ["Date", "Captured At (ET)", "Fund", "Strategy Type", "Series"]
    body = [c for c in CSV_COLUMNS[2:]]  # data.csv data columns in canonical order
    extras = [c for c in out.columns if c not in lead + body]
    out = out[lead + body + extras].sort_values(["Series", "Fund"]).reset_index(drop=True)

    print(f"  series covered: {out['Series'].nunique()} "
          f"({out.groupby('Series')['Fund'].count().to_dict()})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{now_et:%Y-%m-%d}_intraday_{now_et:%H%M}ET"
    xlsx_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.xlsx"
    csv_path = OUTPUT_DIR / f"all_buffer_funds_{tag}.csv"
    out.to_excel(xlsx_path, index=False, sheet_name="Intraday Snapshot")
    out.to_csv(csv_path, index=False)

    print(f"\n✅ {len(out)} funds captured intraday")
    print(f"   → {xlsx_path.relative_to(PROJECT_ROOT)}")
    print(f"   → {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
