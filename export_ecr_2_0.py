"""
ECR 2.0 Exporter — new methodology
==================================

Produces the "website download" in the new format: raw First Trust metrics
converted to decimals, headers remapped to snake_case, and the ECR 2.0
scoring methodology (DBB Score, Buffer Integrity, Cap Integrity, Reset
Proximity, Period Position, ecr_2_0).

Methodology is a 1:1 replication of the reference workbook
(ecr_download_example.xlsx). With columns divided by 100 first (decimals):

  dbb_score       = exp(downside_before_buffer_net * 5)
  buffer_integrity = min(1, max(0, remaining_buffer_net / fund_cap_net))
  cap_integrity    = min(remaining_cap_net / fund_cap_net, 1)
  reset_proximity  = exp(-(reference_asset_return^2) / (2*0.07^2))  if ref_return <= 0
                     exp(-(reference_asset_return^2) / (2*0.05^2))  if ref_return >  0
  period_position  = exp(-((remaining_outcome_days - 180)^2) / (2*75^2))
  ecr_2_0 = (1.5*dbb_score + 1.2*reset_proximity)
            * max(0, buffer_integrity)**0.5
            * max(0, cap_integrity)**0.3
            + 0.3*period_position

NOTE on buffer_integrity: per the reference workbook its denominator is
fund_cap_net (Original Cap Net, column T), NOT buffer_net (Original Buffer
Net, column U). Replicated exactly as specified; flagged here in case it was
meant to be buffer_net.

Validated to reproduce every row of the reference workbook to ~1e-12
(see validate_against_workbook()).

Usage:
    python export_ecr_2_0.py --date 2026-10-07
    python export_ecr_2_0.py --start 2026-09-21 --end 2026-10-06
    python export_ecr_2_0.py --validate   # check against the reference workbook

Output: output/ecr_2_0_<date|range>.xlsx and .csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from export_latest_ecr import fetch_main_table_excel, parse_main_table
from update_input_data import fetch_fund_history

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Columns divided by 100 to become decimal fractions (prices/days excluded).
DIVIDE_BY_100 = [
    "Fund Return (%)",
    "Reference Asset Return (%)",
    "Remaining Cap (%)",
    "Remaining Cap Net (%)",
    "Reference Asset Return to Realize Cap (%)",
    "Remaining Buffer (%)",
    "Remaining Buffer Net (%)",
    "Downside Before Buffer (%)",
    "Downside Before Buffer Net (%)",
    "Reference Asset to Buffer End (%)",
    "Unrealized Option Payoff (%)",
    "Unrealized Option Payoff Net (%)",
    "Original Cap Net (%)",
    "Original Buffer Net (%)",
]

# Header remap (as specified). Reset Proximity / Period Position are new metric
# columns not in the original website download, so they are not in this map;
# they are created directly under their snake_case names.
COLUMN_MAP = {
    "Date": "date",
    "Fund": "ticker",
    "Strategy Type": "fund_sleeve",
    "Series": "series",
    "Fund Value (USD)": "fund_value",
    "Fund Return (%)": "fund_return",
    "Reference Asset Value (USD)": "reference_asset_value",
    "Reference Asset Return (%)": "reference_asset_return",
    "Remaining Outcome Days": "remaining_outcome_days",
    "Remaining Cap (%)": "remaining_cap",
    "Remaining Cap Net (%)": "remaining_cap_net",
    "Reference Asset Return to Realize Cap (%)": "reference_asset_return_to_realize_cap",
    "Remaining Buffer (%)": "remaining_buffer",
    "Remaining Buffer Net (%)": "remaining_buffer_net",
    "Downside Before Buffer (%)": "downside_before_buffer",
    "Downside Before Buffer Net (%)": "downside_before_buffer_net",
    "Reference Asset to Buffer End (%)": "reference_asset_to_buffer_end",
    "Unrealized Option Payoff (%)": "unrealized_option_payoff",
    "Unrealized Option Payoff Net (%)": "unrealized_option_payoff_net",
    "Original Cap Net (%)": "fund_cap_net",
    "Original Buffer Net (%)": "buffer_net",
}

# Final column order (mirrors the reference workbook).
FINAL_COLUMNS = [
    "date", "ticker", "fund_sleeve", "series",
    "fund_value", "fund_return", "reference_asset_value", "reference_asset_return",
    "remaining_outcome_days", "remaining_cap", "remaining_cap_net",
    "reference_asset_return_to_realize_cap", "remaining_buffer", "remaining_buffer_net",
    "downside_before_buffer", "downside_before_buffer_net", "reference_asset_to_buffer_end",
    "unrealized_option_payoff", "unrealized_option_payoff_net", "fund_cap_net", "buffer_net",
    "dbb_score", "buffer_integrity", "cap_integrity", "reset_proximity",
    "period_position", "ecr_2_0",
]


# ============================================================================
# Methodology — operates on DECIMAL columns, replicating the workbook formulas
# ============================================================================

def add_ecr_2_0_metrics(df):
    """Add dbb_score, buffer_integrity, cap_integrity, reset_proximity,
    period_position, ecr_2_0 to a frame that already has the snake_case
    decimal columns. Mirrors the reference workbook cell-for-cell."""
    dbb_net = df["downside_before_buffer_net"]
    rem_buf_net = df["remaining_buffer_net"]
    rem_cap_net = df["remaining_cap_net"]
    cap0 = df["fund_cap_net"]                 # Original Cap Net (col T)
    ref_ret = df["reference_asset_return"]
    days = df["remaining_outcome_days"]

    df["dbb_score"] = np.exp(dbb_net * 5)
    # Buffer Integrity: denominator is fund_cap_net per the reference workbook.
    df["buffer_integrity"] = np.minimum(1.0, np.maximum(0.0, rem_buf_net / cap0))
    df["cap_integrity"] = np.minimum(rem_cap_net / cap0, 1.0)
    # Reset Proximity: asymmetric Gaussian on reference asset return.
    sigma = np.where(ref_ret <= 0, 0.07, 0.05)
    df["reset_proximity"] = np.exp(-(ref_ret ** 2) / (2 * sigma ** 2))
    # Period Position: Gaussian centered at 180 days remaining.
    df["period_position"] = np.exp(-((days - 180) ** 2) / (2 * 75 ** 2))

    df["ecr_2_0"] = (
        (1.5 * df["dbb_score"] + 1.2 * df["reset_proximity"])
        * np.maximum(0.0, df["buffer_integrity"]) ** 0.5
        * np.maximum(0.0, df["cap_integrity"]) ** 0.3
        + 0.3 * df["period_position"]
    )
    return df


def transform(df_raw):
    """Raw FT frame (percent units, FT headers + Strategy Type/Series) ->
    final ECR 2.0 frame (decimals, snake_case headers, new metrics)."""
    df = df_raw.copy()
    for col in DIVIDE_BY_100:
        if col in df.columns:
            df[col] = df[col] / 100.0
    df = df.rename(columns=COLUMN_MAP)
    df = add_ecr_2_0_metrics(df)
    return df[[c for c in FINAL_COLUMNS if c in df.columns]]


# ============================================================================
# Validation against the reference workbook
# ============================================================================

def validate_against_workbook(path):
    """Recompute metrics from the workbook's own (already-decimal) inputs and
    compare to its cached outputs. Returns the worst absolute difference."""
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=list(rows[0]))

    # The workbook is already in decimals, so rename only (no divide).
    expected = df.rename(columns={
        "DBB Score": "x_dbb", "Buffer Integrity": "x_buf", "Cap Integrity": "x_cap",
        "Reset Proximity": "x_reset", "Period Position": "x_period", "ECR 2.0": "x_ecr",
    }).rename(columns=COLUMN_MAP)
    got = add_ecr_2_0_metrics(expected.copy())

    pairs = [("dbb_score", "x_dbb"), ("buffer_integrity", "x_buf"),
             ("cap_integrity", "x_cap"), ("reset_proximity", "x_reset"),
             ("period_position", "x_period"), ("ecr_2_0", "x_ecr")]
    worst = 0.0
    for got_col, exp_col in pairs:
        diff = (got[got_col].astype(float) - got[exp_col].astype(float)).abs().max()
        worst = max(worst, diff)
        print(f"  {got_col:<18} max|Δ| = {diff:.2e}")
    return worst


# ============================================================================
# Data retrieval (reuses the verified fetch machinery)
# ============================================================================

def _fetch_one(ticker, start, end, meta):
    """Fetch one fund's window rows, or None if it has no data in range."""
    hist = fetch_fund_history(ticker, start - pd.Timedelta(days=1))
    hist = (hist.sort_values(["Date", "Remaining Outcome Days"])
                .drop_duplicates("Date", keep="last"))
    win = hist[(hist["Date"] >= start) & (hist["Date"] <= end)].copy()
    if win.empty:
        return None
    win["Strategy Type"] = meta.get(ticker, {}).get("strategy_type")
    win["Series"] = meta.get(ticker, {}).get("series")
    return win


def fetch_raw(start, end, retry_rounds=2):
    """Assemble the raw FT frame (percent units) for all funds across
    [start, end], roll-aware, with Strategy Type / Series attached.

    Funds that error (transient network blips, nsid pulled mid-roll) are
    retried in follow-up rounds so a single hiccup never silently drops a
    fund from a backfill."""
    funds = parse_main_table(fetch_main_table_excel())
    meta = funds.set_index("ticker")[["strategy_type", "series"]].to_dict("index")
    print(f"Fund universe: {len(funds)} tickers")

    frames = []
    pending = list(funds["ticker"])
    no_data = []  # genuinely empty in range (e.g. mid-roll, no nsid) — not retryable
    for attempt in range(retry_rounds + 1):
        if not pending:
            break
        if attempt > 0:
            print(f"  retry round {attempt}: {len(pending)} fund(s) — {', '.join(pending)}")
            time.sleep(5)
        errored = []
        for ticker in pending:
            try:
                win = _fetch_one(ticker, start, end, meta)
                if win is None:
                    no_data.append(ticker)
                else:
                    frames.append(win)
                time.sleep(0.4)
            except Exception as err:  # noqa: BLE001 - retry transient failures next round
                print(f"  {ticker}: ❌ {err}")
                errored.append(ticker)
        pending = errored

    if not frames:
        raise SystemExit("❌ Nothing downloaded.")
    if pending:
        print(f"⚠️  Failed after retries: {', '.join(pending)}")
    if no_data:
        print(f"⚠️  No rows in range (mid-roll / no nsid): {', '.join(no_data)}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Export all funds in the ECR 2.0 format.")
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="single trading day")
    parser.add_argument("--start", metavar="YYYY-MM-DD")
    parser.add_argument("--end", metavar="YYYY-MM-DD")
    parser.add_argument("--validate", action="store_true",
                        help="validate the methodology against the reference workbook and exit")
    args = parser.parse_args()

    if args.validate:
        ref = PROJECT_ROOT / "documents" / "ecr_download_example.xlsx"
        path = ref if ref.exists() else Path(args.date or "")
        print(f"Validating against {path} ...")
        worst = validate_against_workbook(path)
        print(f"\n{'✅ PASS' if worst < 1e-9 else '❌ FAIL'} — worst |Δ| = {worst:.2e}")
        return 0 if worst < 1e-9 else 1

    if args.date:
        start = end = pd.Timestamp(args.date)
        tag = f"{start:%Y-%m-%d}"
    elif args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if end < start:
            raise SystemExit("❌ --end before --start")
        tag = f"{start:%Y-%m-%d}_to_{end:%Y-%m-%d}"
    else:
        raise SystemExit("Provide --date, or --start and --end, or --validate")

    print("=" * 80)
    print(f"ECR 2.0 EXPORT — {start.date()}" + ("" if start == end else f" → {end.date()}"))
    print("=" * 80)

    final = transform(fetch_raw(start, end))
    final = final.sort_values(["date", "ticker"]).reset_index(drop=True)

    n_nan = int(final["ecr_2_0"].isna().sum())
    print(f"\nRows: {len(final)} | funds: {final['ticker'].nunique()} | "
          f"days: {final['date'].nunique()} | null ecr_2_0: {n_nan}")
    print(final.groupby("date").size().to_dict() if start != end else
          f"ecr_2_0 range: {final['ecr_2_0'].min():.3f} – {final['ecr_2_0'].max():.3f}")

    # ISO 8601 dates for unambiguous SQL import (DATE columns expect YYYY-MM-DD).
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUTPUT_DIR / f"ecr_2_0_{tag}.xlsx"
    csv_path = OUTPUT_DIR / f"ecr_2_0_{tag}.csv"
    final.to_excel(xlsx_path, index=False, sheet_name="ECR 2.0")
    final.to_csv(csv_path, index=False)
    print(f"\n✅ {xlsx_path.relative_to(PROJECT_ROOT)}")
    print(f"   {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
