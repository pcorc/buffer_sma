"""
Input Data Updater — F-series buffer ETF data from First Trust
===============================================================

Replaces the data-extraction step that previously lived in the
cost-ratio-enhanced (TypeScript) repo. This is pure Python: it fetches the
same public First Trust endpoints that repo's scraper uses, then updates the
input_data/ files in place:

  1. input_data/data.csv         — daily outcome-period rows per fund
  2. input_data/roll_dates.csv   — 'monthly' column extended with newly
                                   observed roll dates (other columns untouched)
  3. input_data/benchmark_ts.csv — SPY column extended using FT's Reference
                                   Asset Value (= SPY for F-series);
                                   BUFR is left blank for manual entry

Data flow (same as cost-ratio-enhanced):
  - GET ftportfolios.com/Retail/Etf/EtfSummary.aspx?Ticker=XXXX
      → regex the numeric "nsid" fund id out of the page source
  - GET api.ftportfolios.com/api/TargetOutcomePeriodPerformance/UnitedStates/nsid/{nsid}
      → Excel file with the CURRENT outcome period's daily rows
  - same URL + /asofdate/YYYY-MM-DD
      → Excel for a COMPLETED period (used to fill the gap when a fund has
        rolled into a new period since the last update)

"Original Cap Net (%)" / "Original Buffer Net (%)" are derived from each
period's day-1 row (its Remaining Cap Net / Remaining Buffer Net), exactly
like the old pipeline did.

Merge rules:
  - All three files are APPEND-ONLY: existing lines are never rewritten, so
    git diffs show exactly the new data and nothing else.
  - On a roll date the old period ends and the new one starts on the SAME date;
    the new period's day-1 row wins (this matches the existing file and is what
    backtesting/data_pipeline.py expects when it reads caps at roll dates).

Usage:
    python update_input_data.py                    # refresh all input files
    python update_input_data.py --dry-run          # fetch + report, write nothing
    python update_input_data.py --date 2026-06-10  # also export that date's rows
                                                   #   to output/etf_data_2026-06-10.xlsx
    python update_input_data.py --tickers FAPR FJUN  # restrict to specific funds

Requires: pandas, openpyxl (both already used by this repo). Network access
to ftportfolios.com / api.ftportfolios.com.
"""

import argparse
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_CSV = INPUT_DIR / "data.csv"
ROLL_DATES_CSV = INPUT_DIR / "roll_dates.csv"
BENCHMARK_CSV = INPUT_DIR / "benchmark_ts.csv"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SUMMARY_URL = "https://www.ftportfolios.com/Retail/Etf/EtfSummary.aspx?Ticker={ticker}"
API_URL = "https://api.ftportfolios.com/api/TargetOutcomePeriodPerformance/UnitedStates/nsid/{nsid}"
NSID_RE = re.compile(r"TargetOutcomePeriodPerformance/UnitedStates/nsid/(\d+)")

REQUEST_DELAY_SECONDS = 0.5  # politeness delay between First Trust requests
MAX_PERIODS_BACK = 3  # safety cap on how many completed periods to walk back

# Day-1 rows of a new period have ~363-367 remaining outcome days.
DAY1_REMAINING_DAYS_MIN = 350

# Exact column order of input_data/data.csv
CSV_COLUMNS = [
    "Date",
    "Fund",
    "Fund Value (USD)",
    "Fund Return (%)",
    "Reference Asset Value (USD)",
    "Reference Asset Return (%)",
    "Remaining Outcome Days",
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


# ============================================================================
# HTTP helpers
# ============================================================================

def _http_get(url, retries=3):
    """GET a URL with the browser User-Agent, retrying on transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - retry any network failure
            last_err = err
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_err})")


def fetch_nsid(ticker):
    """Extract the numeric nsid fund id from the fund's EtfSummary page."""
    html = _http_get(SUMMARY_URL.format(ticker=ticker)).decode("utf-8", errors="replace")
    match = NSID_RE.search(html)
    if not match:
        raise RuntimeError(f"{ticker}: no nsid found on EtfSummary page "
                           f"(First Trust may have changed the page layout)")
    return match.group(1)


def fetch_period_excel(nsid, asof_date=None):
    """Download one outcome period's Excel file. asof_date=None → current period."""
    url = API_URL.format(nsid=nsid)
    if asof_date is not None:
        url += f"/asofdate/{asof_date:%Y-%m-%d}"
    time.sleep(REQUEST_DELAY_SECONDS)
    return _http_get(url)


# ============================================================================
# Excel parsing
# ============================================================================

def parse_period_excel(content, ticker):
    """
    Parse one First Trust outcome-period Excel into a DataFrame shaped like
    data.csv (Date column as Timestamps; Original Cap/Buffer Net derived
    from the period's first row).
    """
    raw = pd.read_excel(io.BytesIO(content), header=None)

    header_idx = None
    for i in range(min(len(raw), 10)):
        if str(raw.iloc[i, 0]).strip() == "Date":
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"{ticker}: no 'Date' header row found in Excel")

    headers = [str(h).strip() for h in raw.iloc[header_idx]]
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = headers

    # Footer disclaimer rows don't parse as dates → dropped here
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        raise RuntimeError(f"{ticker}: Excel contained no data rows")

    # 'N/A', '--', etc. become NaN, matching the old pipeline's behavior
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    df["Fund"] = ticker

    # Day-1 remaining values ARE the period's original values (no drift yet)
    df["Original Cap Net (%)"] = df["Remaining Cap Net (%)"].iloc[0]
    df["Original Buffer Net (%)"] = df["Remaining Buffer Net (%)"].iloc[0]

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: Excel is missing expected columns {missing} "
                           f"(First Trust may have renamed headers)")
    return df[CSV_COLUMNS]


# ============================================================================
# Per-fund fetching (current period + walk back across rolls if needed)
# ============================================================================

def fetch_fund_history(ticker, last_known_date):
    """
    Fetch the current outcome period for a fund and, if the fund has rolled
    since last_known_date, also the completed period(s) covering the gap.

    A completed period is reached via asofdate = its end date, which equals
    the next period's day-1 date (both periods share the roll date).
    """
    nsid = fetch_nsid(ticker)
    frames = []

    current = parse_period_excel(fetch_period_excel(nsid), ticker)
    frames.append(current)
    period_start = current["Date"].min()
    print(f"    current period: {period_start.date()} → {current['Date'].max().date()} "
          f"({len(current)} rows)")

    walked_back = 0
    while (
        last_known_date is not None
        and period_start > last_known_date + pd.Timedelta(days=1)
        and walked_back < MAX_PERIODS_BACK
    ):
        prev = _fetch_previous_period(ticker, nsid, period_start)
        if prev is None:
            print(f"    ⚠️  {ticker}: could not fetch period before {period_start.date()}; "
                  f"gap after {last_known_date.date()} may remain")
            break
        frames.append(prev)
        print(f"    completed period: {prev['Date'].min().date()} → "
              f"{prev['Date'].max().date()} ({len(prev)} rows)")
        period_start = prev["Date"].min()
        walked_back += 1

    return pd.concat(frames, ignore_index=True)


def _fetch_previous_period(ticker, nsid, period_start):
    """
    Fetch the completed period that ends where the given period starts.
    The end date normally equals period_start exactly; fall back a few
    business days in case the roll landed around a holiday.
    """
    candidates = [period_start] + [period_start - pd.offsets.BDay(n) for n in (1, 2, 3)]
    for asof in candidates:
        try:
            prev = parse_period_excel(fetch_period_excel(nsid, asof_date=asof), ticker)
        except RuntimeError:
            continue
        if prev["Date"].min() < period_start:  # genuinely an earlier period
            return prev
    return None


# ============================================================================
# Merging into the input files
# ============================================================================

def merge_fund_data(df_existing, df_fetched):
    """
    Merge fetched rows into the existing data.

    - Within fetched data, duplicate (Fund, Date) pairs occur on roll dates
      (old period's last row vs new period's day-1 row): keep the day-1 row,
      identified by the larger Remaining Outcome Days.
    - Existing rows are never modified; fetched rows for dates already in the
      file are dropped.

    Returns (df_new_rows, df_merged) with Date still as Timestamps.
    """
    df_fetched = (
        df_fetched.sort_values(["Fund", "Date", "Remaining Outcome Days"])
        .drop_duplicates(["Fund", "Date"], keep="last")
    )

    existing_keys = set(zip(df_existing["Fund"], df_existing["Date"]))
    is_new = [
        (fund, date) not in existing_keys
        for fund, date in zip(df_fetched["Fund"], df_fetched["Date"])
    ]
    df_new = df_fetched[is_new].sort_values(["Date", "Fund"]).reset_index(drop=True)

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    return df_new, df_merged


def format_date_mdy(ts):
    """3/27/2026-style dates (no leading zeros), portable across OSes."""
    return f"{ts.month}/{ts.day}/{ts.year}"


def format_csv_value(v):
    """Numbers the way the existing file writes them: ints without '.0',
    floats in shortest round-trip form, missing values as empty cells."""
    if pd.isna(v):
        return ""
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def append_lines(path, lines):
    """Append lines to a CSV without touching its existing bytes."""
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write("".join(line + "\n" for line in lines))


def append_data_rows(df_new, path):
    lines = []
    for row in df_new.itertuples(index=False):
        record = dict(zip(df_new.columns, row))
        cells = [format_date_mdy(record["Date"]), str(record["Fund"])]
        cells += [format_csv_value(record[col]) for col in CSV_COLUMNS[2:]]
        lines.append(",".join(cells))
    append_lines(path, lines)


def update_roll_dates(df_merged, path, dry_run):
    """
    Append newly observed roll dates to the 'monthly' column.

    Roll dates are detected from the data itself: a fund's Remaining Outcome
    Days jumps back up to ~364 on the first day of a new period. Detecting
    them this way (instead of assuming 3rd Fridays) handles holiday-shifted
    rolls correctly. Only the 'monthly' column is extended; quarterly/
    semi_annual/annual are maintained by hand and left untouched.
    """
    roll_df = pd.read_csv(path)
    monthly = pd.to_datetime(roll_df["monthly"].dropna()).sort_values()
    last_known = monthly.max()

    df_sorted = df_merged.sort_values(["Fund", "Date"])
    days = df_sorted["Remaining Outcome Days"]
    prev_days = days.groupby(df_sorted["Fund"]).shift(1)
    is_period_start = (
        prev_days.notna()
        & (days > prev_days)
        & (days >= DAY1_REMAINING_DAYS_MIN)
    )
    observed = pd.Series(df_sorted.loc[is_period_start, "Date"].unique()).sort_values()
    new_dates = [d for d in observed if d > last_known]

    if not new_dates:
        print("  roll_dates.csv: no new roll dates")
        return

    print(f"  roll_dates.csv: adding {[str(d.date()) for d in new_dates]}")
    if dry_run:
        return

    # Appending "<date>,,," lines only works while monthly is the longest
    # (first) column — true today and asserted so a change can't corrupt the file
    other_cols = [c for c in roll_df.columns if c != "monthly"]
    if list(roll_df.columns)[0] != "monthly" or any(
        roll_df[c].notna().sum() >= len(monthly) for c in other_cols
    ):
        raise RuntimeError("roll_dates.csv layout changed — update this script "
                           "before appending roll dates")
    append_lines(path, [format_date_mdy(d) + "," * len(other_cols) for d in new_dates])


def update_benchmarks(df_merged, path, dry_run):
    """
    Extend benchmark_ts.csv with SPY prices taken from the fund data's
    Reference Asset Value (the F-series reference asset IS SPY).
    BUFR has no automated source and is left blank on new rows.
    """
    bench = pd.read_csv(path)
    bench_dates = pd.to_datetime(bench["Date"])
    last_known = bench_dates.max()

    spy = (
        df_merged[df_merged["Date"] > last_known]
        .dropna(subset=["Reference Asset Value (USD)"])
        .groupby("Date")["Reference Asset Value (USD)"]
        .first()
        .sort_index()
    )
    if spy.empty:
        print("  benchmark_ts.csv: no new SPY rows")
        return

    print(f"  benchmark_ts.csv: adding {len(spy)} SPY rows "
          f"({spy.index.min().date()} → {spy.index.max().date()}); BUFR left blank")
    if dry_run:
        return

    if list(bench.columns) != ["Date", "SPY", "BUFR"]:
        raise RuntimeError("benchmark_ts.csv layout changed — update this script "
                           "before appending benchmark rows")
    append_lines(path, [
        f"{format_date_mdy(d)},{format_csv_value(v)}," for d, v in spy.items()
    ])


def export_date_to_excel(df_merged, target_date, out_dir):
    """Write one date's rows (all funds) to an Excel file — returns the path."""
    rows = df_merged[df_merged["Date"] == target_date].copy()
    if rows.empty:
        available = df_merged.loc[df_merged["Date"] <= target_date, "Date"]
        hint = (f" Nearest earlier date with data: {available.max().date()}."
                if not available.empty else "")
        raise SystemExit(f"❌ No rows for {target_date.date()} (markets closed, or date "
                         f"not yet scraped).{hint}")

    rows = rows.sort_values("Fund")
    rows["Date"] = rows["Date"].map(format_date_mdy)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"etf_data_{target_date:%Y-%m-%d}.xlsx"
    rows.to_excel(out_path, index=False, sheet_name=f"{target_date:%Y-%m-%d}")
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Update input_data/ from First Trust's public API.")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="only update these funds (default: every fund in data.csv)")
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD",
                        help="also export this date's rows to output/etf_data_<date>.xlsx")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, but do not write any files")
    args = parser.parse_args()

    target_date = pd.Timestamp(args.date) if args.date else None

    print("=" * 80)
    print("UPDATING INPUT DATA FROM FIRST TRUST")
    print("=" * 80)

    df_existing = pd.read_csv(DATA_CSV, low_memory=False)
    df_existing["Date"] = pd.to_datetime(df_existing["Date"])
    tickers = args.tickers or sorted(df_existing["Fund"].unique())
    print(f"Existing data: {len(df_existing)} rows through "
          f"{df_existing['Date'].max().date()} | funds: {', '.join(tickers)}\n")

    fetched_frames = []
    failures = []
    for ticker in tickers:
        print(f"  {ticker}:")
        fund_dates = df_existing.loc[df_existing["Fund"] == ticker, "Date"]
        last_known = fund_dates.max() if not fund_dates.empty else None
        try:
            fetched_frames.append(fetch_fund_history(ticker, last_known))
        except Exception as err:  # noqa: BLE001 - keep going, report at the end
            print(f"    ❌ {err}")
            failures.append(ticker)

    if not fetched_frames:
        raise SystemExit("❌ Nothing fetched — aborting without changes.")

    df_new, df_merged = merge_fund_data(df_existing, pd.concat(fetched_frames,
                                                               ignore_index=True))

    print("\n" + "-" * 80)
    if df_new.empty:
        print("✅ data.csv already up to date — no new rows.")
    else:
        summary = df_new.groupby("Fund")["Date"].agg(["min", "max", "count"])
        for fund, row in summary.iterrows():
            print(f"  {fund}: +{row['count']} rows "
                  f"({row['min'].date()} → {row['max'].date()})")
        print(f"  data.csv: +{len(df_new)} rows total → new last date "
              f"{df_merged['Date'].max().date()}")
        if not args.dry_run:
            append_data_rows(df_new, DATA_CSV)

    update_roll_dates(df_merged, ROLL_DATES_CSV, args.dry_run)
    update_benchmarks(df_merged, BENCHMARK_CSV, args.dry_run)

    if target_date is not None:
        out_path = export_date_to_excel(df_merged, target_date, OUTPUT_DIR)
        print(f"  Excel export: {out_path.relative_to(PROJECT_ROOT)}")

    if failures:
        print(f"\n⚠️  Failed funds (unchanged in data.csv): {', '.join(failures)}")
    print("\n" + "=" * 80)
    print("✅ DRY RUN COMPLETE — no files written" if args.dry_run else "✅ UPDATE COMPLETE")
    print("=" * 80)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
