"""
ECR 2.0 full-history export for an EXPLICIT fund list (tolerant parser)
======================================================================

Pulls every outcome period back to inception for a specified set of tickers
(including exotic structures the standard buffer parser can't read) and stacks
them into one ECR 2.0 CSV.

Unlike export_ecr_2_0_history.py (which uses the strict buffer-fund parser and
the strategy-filtered main-table universe), this:
  * targets an explicit ticker list with caller-supplied sleeve/series, so funds
    outside the buffer-strategy universe (Dual Directional, Digital Return,
    Enhance & Moderate Buffer, Premium Income, Max Buffer) are included;
  * parses leniently: maps whatever standard columns a fund's Excel has and
    null-fills the rest, so differing layouts don't error.

ECR 2.0 is only defined where its inputs exist. Buffer / Max Buffer / Enhance &
Moderate / Dual Directional funds get a real ecr_2_0; Digital Return and Premium
Income funds (no cap / no remaining-buffer columns) get null ecr_2_0 and null in
the columns their layout omits.

Usage:
    python export_ecr_2_0_requested.py            # all funds in FUND_LIST below
Output: output/ecr_2_0_history_requested_<asof>.csv
"""

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from update_input_data import CSV_COLUMNS, fetch_nsid, fetch_period_excel
from export_ecr_2_0 import transform

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MAX_PERIODS = 30

# (ticker, fund_sleeve, series) — exactly as requested.
FUND_LIST = [
    ("APXM", "Max Buffer", "April"), ("AUGM", "Max Buffer", "August"),
    ("DECM", "Max Buffer", "December"),
    ("DGAP", "Buffer & Digital Return", "April"), ("DGJA", "Buffer & Digital Return", "January"),
    ("DGJL", "Buffer & Digital Return", "July"), ("DGOC", "Buffer & Digital Return", "October"),
    ("DLAG", "Dual Directional Buffer", "August"), ("DLFE", "Dual Directional Buffer", "February"),
    ("DLMY", "Dual Directional Buffer", "May"), ("DLNV", "Dual Directional Buffer", "November"),
    ("DQJN", "Dual Directional Buffer", "June"), ("DQSE", "Dual Directional Buffer", "September"),
    ("FAPR", "Buffer", "April"), ("FAUG", "Buffer", "August"), ("FDEC", "Buffer", "December"),
    ("FEBM", "Max Buffer", "February"), ("FFEB", "Buffer", "February"), ("FHDG", "Buffer", "Quarterly"),
    ("FJAN", "Buffer", "January"), ("FJUL", "Buffer", "July"), ("FJUN", "Buffer", "June"),
    ("FMAR", "Buffer", "March"), ("FMAY", "Buffer", "May"), ("FNOV", "Buffer", "November"),
    ("FOCT", "Buffer", "October"), ("FSEP", "Buffer", "September"),
    ("JANM", "Max Buffer", "January"), ("JULM", "Max Buffer", "July"), ("JUNM", "Max Buffer", "June"),
    ("MARM", "Max Buffer", "March"), ("MAYM", "Max Buffer", "May"), ("NOVM", "Max Buffer", "November"),
    ("OCTM", "Max Buffer", "October"), ("SEPM", "Max Buffer", "September"),
    ("SQMX", "Max Buffer", "Quarterly"),
    ("XAPR", "Enhance & Moderate Buffer", "April"), ("XAUG", "Enhance & Moderate Buffer", "August"),
    ("XDEC", "Enhance & Moderate Buffer", "December"), ("XFEB", "Enhance & Moderate Buffer", "February"),
    ("XIDE", "Buffer and Premium Income", "December"), ("XIJN", "Buffer and Premium Income", "June"),
    ("XIMR", "Buffer and Premium Income", "March"), ("XISE", "Buffer and Premium Income", "September"),
    ("XJAN", "Enhance & Moderate Buffer", "January"), ("XJUL", "Enhance & Moderate Buffer", "July"),
    ("XJUN", "Enhance & Moderate Buffer", "June"), ("XMAR", "Enhance & Moderate Buffer", "March"),
    ("XMAY", "Enhance & Moderate Buffer", "May"), ("XNOV", "Enhance & Moderate Buffer", "November"),
    ("XOCT", "Enhance & Moderate Buffer", "October"), ("XSEP", "Enhance & Moderate Buffer", "September"),
]


def parse_lenient(content, ticker):
    """Parse a period Excel, mapping standard columns that exist and null-filling
    the rest. Derives Original Cap/Buffer Net from the period's day-1 row when the
    corresponding Remaining columns are present. Returns a frame with CSV_COLUMNS."""
    raw = pd.read_excel(io.BytesIO(content), header=None)
    header_idx = next((i for i in range(min(len(raw), 10))
                       if str(raw.iloc[i, 0]).strip() == "Date"), None)
    if header_idx is None:
        raise RuntimeError(f"{ticker}: no 'Date' header row")

    headers = [str(h).strip() for h in raw.iloc[header_idx]]
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = headers
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        raise RuntimeError(f"{ticker}: no data rows")
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Fund"] = ticker

    # Derive originals from day-1 when available (same rule as the F-series parser).
    if "Remaining Cap Net (%)" in df.columns:
        df["Original Cap Net (%)"] = df["Remaining Cap Net (%)"].iloc[0]
    if "Remaining Buffer Net (%)" in df.columns:
        df["Original Buffer Net (%)"] = df["Remaining Buffer Net (%)"].iloc[0]

    # Ensure every standard column exists; null-fill the ones this layout omits
    # with float NaN (not pd.NA) so numpy ufuncs in the transform work.
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[CSV_COLUMNS].copy()
    for col in CSV_COLUMNS:
        if col not in ("Date", "Fund"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _fetch_prev(nsid, ticker, period_start):
    for asof in [period_start] + [period_start - pd.offsets.BDay(n) for n in (1, 2, 3)]:
        try:
            prev = parse_lenient(fetch_period_excel(nsid, asof_date=asof), ticker)
        except (RuntimeError, ValueError):
            continue
        if prev["Date"].min() < period_start:
            return prev
    return None


def fetch_full_history(ticker):
    nsid = fetch_nsid(ticker)
    frames = [parse_lenient(fetch_period_excel(nsid), ticker)]
    period_start = frames[0]["Date"].min()
    for _ in range(MAX_PERIODS):
        prev = _fetch_prev(nsid, ticker, period_start)
        if prev is None or prev["Date"].min() >= period_start:
            break
        frames.append(prev)
        period_start = prev["Date"].min()
        time.sleep(0.4)
    hist = pd.concat(frames, ignore_index=True)
    return (hist.sort_values(["Date", "Remaining Outcome Days"])
                .drop_duplicates("Date", keep="last"))


def main():
    print("=" * 78)
    print(f"ECR 2.0 FULL-HISTORY — {len(FUND_LIST)} requested funds")
    print("=" * 78)

    frames, pending = [], list(FUND_LIST)
    done = set()
    for attempt in range(3):
        if not pending:
            break
        if attempt > 0:
            print(f"\n  retry round {attempt}: {len(pending)} fund(s)")
            time.sleep(5)
        errored = []
        for ticker, sleeve, series in pending:
            try:
                hist = fetch_full_history(ticker)
                hist["Strategy Type"] = sleeve
                hist["Series"] = series
                frames.append(hist)
                done.add(ticker)
                print(f"  {ticker:5} {sleeve:26} {len(hist):5} rows "
                      f"({hist['Date'].min().date()} → {hist['Date'].max().date()})", flush=True)
                time.sleep(0.6)
            except Exception as err:  # noqa: BLE001
                print(f"  {ticker:5} ❌ {err}", flush=True)
                errored.append((ticker, sleeve, series))
        pending = errored

    if not frames:
        raise SystemExit("❌ Nothing downloaded.")

    final = transform(pd.concat(frames, ignore_index=True)).sort_values(["ticker", "date"]).reset_index(drop=True)
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")

    nn = final.groupby("ticker")["ecr_2_0"].apply(lambda s: s.notna().any())
    with_ecr = sorted(nn[nn].index)
    without_ecr = sorted(nn[~nn].index)
    print("\n" + "-" * 78)
    print(f"Stacked {len(final)} rows | {final['ticker'].nunique()} funds | "
          f"{final['date'].min()} → {final['date'].max()}")
    print(f"  funds WITH ecr_2_0: {len(with_ecr)}")
    print(f"  funds with NULL ecr_2_0 (digital-return / premium-income layout): {without_ecr}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asof = datetime.now().strftime("%Y-%m-%d")
    out = OUTPUT_DIR / f"ecr_2_0_history_requested_{asof}.csv"
    final.to_csv(out, index=False)
    print(f"\n✅ {out}")
    if pending:
        print(f"⚠️  Failed after retries: {[t for t, _, _ in pending]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
