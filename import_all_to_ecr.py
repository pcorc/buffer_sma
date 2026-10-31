"""
Import all input_data sources into the `ecr` MySQL schema
========================================================

Loads each source file into its matching ecr.* table, idempotently (upsert on
each table's primary key, so re-running never duplicates), and records a
provenance row in ecr.data_sources for every load.

Source file  ->  table
---------------------------------------------------------------------------
  input_data/data.csv                             -> ecr.legacy_fund_metrics_pct   (pct values kept as-is)
  input_data/benchmark_ts.csv                     -> ecr.benchmark_prices
  input_data/roll_dates.csv                       -> ecr.roll_dates                (wide -> unpivoted)
  input_data/sp500_regimes.csv                    -> ecr.sp500_regimes
  input_data/drawdown_periods_website_metrics.xlsx-> ecr.drawdown_period_metrics_pct
  input_data/benchmark_definitions.xlsx           -> ecr.benchmark_basket_weights  (list cells -> rows)
  <ECR 2.0 csv>                                   -> ecr.website_metrics           (see --website-csv)
  input_data/fund_catalog.csv                     -> ecr.fund_catalog              (only if the file exists)

Connection (no secrets in this file) — set before running:
    export FT_DB_HOST=... FT_DB_PORT=25060 FT_DB_USER=... FT_DB_PASSWORD=... FT_DB_NAME=ecr
Deps:  pip install pymysql pandas openpyxl

Usage:
    python import_all_to_ecr.py                         # load everything found
    python import_all_to_ecr.py --website-csv output/ecr_2_0_history_nonF_2026-10-31.csv
    python import_all_to_ecr.py --only roll_dates benchmark_prices
    python import_all_to_ecr.py --dry-run              # parse + report, no DB
"""

import argparse
import ast
import glob
import hashlib
import math
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT = PROJECT_ROOT / "input_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prep_dates(df, col, also_required=()):
    """Coerce a date column to YYYY-MM-DD strings and drop rows whose date (or
    any other required key) is missing — handles trailing blank rows / bad dates."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df[df[col].notna()]
    for c in also_required:
        df = df[df[c].notna()]
    df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


def _clean(v, as_int=False):
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(round(v)) if as_int else v


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-table loaders: each returns (columns, rows, provenance_dict)
# ---------------------------------------------------------------------------

PCT_MAP = {
    "Date": "metric_date", "Fund": "fund",
    "Fund Value (USD)": "fund_value_usd", "Fund Return (%)": "fund_return_pct",
    "Reference Asset Value (USD)": "reference_asset_value_usd",
    "Reference Asset Return (%)": "reference_asset_return_pct",
    "Remaining Outcome Days": "remaining_outcome_days",
    "Remaining Cap (%)": "remaining_cap_pct", "Remaining Cap Net (%)": "remaining_cap_net_pct",
    "Reference Asset Return to Realize Cap (%)": "reference_asset_return_to_realize_cap_pct",
    "Remaining Buffer (%)": "remaining_buffer_pct", "Remaining Buffer Net (%)": "remaining_buffer_net_pct",
    "Downside Before Buffer (%)": "downside_before_buffer_pct",
    "Downside Before Buffer Net (%)": "downside_before_buffer_net_pct",
    "Reference Asset to Buffer End (%)": "reference_asset_to_buffer_end_pct",
    "Unrealized Option Payoff (%)": "unrealized_option_payoff_pct",
    "Unrealized Option Payoff Net (%)": "unrealized_option_payoff_net_pct",
    "Original Cap Net (%)": "original_cap_net_pct", "Original Buffer Net (%)": "original_buffer_net_pct",
}


def _load_pct(df, columns, source):
    df = df.rename(columns=PCT_MAP)[columns].copy()
    df = _prep_dates(df, "metric_date", also_required=["fund"])
    rows = [tuple(_clean(v, as_int=(c == "remaining_outcome_days"))
                  for c, v in zip(columns, rec))
            for rec in df.itertuples(index=False, name=None)]
    prov = {"row_count": len(rows), "date_min": df["metric_date"].min(),
            "date_max": df["metric_date"].max(), "source_file": source}
    return columns, rows, prov


def load_legacy_fund_metrics_pct():
    src = INPUT / "data.csv"
    cols = ["metric_date", "fund", "fund_value_usd", "fund_return_pct",
            "reference_asset_value_usd", "reference_asset_return_pct", "remaining_outcome_days",
            "remaining_cap_pct", "remaining_cap_net_pct", "reference_asset_return_to_realize_cap_pct",
            "remaining_buffer_pct", "remaining_buffer_net_pct", "downside_before_buffer_pct",
            "downside_before_buffer_net_pct", "reference_asset_to_buffer_end_pct",
            "unrealized_option_payoff_pct", "unrealized_option_payoff_net_pct",
            "original_cap_net_pct", "original_buffer_net_pct"]
    return _load_pct(pd.read_csv(src), cols, src.name)


def load_drawdown_period_metrics_pct():
    src = INPUT / "drawdown_periods_website_metrics.xlsx"
    cols = ["metric_date", "fund", "fund_value_usd", "fund_return_pct",
            "reference_asset_value_usd", "reference_asset_return_pct", "remaining_outcome_days",
            "remaining_cap_pct", "remaining_cap_net_pct", "reference_asset_return_to_realize_cap_pct",
            "remaining_buffer_pct", "remaining_buffer_net_pct", "downside_before_buffer_pct",
            "downside_before_buffer_net_pct", "reference_asset_to_buffer_end_pct",
            "unrealized_option_payoff_pct", "unrealized_option_payoff_net_pct"]
    return _load_pct(pd.read_excel(src), cols, src.name)


def load_benchmark_prices():
    src = INPUT / "benchmark_ts.csv"
    df = pd.read_csv(src).rename(columns={"Date": "price_date", "SPY": "spy", "BUFR": "bufr"})
    df = _prep_dates(df, "price_date")
    cols = ["price_date", "spy", "bufr"]
    rows = [tuple(_clean(v) for v in rec) for rec in df[cols].itertuples(index=False, name=None)]
    return cols, rows, {"row_count": len(rows), "date_min": df["price_date"].min(),
                        "date_max": df["price_date"].max(), "source_file": src.name}


def load_roll_dates():
    src = INPUT / "roll_dates.csv"
    df = pd.read_csv(src)
    seen, rows = set(), []
    for cadence in ["monthly", "quarterly", "semi_annual", "annual"]:
        if cadence in df.columns:
            for d in pd.to_datetime(df[cadence].dropna()):
                key = (cadence, d.strftime("%Y-%m-%d"))
                if key not in seen:          # roll_dates.csv has intra-column dups
                    seen.add(key)
                    rows.append(key)
    dates = [r[1] for r in rows]
    return ["cadence", "roll_date"], rows, {"row_count": len(rows),
            "date_min": min(dates), "date_max": max(dates), "source_file": src.name}


def load_sp500_regimes():
    src = INPUT / "sp500_regimes.csv"
    df = pd.read_csv(src).rename(columns={"Date": "price_date", "Close": "sp500_close",
                                          "Regimes": "regime"})
    df = _prep_dates(df, "price_date", also_required=["sp500_close"])  # sp500_close is NOT NULL
    cols = ["price_date", "sp500_close", "regime"]
    rows = [(r.price_date, _clean(r.sp500_close), _clean(r.regime, as_int=True))
            for r in df[cols].itertuples(index=False)]
    return cols, rows, {"row_count": len(rows), "date_min": df["price_date"].min(),
                        "date_max": df["price_date"].max(), "source_file": src.name}


def load_benchmark_basket_weights():
    src = INPUT / "benchmark_definitions.xlsx"
    df = pd.read_excel(src).dropna(subset=["Rebalance_Date", "Funds", "Weights"])
    rows = []
    for r in df.itertuples(index=False):
        rebal = pd.to_datetime(r.Rebalance_Date).strftime("%Y-%m-%d")
        funds = ast.literal_eval(r.Funds) if isinstance(r.Funds, str) else r.Funds
        weights = ast.literal_eval(r.Weights) if isinstance(r.Weights, str) else r.Weights
        if len(funds) != len(weights):
            raise SystemExit(f"❌ {src.name}: funds/weights length mismatch on {rebal}")
        for t, w in zip(funds, weights):
            rows.append((rebal, r.Rebalance_Frequency, str(t).strip(), float(w)))
    dates = [x[0] for x in rows]
    return (["rebalance_date", "rebalance_frequency", "ticker", "weight"], rows,
            {"row_count": len(rows), "date_min": min(dates), "date_max": max(dates),
             "source_file": src.name})


def load_fund_catalog():
    src = INPUT / "fund_catalog.csv"
    if not src.exists():
        return None
    df = pd.read_csv(src).rename(columns={"group": "fund_group"})
    cols = ["ticker", "series", "fund_group", "outcome_type", "outcome_month", "strategy",
            "buffer_start", "buffer_end", "reference_asset", "currency"]
    df = df[cols].copy()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    rows = [tuple(_clean(v) for v in rec) for rec in df.itertuples(index=False, name=None)]
    return cols, rows, {"row_count": len(rows), "date_min": None, "date_max": None,
                        "source_file": src.name}


WEBSITE_COLUMNS = [
    "date", "ticker", "fund_sleeve", "series",
    "fund_value", "fund_return", "reference_asset_value", "reference_asset_return",
    "remaining_outcome_days", "remaining_cap", "remaining_cap_net",
    "reference_asset_return_to_realize_cap", "remaining_buffer", "remaining_buffer_net",
    "downside_before_buffer", "downside_before_buffer_net", "reference_asset_to_buffer_end",
    "unrealized_option_payoff", "unrealized_option_payoff_net", "fund_cap_net", "buffer_net",
    "dbb_score", "buffer_integrity", "cap_integrity", "reset_proximity",
    "period_position", "ecr_2_0",
]


def load_website_metrics(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in WEBSITE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ {csv_path}: website_metrics CSV missing columns {missing}")
    df = df[WEBSITE_COLUMNS].copy()
    df = _prep_dates(df, "date", also_required=["ticker"])
    rows = [tuple(_clean(v, as_int=(c == "remaining_outcome_days")) for c, v in zip(WEBSITE_COLUMNS, rec))
            for rec in df.itertuples(index=False, name=None)]
    return WEBSITE_COLUMNS, rows, {"row_count": len(rows), "date_min": df["date"].min(),
                                   "date_max": df["date"].max(), "source_file": Path(csv_path).name}


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def upsert(conn, table, columns, rows, keys, batch=1000):
    col_sql = ", ".join(f"`{c}`" for c in columns)
    ph = ", ".join(["%s"] * len(columns))
    upd = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns if c not in keys)
    if upd:
        sql = f"INSERT INTO `{table}` ({col_sql}) VALUES ({ph}) ON DUPLICATE KEY UPDATE {upd}"
    else:
        # all columns are the primary key — nothing to update, just skip dups
        sql = f"INSERT IGNORE INTO `{table}` ({col_sql}) VALUES ({ph})"
    with conn.cursor() as cur:
        for i in range(0, len(rows), batch):
            cur.executemany(sql, rows[i:i + batch])
    conn.commit()


def record_provenance(conn, table, prov, notes):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO `data_sources` (target_table, source_file, row_count, date_min, date_max, notes) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (table, prov["source_file"], prov["row_count"], prov["date_min"], prov["date_max"], notes))
    conn.commit()


# table -> (loader, primary key columns)
REGISTRY = {
    "legacy_fund_metrics_pct": (load_legacy_fund_metrics_pct, ["fund", "metric_date"]),
    "benchmark_prices": (load_benchmark_prices, ["price_date"]),
    "roll_dates": (load_roll_dates, ["cadence", "roll_date"]),
    "sp500_regimes": (load_sp500_regimes, ["price_date"]),
    "drawdown_period_metrics_pct": (load_drawdown_period_metrics_pct, ["fund", "metric_date"]),
    "benchmark_basket_weights": (load_benchmark_basket_weights,
                                 ["rebalance_date", "rebalance_frequency", "ticker"]),
    "fund_catalog": (load_fund_catalog, ["ticker"]),
}


def connect():
    import pymysql
    miss = [v for v in ("FT_DB_HOST", "FT_DB_USER", "FT_DB_PASSWORD", "FT_DB_NAME") if not os.environ.get(v)]
    if miss:
        raise SystemExit(f"❌ Missing env vars: {', '.join(miss)}")
    return pymysql.connect(host=os.environ["FT_DB_HOST"], port=int(os.environ.get("FT_DB_PORT", 25060)),
                           user=os.environ["FT_DB_USER"], passwd=os.environ["FT_DB_PASSWORD"],
                           db=os.environ["FT_DB_NAME"])


def main():
    ap = argparse.ArgumentParser(description="Load all input_data sources into the ecr schema.")
    ap.add_argument("--website-csv", help="ECR 2.0 CSV to load into ecr.website_metrics")
    ap.add_argument("--only", nargs="+", help="only these tables")
    ap.add_argument("--dry-run", action="store_true", help="parse + report, no DB writes")
    args = ap.parse_args()

    # Resolve website_metrics source (explicit, else newest output/ecr_2_0*.csv)
    website_csv = args.website_csv
    if not website_csv:
        m = sorted(glob.glob(str(PROJECT_ROOT / "output" / "ecr_2_0*.csv")))
        website_csv = m[-1] if m else None

    plan = []  # (table, columns, rows, prov, keys, notes)
    skipped = []

    for table, (loader, keys) in REGISTRY.items():
        if args.only and table not in args.only:
            continue
        result = loader()
        if result is None:
            skipped.append((table, "source file not found (e.g. input_data/fund_catalog.csv)"))
            continue
        cols, rows, prov = result
        plan.append((table, cols, rows, prov, keys, f"loaded from {prov['source_file']}"))

    if (not args.only) or ("website_metrics" in (args.only or [])):
        if website_csv:
            cols, rows, prov = load_website_metrics(website_csv)
            plan.append(("website_metrics", cols, rows, prov, ["date", "ticker"],
                         f"loaded from {prov['source_file']}"))
        else:
            skipped.append(("website_metrics", "no ECR 2.0 CSV found; pass --website-csv"))

    print("=" * 78)
    print("LOAD PLAN")
    print("=" * 78)
    for table, cols, rows, prov, keys, _ in plan:
        span = f"{prov['date_min']}..{prov['date_max']}" if prov["date_min"] else "-"
        print(f"  {table:<30} {prov['row_count']:>7} rows  [{span}]  <- {prov['source_file']}")
    for table, why in skipped:
        print(f"  {table:<30}  SKIPPED — {why}")

    if args.dry_run:
        print("\n✅ DRY RUN — parsed all sources, no DB writes.")
        return 0

    conn = connect()
    try:
        for table, cols, rows, prov, keys, notes in plan:
            upsert(conn, table, cols, rows, keys)
            record_provenance(conn, table, prov, notes)
            print(f"  ✅ {table}: upserted {len(rows)} rows")
    finally:
        conn.close()
    print("\n✅ Done.")
    if skipped:
        print("⚠️  Not loaded: " + "; ".join(f"{t} ({w})" for t, w in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
