"""
Import ECR 2.0 CSV into MySQL  ->  ecr.website_metrics   (pymysql, standalone)
=============================================================================

Loads an ECR 2.0 CSV (from export_ecr_2_0.py / export_ecr_2_0_history.py) into
the `ecr.website_metrics` table, upserting on the (date, ticker) primary key
so re-running on the same or overlapping data never duplicates rows.

Deps:  pip install pymysql pandas

Connection comes from environment variables (no secrets in this file):
    export FT_DB_HOST=your-host.db.ondigitalocean.com
    export FT_DB_PORT=25060
    export FT_DB_USER=youruser
    export FT_DB_PASSWORD=yourpassword
    export FT_DB_NAME=ecr            # schema that holds website_metrics

Usage:
    python import_website_metrics.py --csv "path/to/ecr_2_0_history_nonF_2026-10-31.csv"
    python import_website_metrics.py --csv "path/to/file.csv" --create-table   # first time only
    python import_website_metrics.py --csv "path/to/file.csv" --dry-run         # validate only
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = "ecr"
DEFAULT_TABLE = "website_metrics"

# 27 columns, in the exact order of ecr.website_metrics / the ECR 2.0 CSV.
EXPECTED_COLUMNS = [
    "date", "ticker", "fund_sleeve", "series",
    "fund_value", "fund_return", "reference_asset_value", "reference_asset_return",
    "remaining_outcome_days", "remaining_cap", "remaining_cap_net",
    "reference_asset_return_to_realize_cap", "remaining_buffer", "remaining_buffer_net",
    "downside_before_buffer", "downside_before_buffer_net", "reference_asset_to_buffer_end",
    "unrealized_option_payoff", "unrealized_option_payoff_net", "fund_cap_net", "buffer_net",
    "dbb_score", "buffer_integrity", "cap_integrity", "reset_proximity",
    "period_position", "ecr_2_0",
]
UPSERT_KEYS = ["date", "ticker"]          # matches PRIMARY KEY (`date`, ticker)
INTEGER_COLUMNS = {"remaining_outcome_days"}

# Exact DDL supplied for the table (used only by --create-table).
CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  `date`                                 DATE           NOT NULL,
  ticker                                 VARCHAR(10)    NOT NULL,
  fund_sleeve                            VARCHAR(32)    NOT NULL,
  series                                 VARCHAR(16)    NULL,
  fund_value                             DECIMAL(12,4)  NULL,
  fund_return                            DECIMAL(10,6)  NULL,
  reference_asset_value                  DECIMAL(12,4)  NULL,
  reference_asset_return                 DECIMAL(10,6)  NULL,
  remaining_outcome_days                 SMALLINT       NULL,
  remaining_cap                          DECIMAL(10,6)  NULL,
  remaining_cap_net                      DECIMAL(10,6)  NULL,
  reference_asset_return_to_realize_cap  DECIMAL(10,6)  NULL,
  remaining_buffer                       DECIMAL(10,6)  NULL,
  remaining_buffer_net                   DECIMAL(10,6)  NULL,
  downside_before_buffer                 DECIMAL(10,6)  NULL,
  downside_before_buffer_net             DECIMAL(10,6)  NULL,
  reference_asset_to_buffer_end          DECIMAL(10,6)  NULL,
  unrealized_option_payoff               DECIMAL(10,6)  NULL,
  unrealized_option_payoff_net           DECIMAL(10,6)  NULL,
  fund_cap_net                           DECIMAL(10,6)  NULL,
  buffer_net                             DECIMAL(10,6)  NULL,
  dbb_score                              DECIMAL(12,8)  NULL,
  buffer_integrity                       DECIMAL(12,8)  NULL,
  cap_integrity                          DECIMAL(12,8)  NULL,
  reset_proximity                        DECIMAL(12,8)  NULL,
  period_position                        DECIMAL(12,8)  NULL,
  ecr_2_0                                DECIMAL(12,8)  NULL,
  PRIMARY KEY (`date`, ticker),
  KEY ix_website_metrics_ticker_date (ticker, `date`)
)
"""


def connect():
    import pymysql
    missing = [v for v in ("FT_DB_HOST", "FT_DB_USER", "FT_DB_PASSWORD", "FT_DB_NAME")
               if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"❌ Missing env vars: {', '.join(missing)}. See this file's header.")
    return pymysql.connect(
        host=os.environ["FT_DB_HOST"],
        port=int(os.environ.get("FT_DB_PORT", 25060)),
        user=os.environ["FT_DB_USER"],
        passwd=os.environ["FT_DB_PASSWORD"],
        db=os.environ["FT_DB_NAME"],
    )


def upsert_sql(table):
    cols = ", ".join(f"`{c}`" for c in EXPECTED_COLUMNS)
    placeholders = ", ".join(["%s"] * len(EXPECTED_COLUMNS))
    updates = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in EXPECTED_COLUMNS if c not in UPSERT_KEYS)
    return f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"


def load_rows(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ CSV missing expected columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    def clean(col, v):
        if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            return None
        if col in INTEGER_COLUMNS:
            return int(v)
        return v

    rows = [tuple(clean(c, v) for c, v in zip(EXPECTED_COLUMNS, rec))
            for rec in df.itertuples(index=False, name=None)]
    return df, rows


def main():
    parser = argparse.ArgumentParser(description="Load ECR 2.0 CSV into ecr.website_metrics (MySQL).")
    parser.add_argument("--csv", help="CSV to load (default: latest output/ecr_2_0*.csv)")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--create-table", action="store_true", help="create the table first (first run only)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="validate CSV only; no DB connection")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        matches = sorted(glob.glob(str(PROJECT_ROOT / "output" / "ecr_2_0*.csv")))
        if not matches:
            raise SystemExit("❌ No output/ecr_2_0*.csv found; pass --csv.")
        csv_path = Path(matches[-1])
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV not found: {csv_path}")

    df, rows = load_rows(csv_path)
    print(f"CSV: {csv_path}")
    print(f"  {len(rows)} rows | {df['ticker'].nunique()} funds | "
          f"{df['date'].nunique()} dates ({df['date'].min()} → {df['date'].max()})")
    print(f"  target: {args.schema}.{args.table} (upsert key: {', '.join(UPSERT_KEYS)})")

    if args.dry_run:
        print("\n✅ DRY RUN — CSV valid, 27 columns present. No DB connection made.")
        return 0

    conn = connect()
    try:
        with conn.cursor() as cur:
            if args.create_table:
                cur.execute(CREATE_TABLE_DDL.format(table=args.table))
                print(f"  ensured {args.schema}.{args.table} exists")
            sql = upsert_sql(args.table)
            for i in range(0, len(rows), args.batch_size):
                cur.executemany(sql, rows[i:i + args.batch_size])
                print(f"    upserted {min(i + args.batch_size, len(rows))}/{len(rows)}", flush=True)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{args.table}`")
            total = cur.fetchone()[0]
        print(f"\n✅ Upserted {len(rows)} rows. {args.schema}.{args.table} now holds {total} rows.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
