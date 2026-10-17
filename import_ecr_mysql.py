"""
Import ECR 2.0 CSV into MySQL  ->  first_trust_usa.ecr   (pymysql, standalone)
==============================================================================

Loads an ecr_2_0_*.csv (from export_ecr_2_0.py) into a MySQL table, upserting
on (ticker, date) so re-running a date range never duplicates rows.

Matches the project's existing pymysql connection style. Only deps: pymysql, pandas
    pip install pymysql pandas

--------------------------------------------------------------------------
CONNECTION
--------------------------------------------------------------------------
Credentials are read from environment variables so nothing secret lives in
this file (it is committed to git). Set them before running:

    export FT_DB_HOST=your-host.db.ondigitalocean.com
    export FT_DB_PORT=25060
    export FT_DB_USER=youruser
    export FT_DB_PASSWORD=yourpassword
    export FT_DB_NAME=first_trust_usa      # the schema/database to write to

(On Windows use `set FT_DB_HOST=...` per line, or set them in the System
Environment Variables dialog.)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    # validate the CSV only (no DB connection)
    python import_ecr_mysql.py --csv "path\\to\\ecr_2_0_2026-10-02_to_2026-10-15.csv" --dry-run

    # create the table if needed, then load
    python import_ecr_mysql.py --csv "path\\to\\file.csv" --create-table

    # load into an existing table
    python import_ecr_mysql.py --csv "path\\to\\file.csv"

Defaults to the most recent output/ecr_2_0_*.csv if --csv is omitted.
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = "ecr"

# 27 columns produced by export_ecr_2_0.py, with MySQL types.
TEXT_COLUMNS = {"ticker": "VARCHAR(16)", "fund_sleeve": "VARCHAR(64)", "series": "VARCHAR(32)"}
INTEGER_COLUMNS = {"remaining_outcome_days"}
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
UPSERT_KEYS = ["ticker", "date"]


def connect():
    """Open a pymysql connection from FT_DB_* environment variables."""
    import pymysql
    missing = [v for v in ("FT_DB_HOST", "FT_DB_USER", "FT_DB_PASSWORD", "FT_DB_NAME")
               if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"❌ Missing env vars: {', '.join(missing)}. See the header of this file.")
    return pymysql.connect(
        host=os.environ["FT_DB_HOST"],
        port=int(os.environ.get("FT_DB_PORT", 25060)),
        user=os.environ["FT_DB_USER"],
        passwd=os.environ["FT_DB_PASSWORD"],
        db=os.environ["FT_DB_NAME"],
        # DigitalOcean managed MySQL: add ssl={"ssl": {}} here if the server
        # rejects a plaintext connection. The project's existing code omits it.
    )


def column_ddl(name):
    if name == "date":
        return "`date` DATE NOT NULL"
    if name in TEXT_COLUMNS:
        return f"`{name}` {TEXT_COLUMNS[name]} NOT NULL"
    if name in INTEGER_COLUMNS:
        return f"`{name}` INT"
    return f"`{name}` DOUBLE"


def create_table_sql(table):
    cols = ",\n  ".join(column_ddl(c) for c in EXPECTED_COLUMNS)
    return (f"CREATE TABLE IF NOT EXISTS `{table}` (\n  {cols},\n"
            f"  UNIQUE KEY `uq_{table}_ticker_date` (`ticker`, `date`)\n)")


def upsert_sql(table):
    cols = ", ".join(f"`{c}`" for c in EXPECTED_COLUMNS)
    placeholders = ", ".join(["%s"] * len(EXPECTED_COLUMNS))
    updates = ", ".join(f"`{c}`=VALUES(`{c}`)"
                        for c in EXPECTED_COLUMNS if c not in UPSERT_KEYS)
    return f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"


def load_rows(csv_path):
    """Read CSV, validate columns, return (df, list-of-tuples) with NaN->None."""
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ CSV missing expected columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    def clean(v):
        if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            return None
        return v

    rows = [tuple(clean(v) for v in rec) for rec in df.itertuples(index=False, name=None)]
    return df, rows


def main():
    parser = argparse.ArgumentParser(description="Load ECR 2.0 CSV into first_trust_usa.ecr (MySQL).")
    parser.add_argument("--csv", help="CSV to load (default: latest output/ecr_2_0_*.csv)")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--create-table", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate CSV only; no DB connection")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        matches = sorted(glob.glob(str(PROJECT_ROOT / "output" / "ecr_2_0_*.csv")))
        if not matches:
            raise SystemExit("❌ No output/ecr_2_0_*.csv found; pass --csv.")
        csv_path = Path(matches[-1])
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV not found: {csv_path}")

    df, rows = load_rows(csv_path)
    print(f"CSV: {csv_path}")
    print(f"  {len(rows)} rows | {df['ticker'].nunique()} funds | "
          f"{df['date'].nunique()} dates ({df['date'].min()} → {df['date'].max()})")
    print(f"  target table: `{args.table}` (upsert key: {', '.join(UPSERT_KEYS)})")

    if args.dry_run:
        print("\n✅ DRY RUN — CSV valid, 27 columns present. No DB connection made.")
        return 0

    conn = connect()
    try:
        with conn.cursor() as cur:
            if args.create_table:
                cur.execute(create_table_sql(args.table))
                print(f"  ensured table `{os.environ['FT_DB_NAME']}`.`{args.table}` exists")
            cur.executemany(upsert_sql(args.table), rows)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{args.table}`")
            total = cur.fetchone()[0]
        print(f"\n✅ Upserted {len(rows)} rows. Table now holds {total} rows total.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
