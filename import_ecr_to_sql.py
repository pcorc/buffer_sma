"""
Import ECR 2.0 CSV into SQL  ->  first_trust_usa.ecr
====================================================

Loads an ecr_2_0_*.csv (produced by export_ecr_2_0.py) into a SQL table,
upserting on (ticker, date) so re-running a date range NEVER creates
duplicate rows — it updates the existing ones. Works on PostgreSQL and
MySQL.

--------------------------------------------------------------------------
CONNECTION (no credentials in code or chat)
--------------------------------------------------------------------------
Set a SQLAlchemy URL in the DATABASE_URL environment variable, e.g.

  PostgreSQL:  export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/dbname"
  MySQL:       export DATABASE_URL="mysql+pymysql://user:pass@host:3306/dbname"

GCP Cloud SQL note: either point host at the Cloud SQL Auth Proxy
(127.0.0.1) or use the Cloud SQL Python Connector. For a quick run the
Auth Proxy is simplest:
  cloud-sql-proxy <PROJECT:REGION:INSTANCE> &
  export DATABASE_URL="postgresql+psycopg2://USER:PASS@127.0.0.1:5432/DBNAME"

Dependencies:
  pip install sqlalchemy pandas
  pip install psycopg2-binary      # PostgreSQL
  pip install pymysql              # MySQL

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  # dry run: validate the CSV and show what WOULD happen (no DB needed)
  python import_ecr_to_sql.py --csv output/ecr_2_0_2026-10-02_to_2026-10-15.csv --dry-run

  # create the table if it doesn't exist, then load
  python import_ecr_to_sql.py --csv output/ecr_2_0_2026-10-02_to_2026-10-15.csv --create-table

  # load into an existing table (default schema/table: first_trust_usa.ecr)
  python import_ecr_to_sql.py --csv output/ecr_2_0_2026-10-02_to_2026-10-15.csv

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
DEFAULT_SCHEMA = "first_trust_usa"
DEFAULT_TABLE = "ecr"
UPSERT_KEYS = ["ticker", "date"]

# Expected columns (from export_ecr_2_0.py) and their SQL types.
INTEGER_COLUMNS = {"remaining_outcome_days"}
TEXT_COLUMNS = {"ticker": 16, "fund_sleeve": 64, "series": 32}
# everything else below is DATE or FLOAT
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


def build_table(metadata, schema, table):
    """Define the ecr table: date/ticker key, text metadata, float metrics."""
    from sqlalchemy import Table, Column, Date, String, Float, Integer, UniqueConstraint

    cols = []
    for name in EXPECTED_COLUMNS:
        if name == "date":
            cols.append(Column("date", Date, nullable=False))
        elif name in TEXT_COLUMNS:
            cols.append(Column(name, String(TEXT_COLUMNS[name]), nullable=False))
        elif name in INTEGER_COLUMNS:
            cols.append(Column(name, Integer))
        else:
            cols.append(Column(name, Float))
    cols.append(UniqueConstraint("ticker", "date", name=f"uq_{table}_ticker_date"))
    return Table(table, metadata, *cols, schema=schema)


def load_rows(csv_path):
    """Read the CSV, validate columns, return list-of-dict rows with NaN->None."""
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ CSV is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        print(f"⚠️  Ignoring unexpected columns: {extra}")
    df = df[EXPECTED_COLUMNS]

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["remaining_outcome_days"] = (
        df["remaining_outcome_days"].astype("Int64")  # nullable int
    )
    records = df.to_dict("records")
    # Normalize NaN/NA -> None so SQL stores NULL
    for rec in records:
        for key, val in rec.items():
            if val is None or (isinstance(val, float) and math.isnan(val)) or val is pd.NA:
                rec[key] = None
    return df, records


def upsert(engine, table, records, batch_size=500):
    """Idempotent insert-or-update on (ticker, date), dialect-aware."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.mysql import insert as mysql_insert

    dialect = engine.dialect.name
    update_cols = [c for c in EXPECTED_COLUMNS if c not in UPSERT_KEYS]
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            if dialect == "postgresql":
                stmt = pg_insert(table).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=UPSERT_KEYS,
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            elif dialect in ("mysql", "mariadb"):
                stmt = mysql_insert(table).values(batch)
                stmt = stmt.on_duplicate_key_update(
                    {c: stmt.inserted[c] for c in update_cols}
                )
            else:
                raise SystemExit(f"❌ Unsupported dialect '{dialect}'. Use PostgreSQL or MySQL.")
            conn.execute(stmt)
            total += len(batch)
    return total


def main():
    parser = argparse.ArgumentParser(description="Load ECR 2.0 CSV into first_trust_usa.ecr.")
    parser.add_argument("--csv", help="CSV to load (default: latest output/ecr_2_0_*.csv)")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--create-table", action="store_true",
                        help="create the schema/table (and unique index) if absent")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the CSV and report; do not connect or write")
    args = parser.parse_args()

    # Resolve CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        matches = sorted(glob.glob(str(PROJECT_ROOT / "output" / "ecr_2_0_*.csv")))
        if not matches:
            raise SystemExit("❌ No output/ecr_2_0_*.csv found; pass --csv.")
        csv_path = Path(matches[-1])
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV not found: {csv_path}")

    df, records = load_rows(csv_path)
    print(f"CSV: {csv_path}")
    print(f"  {len(records)} rows | {df['ticker'].nunique()} funds | "
          f"{df['date'].nunique()} dates ({df['date'].min()} → {df['date'].max()})")
    print(f"  target: {args.schema}.{args.table} (upsert key: {', '.join(UPSERT_KEYS)})")

    if args.dry_run:
        print("\n✅ DRY RUN — CSV valid, 27 columns present. No DB connection made.")
        return 0

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "❌ DATABASE_URL not set. Example:\n"
            '   export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/dbname"\n'
            '   export DATABASE_URL="mysql+pymysql://user:pass@host:3306/dbname"')

    try:
        from sqlalchemy import create_engine, MetaData, schema as sa_schema
    except ImportError:
        raise SystemExit("❌ SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary")

    engine = create_engine(url)
    metadata = MetaData()
    table = build_table(metadata, args.schema, args.table)

    if args.create_table:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(sa_schema.CreateSchema(args.schema, if_not_exists=True))
        metadata.create_all(engine, tables=[table])
        print(f"  ensured {args.schema}.{args.table} exists")

    print(f"  connected: {engine.dialect.name}")
    n = upsert(engine, table, records)
    print(f"\n✅ Upserted {n} rows into {args.schema}.{args.table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
